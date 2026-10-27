#!/usr/bin/env python3
"""
Nastran BDF -> Legacy-VTK fuer den FEM-direkt-Pfad.

    python3 tools/import_bdf.py netz.bdf --map 1=1,2=2,3=3 -o team28_fem.vtk

WOZU
----
`VtkReader::read_hexahedral_mesh` liest ein sehr enges Format, und fast jede
Abweichung fuehrt nicht zu einem Fehler, sondern zu einem falschen Netz. Dieses
Werkzeug erzeugt genau das erwartete Format und prueft vorher alles, was still
schiefgehen wuerde.

WARUM KEIN BDF-PARSER IN C++
----------------------------
Das Format hat zu viele Fallen fuer den Aufwand: Small/Large/Free Field
nebeneinander, Fortsetzungszeilen, und vor allem das Feld CP auf GRID, mit dem
Knotenkoordinaten in einem lokalen System (CORD2R, CORD2C, CORD2S) stehen
koennen. Wer das uebersieht, bekommt ein Netz, das an einer Stelle verdreht ist,
ohne dass irgendetwas abstuerzt. pyNastran loest das korrekt auf.

    pip install pyNastran

WAS GEPRUEFT WIRD
-----------------
  * Elemente zweiter Ordnung (CPENTA15, CHEXA20, CTETRA10) -> Abbruch.
    Der Loeser ist erster Ordnung. Mittenknoten wegzuwerfen waere bei
    gekruemmten Raendern (Spulenradien, Plattenrand) eine Geometrieaenderung.
  * Jacobi-Determinante in allen Elementen positiv. Negativ orientierte
    Elemente werden gemeldet und auf Wunsch (--fix-orientation) gedreht.
  * Doppelte Knoten, verwaiste Knoten.
  * Konformitaet: jede innere Flaeche gehoert genau zwei Elementen. Haengende
    Knoten erzeugen keinen Fehler, sondern ein falsches Feld.
  * Affinitaet der Prismen. Nedelec-Elemente niedrigster Ordnung reproduzieren
    ein konstantes B nur auf affin abgebildeten Elementen exakt; sonst bleibt
    ein Fehler linear in der Verzerrung (siehe tests/test_nedelec_prism.cu).
    Bei einem extrudierten Netz ist die Abbildung affin, wenn die Deckflaeche
    eine reine Parallelverschiebung der Bodenflaeche ist.
  * PID -> component_id, vollstaendig und sichtbar als Tabelle.
  * Nach dem Schreiben eine zweite, unabhaengige Pruefung der erzeugten Datei
    mit tools/meshcheck.py (deckt den Schreibschritt selbst ab; braucht
    numpy, sonst mit Hinweis uebersprungen).
  * Abmessungen (Einheitenverdacht mm gegen m).

AUSGABEFORMAT
-------------
Legacy-VTK ASCII im Layout VOR Version 5.1, also

    CELLS n_cells total_ints
    8 v0 v1 ... v7

und NICHT die neuere Form mit OFFSETS/CONNECTIVITY, die meshio inzwischen
schreibt. Genau daran scheitert VtkReader.

Geschrieben wird GENAU EIN SCALARS-Array in CELL_DATA. VtkReader nimmt das
erste und ignoriert den Namen; mehrere Zellarrays hiessen stillschweigend
falsche Materialien.
"""

import argparse
import math
import os
import sys
from collections import defaultdict

# VTK-Zelltypen
VTK_TETRA = 10
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14

# Nastran-Elementname -> (Knotenzahl, VTK-Typ)
SUPPORTED = {
    "CTETRA": (4, VTK_TETRA),
    "CPENTA": (6, VTK_WEDGE),
    "CHEXA": (8, VTK_HEXAHEDRON),
    "CPYRAM": (5, VTK_PYRAMID),
}

# Nur diese Typen kann der Loeser derzeit rechnen. CPYRAM wird gelesen und
# gemeldet, aber abgelehnt: Kantenelemente erster Ordnung auf der Pyramide
# haben rationale Basisfunktionen, und die sind nicht implementiert.
IMPLEMENTED = {"CTETRA", "CPENTA", "CHEXA"}


class Problem(Exception):
    pass


# -----------------------------------------------------------------------------
# Geometrie
# -----------------------------------------------------------------------------
def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a):
    return math.sqrt(dot(a, a))


def prism_shape_derivatives(r, s, zeta):
    """Identisch zu prism_shape_derivatives() in include/NedelecPrism.cuh."""
    L = (1.0 - r - s, r, s)
    dLr = (-1.0, 1.0, 0.0)
    dLs = (-1.0, 0.0, 1.0)
    zm, zp = 0.5 * (1.0 - zeta), 0.5 * (1.0 + zeta)
    dr = [dLr[i] * zm for i in range(3)] + [dLr[i] * zp for i in range(3)]
    ds = [dLs[i] * zm for i in range(3)] + [dLs[i] * zp for i in range(3)]
    dz = [-0.5 * L[i] for i in range(3)] + [0.5 * L[i] for i in range(3)]
    return dr, ds, dz


def det3(J):
    return (J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
            - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
            + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]))


def prism_detJ(xyz, r, s, zeta):
    dr, ds, dz = prism_shape_derivatives(r, s, zeta)
    J = [[0.0] * 3 for _ in range(3)]
    for i in range(6):
        for c in range(3):
            J[0][c] += dr[i] * xyz[i][c]
            J[1][c] += ds[i] * xyz[i][c]
            J[2][c] += dz[i] * xyz[i][c]
    return det3(J)


def hex_detJ(xyz, xi, eta, zeta):
    sx = (-1, 1, 1, -1, -1, 1, 1, -1)
    sy = (-1, -1, 1, 1, -1, -1, 1, 1)
    sz = (-1, -1, -1, -1, 1, 1, 1, 1)
    J = [[0.0] * 3 for _ in range(3)]
    for i in range(8):
        dxi = 0.125 * sx[i] * (1 + sy[i] * eta) * (1 + sz[i] * zeta)
        det_ = 0.125 * sy[i] * (1 + sx[i] * xi) * (1 + sz[i] * zeta)
        dze = 0.125 * sz[i] * (1 + sx[i] * xi) * (1 + sy[i] * eta)
        for c in range(3):
            J[0][c] += dxi * xyz[i][c]
            J[1][c] += det_ * xyz[i][c]
            J[2][c] += dze * xyz[i][c]
    return det3(J)


PRISM_QUAD = [(1 / 6, 1 / 6), (2 / 3, 1 / 6), (1 / 6, 2 / 3)]
GP = 0.5773502691896257


def tet_detJ(xyz):
    """Konstant ueber das Element: das sechsfache vorzeichenbehaftete Volumen."""
    e1 = sub(xyz[1], xyz[0])
    e2 = sub(xyz[2], xyz[0])
    e3 = sub(xyz[3], xyz[0])
    return dot(cross(e1, e2), e3)


def element_min_detJ(kind, xyz):
    """Kleinste Determinante ueber die Quadraturpunkte des Elements."""
    if kind == "CTETRA":
        return tet_detJ(xyz)
    if kind == "CPENTA":
        return min(prism_detJ(xyz, r, s, z)
                   for (r, s) in PRISM_QUAD for z in (-GP, GP))
    if kind == "CHEXA":
        return min(hex_detJ(xyz, a, b, c)
                   for a in (-GP, GP) for b in (-GP, GP) for c in (-GP, GP))
    return None


def flip(kind, nodes):
    """Orientierung umkehren, ohne die Elementform zu aendern."""
    if kind == "CTETRA":
        return [nodes[0], nodes[2], nodes[1], nodes[3]]
    if kind == "CPENTA":
        return [nodes[0], nodes[2], nodes[1], nodes[3], nodes[5], nodes[4]]
    if kind == "CHEXA":
        return [nodes[4], nodes[5], nodes[6], nodes[7],
                nodes[0], nodes[1], nodes[2], nodes[3]]
    return nodes


def prism_affinity_defect(xyz):
    """
    0 fuer ein affin abgebildetes Prisma. Affin heisst: die Deckflaeche ist
    eine reine Parallelverschiebung der Bodenflaeche, also
    x3-x0 = x4-x1 = x5-x2. Rueckgabe relativ zur Elementgroesse.
    """
    d = [sub(xyz[i + 3], xyz[i]) for i in range(3)]
    scale = max(norm(sub(xyz[1], xyz[0])), norm(sub(xyz[2], xyz[0])), norm(d[0]))
    if scale <= 0.0:
        return float("inf")
    worst = max(norm(sub(d[1], d[0])), norm(sub(d[2], d[0])), norm(sub(d[2], d[1])))
    return worst / scale


# -----------------------------------------------------------------------------
# Flaechen fuer die Konformitaetspruefung
# -----------------------------------------------------------------------------
FACES = {
    "CPENTA": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "CHEXA": [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    "CTETRA": [(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)],
    "CPYRAM": [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
}


# -----------------------------------------------------------------------------
# Einlesen
# -----------------------------------------------------------------------------
def read_bdf(path):
    try:
        from pyNastran.bdf.bdf import BDF
    except ImportError:
        raise Problem(
            "pyNastran ist nicht installiert.\n"
            "    pip install pyNastran\n"
            "Grund fuer die Abhaengigkeit: nur so werden die lokalen\n"
            "Koordinatensysteme (GRID-Feld CP, CORD2R/C/S) korrekt aufgeloest.")

    # Nicht jeder Export bringt ein brauchbares Executive- und Case-Control-Deck
    # mit. HyperMesh schreibt beides als reine Kommentarbloecke, pyNastran
    # lehnt das ab ("No executive/case control deck was defined") -- und der
    # punch-Weg scheitert seinerseits am vorhandenen BEGIN BULK.
    #
    # Dazu kommt: ein reines Geometrie-Deck hat oft gar keine PSOLID- und
    # MAT1-Karten. Fuer Nastran ist das ein Querverweisfehler, fuer uns nicht --
    # wir brauchen Knoten, Elemente und PIDs, keine Werkstoffdaten. Deshalb
    # notfalls ohne Querverweise lesen und die Koordinatensysteme selbst
    # aufloesen.
    attempts = []

    def try_read(**kwargs):
        m = BDF(debug=None)
        m.read_bdf(**kwargs)
        return m

    bulk_path = _write_bulk_section(path)
    sources = [(path, "Datei")]
    if bulk_path:
        sources.append((bulk_path, "Bulk-Abschnitt"))

    try:
        for src, label in sources:
            for punch in (False, True):
                for xref in (True, False):
                    try:
                        model = try_read(bdf_filename=src, xref=xref, punch=punch)
                    except Exception as exc:
                        attempts.append("%s, punch=%s, xref=%s: %s"
                                        % (label, punch, xref,
                                           str(exc).splitlines()[0]))
                        continue
                    if not xref:
                        attempts.append(
                            "%s, punch=%s: ohne Querverweise gelesen "
                            "(fehlende PSOLID/MAT1 sind fuer die Geometrie "
                            "belanglos)" % (label, punch))
                        print("  Hinweis: %s" % attempts[-1])
                    return _collect(model)
    finally:
        if bulk_path:
            import os
            os.unlink(bulk_path)

    raise Problem("pyNastran kann %s nicht lesen. Versuche:\n    %s"
                  % (path, "\n    ".join(attempts)))


def _write_bulk_section(path):
    """
    Alles zwischen BEGIN BULK und ENDDATA in eine temporaere .pch-Datei.
    Zeilenenden werden dabei auf Unix normiert: HyperMesh schreibt CRLF, und
    ein Wagenruecklauf am Feldende verschiebt die Spaltenaufteilung im
    Small-Field-Format.
    """
    import os
    import tempfile
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None

    start = None
    for i, ln in enumerate(lines):
        if ln.strip().upper().startswith("BEGIN BULK"):
            start = i + 1
            break
    if start is None:
        return None

    out = []
    for ln in lines[start:]:
        if ln.strip().upper().startswith("ENDDATA"):
            break
        out.append(ln.rstrip("\r"))

    fd, tmp = tempfile.mkstemp(suffix=".pch")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(out) + "\n")
    return tmp


def _node_position(model, node):
    """
    Knotenkoordinate im Basissystem. Mit Querverweisen erledigt das
    get_position(); ohne sie muss das Feld CP von Hand aufgeloest werden --
    genau der Fall, der sonst ein verdrehtes Netz ohne Fehlermeldung ergibt.
    """
    try:
        return tuple(float(v) for v in node.get_position())
    except Exception:
        pass

    cp = node.cp if isinstance(node.cp, int) else getattr(node.cp, "cid", 0)
    xyz = tuple(float(v) for v in node.xyz)
    if not cp:
        return xyz
    coord = model.coords.get(cp)
    if coord is None:
        raise Problem("Knoten %s verweist auf das Koordinatensystem %s, das im "
                      "Deck fehlt. Die Koordinate laesst sich nicht aufloesen."
                      % (node.nid, cp))
    return tuple(float(v) for v in coord.transform_node_to_global(xyz))


def _collect(model):
    nodes = {nid: _node_position(model, node)
             for nid, node in model.nodes.items()}

    elements = []      # (kind, pid, [node ids])
    shells = []        # (kind, pid, [node ids])
    rejected = defaultdict(int)

    for eid, elem in sorted(model.elements.items()):
        kind = elem.type
        raw = list(elem.node_ids)

        if kind in ("CTRIA3", "CQUAD4", "CTRIA6", "CQUAD8"):
            shells.append((kind, elem.Pid(), [n for n in raw if n is not None]))
            continue

        if kind not in SUPPORTED:
            rejected[kind] += 1
            continue

        # Die ROHE Feldzahl entscheidet, nicht die Zahl der besetzten Felder.
        # Ein CPENTA15 mit teilweise leeren Mittenknoten haette sonst als
        # penta6 durchrutschen koennen -- genau die stille Fehlinterpretation,
        # die dieses Werkzeug verhindern soll.
        expected, _ = SUPPORTED[kind]
        if len(raw) != expected:
            rejected["%s mit %d Knotenfeldern (Ordnung 2)" % (kind, len(raw))] += 1
            continue
        if any(n is None for n in raw):
            rejected["%s mit leeren Knotenfeldern" % kind] += 1
            continue

        elements.append((kind, elem.Pid(), raw))

    return nodes, elements, shells, rejected


# -----------------------------------------------------------------------------
# Pruefungen
# -----------------------------------------------------------------------------
def check_mesh(nodes, elements, args, log):
    fatal = []

    # --- Elementbestand -------------------------------------------------------
    inventory = defaultdict(int)
    for kind, _, _ in elements:
        inventory[kind] += 1
    log("Elementbestand:")
    for kind in sorted(inventory):
        mark = "" if kind in IMPLEMENTED else "   <- NICHT IMPLEMENTIERT"
        log("  %-8s %8d%s" % (kind, inventory[kind], mark))
    not_impl = sorted(set(inventory) - IMPLEMENTED)
    if not_impl:
        fatal.append(
            "Elementtypen ohne Kantenelement im Loeser: %s. "
            "CTETRA, CPENTA und CHEXA sind implementiert; fuer CPYRAM "
            "braeuchte es rationale Basisfunktionen."
            % ", ".join(not_impl))

    # --- Abmessungen ----------------------------------------------------------
    used = sorted({n for _, _, nl in elements for n in nl})
    if not used:
        raise Problem("Keine verwertbaren Volumenelemente gefunden.")
    xs = [nodes[n] for n in used]
    lo = [min(p[c] for p in xs) for c in range(3)]
    hi = [max(p[c] for p in xs) for c in range(3)]
    span = max(hi[c] - lo[c] for c in range(3))
    log("\nAbmessungen: x [%.4g, %.4g]  y [%.4g, %.4g]  z [%.4g, %.4g]"
        % (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))
    if span < 10.0:
        log("  WARNUNG: groesste Ausdehnung %.4g. Das TEAM-28-Netz ist in mm "
            "angegeben; sieht nach Metern aus." % span)

    # --- verwaiste und doppelte Knoten ---------------------------------------
    orphans = len(nodes) - len(used)
    if orphans:
        log("\n%d verwaiste Knoten (in keinem Volumenelement), werden verworfen."
            % orphans)

    tol = args.merge_tol
    buckets = defaultdict(list)
    for n in used:
        p = nodes[n]
        key = (round(p[0] / tol), round(p[1] / tol), round(p[2] / tol))
        buckets[key].append(n)
    dupes = [v for v in buckets.values() if len(v) > 1]
    if dupes:
        fatal.append("%d Knotenpaare liegen naeher als %g beieinander. "
                     "Doppelte Knoten trennen das Netz, ohne dass es auffaellt."
                     % (len(dupes), tol))

    # --- Orientierung ---------------------------------------------------------
    negative, degenerate = [], []
    for idx, (kind, pid, nl) in enumerate(elements):
        xyz = [nodes[n] for n in nl]
        d = element_min_detJ(kind, xyz)
        if d is None:
            continue
        if abs(d) < 1e-20:
            degenerate.append(idx)
        elif d < 0.0:
            negative.append(idx)

    if degenerate:
        fatal.append("%d entartete Elemente (Jacobi-Determinante ~ 0)."
                     % len(degenerate))
    if negative:
        if args.fix_orientation:
            for idx in negative:
                kind, pid, nl = elements[idx]
                elements[idx] = (kind, pid, flip(kind, nl))
            log("\n%d negativ orientierte Elemente gedreht (--fix-orientation)."
                % len(negative))
            still = [i for i in negative
                     if element_min_detJ(elements[i][0],
                                         [nodes[n] for n in elements[i][2]]) < 0]
            if still:
                fatal.append("%d Elemente bleiben nach dem Drehen negativ. "
                             "Das ist keine Frage der Knotenreihenfolge, "
                             "sondern eine verschraenkte Geometrie." % len(still))
        else:
            fatal.append("%d negativ orientierte Elemente. Mit "
                         "--fix-orientation drehen." % len(negative))

    # --- Konformitaet ---------------------------------------------------------
    face_count = defaultdict(int)
    for kind, _, nl in elements:
        for f in FACES.get(kind, []):
            key = tuple(sorted(nl[i] for i in f))
            face_count[key] += 1
    boundary = sum(1 for c in face_count.values() if c == 1)
    interior = sum(1 for c in face_count.values() if c == 2)
    broken = sum(1 for c in face_count.values() if c > 2)
    log("\nFlaechen: %d innen (genau 2 Nachbarn), %d am Rand (1 Nachbar)"
        % (interior, boundary))
    if broken:
        fatal.append("%d Flaechen mit mehr als zwei Nachbarelementen. "
                     "Das Netz ist nicht konform." % broken)
    if boundary == 0:
        fatal.append("Kein einziger Rand gefunden. Fuer ein abgeschnittenes "
                     "Luftgebiet muss es einen aeusseren Rand geben.")

    # --- Affinitaet der Prismen ----------------------------------------------
    n_tet = sum(1 for kind, _, _ in elements if kind == "CTETRA")
    if n_tet:
        log("\nTetraeder sind immer affin abgebildet: auf ihnen wird ein "
            "konstantes B\nexakt reproduziert, unabhaengig von der "
            "Elementform. Schlecht geformte\nTetraeder verschlechtern die "
            "Kondition, nicht die Konsistenz.")

        # Formguete, normiert auf den regelmaessigen Tetraeder (V/l^3 =
        # 1/(6*sqrt(2))). 1.0 ist ideal, 0 ist entartet. Fuer einen
        # Direktloeser ist die Kondition zweitrangig, fuer den
        # Interpolationsfehler nicht.
        ideal = 1.0 / (6.0 * math.sqrt(2.0))
        q = []
        for kind, _, nl in elements:
            if kind != "CTETRA":
                continue
            xyz = [nodes[n] for n in nl]
            lmax = max(norm(sub(xyz[a], xyz[b]))
                       for a in range(4) for b in range(a + 1, 4))
            if lmax > 0.0:
                q.append(abs(tet_detJ(xyz) / 6.0) / (lmax ** 3) / ideal)
        if q:
            q.sort()
            m = len(q)
            log("Formguete (1 = regelmaessig, 0 = entartet): "
                "min %.3f  5%% %.3f  Median %.3f"
                % (q[0], q[int(0.05 * (m - 1))], q[m // 2]))
            bad = sum(1 for v in q if v < 0.05)
            if bad:
                log("  %d Tetraeder unter 0.05 (%.1f %%)" % (bad, 100.0 * bad / m))

    defects = [prism_affinity_defect([nodes[n] for n in nl])
               for kind, _, nl in elements if kind == "CPENTA"]
    if defects:
        defects.sort()
        n = len(defects)
        worst = defects[-1]
        over = sum(1 for d in defects if d > args.affinity_tol)
        log("\nAffinitaet der Prismen (0 = affin, Fehler in B ist linear darin):")
        log("  Median %8.2e   95%% %8.2e   groesster %8.2e"
            % (defects[n // 2], defects[int(0.95 * (n - 1))], worst))
        log("  %d von %d ueber der Marke %g (%.1f %%)"
            % (over, n, args.affinity_tol, 100.0 * over / n))
        if over:
            log("  Hinweis: auf diesen Elementen wird ein konstantes B nicht")
            log("  exakt reproduziert. Das ist zulaessig und konvergiert, aber")
            log("  ein extrudiertes Netz mit konstanter Extrusionsrichtung je")
            log("  Element waere hier exakt. Siehe tests/test_nedelec_prism.cu.")

    return fatal


# -----------------------------------------------------------------------------
# Materialzuordnung
# -----------------------------------------------------------------------------
def build_component_map(elements, args, log):
    mapping = {}
    if args.map:
        for item in args.map.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise Problem("--map erwartet PID=ID, bekommen: %r" % item)
            k, v = item.split("=", 1)
            mapping[int(k)] = int(v)

    pids = defaultdict(int)
    for _, pid, _ in elements:
        pids[pid] += 1

    log("\nPID -> component_id:")
    unmapped = []
    for pid in sorted(pids):
        cid = mapping.get(pid, args.default_id)
        note = "" if pid in mapping else "   (nicht in --map, Vorgabe)"
        log("  PID %-6d -> %-3d  %8d Elemente%s" % (pid, cid, pids[pid], note))
        if pid not in mapping:
            unmapped.append(pid)

    if unmapped and not args.allow_default:
        raise Problem(
            "Nicht zugeordnete PIDs: %s. Ohne Zuordnung landen sie auf %d "
            "(Luft). Wenn das gewollt ist: --allow-default."
            % (", ".join(str(p) for p in unmapped), args.default_id))

    log("\n  Erinnerung: 3 = Platte, 2 = Innenspule, 1 = Aussenspule, 0 = Luft.")
    return mapping


# -----------------------------------------------------------------------------
# Schreiben
# -----------------------------------------------------------------------------
def write_vtk(path, nodes, elements, mapping, args):
    used = sorted({n for _, _, nl in elements for n in nl})
    renum = {nid: i for i, nid in enumerate(used)}

    total_ints = sum(1 + len(nl) for _, _, nl in elements)

    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEMBEM FEM-direkt, erzeugt von tools/import_bdf.py\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        f.write("POINTS %d double\n" % len(used))
        for nid in used:
            p = nodes[nid]
            f.write("%.17g %.17g %.17g\n" % p)

        # Layout VOR VTK 5.1. Genau das erwartet VtkReader.
        f.write("\nCELLS %d %d\n" % (len(elements), total_ints))
        for kind, _, nl in elements:
            f.write("%d %s\n" % (len(nl), " ".join(str(renum[n]) for n in nl)))

        f.write("\nCELL_TYPES %d\n" % len(elements))
        for kind, _, _ in elements:
            f.write("%d\n" % SUPPORTED[kind][1])

        # GENAU EIN SCALARS-Array. VtkReader nimmt das erste und ignoriert
        # den Namen; ein zweites Array hiesse stillschweigend falsche
        # Materialien.
        f.write("\nCELL_DATA %d\n" % len(elements))
        f.write("SCALARS component_id int 1\nLOOKUP_TABLE default\n")
        for _, pid, _ in elements:
            f.write("%d\n" % mapping.get(pid, args.default_id))

    return len(used), len(elements)


def write_surface_vtk(path, nodes, shells, used_nodes):
    """
    Schalenelemente als eigene Datei. Sie tragen keine Freiheitsgrade, sind aber
    die zuverlaessigste Quelle fuer den Dirichlet-Rand: wenn HyperMesh damit die
    Aussenflaeche markiert, muss der Loeser den Rand nicht aus der Topologie
    ableiten.
    """
    keep = [(k, pid, nl) for k, pid, nl in shells
            if all(n in used_nodes for n in nl) and k in ("CTRIA3", "CQUAD4")]
    if not keep:
        return 0

    renum = {nid: i for i, nid in enumerate(sorted(used_nodes))}
    total = sum(1 + len(nl) for _, _, nl in keep)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEMBEM Randflaechen (Dirichlet-Kandidaten)\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write("POINTS %d double\n" % len(renum))
        for nid in sorted(used_nodes):
            f.write("%.17g %.17g %.17g\n" % nodes[nid])
        f.write("\nCELLS %d %d\n" % (len(keep), total))
        for _, _, nl in keep:
            f.write("%d %s\n" % (len(nl), " ".join(str(renum[n]) for n in nl)))
        f.write("\nCELL_TYPES %d\n" % len(keep))
        for k, _, _ in keep:
            f.write("%d\n" % (5 if k == "CTRIA3" else 9))
        f.write("\nCELL_DATA %d\n" % len(keep))
        f.write("SCALARS shell_pid int 1\nLOOKUP_TABLE default\n")
        for _, pid, _ in keep:
            f.write("%d\n" % pid)
    return len(keep)


# -----------------------------------------------------------------------------
def verify_written(path, log):
    """
    Zweite, unabhaengige Pruefung der GESCHRIEBENEN Datei mit
    tools/meshcheck.py. check_mesh() prueft die Elemente vor dem Schreiben;
    diese Pruefung deckt den Schreibschritt selbst ab (Knotenreihenfolge,
    Nummerierung, Zelltypen, component_id), so wie VtkReader ihn liest.
    Braucht numpy; ohne numpy wird sie mit Hinweis uebersprungen.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import meshcheck
    except ImportError as exc:
        log("\nZweite Pruefung uebersprungen (%s). Mit numpy laeuft sie mit." % exc)
        return True
    log("\nZweite Pruefung der geschriebenen Datei (tools/meshcheck.py):")
    ok = meshcheck.print_results(meshcheck.run_checks(meshcheck.read_vtk(path)))
    for cid, vol in sorted(meshcheck.volumes(meshcheck.read_vtk(path)).items()):
        log("  Volumen component_id %d: %.6g" % (cid, vol))
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Nastran BDF -> Legacy-VTK fuer den FEM-direkt-Pfad",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bdf")
    ap.add_argument("-o", "--out", default=None,
                    help="Ausgabedatei (Vorgabe: <bdf-Name>.vtk)")
    ap.add_argument("--map", default="",
                    help="PID-Zuordnung, z. B. 1=3,2=2,3=1 "
                         "(3 Platte, 2 Innenspule, 1 Aussenspule, 0 Luft)")
    ap.add_argument("--default-id", type=int, default=0,
                    help="component_id fuer nicht zugeordnete PIDs (Vorgabe 0 = Luft)")
    ap.add_argument("--allow-default", action="store_true",
                    help="nicht zugeordnete PIDs zulassen statt abzubrechen")
    ap.add_argument("--fix-orientation", action="store_true",
                    help="negativ orientierte Elemente drehen")
    ap.add_argument("--merge-tol", type=float, default=1e-9,
                    help="Abstand, ab dem zwei Knoten als doppelt gelten")
    ap.add_argument("--affinity-tol", type=float, default=1e-10,
                    help="Marke fuer die Affinitaetsstatistik der Prismen")
    ap.add_argument("--surface-out", default=None,
                    help="Schalenelemente zusaetzlich hierhin schreiben")
    ap.add_argument("--force", action="store_true",
                    help="trotz Beanstandungen schreiben")
    args = ap.parse_args(argv)

    out = args.out or (args.bdf.rsplit(".", 1)[0] + ".vtk")

    def log(msg=""):
        print(msg)

    log("BDF: %s" % args.bdf)
    nodes, elements, shells, rejected = read_bdf(args.bdf)
    log("%d Knoten, %d Volumenelemente, %d Schalenelemente"
        % (len(nodes), len(elements), len(shells)))

    if rejected:
        log("\nVerworfen:")
        for k in sorted(rejected):
            log("  %-40s %6d" % (k, rejected[k]))
        second_order = [k for k in rejected if "Ordnung 2" in k]
        if second_order:
            log("\n  Elemente zweiter Ordnung koennen NICHT auf erste Ordnung")
            log("  heruntergesetzt werden: bei gekruemmten Raendern (Spulen-")
            log("  radien 30/51/84/91, Plattenrand 63) waere das Wegwerfen der")
            log("  Mittenknoten eine Geometrieaenderung. Aus HyperMesh direkt")
            log("  penta6 exportieren.")

    if not elements:
        raise Problem("Keine verwertbaren Volumenelemente uebrig.")

    fatal = check_mesh(nodes, elements, args, log)
    mapping = build_component_map(elements, args, log)

    if fatal:
        log("\nBEANSTANDUNGEN:")
        for f in fatal:
            log("  * %s" % f)
        if not args.force:
            log("\nNichts geschrieben. Mit --force trotzdem schreiben.")
            return 1
        log("\n--force: wird trotzdem geschrieben.")

    n_nodes, n_cells = write_vtk(out, nodes, elements, mapping, args)
    log("\nGeschrieben: %s  (%d Knoten, %d Zellen)" % (out, n_nodes, n_cells))

    if not verify_written(out, log):
        if not args.force:
            log("\nDie geschriebene Datei besteht die zweite Pruefung NICHT -- nicht")
            log("verwenden. Das deutet auf einen Fehler beim Schreiben hin, nicht im")
            log("BDF (das hat check_mesh() oben bereits geprueft).")
            return 1
        log("\n--force: zweite Pruefung beanstandet, Datei bleibt trotzdem stehen.")

    if args.surface_out:
        used = {n for _, _, nl in elements for n in nl}
        n_sh = write_surface_vtk(args.surface_out, nodes, shells, used)
        log("Geschrieben: %s  (%d Randflaechen)" % (args.surface_out, n_sh))

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Problem as exc:
        print("\nFEHLER: %s" % exc, file=sys.stderr)
        sys.exit(2)
