#!/usr/bin/env python3
"""
Eigene Gmsh-Netze (.msh) oder Geometrien (.geo) fuer FEMBEM importieren.

    python3 tools/gmsh/import_msh.py netz.msh -o netz.vtk
    python3 tools/gmsh/import_msh.py modell.geo -o netz.vtk          # wird 3D vernetzt
    python3 tools/gmsh/import_msh.py netz.msh --map Ring_A=1 --map Ring_B=2 --map Scheibe=3
    python3 tools/gmsh/import_msh.py netz.msh --bem --check-map      # BEM-FEM, TEAM 28

ZUORDNUNG Physical Group -> component_id
  Massgeblich ist der NAME der Volumen-Gruppe (Physical Volume), nicht ihre
  Nummer: Die Vorzeichen der Spulenstroeme haengen an der component_id, und
  vertauschte Spulen rechnen ohne Fehlermeldung falsch. Eine Nummer sagt
  nichts darueber, was gemeint war; ein Name schon. Vorgabe (Gross-/
  Kleinschreibung, "-", "_" und Leerzeichen egal):

    1 Aussenspule  outer_coil, coil_outer, aussenspule, outer
    2 Innenspule   inner_coil, coil_inner, innenspule, inner
    3 Platte       plate, platte
    0 Luft         air, luft

  team28_hex.py und team28_fem_air.py benutzen solche Namen. --map Name=ID
  (wiederholbar) ergaenzt oder ueberschreibt die Vorgabe; --no-default-names
  laesst nur --map gelten. Eine Gruppe ohne Namen wird ueber ihre Nummer
  angesprochen (--map 7=3).

ABGELEHNT (Meldung, Exit-Code 1) -- lieber kein Netz als ein still falsches
  * keine Volumen-Gruppe, Gruppe mit unbekanntem Namen, Name aus --map nicht
    in der Datei, Gruppe ohne Elemente, verlangte component_id fehlt
    (--require, Vorgabe 1,2,3);
  * ein Volumen in zwei Gruppen mit verschiedener component_id;
  * Volumenelemente ohne Gruppe (sie fehlten sonst ohne Meldung im Netz --
    typisch die vergessene Luft, siehe "Risiken" im Plan);
  * Elemente zweiter Ordnung (der Loeser ist erster Ordnung; die
    Mittenknoten wegzulassen aenderte die Geometrie an gekruemmten Raendern);
  * andere Zelltypen als Tetraeder, Hexaeder, Prismen (Pyramiden brauchen
    rationale Formfunktionen und sind im Loeser bewusst nicht vorgesehen).

WAS DAS WERKZEUG AENDERT
  * Orientierung: jede Zelle mit negativer Determinante wird in die
    Konvention des Loesers umsortiert (solver_vtk.py, wie team28_fem_air.py).
    Gmsh schreibt Prismen umgekehrt orientiert (gmsh-meshing-plan.md).
  * Knoten ohne Volumenzelle entfallen, die uebrigen werden fortlaufend
    nummeriert. Knoten werden NICHT verschmolzen: das vereinte Knoten
    verschiedener Koerper, sobald sie zufaellig zusammenfallen, und
    verbaende Platte und Spule leitend. Doppelte Knoten meldet die Pruefung.

PRUEFUNG NACH DEM SCHREIBEN (tools/meshcheck.py; Fehler -> Exit-Code 1)
  Immer: Zelltypen, doppelte/verwaiste Knoten, Konformitaet, Orientierung.
  --bem: nur Hexaeder, Koerper getrennt (BEM-FEM-Pfad).
  --fem-direct: Luft umschliesst alles, Prismen affin (FEM-direkt-Pfad).
  --check-map: zusaetzlich tools/check_map.py (nur fuer TEAM-28-Geometrie).

Einheiten werden nicht umgerechnet: Der Loeser erwartet Millimeter.
"""
import argparse
import contextlib
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

NAMES = {0: "Luft", 1: "Aussenspule", 2: "Innenspule", 3: "Platte"}
DEFAULT_NAMES = {
    1: ["outer_coil", "coil_outer", "aussenspule", "außenspule", "outer"],
    2: ["inner_coil", "coil_inner", "innenspule", "inner"],
    3: ["plate", "platte"],
    0: ["air", "luft"],
}
# Gmsh-Elementtypen erster Ordnung, die der Loeser kennt -> VTK-Zelltyp. Die
# Knotenreihenfolge ist bei diesen drei Typen in Gmsh und VTK dieselbe; nur
# die Orientierung kann abweichen (wird danach je Zelle korrigiert).
GMSH_TO_VTK = {4: 10, 5: 12, 6: 13}       # Tetraeder, Hexaeder, Prisma


class MshImportError(Exception):
    """Netz oder Zuordnung unbrauchbar; die Meldung sagt, was zu tun ist."""


def norm(name):
    """Name vergleichbar machen: Klein, ohne "-", "_", Leerzeichen."""
    return re.sub(r"[\s_\-]+", "", name.strip().lower())


def parse_map(items, use_defaults=True):
    """--map-Eintraege und Vorgabe zu {normierter Name: component_id}."""
    table = {}
    if use_defaults:
        for cid, names in DEFAULT_NAMES.items():
            for n in names:
                table[norm(n)] = cid
    explicit = {}
    for item in items:
        name, sep, value = item.rpartition("=")
        if not sep or not name.strip():
            raise MshImportError(f"--map {item!r}: erwartet Name=ID, z. B. --map Scheibe=3")
        try:
            cid = int(value)
        except ValueError:
            raise MshImportError(f"--map {item!r}: ID {value!r} ist keine ganze Zahl") from None
        if cid not in NAMES:
            raise MshImportError(f"--map {item!r}: component_id {cid} gibt es nicht "
                                 "(0 Luft, 1 Aussenspule, 2 Innenspule, 3 Platte)")
        table[norm(name)] = cid
        explicit[norm(name)] = name.strip()
    return table, explicit


def load(path, gmsh):
    """Datei in Gmsh laden; .geo wird dreidimensional vernetzt."""
    if not os.path.isfile(path):
        raise MshImportError(f"{path}: Datei nicht gefunden")
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".msh", ".geo"):
        raise MshImportError(f"{path}: erwartet .msh oder .geo")
    gmsh.open(path)
    if ext == ".geo":
        # Die Elementordnung bewusst NICHT ueberschreiben: Setzt die .geo
        # zweite Ordnung, soll das als Fehler sichtbar werden, statt dass hier
        # still ein anderes Netz entsteht als das, was die Datei beschreibt.
        gmsh.model.mesh.generate(3)


def collect(gmsh, table, explicit):
    """Volumenelemente je Gruppe holen; liefert (Knoten, Bloecke, Gruppenbericht).

    Bloecke: Liste von (vtk_type, conn mit Gmsh-Knotenindizes, component_id).
    """
    import numpy as np

    groups = gmsh.model.getPhysicalGroups(3)
    if not groups:
        raise MshImportError("keine Volumen-Gruppe (Physical Volume) in der Datei. "
                             "Jedes Volumen braucht eine benannte Gruppe, z. B. "
                             'Physical Volume("plate") = {...};')

    entity_cid, entity_group, report, unknown = {}, {}, [], []
    seen = set()
    for _, tag in groups:
        name = gmsh.model.getPhysicalName(3, tag)
        key = norm(name) if name else str(tag)
        label = name if name else f"(ohne Namen, Nummer {tag})"
        seen.add(key)
        if key not in table:
            unknown.append(label)
            continue
        cid = table[key]
        for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
            if ent in entity_cid and entity_cid[ent] != cid:
                raise MshImportError(
                    f"Volumen {ent} liegt in den Gruppen {entity_group[ent]!r} "
                    f"(component_id {entity_cid[ent]}) und {label!r} (component_id {cid})")
            entity_cid[ent] = cid
            entity_group[ent] = label
        report.append([label, tag, cid, 0])
    if unknown:
        raise MshImportError(
            "Gruppe(n) ohne Zuordnung: " + ", ".join(repr(u) for u in unknown) +
            ". Mit --map Name=ID zuordnen (0 Luft, 1 Aussenspule, 2 Innenspule, "
            "3 Platte); ohne Namen ueber die Nummer, z. B. --map 7=3.")
    missing = [orig for key, orig in explicit.items() if key not in seen]
    if missing:
        raise MshImportError("Gruppe(n) aus --map nicht in der Datei: " +
                             ", ".join(repr(m) for m in missing) +
                             ". Vorhanden: " + ", ".join(
                                 repr(gmsh.model.getPhysicalName(3, t) or str(t))
                                 for _, t in groups))

    ntags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    pos = np.full(int(ntags.max()) + 1 if len(ntags) else 1, -1, dtype=np.int64)
    pos[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))

    blocks, ungrouped, per_group = [], {}, {}
    for _, ent in gmsh.model.getEntities(3):
        types, _, nodes = gmsh.model.mesh.getElements(3, ent)
        for t, nn in zip(types, nodes):
            t = int(t)
            name, _, order, n_nodes, _, _ = gmsh.model.mesh.getElementProperties(t)
            count = len(nn) // n_nodes
            if count == 0:
                continue
            if ent not in entity_cid:
                ungrouped[ent] = ungrouped.get(ent, 0) + count
                continue
            if order != 1:
                raise MshImportError(
                    f"{count} Elemente '{name}' (Ordnung {order}) in Volumen {ent}: "
                    "der Loeser ist erster Ordnung. Mit Mesh.ElementOrder = 1 neu vernetzen.")
            if t not in GMSH_TO_VTK:
                raise MshImportError(
                    f"{count} Elemente '{name}' in Volumen {ent}: unterstuetzt sind nur "
                    "Tetraeder, Hexaeder und Prismen erster Ordnung.")
            conn = pos[np.asarray(nn, dtype=np.int64)].reshape(-1, n_nodes)
            blocks.append((GMSH_TO_VTK[t], conn, entity_cid[ent]))
            per_group[entity_group[ent]] = per_group.get(entity_group[ent], 0) + count
    if ungrouped:
        n = sum(ungrouped.values())
        raise MshImportError(
            f"{n} Volumenelemente ohne Gruppe (Volumen {sorted(ungrouped)}). Sie fehlten im "
            "Netz ohne Meldung -- typisch die vergessene Luft. Jedes Volumen einer "
            "Gruppe zuordnen.")
    for row in report:
        row[3] = per_group.get(row[0], 0)
    empty = [r[0] for r in report if r[3] == 0]
    if empty:
        raise MshImportError("Gruppe(n) ohne Volumenelemente: " +
                             ", ".join(repr(e) for e in empty) + " (nicht vernetzt?)")
    return xyz, blocks, report


def convert(path, out, table, explicit, require):
    """Laden, zuordnen, drehen, schreiben. Liefert eine Zusammenfassung."""
    import gmsh
    import numpy as np
    import solver_vtk

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        load(path, gmsh)
        xyz, blocks, report = collect(gmsh, table, explicit)
    finally:
        gmsh.finalize()

    present = {cid for _, _, cid in blocks}
    lacking = [c for c in require if c not in present]
    if lacking:
        raise MshImportError("verlangte component_id fehlt: " +
                             ", ".join(f"{c} ({NAMES[c]})" for c in lacking) +
                             ". Gruppe anlegen oder --require anpassen.")

    # Bloecke gleichen Typs zusammenfassen, damit jeder Typ einmal gedreht
    # wird; die Massstabsschwelle fuer "entartet" gilt dann je Typ.
    merged = {}
    for t, conn, cid in blocks:
        merged.setdefault(t, ([], []))
        merged[t][0].append(conn)
        merged[t][1].append(np.full(len(conn), cid, dtype=np.int64))
    out_blocks, flipped = [], {}
    for t in sorted(merged):
        conn = np.vstack(merged[t][0])
        try:
            conn, n = solver_vtk.orient_to_solver(xyz, t, conn)
        except ValueError as e:
            raise MshImportError(str(e)) from None
        flipped[t] = (n, len(conn))
        out_blocks.append((t, conn, np.concatenate(merged[t][1])))
    points, conns = solver_vtk.compact_nodes(xyz, [b[1] for b in out_blocks])
    out_blocks = [(t, c, comp) for (t, _, comp), c in zip(out_blocks, conns)]
    solver_vtk.write_legacy_vtk(
        out, points, out_blocks,
        f"FEMBEM, importiert von tools/gmsh/import_msh.py aus {os.path.basename(path)}")
    return dict(n_nodes=len(points), n_all_nodes=len(xyz), report=report, flipped=flipped,
                gmsh=gmsh.__version__)


def check(out, a):
    import meshcheck as mc
    mesh = mc.read_vtk(out)
    res = mc.run_checks(mesh, require_hex=a.bem, separate_bodies=a.bem,
                        air_enclosed=a.fem_direct, prism_affinity=a.fem_direct)
    log = ""
    if a.check_map:
        import check_map
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = check_map.main(out)
        res.append(mc.Result("check_map (Zuordnung, Masse, Vollscheibe)", rc == 0,
                             "besteht" if rc == 0 else "siehe Ausgabe"))
        log = buf.getvalue() if rc != 0 else ""
    # stream ausdruecklich: die Vorgabe in meshcheck ist beim Import gebunden
    # und liefe an einer Umleitung von sys.stdout (Tests) vorbei
    ok = mc.print_results(res, stream=sys.stdout)
    for cid, v in sorted(mc.volumes(mesh).items()):
        print(f"  Volumen {NAMES.get(cid, cid)} (component_id {cid}): {v:.6g} mm^3")
    if log:
        print(log)
    return ok


def parse_require(text):
    if not text.strip():
        return []
    try:
        ids = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise MshImportError(f"--require {text!r}: erwartet z. B. 1,2,3") from None
    bad = [i for i in ids if i not in NAMES]
    if bad:
        raise MshImportError(f"--require: unbekannte component_id {bad}")
    return ids


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Gmsh-Netz (.msh) oder -Geometrie (.geo) nach Legacy-VTK fuer FEMBEM; "
                    "Physical Volumes werden ueber ihren Namen auf component_id abgebildet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Vorgabe-Namen: 1 outer_coil/aussenspule/outer, 2 inner_coil/innenspule/inner, "
               "3 plate/platte, 0 air/luft (Gross-/Kleinschreibung, -, _ egal).")
    ap.add_argument("input", help=".msh oder .geo")
    ap.add_argument("-o", "--out", default=None, help="Ausgabedatei (Vorgabe: <Eingabe>.vtk)")
    ap.add_argument("--map", action="append", default=[], metavar="NAME=ID",
                    help="Gruppe NAME auf component_id ID abbilden (wiederholbar)")
    ap.add_argument("--no-default-names", action="store_true",
                    help="nur Zuordnungen aus --map verwenden")
    ap.add_argument("--require", default="1,2,3",
                    help="component_ids, die vorkommen muessen (leer: keine)")
    ap.add_argument("--bem", action="store_true",
                    help="BEM-FEM-Pfad: nur Hexaeder, Koerper getrennt")
    ap.add_argument("--fem-direct", action="store_true",
                    help="FEM-direkt-Pfad: Luft umschliesst alles, Prismen affin")
    ap.add_argument("--check-map", action="store_true",
                    help="zusaetzlich tools/check_map.py (TEAM-28-Geometrie)")
    a = ap.parse_args(argv)
    if a.bem and a.fem_direct:
        ap.error("--bem und --fem-direct schliessen sich aus")
    out = a.out or os.path.splitext(a.input)[0] + ".vtk"

    try:
        import gmsh  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        print(f"FEHLER: {e}. pip install -r tools/requirements-gmsh.txt; ohne Grafik\n"
              "  zusaetzlich die GL-Bibliotheken, siehe docs/meshing.md.", file=sys.stderr)
        return 1

    try:
        table, explicit = parse_map(a.map, not a.no_default_names)
        require = parse_require(a.require)
        info = convert(a.input, out, table, explicit, require)
    except MshImportError as e:
        print(f"FEHLER: {e}", file=sys.stderr)
        return 1

    n_cells = sum(r[3] for r in info["report"])
    print(f"{out}: {n_cells} Zellen, {info['n_nodes']} Knoten "
          f"({info['n_all_nodes'] - info['n_nodes']} ohne Volumenzelle entfernt), "
          f"Gmsh {info['gmsh']}")
    for label, tag, cid, count in info["report"]:
        print(f"  Gruppe {label!r} (Nummer {tag}) -> component_id {cid} {NAMES[cid]}: {count} Zellen")
    tn = {10: "Tetraeder", 12: "Hexaeder", 13: "Prismen"}
    print("  in die Loeser-Konvention gedreht: " +
          ", ".join(f"{tn[t]} {n} von {m}" for t, (n, m) in info["flipped"].items()))
    ok = check(out, a)
    print("Netz in Ordnung." if ok else "NETZ FEHLERHAFT -- nicht verwenden.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
