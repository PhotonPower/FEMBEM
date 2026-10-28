#!/usr/bin/env python3
"""
TEAM-28-Netz fuer den FEM-direkt-Pfad: Leiter UND Luftgebiet, aus Prismen.

    python3 tools/gmsh/team28_fem_air.py -o team28_fem.vtk
    python3 tools/gmsh/team28_fem_air.py --truncation 5 -o kasten5.vtk   # Abschneidestudie
    python3 tools/gmsh/team28_fem_air.py --plate-z0 11.3 -o gleichgewicht.vtk

Fuer ./fembem --fem-direct (bzw. --fem-direct-iterative). Fuer den BEM-FEM-Pfad
(nur Leiter, reine Hexaeder) ist tools/gmsh/team28_hex.py zustaendig.

AUFBAU (docs/design/fem-direct-plan.md, AP2)
  * Dreiecksnetz in der Ebene. Alle Radien sind Kanten: Spulen 27/55 und
    80/95 mm, Plattenrand 65 mm, Luftzylinder aussen.
  * Schichtweise SENKRECHT extrudiert. Schichtgrenzen an den Spulen
    (z = -52, 0), am Spalt und an der Platte. Dadurch sind alle Prismen affin
    (Deckflaeche = verschobene Bodenflaeche) -- Nedelec-Elemente niedrigster
    Ordnung reproduzieren ein konstantes B nur dann exakt -- und das Netz ist
    an allen Materialgrenzen konform.
  * Die Platte hat mehrere Lagen ueber die Dicke (AP2: vier bis sechs).
  * Die Luftschichten werden zum Rand hin geometrisch groeber.
  * Luftgebiet: Zylinder um die Leiter. Seine Groesse wird ueber den
    Abschneidefaktor eingestellt, genau wie ihn das Programm berechnet:
    Ausdehnung der Luft / Ausdehnung der Leiter, je Achse (FemDirectRun.cpp).

FESTE REGELN (docs/design/gmsh-meshing-plan.md)
  * Die Platte ist eine VOLLSCHEIBE, kein Loch. Dafuer gibt es keine Option.
  * Die Platte liegt echt oberhalb der Spulen.
  * Prismen werden vor dem Schreiben in die Konvention des Loesers gedreht
    (include/NedelecPrism.cuh); Gmsh liefert sie umgekehrt orientiert.

component_id: 0 Luft, 1 Aussenspule, 2 Innenspule, 3 Platte. Einheiten: mm.

PRUEFUNG NACH DEM SCHREIBEN (Exit-Code 1, wenn etwas nicht stimmt)
  * tools/meshcheck.py: nur Prismen, keine doppelten/verwaisten Knoten,
    konform, Orientierung positiv, Prismen affin, Luft umschliesst alles.
  * Volumen je Koerper gegen das exakte Vieleck aus den Knoten auf jedem
    Kreis (bis auf Rundung; prueft Radien, Hoehen und Zuordnung zugleich).
  * tools/check_map.py: Zuordnung, Masse, Platte ueberdeckt die Achse.
  * Abschneidefaktoren wie im Programm.
  Nicht hier geprueft, weil es das Programm braucht: die Zahl der
  Kohomologie-Generatoren. Mit Luft ist das leitende Gebiet nur die Platte;
  als Vollscheibe traegt sie keinen bei, erwartet ist also 0.
"""
import argparse
import contextlib
import io
import math
import os
import sys
import time

TOOLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, TOOLS)

AIR, OUTER, INNER, PLATE = 0, 1, 2, 3
NAMES = {AIR: "Luft", OUTER: "Aussenspule", INNER: "Innenspule", PLATE: "Platte"}
R_IN0, R_IN1 = 27.0, 55.0          # Innenspule
R_OUT0, R_OUT1 = 80.0, 95.0        # Aussenspule
COIL_Z0, COIL_Z1 = -52.0, 0.0
PLATE_R, PLATE_T = 65.0, 3.0
PHYS = {OUTER: 1, INNER: 2, PLATE: 3, AIR: 10}      # Gmsh-Gruppen (0 ist dort nicht erlaubt)


def graded_layers(length, first, growth):
    """Schichtdicken, beginnend mit `first`, je Schicht * growth, Summe = length."""
    if growth <= 1.0 + 1e-12:
        n = max(1, int(math.ceil(length / first)))
        return [length / n] * n
    t, total = [], 0.0
    while total < length - 1e-12:
        t.append(first * growth ** len(t))
        total += t[-1]
    return [x * length / total for x in t]


def cumulative(thick):
    s, out = 0.0, []
    for x in thick:
        s += x
        out.append(s)
    return [v / s for v in out]


def build(a):
    """Geometrie und Netz erzeugen; liefert Luftradius und z-Grenzen."""
    import gmsh
    g = gmsh.model.geo

    # Abschneidefaktor wie im Programm: Ausdehnung Luft / Ausdehnung Leiter
    r_air = a.truncation_radial * R_OUT1
    zc_lo, zc_hi = COIL_Z0, a.plate_z0 + PLATE_T
    zmid, half = 0.5 * (zc_lo + zc_hi), 0.5 * (zc_hi - zc_lo)
    z_bot, z_top = zmid - a.truncation_axial * half, zmid + a.truncation_axial * half

    # 2D-Geometrie bei z = z_bot: konzentrische Kreise, jeweils 4 Boegen
    radii = [R_IN0, R_IN1, PLATE_R, R_OUT0, R_OUT1, r_air]
    sizes = [a.h, a.h, a.h, a.h, a.h, a.h_air]
    centre = g.addPoint(0, 0, z_bot, a.h)
    loops = []
    for r, h in zip(radii, sizes):
        p = [g.addPoint(r * math.cos(k * math.pi / 2), r * math.sin(k * math.pi / 2), z_bot, h)
             for k in range(4)]
        arcs = [g.addCircleArc(p[k], centre, p[(k + 1) % 4]) for k in range(4)]
        loops.append(g.addCurveLoop(arcs))
    # Regionen: 0 r<27, 1 Innenspule, 2 bis Plattenrand, 3 bis Aussenspule,
    #           4 Aussenspule, 5 Luft bis zum Zylinder
    surfaces = [g.addPlaneSurface([loops[0]])]
    surfaces += [g.addPlaneSurface([loops[i], loops[i - 1]]) for i in range(1, len(loops))]

    # Schichten von unten nach oben; Luft zum Leiter hin fein
    below = graded_layers(COIL_Z0 - z_bot, a.air_first, a.air_growth)[::-1]
    above = graded_layers(z_top - zc_hi, a.air_first, a.air_growth)
    segments = [  # (Name, Hoehe, Schichtdicken oder Anzahl)
        ("Luft unten", COIL_Z0 - z_bot, below),
        ("Spulen", COIL_Z1 - COIL_Z0, a.coil_nz),
        ("Spalt", a.plate_z0 - COIL_Z1, a.gap_nz),
        ("Platte", PLATE_T, a.plate_nz),
        ("Luft oben", z_top - zc_hi, above),
    ]

    current = [(2, s) for s in surfaces]
    volumes = []                         # volumes[Segment][Region]
    layer_counts = []
    for name, height, layers in segments:
        if isinstance(layers, int):
            ne, hs = [layers], []
            layer_counts.append((name, layers))
        else:
            ne, hs = [1] * len(layers), cumulative(layers)
            layer_counts.append((name, len(layers)))
        out = g.extrude(current, 0, 0, height, numElements=ne, heights=hs, recombine=True)
        tops, vols = [], []
        for k, (dim, tag) in enumerate(out):
            if dim == 3:
                vols.append(tag)
                tops.append(out[k - 1])
        assert len(vols) == len(surfaces), "Extrusion lieferte unerwartete Volumina"
        volumes.append(vols)
        current = tops

    g.synchronize()
    groups = {OUTER: [], INNER: [], PLATE: [], AIR: []}
    for s, (name, _, _) in enumerate(segments):
        for region, vol in enumerate(volumes[s]):
            if name == "Spulen" and region == 1:
                groups[INNER].append(vol)
            elif name == "Spulen" and region == 4:
                groups[OUTER].append(vol)
            elif name == "Platte" and region in (0, 1, 2):
                groups[PLATE].append(vol)
            else:
                groups[AIR].append(vol)
    for cid, vols in groups.items():
        gmsh.model.addPhysicalGroup(3, vols, PHYS[cid], NAMES[cid])
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(3)
    return r_air, z_bot, z_top, layer_counts


def collect_and_write(path):
    """Elemente je Gruppe holen, Prismen in die Loeser-Konvention drehen, VTK schreiben."""
    import gmsh
    import numpy as np
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = np.asarray(coords).reshape(-1, 3)
    pos = {int(t): i for i, t in enumerate(ntags)}

    cells, comps, other = [], [], 0
    for cid, ptag in PHYS.items():
        for ent in gmsh.model.getEntitiesForPhysicalGroup(3, ptag):
            types, _, nodes = gmsh.model.mesh.getElements(3, ent)
            for t, nn in zip(types, nodes):
                if int(t) != 6:                     # 6 = Prisma (Gmsh)
                    other += len(nn) // gmsh.model.mesh.getElementProperties(int(t))[3]
                    continue
                conn = np.array([pos[int(v)] for v in nn], dtype=np.int64).reshape(-1, 6)
                cells.append(conn)
                comps.append(np.full(len(conn), cid))
    if other:
        raise RuntimeError(f"{other} Elemente sind keine Prismen -- Extrusion fehlerhaft")
    conn = np.vstack(cells)
    comp = np.concatenate(comps)

    # Orientierung in der Konvention des Loesers: (x1-x0, x2-x0, x3-x0) rechtshaendig
    P = xyz
    det = np.linalg.det(np.stack([P[conn[:, 1]] - P[conn[:, 0]],
                                  P[conn[:, 2]] - P[conn[:, 0]],
                                  P[conn[:, 3]] - P[conn[:, 0]]], axis=1))
    if (np.abs(det) < 1e-12 * np.abs(det).max()).any():
        raise RuntimeError("entartete Prismen (Volumen ~ 0)")
    flip = det < 0
    conn[flip] = conn[flip][:, [0, 2, 1, 3, 5, 4]]

    used = np.unique(conn)
    renum = np.full(len(P), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    conn = renum[conn]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("FEMBEM FEM-direkt, erzeugt von tools/gmsh/team28_fem_air.py\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(used)} double\n")
        for p in P[used]:
            f.write("%.17g %.17g %.17g\n" % tuple(p))
        f.write(f"\nCELLS {len(conn)} {7 * len(conn)}\n")
        for c in conn:
            f.write("6 %d %d %d %d %d %d\n" % tuple(c))
        f.write(f"\nCELL_TYPES {len(conn)}\n")
        f.write("13\n" * len(conn))
        # GENAU EIN SCALARS-Array: VtkReader nimmt das erste
        f.write(f"\nCELL_DATA {len(conn)}\nSCALARS component_id int 1\nLOOKUP_TABLE default\n")
        f.writelines(f"{int(c)}\n" for c in comp)
    return len(used), len(conn), int(flip.sum()), {k: int((comp == k).sum()) for k in NAMES}


def polygon_areas(mesh, radii, z_bot):
    """Exakte Flaeche des Vielecks, das die Knoten auf jedem Kreis bilden.

    Die Kreise sind durch Netzknoten auf dem Kreis angenaehert; bei grobem Netz
    fehlt dem Vieleck merklich Flaeche (17 Segmente: 2 %). Gemessen wird deshalb
    gegen das Vieleck aus den tatsaechlichen Knoten -- das ist exakt, bis auf
    Rundung. Alle Schichten haben dieselben Knoten in der Ebene (Extrusion),
    die Unterseite z = z_bot genuegt.
    """
    import numpy as np
    P = mesh.points
    plane = P[np.abs(P[:, 2] - z_bot) < 1e-9 * max(1.0, abs(z_bot))]
    r = np.hypot(plane[:, 0], plane[:, 1])
    out = {}
    for R in radii:
        on = plane[np.abs(r - R) < 1e-9 * R]
        ang = np.sort(np.arctan2(on[:, 1], on[:, 0]))
        d = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
        out[R] = 0.5 * R * R * float(np.sin(d).sum())
    return out


def expected_volumes(mesh, r_air, z_bot, z_top):
    A = polygon_areas(mesh, [R_IN0, R_IN1, PLATE_R, R_OUT0, R_OUT1, r_air], z_bot)
    h = COIL_Z1 - COIL_Z0
    v = {OUTER: (A[R_OUT1] - A[R_OUT0]) * h, INNER: (A[R_IN1] - A[R_IN0]) * h,
         PLATE: A[PLATE_R] * PLATE_T}
    v[AIR] = A[r_air] * (z_top - z_bot) - sum(v.values())
    return v, None


def truncation_factors(mesh):
    """Wie FemDirectRun.cpp: Ausdehnung der Luft / Ausdehnung der Leiter, je Achse."""
    import numpy as np
    def box(mask):
        nodes = np.unique(np.concatenate([mesh.cells[i] for i in np.flatnonzero(mask)]))
        p = mesh.points[nodes]
        return p.max(axis=0) - p.min(axis=0)
    air, cond = box(mesh.comp == AIR), box(mesh.comp != AIR)
    return air / cond


def check(path, r_air, z_bot, z_top, a, verbose):
    import meshcheck as mc
    import check_map
    mesh = mc.read_vtk(path)
    res = mc.run_checks(mesh, air_enclosed=True, prism_affinity=True)
    got, (want, _) = mc.volumes(mesh), expected_volumes(mesh, r_air, z_bot, z_top)
    for cid in (OUTER, INNER, PLATE, AIR):
        rel = abs(got.get(cid, 0.0) - want[cid]) / want[cid]
        res.append(mc.Result(f"Volumen {NAMES[cid]} (exaktes Vieleck)", rel <= 1e-9,
                             f"{got.get(cid, 0.0):.1f} mm^3, Soll {want[cid]:.1f}, rel. {rel:.1e}"))
    tf = truncation_factors(mesh)
    ok_tf = (abs(tf[0] - a.truncation_radial) / a.truncation_radial < 0.02
             and abs(tf[2] - a.truncation_axial) / a.truncation_axial < 0.02)
    res.append(mc.Result("Abschneidefaktor x/y/z", ok_tf,
                         f"{tf[0]:.3f} / {tf[1]:.3f} / {tf[2]:.3f} (verlangt {a.truncation_radial:g} "
                         f"radial, {a.truncation_axial:g} axial)"))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = check_map.main(path)
    res.append(mc.Result("check_map (Zuordnung, Masse, Vollscheibe)", rc == 0,
                         "besteht" if rc == 0 else "siehe Ausgabe"))
    ok = mc.print_results(res)
    if verbose or rc != 0:
        print(buf.getvalue())
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="TEAM-28-Netz mit Luftgebiet (Prismen) fuer FEM-direkt erzeugen und pruefen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-o", "--out", default="team28_fem_air.vtk", help="Ausgabedatei (Legacy-VTK)")
    ap.add_argument("--h", type=float, default=5.0, help="Kantenlaenge in der Ebene an Leitern und Platte, mm")
    ap.add_argument("--h-air", type=float, default=35.0, help="Kantenlaenge am Aussenrand der Luft, mm")
    ap.add_argument("--coil-nz", type=int, default=6, help="Schichten ueber die Spulenhoehe (52 mm)")
    ap.add_argument("--gap-nz", type=int, default=3, help="Schichten im Spalt Spule-Platte")
    ap.add_argument("--plate-nz", type=int, default=4, help="Schichten ueber die Plattendicke (3 mm)")
    ap.add_argument("--air-first", type=float, default=3.0, help="erste Luftschicht am Leiter, mm")
    ap.add_argument("--air-growth", type=float, default=1.4, help="Wachstum der Luftschichten nach aussen")
    ap.add_argument("--truncation", type=float, default=None,
                    help="Abschneidefaktor radial UND axial gleich (Kurzform)")
    # Vorgaben aus der Messreihe in docs/design/gmsh-phase6-report.md: am
    # Schwebegleichgewicht ist F_z axial bei 10 ausgeschoepft (10 -> 15:
    # -0,08 %), radial bei 7 (7 -> 10: -0,1 %); radial 5 liegt 0,7 % daneben
    # und kostet deutlich weniger. Mit 3 / 3 lag F_z 24 % zu hoch.
    ap.add_argument("--truncation-radial", type=float, default=5.0, help="Abschneidefaktor in x und y")
    ap.add_argument("--truncation-axial", type=float, default=10.0, help="Abschneidefaktor in z")
    ap.add_argument("--plate-z0", type=float, default=3.8, help="Unterkante der Platte in mm, > 0")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Feinheit: teilt Kantenlaengen, vervielfacht Schichtzahlen (Netzserie)")
    ap.add_argument("--no-check", action="store_true", help="Pruefung nach dem Schreiben auslassen")
    ap.add_argument("-v", "--verbose", action="store_true", help="Ausgabe von check_map zeigen")
    a = ap.parse_args()

    if a.truncation is not None:
        a.truncation_radial = a.truncation_axial = a.truncation
    if a.truncation_radial <= 1.05 or a.truncation_axial <= 1.05:
        ap.error("Abschneidefaktoren muessen deutlich ueber 1 liegen (Luft um die Leiter)")
    if a.plate_z0 <= 0.0:
        ap.error("--plate-z0 muss > 0 sein: die Platte liegt oberhalb der Spulen (z = 0)")
    if a.scale <= 0 or a.h <= 0 or a.h_air <= 0 or a.air_first <= 0 or a.air_growth < 1.0:
        ap.error("Groessen muessen positiv sein, --air-growth >= 1")
    a.h /= a.scale
    a.h_air /= a.scale
    a.air_first /= a.scale
    for name in ("coil_nz", "gap_nz", "plate_nz"):
        setattr(a, name, max(1, int(round(getattr(a, name) * a.scale))))

    try:
        import gmsh
        import numpy  # noqa: F401
    except ImportError as e:
        sys.exit(f"FEHLER: {e}. pip install -r tools/requirements-gmsh.txt; ohne Grafik\n"
                 "  zusaetzlich die GL-Bibliotheken, siehe docs/meshing.md.")

    t0 = time.time()
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("FEMBEM team28_fem_air")
        r_air, z_bot, z_top, layers = build(a)
        n_nodes, n_cells, n_flip, per = collect_and_write(a.out)
    finally:
        gmsh.finalize()
    print(f"{a.out}: {n_cells} Prismen, {n_nodes} Knoten ({time.time() - t0:.1f} s), "
          f"Gmsh {gmsh.__version__}")
    print("  " + ", ".join(f"{NAMES[k]} {v}" for k, v in per.items()))
    print(f"  Luftzylinder r = {r_air:.1f} mm, z = {z_bot:.1f} .. {z_top:.1f} mm; Schichten: "
          + ", ".join(f"{n} {k}" for n, k in layers))
    print(f"  in die Loeser-Konvention gedrehte Prismen: {n_flip} von {n_cells}")
    if a.no_check:
        return 0
    ok = check(a.out, r_air, z_bot, z_top, a, a.verbose)
    print("Netz in Ordnung." if ok else "NETZ FEHLERHAFT -- nicht verwenden.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
