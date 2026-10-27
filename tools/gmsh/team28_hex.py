#!/usr/bin/env python3
"""
TEAM-28-Netz aus reinen Hexaedern mit Gmsh erzeugen und pruefen.

    python3 tools/gmsh/team28_hex.py -o team28_gmsh.vtk
    python3 tools/gmsh/team28_hex.py --scale 2 -o fein.vtk       # doppelt so fein
    python3 tools/gmsh/team28_hex.py --plate-z0 9.8 -o hoch.vtk  # andere Plattenlage

Braucht das Python-Paket gmsh (pip install gmsh numpy; auf Servern ohne
Grafik zusaetzlich die GL-Laufzeitbibliotheken, siehe
docs/design/gmsh-phase0-report.md).

GEOMETRIE (mm, wie das Standardnetz data/team28/team28.vtk)
  Aussenspule  r 80..95, z -52..0     component_id 1
  Innenspule   r 27..55, z -52..0     component_id 2
  Platte       r 0..65, Dicke 3,
               Unterkante --plate-z0  component_id 3

AUFBAU
  * Spulen: Rechteck-Querschnitt, viermal um 90 Grad gedreht (der Geo-Kern
    dreht nur um weniger als 180 Grad). Die Viertel teilen sich ihre
    Schnittflaechen: konform, ohne Naht.
  * Platte: Vollscheibe mit Hexaederkern (O-Gitter) -- ein Viereck um die
    Achse, vier Randstuecke bis zum Aussenradius, in z extrudiert.

FESTE REGELN (docs/design/gmsh-meshing-plan.md)
  * KEIN LOCH in der Platte. Ein Loch aendert die Topologie (zusaetzlicher
    Kohomologie-Generator) und die Geometrie und ist ausgeschlossen. Es gibt
    dafuer bewusst keine Option.
  * Nur Hexaeder: die BEM wird nur bei reinen Hexaedernetzen aufgebaut.

PRUEFUNG NACH DEM SCHREIBEN (Exit-Code 1, wenn etwas nicht stimmt)
  * tools/meshcheck.py: nur Hexaeder, keine doppelten oder verwaisten
    Knoten, Koerper getrennt (kein gemeinsamer Knoten), konform, Jacobi-Determinante an allen Ecken positiv, Volumen je
    Koerper gleich dem exakten N-Eck (N = 4 * --n-quarter Segmente).
  * tools/check_map.py: Zuordnung der Spulen, Masse, Platte ueberdeckt die
    Achse (Vollscheibe).
  Nicht hier geprueft, weil es das Programm braucht: dass der Loeser genau
  2 Kohomologie-Generatoren findet (je Ringspule einer).
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

OUTER = dict(r0=80.0, r1=95.0, z0=-52.0, z1=0.0)
INNER = dict(r0=27.0, r1=55.0, z0=-52.0, z1=0.0)
PLATE_R, PLATE_T = 65.0, 3.0
NAMES = {1: "Aussenspule", 2: "Innenspule", 3: "Platte"}


def coil(g, c, n_r, n_z, n_q):
    """Rechteck in der xz-Ebene, viermal um 90 Grad um z gedreht."""
    p = [g.addPoint(c["r0"], 0, c["z0"]), g.addPoint(c["r1"], 0, c["z0"]),
         g.addPoint(c["r1"], 0, c["z1"]), g.addPoint(c["r0"], 0, c["z1"])]
    l = [g.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    s = g.addPlaneSurface([g.addCurveLoop(l)])
    # Netzvorgaben ueber g.mesh (Geo-Kern) setzen, NICHT ueber gmsh.model.mesh:
    # letztere gehen beim naechsten synchronize() verloren, und aus Vierecken
    # werden ohne Meldung Dreiecke -> Prismen statt Hexaeder.
    for k in (l[0], l[2]):
        g.mesh.setTransfiniteCurve(k, n_r + 1)
    for k in (l[1], l[3]):
        g.mesh.setTransfiniteCurve(k, n_z + 1)
    g.mesh.setTransfiniteSurface(s)
    g.mesh.setRecombine(2, s)
    vols, cur = [], [(2, s)]
    for _ in range(4):
        out = g.revolve(cur, 0, 0, 0, 0, 0, 1, math.pi / 2, [n_q], recombine=True)
        cur = [out[0]]                        # Deckflaeche dieses Viertels
        vols.append(out[1][1])
    return vols


def plate(g, z0, n_q, n_r, n_z, core, grading):
    """Vollscheibe mit Hexaederkern; core = Abstand der Kern-Ecken von der Achse / R."""
    R = PLATE_R
    a = core * R
    ang = [math.pi / 4 + k * math.pi / 2 for k in range(4)]
    o = g.addPoint(0, 0, z0)
    cin = [g.addPoint(a * math.cos(t), a * math.sin(t), z0) for t in ang]
    cout = [g.addPoint(R * math.cos(t), R * math.sin(t), z0) for t in ang]
    sq = [g.addLine(cin[k], cin[(k + 1) % 4]) for k in range(4)]
    arc = [g.addCircleArc(cout[k], o, cout[(k + 1) % 4]) for k in range(4)]
    rad = [g.addLine(cin[k], cout[k]) for k in range(4)]   # vom Kern nach aussen
    surf = [g.addPlaneSurface([g.addCurveLoop(sq)])]
    for k in range(4):
        surf.append(g.addPlaneSurface([g.addCurveLoop(
            [sq[k], rad[(k + 1) % 4], -arc[k], -rad[k]])]))
    for k in sq + arc:
        g.mesh.setTransfiniteCurve(k, n_q + 1)
    for k in rad:
        # Progression < 1: Zellen werden zum Plattenrand hin kleiner
        g.mesh.setTransfiniteCurve(k, n_r + 1, "Progression", grading)
    for s in surf:
        g.mesh.setTransfiniteSurface(s)
        g.mesh.setRecombine(2, s)
    out = g.extrude([(2, s) for s in surf], 0, 0, PLATE_T, [n_z], recombine=True)
    return [e[1] for e in out if e[0] == 3]


def ngon_volumes(n_seg):
    """Exakte Volumen bei n_seg geraden Segmenten am Umfang (Knoten auf dem Kreis)."""
    f = n_seg / (2 * math.pi) * math.sin(2 * math.pi / n_seg)
    ring = lambda c: math.pi * (c["r1"] ** 2 - c["r0"] ** 2) * (c["z1"] - c["z0"]) * f
    return {1: ring(OUTER), 2: ring(INNER), 3: math.pi * PLATE_R ** 2 * PLATE_T * f}


def generate(a):
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        title = (f"FEMBEM team28_hex nq={a.n_quarter} coil={a.coil_nr}x{a.coil_nz} "
                 f"plate={a.plate_nr}x{a.plate_nz} core={a.core} grading={a.plate_grading} "
                 f"z0={a.plate_z0} gmsh={gmsh.__version__}")
        gmsh.model.add(title)
        g = gmsh.model.geo
        v_out = coil(g, OUTER, a.coil_nr, a.coil_nz, a.n_quarter)
        v_in = coil(g, INNER, a.coil_nr, a.coil_nz, a.n_quarter)
        v_pl = plate(g, a.plate_z0, a.n_quarter, a.plate_nr, a.plate_nz, a.core, a.plate_grading)
        g.synchronize()
        gmsh.model.addPhysicalGroup(3, v_out, 1, "outer_coil")
        gmsh.model.addPhysicalGroup(3, v_in, 2, "inner_coil")
        gmsh.model.addPhysicalGroup(3, v_pl, 3, "plate")
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.option.setNumber("Mesh.Binary", 0)
        gmsh.option.setNumber("Mesh.SaveAll", 0)     # nur Elemente in Physical Groups
        gmsh.model.mesh.generate(3)
        # KEIN removeDuplicateNodes(): der Geo-Kern liefert die Viertel schon
        # konform. Das Verschmelzen wuerde zudem Knoten verschiedener Koerper
        # vereinen, wenn sie zufaellig zusammenfallen -- Platte und Spule waeren
        # dann ein leitender Koerper. Doppelte Knoten meldet die Pruefung.
        quality = {}
        for tag in (1, 2, 3):
            lo = 1.0
            for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
                _, el, _ = gmsh.model.mesh.getElements(3, ent)
                for tags in el:
                    lo = min(lo, min(gmsh.model.mesh.getElementQualities(tags, "minSJ")))
            quality[tag] = lo
        gmsh.write(a.out)
        return title, quality
    finally:
        gmsh.finalize()


def check(path, n_seg, verbose):
    import meshcheck as mc
    import check_map
    mesh = mc.read_vtk(path)
    results = mc.run_checks(mesh, require_hex=True, separate_bodies=True,
                            expected_volumes=ngon_volumes(n_seg),
                            names={k: f"{v} ({n_seg}-Eck)" for k, v in NAMES.items()})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = check_map.main(path)
    results.append(mc.Result("check_map (Zuordnung, Masse, Vollscheibe)", rc == 0,
                             "besteht" if rc == 0 else "siehe Ausgabe"))
    ok = mc.print_results(results)
    if verbose or rc != 0:
        print(buf.getvalue())
    return ok


def scaled(n, s):
    return max(1, int(round(n * s)))


def main():
    ap = argparse.ArgumentParser(
        description="TEAM-28-Netz aus reinen Hexaedern mit Gmsh erzeugen und pruefen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-o", "--out", default="team28_gmsh.vtk", help="Ausgabedatei (Legacy-VTK)")
    ap.add_argument("--n-quarter", type=int, default=12, help="Zellen je Viertelkreis (alle Koerper)")
    ap.add_argument("--coil-nr", type=int, default=4, help="Spulenzellen radial")
    ap.add_argument("--coil-nz", type=int, default=8, help="Spulenzellen in z")
    ap.add_argument("--plate-nr", type=int, default=10, help="Plattenzellen radial, Kern bis Rand")
    ap.add_argument("--plate-nz", type=int, default=3, help="Plattenzellen in z (Dicke 3 mm)")
    ap.add_argument("--core", type=float, default=0.4,
                    help="Kerngroesse: Abstand der Kern-Ecken von der Achse relativ zu R = 65 mm")
    ap.add_argument("--plate-grading", type=float, default=1.0,
                    help="radiale Progression der Platte; < 1 verfeinert zum Rand hin")
    ap.add_argument("--plate-z0", type=float, default=3.8, help="Unterkante der Platte in mm")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Faktor auf alle Teilungen (Netzserie: 0.5, 1, 2, ...)")
    ap.add_argument("--no-check", action="store_true", help="Pruefung nach dem Schreiben auslassen")
    ap.add_argument("-v", "--verbose", action="store_true", help="Ausgabe von check_map zeigen")
    a = ap.parse_args()

    for name in ("n_quarter", "coil_nr", "coil_nz", "plate_nr", "plate_nz"):
        setattr(a, name, scaled(getattr(a, name), a.scale))
    if not 0.05 <= a.core <= 0.8:
        ap.error("--core muss zwischen 0.05 und 0.8 liegen")
    if a.plate_grading <= 0:
        ap.error("--plate-grading muss positiv sein")
    if a.plate_z0 <= 0.0:
        ap.error("--plate-z0 muss > 0 sein: die Spulen enden bei z = 0, die Platte "
                 "muss mit Abstand darueber liegen (beruehrende Flaechen zweier "
                 "Koerper sind fuer die BEM unbrauchbar)")

    try:
        import gmsh  # noqa: F401
    except ImportError as e:
        sys.exit(f"FEHLER: Python-Paket gmsh fehlt oder laedt nicht ({e}).\n"
                 "  pip install gmsh numpy; ohne Grafik zusaetzlich die GL-Bibliotheken,\n"
                 "  siehe docs/design/gmsh-phase0-report.md.")

    t0 = time.time()
    title, quality = generate(a)
    print(f"{a.out}: {title}")
    print("  min. skalierte Jacobi-Determinante: " +
          ", ".join(f"{NAMES[k]} {v:.3f}" for k, v in quality.items()) +
          f"   ({time.time() - t0:.1f} s)")
    if a.no_check:
        return 0
    ok = check(a.out, 4 * a.n_quarter, a.verbose)
    print("Netz in Ordnung." if ok else "NETZ FEHLERHAFT -- nicht verwenden.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
