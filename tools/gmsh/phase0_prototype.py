#!/usr/bin/env python3
"""
PROTOTYP aus Phase 0 der Gmsh-Vernetzung -- kein fertiges Werkzeug.

Erzeugt ein reines Hexaedernetz der TEAM-28-Geometrie mit Gmsh und misst,
ob es die Anforderungen erfuellt. Grundlage fuer tools/gmsh/team28_hex.py
(Phase 2), siehe docs/design/gmsh-meshing-plan.md.

    python3 tools/gmsh/phase0_prototype.py -o team28_gmsh.vtk

AUFBAU
  * Spulen: Rechteck-Querschnitt (r, z), viermal um 90 Grad gedreht. Der
    eingebaute Geo-Kern dreht hoechstens um weniger als 180 Grad; vier
    Viertel teilen sich ihre Schnittflaechen, das Netz ist damit konform
    und ohne Naht.
  * Platte: VOLLSCHEIBE mit Hexaederkern (O-Gitter): ein Viereck um die
    Achse, darum vier Randstuecke bis zum Aussenradius; dann in z extrudiert.
    KEIN Loch: ein Loch aendert die Topologie (zusaetzlicher
    Kohomologie-Generator) und die Geometrie. Das ist ausdruecklich
    ausgeschlossen.

Einheiten: mm. Komponenten: 1 Aussenspule, 2 Innenspule, 3 Platte.
"""
import argparse
import math
import sys

import gmsh

# TEAM-28-Abmessungen in mm (wie im Standardnetz, siehe tools/check_map.py)
OUTER = dict(r0=80.0, r1=95.0, z0=-52.0, z1=0.0)
INNER = dict(r0=27.0, r1=55.0, z0=-52.0, z1=0.0)
PLATE_R, PLATE_Z0, PLATE_T = 65.0, 3.8, 3.0


def coil(g, c, n_r, n_z, n_q):
    """Rechteck in der xz-Ebene, viermal um 90 Grad um z gedreht."""
    p = [g.addPoint(c["r0"], 0, c["z0"]), g.addPoint(c["r1"], 0, c["z0"]),
         g.addPoint(c["r1"], 0, c["z1"]), g.addPoint(c["r0"], 0, c["z1"])]
    l = [g.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    s = g.addPlaneSurface([g.addCurveLoop(l)])
    # Netzvorgaben ueber g.mesh setzen, nicht ueber gmsh.model.mesh: beim
    # Geo-Kern gehen letztere beim naechsten synchronize() verloren, und aus
    # den Vierecken werden Dreiecke -> Prismen statt Hexaeder.
    for k in (l[0], l[2]):
        g.mesh.setTransfiniteCurve(k, n_r + 1)
    for k in (l[1], l[3]):
        g.mesh.setTransfiniteCurve(k, n_z + 1)
    g.mesh.setTransfiniteSurface(s)
    g.mesh.setRecombine(2, s)
    vols, cur = [], [(2, s)]
    for _ in range(4):
        out = g.revolve(cur, 0, 0, 0, 0, 0, 1, math.pi / 2, [n_q], recombine=True)
        cur = [out[0]]                       # Deckflaeche dieses Viertels
        vols.append(out[1][1])
    return vols


def plate(g, n_q, n_r, n_z, core):
    """Vollscheibe mit Hexaederkern. core = halbe Diagonale des Kernvierecks / R."""
    R, z0 = PLATE_R, PLATE_Z0
    a = core * R                              # Abstand der Kern-Ecken von der Achse
    ang = [math.pi / 4 + k * math.pi / 2 for k in range(4)]
    o = g.addPoint(0, 0, z0)
    cin = [g.addPoint(a * math.cos(t), a * math.sin(t), z0) for t in ang]
    cout = [g.addPoint(R * math.cos(t), R * math.sin(t), z0) for t in ang]
    sq = [g.addLine(cin[k], cin[(k + 1) % 4]) for k in range(4)]
    arc = [g.addCircleArc(cout[k], o, cout[(k + 1) % 4]) for k in range(4)]
    rad = [g.addLine(cin[k], cout[k]) for k in range(4)]
    surf = [g.addPlaneSurface([g.addCurveLoop(sq)])]
    for k in range(4):
        surf.append(g.addPlaneSurface([g.addCurveLoop(
            [sq[k], rad[(k + 1) % 4], -arc[k], -rad[k]])]))
    for k in sq + arc:
        g.mesh.setTransfiniteCurve(k, n_q + 1)
    for k in rad:
        g.mesh.setTransfiniteCurve(k, n_r + 1)
    for s in surf:
        g.mesh.setTransfiniteSurface(s)
        g.mesh.setRecombine(2, s)
    out = g.extrude([(2, s) for s in surf], 0, 0, PLATE_T, [n_z], recombine=True)
    return [e[1] for e in out if e[0] == 3]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("-o", "--out", default="team28_gmsh.vtk")
    ap.add_argument("--n-quarter", type=int, default=12, help="Zellen je Viertelkreis")
    ap.add_argument("--plate-nr", type=int, default=10, help="Plattenzellen radial (Kern bis Rand)")
    ap.add_argument("--plate-nz", type=int, default=3)
    ap.add_argument("--core", type=float, default=0.3, help="Kerngroesse relativ zum Radius")
    ap.add_argument("--coil-nr", type=int, default=4)
    ap.add_argument("--coil-nz", type=int, default=8)
    a = ap.parse_args()

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("team28")
    g = gmsh.model.geo
    v_out = coil(g, OUTER, a.coil_nr, a.coil_nz, a.n_quarter)
    v_in = coil(g, INNER, a.coil_nr, a.coil_nz, a.n_quarter)
    v_pl = plate(g, a.n_quarter, a.plate_nr, a.plate_nz, a.core)
    g.synchronize()
    gmsh.model.addPhysicalGroup(3, v_out, 1, "outer_coil")
    gmsh.model.addPhysicalGroup(3, v_in, 2, "inner_coil")
    gmsh.model.addPhysicalGroup(3, v_pl, 3, "plate")
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.write(a.out)

    # Kennzahlen je Koerper
    print(f"geschrieben: {a.out}")
    for tag, name in ((1, "Aussenspule"), (2, "Innenspule"), (3, "Platte")):
        types, counts, minsj, vol = set(), 0, 1.0, 0.0
        for ent in gmsh.model.getEntitiesForPhysicalGroup(3, tag):
            et, el, _ = gmsh.model.mesh.getElements(3, ent)
            for t, tags in zip(et, el):
                types.add(int(t)); counts += len(tags)
                q = gmsh.model.mesh.getElementQualities(tags, "minSJ")
                minsj = min(minsj, min(q))
                vol += sum(gmsh.model.mesh.getElementQualities(tags, "volume"))
        print(f"  {name:12s} Elemente {counts:6d}  Typen {sorted(types)}  "
              f"min. skal. Jacobi {minsj:6.3f}  Volumen {vol:12.2f} mm^3")
    gmsh.finalize()


if __name__ == "__main__":
    sys.exit(main())
