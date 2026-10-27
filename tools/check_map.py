#!/usr/bin/env python3
"""Prueft, ob die PID-Zuordnung in der konvertierten VTK zur Konvention passt,
die src/fem.cu erwartet (siehe assign_materials_by_component_id dort):

    component_id 1 = Aussenspule   (r 80..95)   Strom -J_OUTER
    component_id 2 = Innenspule    (r 27..55)   Strom +J_INNER
    component_id 3 = Platte        (z  3.8..6.8)
    component_id 0 = Luft

Vertauschte Spulen erzeugen keinen Absturz, sondern ein falsches Feld: die
Vorzeichen der beiden Stroeme sind in fem.cu fest verdrahtet und haengen an
der component_id, nicht an der Geometrie.

    python check_map.py team28_fem.vtk
"""
import sys
import numpy as np


def read_legacy_vtk(path):
    lines = open(path, "r", errors="replace").read().replace("\r", "").split("\n")

    def block(keyword):
        return next(k for k, l in enumerate(lines) if l.startswith(keyword))

    def gather(start, count):
        vals, k = [], start
        while len(vals) < count:
            vals += lines[k].split()
            k += 1
        return vals[:count]

    i = block("POINTS")
    n_pts = int(lines[i].split()[1])
    pts = np.array(gather(i + 1, 3 * n_pts), dtype=float).reshape(n_pts, 3)

    i = block("CELLS")
    n_cells, n_ints = (int(x) for x in lines[i].split()[1:3])
    flat = [int(v) for v in gather(i + 1, n_ints)]

    cells, p = [], 0
    while p < n_ints:
        n = flat[p]
        cells.append(flat[p + 1:p + 1 + n])
        p += 1 + n

    i = block("CELL_TYPES")
    ctypes = np.array(gather(i + 1, n_cells), dtype=int)

    i = block("SCALARS")
    j = i + 1
    if lines[j].startswith("LOOKUP"):
        j += 1
    comp = np.array(gather(j, n_cells), dtype=int)

    return pts, cells, ctypes, comp


VTK_NAMES = {10: "Tetraeder", 12: "Hexaeder", 13: "Prisma", 14: "Pyramide"}

# component_id -> (Name, r-Bereich, z-Bereich) laut fem.cu. None heisst
# "nicht pruefen".
#
# Die Spulen stehen fest, ihre Masse sind die Pruefung. Bei der PLATTE wird
# nur der Aussenradius und die DICKE geprueft, nicht die Hoehe: die Hoehe ist
# genau die Groesse, die zwischen Startlage und Gleichgewichtspunkt variiert
# (team28_simple_air.bdf gegen team28_simple_air_113.bdf). Eine feste
# z-Erwartung dort wuerde bei jedem neuen Netz falsch anschlagen.
#
# r_min der Platte wird nicht als Mass geprueft. Der kleinste Knotenradius
# sagt nichts ueber ein Loch: im Standardnetz team28.vtk liegt er bei
# r = 1,84 mm, weil dort eine Mittelzelle die Achse ueberdeckt, deren Ecken so
# weit aussen liegen -- die Platte ist trotzdem eine Vollscheibe.
#
# Stattdessen wird geprueft, dass die Platte die Achse UEBERDECKT (Knoten auf
# der Achse oder eine Zelle, deren Grundriss die Achse umschliesst). Ein Loch
# ist ausgeschlossen: es aendert die Topologie (ein zusaetzlicher
# Kohomologie-Generator, die Platte wird zum Volltorus) und die Geometrie.
# Siehe docs/design/gmsh-meshing-plan.md.
EXPECTED = {
    1: ("Aussenspule", (80.0, 95.0), (-52.0, 0.0)),
    2: ("Innenspule", (27.0, 55.0), (-52.0, 0.0)),
    3: ("Platte", (None, 65.0), None),
}
PLATTENDICKE = 3.0   # mm


def covers_axis(pts, cells, idx, tol=1e-9):
    """True, wenn die Zellen idx die z-Achse ueberdecken (Vollscheibe).

    Entweder liegt ein Knoten auf der Achse, oder der Grundriss einer Zelle
    umschliesst sie: dann haben die Knotenwinkel um die Achse keine Luecke von
    pi oder mehr (die Achse liegt in der konvexen Huelle).
    """
    for i in idx:
        xy = pts[cells[i]][:, :2]
        r = np.hypot(xy[:, 0], xy[:, 1])
        if (r < tol).any():
            return True
        ang = np.sort(np.arctan2(xy[:, 1], xy[:, 0]))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
        if gaps.max() < np.pi - 1e-12:
            return True
    return False


def main(path):
    pts, cells, ctypes, comp = read_legacy_vtk(path)

    print(f"{path}: {len(pts)} Knoten, {len(cells)} Zellen")
    for t, n in zip(*np.unique(ctypes, return_counts=True)):
        print(f"  Typ {t} ({VTK_NAMES.get(int(t), '?')}): {n}")

    unknown = set(int(t) for t in np.unique(ctypes)) - set(VTK_NAMES)
    if unknown:
        print(f"  ACHTUNG: VtkReader lehnt Zelltypen {sorted(unknown)} ab.")

    print("\ncomponent_id   Elemente        r-Bereich         z-Bereich   erwartet")
    beanstandet = []

    for cid in sorted(np.unique(comp)):
        mask = comp == cid
        idx = [i for i, m in enumerate(mask) if m]
        nodes = pts[np.concatenate([cells[i] for i in idx])]
        r = np.hypot(nodes[:, 0], nodes[:, 1])
        z = nodes[:, 2]

        name, r_exp, z_exp = EXPECTED.get(int(cid), ("Luft", None, None))
        verdict = name

        if r_exp is not None:
            # 1,5 mm Toleranz, das Netz loest die Radien als Kantenzuege auf
            ok = abs(r.max() - r_exp[1]) < 1.5
            if r_exp[0] is not None:
                ok = ok and abs(r.min() - r_exp[0]) < 1.5
            if z_exp is not None:
                ok = ok and abs(z.min() - z_exp[0]) < 1.5 \
                        and abs(z.max() - z_exp[1]) < 1.5
            elif int(cid) == 3:
                # Statt der Hoehe die Dicke pruefen -- die aendert sich nicht
                ok = ok and abs((z.max() - z.min()) - PLATTENDICKE) < 0.5
                # Vollscheibe: die Platte muss die Achse ueberdecken (kein Loch)
                if not covers_axis(pts, cells, idx):
                    ok = False
                    name = "Platte (LOCH auf der Achse -- ausgeschlossen)"
            if not ok:
                verdict = f"{name}  <== PASST NICHT"
                beanstandet.append(int(cid))

        print(f"{cid:>10}   {mask.sum():8d}   "
              f"{r.min():6.1f}..{r.max():6.1f}   "
              f"{z.min():7.1f}..{z.max():7.1f}   {verdict}")

    # Die Plattenhoehe ist keine Pruefung, aber die Zahl, die den Lauf
    # einordnet: bei welcher Hoehe die Kraft gemessen wird.
    for cid in sorted(np.unique(comp)):
        if int(cid) != 3:
            continue
        idx = [i for i, m in enumerate(comp == cid) if m]
        nodes = pts[np.concatenate([cells[i] for i in idx])]
        z = nodes[:, 2]
        print()
        print(f"Plattenlage: z = {z.min():.2f} .. {z.max():.2f} mm, "
              f"Mitte {0.5*(z.min()+z.max()):.2f} mm, "
              f"Dicke {z.max()-z.min():.2f} mm")
        print("  Bei 11,3 mm nennt HarmonicLorentz.cuh 1,0497 N als Sollwert,")
        print("  das Plattengewicht ist 1,026 N. Tiefer -> groessere Kraft.")

    print()
    if not beanstandet:
        print("Zuordnung passt zur Konvention in src/fem.cu.")
        return 0

    print(f"Beanstandet: component_id {beanstandet}.")
    if set(beanstandet) == {1, 2}:
        print("Beide Spulen beanstandet -- das ist der Vertauschungsfall.")
        print("Nochmal konvertieren mit vertauschten Spulen-PIDs, also")
        print("  --map 1=2,2=1,3=3,4=0   statt   --map 1=1,2=2,3=3,4=0")
    else:
        print("Kein reiner Vertauschungsfall. Die Geometrie des Netzes weicht von")
        print("den in EXPECTED eingetragenen TEAM-28-Massen ab -- erst klaeren,")
        print("welche der beiden stimmt, bevor du weiterrechnest.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
