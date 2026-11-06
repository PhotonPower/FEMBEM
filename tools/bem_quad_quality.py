#!/usr/bin/env python3
"""Qualitaet der BEM-Randquads je Koerper und Lage (Seitenverhaeltnis, Winkel, Windschiefe).

Je Netz (TEAM-28-Hexaeder, Komponenten 1 Aussenspule, 2 Innenspule, 3 Platte) und je Gruppe
Deckel/Boden (|n_z| > 0.99) bzw. Seiten:
  AR     Seitenverhaeltnis laengste / kuerzeste Kante des Quads
  Winkel groesste Abweichung eines Innenwinkels von 90 Grad
  Warp   Abstand der vierten Ecke von der Ebene der ersten drei, durch mittlere Kante
und je Gruppe Minimum / Mittel / Maximum. Ausserdem je Koerper das Seitenverhaeltnis der
Hexaederzellen (laengste / kuerzeste der 12 Kanten).

    python3 tools/bem_quad_quality.py NETZ.vtk [NETZ2.vtk ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshcheck  # noqa: E402
from bem_panel_sizes import HEX_FACES, NAMES  # noqa: E402

HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def quad_metrics(p):
    e = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    ar = max(e) / min(e)
    dev = 0.0
    for i in range(4):
        a, b = p[i - 1] - p[i], p[(i + 1) % 4] - p[i]
        ang = np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)))
        dev = max(dev, abs(ang - 90.0))
    n = np.cross(p[1] - p[0], p[2] - p[0])
    n /= np.linalg.norm(n)
    warp = abs((p[3] - p[0]) @ n) / np.mean(e)
    return ar, dev, warp


def main():
    print("| Netz | Koerper | Lage | Quads | AR min / mittel / max | Winkelabw. max [Grad] | Warp max |")
    print("|---|---|---|---:|---|---:|---:|")
    cells = []
    for path in sys.argv[1:]:
        mesh = meshcheck.read_vtk(path)
        pts = mesh.points
        owners = {}
        for ci, conn in enumerate(mesh.cells):
            for f in HEX_FACES:
                owners.setdefault(tuple(sorted(int(conn[k]) for k in f)), []).append((ci, [int(conn[k]) for k in f]))
        groups = {}
        for lst in owners.values():
            if len(lst) != 1:
                continue
            ci, quad = lst[0]
            body = int(mesh.comp[ci])
            p = pts[quad]
            nn = np.cross(p[1] - p[0], p[3] - p[0])
            lid = abs(nn[2] / np.linalg.norm(nn)) > 0.99
            groups.setdefault((body, "Deckel/Boden" if lid else "Seiten"), []).append(quad_metrics(p))
        name = os.path.basename(path)
        for (body, kind), g in sorted(groups.items()):
            a = np.array(g)
            print(f"| {name} | {NAMES[body]} | {kind} | {len(g)} | {a[:,0].min():.2f} / {a[:,0].mean():.2f} / {a[:,0].max():.2f} "
                  f"| {a[:,1].max():.2f} | {a[:,2].max():.3f} |")
        for body in (1, 2, 3):
            ars = []
            for ci, conn in enumerate(mesh.cells):
                if mesh.comp[ci] != body:
                    continue
                ln = [np.linalg.norm(pts[conn[a]] - pts[conn[b]]) for a, b in HEX_EDGES]
                ars.append(max(ln) / min(ln))
            cells.append((name, NAMES[body], len(ars), min(ars), float(np.mean(ars)), max(ars)))
    print("\n| Netz | Koerper | Hexaeder | AR min / mittel / max (12 Kanten) |")
    print("|---|---|---:|---|")
    for name, b, n, lo, me, hi in cells:
        print(f"| {name} | {b} | {n} | {lo:.2f} / {me:.2f} / {hi:.2f} |")


if __name__ == "__main__":
    main()
