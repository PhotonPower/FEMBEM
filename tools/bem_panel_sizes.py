#!/usr/bin/env python3
"""Misst die Panelkanten der BEM-Randflaechen je Koerper und Lage aus einem Hexaedernetz.

Fuer jedes Netz (TEAM-28-Hexaeder, Komponenten 1 Aussenspule, 2 Innenspule, 3 Platte):
Randquads je Koerper, getrennt nach Deckel/Boden (|n_z| > 0.99) und Seiten; je Gruppe
kleinste, mittlere und groesste Kantenlaenge sowie sqrt(Flaeche) in mm. Fuer die Platte
zusaetzlich die Dicke (z-Ausdehnung) und das Verhaeltnis mittlere Deckelkante / Dicke.

    python3 tools/bem_panel_sizes.py NETZ.vtk [NETZ2.vtk ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshcheck  # noqa: E402

NAMES = {1: "Aussenspule", 2: "Innenspule", 3: "Platte"}
HEX_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]


def quad_stats(pts, quad):
    p = pts[quad]
    edges = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    n = np.cross(p[1] - p[0], p[3] - p[0]) + np.cross(p[2] - p[1], p[0] - p[1])
    area = 0.5 * np.linalg.norm(n)
    nn = n / np.linalg.norm(n)
    return edges, area, nn


def analyse(path):
    mesh = meshcheck.read_vtk(path)
    pts = mesh.points
    scale = 1.0   # Netz in mm
    owners = {}
    for ci, conn in enumerate(mesh.cells):
        if len(conn) != 8:
            continue
        for f in HEX_FACES:
            key = tuple(sorted(int(conn[k]) for k in f))
            owners.setdefault(key, []).append((ci, [int(conn[k]) for k in f]))
    groups = {}
    for key, lst in owners.items():
        if len(lst) != 1:
            continue
        ci, quad = lst[0]
        body = int(mesh.comp[ci])
        if body not in NAMES:
            continue
        edges, area, nn = quad_stats(pts, quad)
        lid = abs(nn[2]) > 0.99
        g = groups.setdefault((body, "Deckel/Boden" if lid else "Seiten"), {"e": [], "a": []})
        g["e"].extend(edges)
        g["a"].append(area)
    out = {}
    for (body, kind), g in sorted(groups.items()):
        e = np.array(g["e"]) * scale
        out[(body, kind)] = (len(g["a"]), e.min(), e.mean(), e.max(), float(np.sqrt(np.mean(g["a"]))))
    plate = np.array([c for ci, c in enumerate(mesh.cells) if mesh.comp[ci] == 3 and len(c) == 8])
    zs = pts[np.unique(plate.ravel()), 2]
    return out, zs.max() - zs.min()


def main():
    print("| Netz | Koerper | Lage | Quads | Kante min [mm] | Kante mittel [mm] | Kante max [mm] | sqrt(Flaeche) [mm] |")
    print("|---|---|---|---:|---:|---:|---:|---:|")
    thick = {}
    for path in sys.argv[1:]:
        out, t = analyse(path)
        name = os.path.basename(path)
        thick[name] = (t, out)
        for (body, kind), (n, emin, emean, emax, sq) in out.items():
            print(f"| {name} | {NAMES[body]} | {kind} | {n} | {emin:.3f} | {emean:.3f} | {emax:.3f} | {sq:.3f} |")
    print()
    print("| Netz | Plattendicke [mm] | Deckelkante mittel [mm] | sqrt(Flaeche) Deckel [mm] | Kante/Dicke | sqrt(A)/Dicke |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, (t, out) in thick.items():
        n, emin, emean, emax, sq = out[(3, "Deckel/Boden")]
        print(f"| {name} | {t:.3f} | {emean:.3f} | {sq:.3f} | {emean / t:.3f} | {sq / t:.3f} |")


if __name__ == "__main__":
    main()
