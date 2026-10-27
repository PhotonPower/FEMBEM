#!/usr/bin/env python3
"""
Tests fuer tools/meshcheck.py: jede Pruefung muss an einem absichtlich
kaputten Netz anschlagen und an einem korrekten bestehen.

    python3 tools/tests/test_meshcheck.py

Braucht nur numpy. Endet mit 0, wenn alles bestanden ist.
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import meshcheck as mc  # noqa: E402

FAILS = []


def expect(name, cond):
    print(f"  {'ok' if cond else 'FEHLGESCHLAGEN':14s} {name}")
    if not cond:
        FAILS.append(name)


def cube(origin=(0.0, 0.0, 0.0), h=1.0):
    """Knoten eines Wuerfels in VTK-Reihenfolge (unten 0-3 gegen den Uhrzeigersinn, oben 4-7)."""
    x, y, z = origin
    return np.array([[x, y, z], [x + h, y, z], [x + h, y + h, z], [x, y + h, z],
                     [x, y, z + h], [x + h, y, z + h], [x + h, y + h, z + h], [x, y + h, z + h]])


def two_cubes_shared():
    """Zwei Wuerfel, die sich eine Flaeche (x = 1) und deren 4 Knoten teilen."""
    pts = np.vstack([cube(), cube((1, 0, 0))[[1, 2, 5, 6]]])     # 8 + 4 Knoten
    c0 = np.arange(8)
    c1 = np.array([1, 8, 9, 2, 5, 10, 11, 6])
    return pts, [c0, c1]


def mesh(pts, cells, types=None, comp=None):
    n = len(cells)
    return mc.Mesh(np.asarray(pts, float), [np.asarray(c) for c in cells],
                   np.array(types if types is not None else [mc.VTK_HEX] * n),
                   np.array(comp if comp is not None else [1] * n))


def by_name(results, prefix):
    return next(r for r in results if r.name.startswith(prefix))


def main():
    print("Korrektes Netz (zwei Wuerfel, gemeinsame Flaeche):")
    pts, cells = two_cubes_shared()
    res = mc.run_checks(mesh(pts, cells), require_hex=True, expected_volumes={1: 2.0})
    for r in res:
        expect(f"{r.name} besteht ({r.detail})", r.ok)

    print("Doppelte Knoten (Naht, nicht verschmolzen):")
    pts2 = np.vstack([cube(), cube((1, 0, 0))])                  # 16 Knoten, 4 doppelt
    m = mesh(pts2, [np.arange(8), np.arange(8, 16)])
    r = mc.check_duplicate_nodes(m)
    expect(f"erkannt ({r.detail})", not r.ok and r.detail == "4")

    print("Verwaister Knoten:")
    m = mesh(np.vstack([pts, [[9.0, 9.0, 9.0]]]), cells)
    expect("erkannt", not mc.check_orphan_nodes(m).ok)

    print("Koerper getrennt:")
    expect("zwei Koerper ohne gemeinsamen Knoten bestehen",
           mc.check_separate_bodies(mesh(pts2, [np.arange(8), np.arange(8, 16)], comp=[1, 2])).ok)
    r = mc.check_separate_bodies(mesh(pts, cells, comp=[1, 3]))
    expect(f"gemeinsame Flaeche zweier Koerper erkannt ({r.detail})", not r.ok and r.detail.startswith("4 "))

    print("Flaeche in drei Zellen (doppelte Zelle):")
    m = mesh(pts, cells + [cells[1]])
    expect("erkannt", not mc.check_conformity(m).ok)

    print("Falsch orientierter Hexaeder (oben/unten vertauscht):")
    flipped = np.concatenate([cells[1][4:], cells[1][:4]])
    m = mesh(pts, [cells[0], flipped])
    r = mc.check_orientation(m)
    expect(f"erkannt ({r.detail})", not r.ok)

    print("Zelltypen:")
    m = mesh(pts, cells, types=[mc.VTK_HEX, 25])                 # 25 = Hexaeder 2. Ordnung
    expect("unbekannter Typ erkannt", not mc.check_types(m).ok)
    tet_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    m = mesh(tet_pts, [np.arange(4)], types=[mc.VTK_TET])
    expect("Tetraeder ohne --require-hex zulaessig", mc.check_types(m).ok)
    expect("Tetraeder mit --require-hex abgelehnt", not mc.check_types(m, require_hex=True).ok)
    expect("Tetraeder-Volumen 1/6", abs(mc.volumes(m)[1] - 1 / 6) < 1e-15)

    print("Prismen (Konvention des Loesers, include/NedelecPrism.cuh):")
    pr_pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]], float)
    m = mesh(pr_pts, [np.arange(6)], types=[mc.VTK_PRISM])
    r = mc.check_orientation(m)
    expect(f"richtig orientiertes Prisma besteht ({r.detail})", r.ok)
    expect("Prisma-Volumen 1/2", abs(mc.volumes(m)[1] - 0.5) < 1e-15)
    m = mesh(pr_pts, [np.array([0, 2, 1, 3, 5, 4])], types=[mc.VTK_PRISM])
    r = mc.check_orientation(m)
    expect(f"umgedrehtes Prisma erkannt, wie Gmsh es schreibt ({r.detail})",
           not r.ok and r.detail.startswith("6 "))
    sheared = pr_pts.copy(); sheared[3:] += [0.3, 0.2, 0.0]      # affin geschert
    m = mesh(sheared, [np.arange(6)], types=[mc.VTK_PRISM])
    expect("geschertes Prisma: Volumen unveraendert 1/2", abs(mc.volumes(m)[1] - 0.5) < 1e-15)

    print("Volumen gegen Sollwert:")
    r = mc.check_volumes(mesh(pts, cells), {1: 2.0 * (1 + 1e-6)})[0]
    expect(f"Abweichung 1e-6 erkannt ({r.detail})", not r.ok)

    print("Lesen einer VTK-Datei (Rundreise, component_id):")
    with tempfile.NamedTemporaryFile("w", suffix=".vtk", delete=False) as f:
        f.write("# vtk DataFile Version 2.0\ntest\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(pts)} double\n")
        f.writelines(" ".join(map(str, p)) + "\n" for p in pts)
        f.write(f"CELLS 2 18\n8 {' '.join(map(str, cells[0]))}\n8 {' '.join(map(str, cells[1]))}\n")
        f.write("CELL_TYPES 2\n12\n12\nCELL_DATA 2\nSCALARS CellEntityIds int 1\nLOOKUP_TABLE default\n3\n1\n")
        name = f.name
    m = mc.read_vtk(name)
    os.unlink(name)
    expect("Knoten und Zellen gelesen", len(m.points) == 12 and len(m.cells) == 2)
    expect("component_id gelesen (3, 1)", m.comp.tolist() == [3, 1])

    print()
    print("Alle Tests bestanden." if not FAILS else f"{len(FAILS)} Test(s) fehlgeschlagen.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
