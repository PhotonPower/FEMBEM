"""
Gemeinsame Bausteine der Gmsh-Werkzeuge: Zellen in die Konvention des Loesers
drehen und Legacy-VTK so schreiben, wie src/VtkReader.cpp es liest.

Benutzt von tools/gmsh/team28_fem_air.py und tools/gmsh/import_msh.py.
team28_hex.py schreibt mit gmsh.write(), weil Gmsh Hexaeder bereits in der
Reihenfolge des Loesers liefert und kein Umsortieren noetig ist.

WARUM DREHEN
  Der Loeser erwartet je Zelle eine positive Orientierung in seiner eigenen
  Konvention (Prismen: include/NedelecPrism.cuh, Knoten 0-2 unten, 3-5 oben,
  (x1-x0, x2-x0, x3-x0) rechtshaendig). Gmsh schreibt Prismen in dieser
  Konvention umgekehrt orientiert (docs/design/gmsh-meshing-plan.md,
  "Ergebnis Phase 1"); falsch orientierte Zellen rechnen ohne Fehlermeldung
  mit falschem Vorzeichen. Gedreht wird jede Zelle EINZELN nach ihrer
  Determinante, nicht pauschal je Typ: Ob Gmsh umgekehrt schreibt, haengt vom
  Weg ab, auf dem das Prisma entstand (die Extrusion in team28_fem_air.py
  liefert sie richtig), und fremde .msh-Dateien koennen gemischt sein.

WARUM GENAU DIESES VTK-LAYOUT
  CELLS im Layout vor VTK 5.1 ("n v0 v1 ..."), weil VtkReader die neuere Form
  mit OFFSETS/CONNECTIVITY nicht liest, und GENAU EIN SCALARS-Array unter
  CELL_DATA, weil VtkReader das erste nimmt und den Namen ignoriert.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from meshcheck import NODES_PER_TYPE, VTK_HEX, VTK_PRISM, VTK_TET  # noqa: E402

# Knoten, aus denen die Orientierung bestimmt wird: Ecke 0 und ihre drei
# Nachbarn, so dass (x_a-x_0, x_b-x_0, x_c-x_0) bei richtiger Orientierung ein
# Rechtssystem ist (wie meshcheck.HEX_CORNERS[0] bzw. PRISM_CORNERS[0]).
REF_CORNER = {VTK_TET: (0, 1, 2, 3), VTK_HEX: (0, 1, 3, 4), VTK_PRISM: (0, 1, 2, 3)}
# Umsortierung, die die Orientierung umkehrt und die Rollen der Knoten
# (unten/oben beim Prisma und Hexaeder) erhaelt.
FLIP = {VTK_TET: [0, 2, 1, 3], VTK_HEX: [0, 3, 2, 1, 4, 7, 6, 5],
        VTK_PRISM: [0, 2, 1, 3, 5, 4]}


def corner_dets(points, vtk_type, conn):
    """Determinante an Ecke 0 je Zelle (Vorzeichen = Orientierung)."""
    v, a, b, c = REF_CORNER[vtk_type]
    P = points
    return np.linalg.det(np.stack([P[conn[:, a]] - P[conn[:, v]],
                                   P[conn[:, b]] - P[conn[:, v]],
                                   P[conn[:, c]] - P[conn[:, v]]], axis=1))


def orient_to_solver(points, vtk_type, conn, rel_tol=1e-12):
    """Negativ orientierte Zellen umsortieren; liefert (conn, Anzahl gedreht).

    Entartete Zellen (Determinante ~ 0) sind kein Fall zum Drehen, sondern ein
    Netzfehler -> ValueError. Beim Hexaeder entscheidet nur Ecke 0; ein in sich
    verdrehter Hexaeder bleibt an den anderen Ecken negativ, das meldet
    meshcheck.check_orientation.
    """
    conn = np.array(conn, dtype=np.int64, copy=True)
    if len(conn) == 0:
        return conn, 0
    det = corner_dets(points, vtk_type, conn)
    scale = np.abs(det).max()
    degenerate = np.abs(det) <= rel_tol * scale
    if degenerate.any():
        raise ValueError(f"{int(degenerate.sum())} entartete Zellen vom VTK-Typ {vtk_type} "
                         "(Volumen ~ 0)")
    flip = det < 0
    conn[flip] = conn[flip][:, FLIP[vtk_type]]
    return conn, int(flip.sum())


def compact_nodes(points, conns):
    """Nur benutzte Knoten behalten und fortlaufend nummerieren.

    Gmsh liefert alle Knoten des Modells, auch solche auf Kurven und Flaechen
    ohne Volumenzelle; VtkReader und meshcheck wuerden sie als verwaist
    fuehren. Knoten werden NICHT verschmolzen (siehe gmsh-meshing-plan.md).
    """
    used = np.unique(np.concatenate([c.ravel() for c in conns]))
    renum = np.full(len(points), -1, dtype=np.int64)
    renum[used] = np.arange(len(used))
    return points[used], [renum[c] for c in conns]


def write_legacy_vtk(path, points, blocks, title):
    """Schreibt Legacy-VTK ASCII; blocks = Liste von (vtk_type, conn, comp)."""
    n = sum(len(conn) for _, conn, _ in blocks)
    n_ints = sum(len(conn) * (NODES_PER_TYPE[t] + 1) for t, conn, _ in blocks)
    with open(path, "w", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(title.replace("\n", " ") + "\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(points)} double\n")
        for p in points:
            f.write("%.17g %.17g %.17g\n" % tuple(p))
        f.write(f"\nCELLS {n} {n_ints}\n")
        for t, conn, _ in blocks:
            k = NODES_PER_TYPE[t]
            fmt = f"{k}" + " %d" * k + "\n"
            for c in conn:
                f.write(fmt % tuple(c))
        f.write(f"\nCELL_TYPES {n}\n")
        for t, conn, _ in blocks:
            f.write(f"{t}\n" * len(conn))
        # GENAU EIN SCALARS-Array: VtkReader nimmt das erste
        f.write(f"\nCELL_DATA {n}\nSCALARS component_id int 1\nLOOKUP_TABLE default\n")
        for _, conn, comp in blocks:
            f.writelines(f"{int(c)}\n" for c in np.broadcast_to(comp, (len(conn),)))
