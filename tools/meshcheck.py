#!/usr/bin/env python3
"""
Pruefungen fuer Netze im Legacy-VTK-Format, das src/VtkReader.cpp liest.

    python3 tools/meshcheck.py netz.vtk [--require-hex]

Die meisten Netzfehler fuehren nicht zum Absturz, sondern zu einem falschen
Feld. Dieses Modul prueft deshalb alles, was still schiefgehen kann, und
liefert eine Tabelle mit Befund je Pruefung.

GEPRUEFT
  * Zelltypen: nur Tetraeder (10), Hexaeder (12), Prismen (13), erster
    Ordnung -- wie VtkReader. Mit --require-hex nur Hexaeder: die BEM wird
    nur bei reinen Hexaedernetzen aufgebaut (mesh.is_pure_hex() in fem.cu).
  * Doppelte Knoten (gleiche Koordinaten, verschiedene Nummern). Typisch an
    der Naht eines gedrehten Netzes; erzeugt einen unsichtbaren Schnitt.
  * Verwaiste Knoten (von keiner Zelle benutzt).
  * Mit --separate-bodies: kein Knoten gehoert zu zwei component_ids
    (Leiter ohne Luft, BEM-Pfad).
  * Konformitaet: jede Flaeche gehoert hoechstens zwei Zellen.
  * Orientierung: Jacobi-Determinante an allen Ecken positiv (Hexaeder,
    Tetraeder). Prismen: noch nicht (Phase 3 der Gmsh-Vernetzung).
  * Volumen je component_id, auf Wunsch gegen Sollwerte.

Phase 1 (docs/design/gmsh-meshing-plan.md) ergaenzt die Pruefungen aus
tools/import_bdf.py (Prismen-Affinitaet, Einheiten) und laesst beide
Importeure dieses Modul benutzen.
"""
import argparse
import sys
from collections import namedtuple

import numpy as np

VTK_TET, VTK_HEX, VTK_PRISM = 10, 12, 13
TYPE_NAMES = {VTK_TET: "Tetraeder", VTK_HEX: "Hexaeder", VTK_PRISM: "Prisma"}
NODES_PER_TYPE = {VTK_TET: 4, VTK_HEX: 8, VTK_PRISM: 6}

# Flaechen je Zelltyp in VTK-Knotenreihenfolge
FACES = {
    VTK_TET: [(0, 1, 2), (0, 1, 3), (1, 2, 3), (0, 2, 3)],
    VTK_HEX: [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
              (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
    VTK_PRISM: [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
}
# Hexaeder: je Ecke die drei Nachbarn in rechtshaendiger Reihenfolge
HEX_CORNERS = [(0, 1, 3, 4), (1, 2, 0, 5), (2, 3, 1, 6), (3, 0, 2, 7),
               (4, 7, 5, 0), (5, 4, 6, 1), (6, 5, 7, 2), (7, 6, 4, 3)]
# Hexaeder-Volumen: Zerlegung in 5 Tetraeder
HEX_TETS = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)]

Mesh = namedtuple("Mesh", "points cells types comp")
Result = namedtuple("Result", "name ok detail")


def read_vtk(path):
    """Legacy-VTK (ASCII, UNSTRUCTURED_GRID) lesen, wie VtkReader es erwartet."""
    tokens = open(path, "r", errors="replace").read().split()

    def find(word, start=0):
        try:
            return tokens.index(word, start)
        except ValueError:
            raise ValueError(f"{path}: Abschnitt {word} fehlt") from None

    i = find("POINTS")
    n_pts = int(tokens[i + 1])
    pts = np.array(tokens[i + 3:i + 3 + 3 * n_pts], dtype=float).reshape(n_pts, 3)

    i = find("CELLS", i)
    n_cells, n_ints = int(tokens[i + 1]), int(tokens[i + 2])
    flat = np.array(tokens[i + 3:i + 3 + n_ints], dtype=np.int64)
    cells, p = [], 0
    while p < n_ints:
        n = int(flat[p])
        cells.append(flat[p + 1:p + 1 + n])
        p += 1 + n

    i = find("CELL_TYPES", i)
    types = np.array(tokens[i + 2:i + 2 + n_cells], dtype=int)

    comp = np.zeros(n_cells, dtype=int)
    if "CELL_DATA" in tokens:
        # "SCALARS <name> <typ> [<anzahl>]", danach "LOOKUP_TABLE <name>"
        # (bei VtkReader und Gmsh immer vorhanden), dann die Werte.
        j = find("SCALARS", find("CELL_DATA"))
        k = find("LOOKUP_TABLE", j)
        if k > j + 4:
            raise ValueError(f"{path}: SCALARS ohne LOOKUP_TABLE wird nicht unterstuetzt")
        comp = np.array(tokens[k + 2:k + 2 + n_cells], dtype=float).astype(int)
    return Mesh(pts, cells, types, comp)


def _by_type(mesh, t):
    idx = np.flatnonzero(mesh.types == t)
    if len(idx) == 0:
        return idx, np.empty((0, NODES_PER_TYPE[t]), dtype=np.int64)
    return idx, np.array([mesh.cells[i] for i in idx], dtype=np.int64)


def check_types(mesh, require_hex=False):
    bad = sorted(set(mesh.types.tolist()) - set(NODES_PER_TYPE))
    if bad:
        return Result("Zelltypen", False, f"nicht unterstuetzt: {bad} (VtkReader: 10, 12, 13)")
    wrong_len = [i for i, c in enumerate(mesh.cells) if len(c) != NODES_PER_TYPE[mesh.types[i]]]
    if wrong_len:
        return Result("Zelltypen", False, f"{len(wrong_len)} Zellen mit falscher Knotenzahl (zweite Ordnung?)")
    counts = {TYPE_NAMES[t]: int((mesh.types == t).sum()) for t in NODES_PER_TYPE if (mesh.types == t).any()}
    if require_hex and set(counts) != {"Hexaeder"}:
        return Result("Zelltypen", False, f"{counts} -- BEM braucht reine Hexaedernetze")
    return Result("Zelltypen", True, ", ".join(f"{v} {k}" for k, v in counts.items()))


def check_duplicate_nodes(mesh, rel_tol=1e-9):
    span = float(np.ptp(mesh.points, axis=0).max()) or 1.0
    q = np.round(mesh.points / (rel_tol * span)).astype(np.int64)
    n_unique = len(np.unique(q, axis=0))
    dup = len(mesh.points) - n_unique
    return Result("Doppelte Knoten", dup == 0, f"{dup}")


def check_orphan_nodes(mesh):
    used = np.zeros(len(mesh.points), dtype=bool)
    for c in mesh.cells:
        used[c] = True
    n = int((~used).sum())
    return Result("Verwaiste Knoten", n == 0, f"{n}")


def check_separate_bodies(mesh):
    """Kein Knoten gehoert zu zwei component_ids.

    Fuer den BEM-Pfad (Leiter ohne Luft) muessen die Koerper getrennt sein;
    ein gemeinsamer Knoten verbindet sie leitend. Fuer Netze mit Luft
    (FEM-direkt) gilt das nicht -- dort die Pruefung nicht verwenden.
    """
    owner = np.full(len(mesh.points), -1, dtype=np.int64)
    shared = set()
    for c, cid in zip(mesh.cells, mesh.comp):
        prev = owner[c]
        clash = (prev >= 0) & (prev != cid)
        if clash.any():
            shared.update(c[clash].tolist())
        owner[c] = cid
    return Result("Koerper getrennt", not shared, f"{len(shared)} gemeinsame Knoten")


def check_conformity(mesh):
    counts = {}
    for t, faces in FACES.items():
        _, conn = _by_type(mesh, t)
        if len(conn) == 0:
            continue
        for f in faces:
            keys = np.sort(conn[:, list(f)], axis=1)
            for row in map(tuple, keys):
                counts[row] = counts.get(row, 0) + 1
    vals = np.fromiter(counts.values(), dtype=int)
    over = int((vals > 2).sum())
    detail = f"innen {int((vals == 2).sum())}, Rand {int((vals == 1).sum())}, mehr als zwei Zellen {over}"
    return Result("Konformitaet", over == 0, detail)


def check_orientation(mesh):
    P = mesh.points
    neg, smallest = 0, np.inf
    _, hexes = _by_type(mesh, VTK_HEX)
    if len(hexes):
        for v, a, b, d in HEX_CORNERS:
            J = np.linalg.det(np.stack([P[hexes[:, a]] - P[hexes[:, v]],
                                        P[hexes[:, b]] - P[hexes[:, v]],
                                        P[hexes[:, d]] - P[hexes[:, v]]], axis=1))
            neg += int((J <= 0).sum())
            smallest = min(smallest, float(J.min()))
    _, tets = _by_type(mesh, VTK_TET)
    if len(tets):
        J = np.linalg.det(np.stack([P[tets[:, 1]] - P[tets[:, 0]],
                                    P[tets[:, 2]] - P[tets[:, 0]],
                                    P[tets[:, 3]] - P[tets[:, 0]]], axis=1))
        neg += int((J <= 0).sum())
        smallest = min(smallest, float(J.min()))
    note = " (Prismen nicht geprueft)" if (mesh.types == VTK_PRISM).any() else ""
    return Result("Jacobi-Determinante > 0", neg == 0,
                  f"{neg} Ecken nicht positiv, kleinste {smallest:.3g}{note}")


def volumes(mesh):
    """Volumen je component_id (Hexaeder und Tetraeder)."""
    P, vol = mesh.points, np.zeros(len(mesh.cells))
    idx, hexes = _by_type(mesh, VTK_HEX)
    if len(hexes):
        v = np.zeros(len(hexes))
        for a, b, c, d in HEX_TETS:
            v += np.abs(np.linalg.det(np.stack([P[hexes[:, b]] - P[hexes[:, a]],
                                                P[hexes[:, c]] - P[hexes[:, a]],
                                                P[hexes[:, d]] - P[hexes[:, a]]], axis=1))) / 6.0
        vol[idx] = v
    idx, tets = _by_type(mesh, VTK_TET)
    if len(tets):
        vol[idx] = np.abs(np.linalg.det(np.stack([P[tets[:, 1]] - P[tets[:, 0]],
                                                  P[tets[:, 2]] - P[tets[:, 0]],
                                                  P[tets[:, 3]] - P[tets[:, 0]]], axis=1))) / 6.0
    return {int(c): float(vol[mesh.comp == c].sum()) for c in np.unique(mesh.comp)}


def check_volumes(mesh, expected, rel_tol=1e-9, names=None):
    got, out = volumes(mesh), []
    for cid, want in expected.items():
        have = got.get(cid, 0.0)
        rel = abs(have - want) / want
        label = (names or {}).get(cid, f"component_id {cid}")
        out.append(Result(f"Volumen {label}", rel <= rel_tol,
                          f"{have:.4f} mm^3, Soll {want:.4f}, rel. {rel:.1e}"))
    return out


def run_checks(mesh, require_hex=False, expected_volumes=None, names=None,
               separate_bodies=False):
    res = [check_types(mesh, require_hex), check_duplicate_nodes(mesh),
           check_orphan_nodes(mesh)]
    if separate_bodies:
        res.append(check_separate_bodies(mesh))
    res += [check_conformity(mesh), check_orientation(mesh)]
    if expected_volumes:
        res += check_volumes(mesh, expected_volumes, names=names)
    return res


def print_results(results, stream=sys.stdout):
    width = max(len(r.name) for r in results)
    for r in results:
        print(f"  {r.name:<{width}}  {'ok' if r.ok else 'FEHLER':6s}  {r.detail}", file=stream)
    return all(r.ok for r in results)


def main():
    ap = argparse.ArgumentParser(description="Netzpruefungen fuer FEMBEM (Legacy-VTK).")
    ap.add_argument("vtk")
    ap.add_argument("--require-hex", action="store_true", help="nur Hexaeder zulassen (BEM-Pfad)")
    ap.add_argument("--separate-bodies", action="store_true",
                    help="Koerper duerfen keine Knoten teilen (BEM-Pfad, ohne Luft)")
    a = ap.parse_args()
    mesh = read_vtk(a.vtk)
    print(f"{a.vtk}: {len(mesh.points)} Knoten, {len(mesh.cells)} Zellen")
    ok = print_results(run_checks(mesh, require_hex=a.require_hex,
                                  separate_bodies=a.separate_bodies))
    for cid, v in volumes(mesh).items():
        print(f"  Volumen component_id {cid}: {v:.4f}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
