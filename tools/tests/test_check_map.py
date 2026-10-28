#!/usr/bin/env python3
"""
Tests fuer die Vollscheiben-Pruefung in tools/check_map.py (covers_axis).

Die Platte muss die Achse ueberdecken -- ein Loch ist ausgeschlossen
(docs/design/gmsh-meshing-plan.md). Die Pruefung darf eine Vollscheibe nie
als Loch melden, egal ob die Achse auf einem Knoten, in einer Zelle oder auf
einer Kante zwischen zwei Zellen liegt, und muss jedes Loch erkennen.

    python3 tools/tests/test_check_map.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import check_map  # noqa: E402

FAILS = []


def expect(name, cond):
    print(f"  {'ok' if cond else 'FEHLGESCHLAGEN':14s} {name}")
    if not cond:
        FAILS.append(name)


def box(x0, x1, y0, y1, z0=0.0, z1=1.0):
    """Hexaeder-Knoten (VTK-Reihenfolge) eines achsparallelen Quaders."""
    return np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                     [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])


def as_mesh(boxes):
    pts = np.vstack(boxes)
    cells = [np.arange(8 * i, 8 * i + 8) for i in range(len(boxes))]
    return pts, cells, list(range(len(cells)))


def covers(boxes):
    pts, cells, idx = as_mesh(boxes)
    return check_map.covers_axis(pts, cells, idx)


def ring(r0, r1, n=24):
    """Ring aus n Hexaedern um die Achse, Innenradius r0 (Loch)."""
    out = []
    for k in range(n):
        a, b = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        q = [(r0 * math.cos(a), r0 * math.sin(a)), (r1 * math.cos(a), r1 * math.sin(a)),
             (r1 * math.cos(b), r1 * math.sin(b)), (r0 * math.cos(b), r0 * math.sin(b))]
        out.append(np.array([[x, y, z] for z in (0.0, 1.0) for x, y in q]))
    return out


def main():
    print("Vollscheibe:")
    expect("Achse im Inneren einer Zelle", covers([box(-1, 1, -1, 1)]))
    expect("Achse auf einem Knoten", covers([box(0, 1, 0, 1), box(-1, 0, 0, 1)]))
    expect("Achse auf einer Kante zwischen zwei Zellen",
           covers([box(-1, 1, 0, 2), box(-1, 1, -2, 0)]))
    expect("Mittelzelle ueber der Achse, kein Knoten auf ihr (wie im Standardnetz)",
           covers([box(-1.3, 1.3, -1.3, 1.3)]))
    print("Loch:")
    expect("Ring mit Innenradius 5 erkannt", not covers(ring(5.0, 10.0)))
    expect("Ring mit Innenradius 0,5 erkannt", not covers(ring(0.5, 10.0)))
    expect("Zelle neben der Achse, Kantenverlaengerung durch die Achse",
           not covers([box(1, 2, 0, 1)]))
    print()
    print("Alle Tests bestanden." if not FAILS else f"{len(FAILS)} Test(s) fehlgeschlagen.")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
