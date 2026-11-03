#!/usr/bin/env python3
"""Pruefung von tools/compare_fields.py mit synthetischen Feldern.

Drei Koerper aus je einem 5x5x5-Hexaederblock (Randtiefe bekannt: 98 Rand-,
26 Tiefe-1-, 1 Tiefe-2-Zelle, je Block 125 Zellen). Das Testfeld
ist das Referenzfeld, im Inneren (Tiefe >= 1) mit 0,5 multipliziert und in
der Randschicht um eine reine B_phi-Komponente ergaenzt. Die Kennzahlen
muessen genau diese Struktur wiedergeben, und die Gegenproben
(vertauschte Phasen, vertauschte Koerper, ein Dump eines anderen Netzes)
muessen erkannt werden.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
import compare_fields as cf  # noqa: E402

fails = 0


def check(ok, what):
    global fails
    print(f"  [{'OK ' if ok else 'FEHL'}] {what}")
    if not ok:
        fails += 1


def block(n, origin):
    """n^3 Hexaeder in VTK-Reihenfolge; Rueckgabe Zellen (Knotenindex-Offset 0), Knoten."""
    pts = np.array([[i, j, k] for k in range(n + 1) for j in range(n + 1)
                    for i in range(n + 1)], float) + origin
    idx = lambda i, j, k: i + (n + 1) * (j + (n + 1) * k)  # noqa: E731
    cells = []
    for k in range(n):
        for j in range(n):
            for i in range(n):
                cells.append([idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k),
                              idx(i, j + 1, k), idx(i, j, k + 1), idx(i + 1, j, k + 1),
                              idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)])
    return np.array(cells), pts


def make():
    cells_all, centers, body = [], [], []
    off = 0
    for b, origin in ((1, (10, 10, 0)), (2, (-10, 10, 0)), (3, (10, -10, 0))):
        c, p = block(5, np.array(origin, float))
        cells_all.append(c + off)
        centers.append(p[c].mean(axis=1))
        body += [b] * len(c)
        off += len(p)
    return np.vstack(cells_all), np.vstack(centers), np.array(body)


cells, cen, body = make()
depth = cf.element_depths(cells)
x, y = cen[:, 0], cen[:, 1]
rng = np.random.default_rng(3)
Bre_ref = rng.normal(size=(len(x), 3))
Bim_ref = rng.normal(size=(len(x), 3))
r = np.hypot(x, y)
ephi = np.stack([-y / r, x / r, np.zeros_like(x)], axis=1)


def make_test(phi_boundary=0.3):
    out = []
    for B in (Bre_ref, Bim_ref):
        T = B.copy()
        inner = depth >= 1
        T[inner] *= 0.5
        T[depth == 0] += phi_boundary * np.linalg.norm(B[depth == 0], axis=1)[:, None] * ephi[depth == 0]
        out.append(T)
    return tuple(out)


def row(rows, bodyname, layer, part):
    return [q for q in rows if q["body"] == bodyname and q["layer"] == layer and q["part"] == part][0]


print("Test 1: Randtiefe")
check(np.bincount(depth[body == 1]).tolist() == [98, 26, 1],
      f"5x5x5-Block: 98 Rand-, 26 Tiefe-1-, 1 Tiefe-2-Zelle ({np.bincount(depth[body == 1]).tolist()})")

print("Test 2: Kennzahlen")
Bt = make_test()
rows = cf.compare_fields(Bt, (Bre_ref, Bim_ref), body, x, y, depth)
for name in ("Platte", "Innenspule", "Aussenspule"):
    ri = row(rows, name, "Innen", "Re")
    check(abs(ri["proj"] - 0.5) < 1e-12 and abs(ri["ratio"] - 0.5) < 1e-12 and
          abs(ri["l2"] - 0.5) < 1e-12,
          f"{name}, Innen: proj = ratio = L2 = 0,5")
    rr = row(rows, name, "Rand", "Im")
    check(abs(rr["proj"] - 1.0) < 0.05 and rr["bphi_t"] > 0.2 and rr["bphi_r"] < 0.9,
          f"{name}, Rand: proj ~ 1, B_phi-Anteil sichtbar ({rr['bphi_t']:.3f})")
    check(row(rows, name, "Rand", "Re")["n"] == 98, f"{name}: 98 Randzellen")

print("Test 3: Gegenproben")
swapped = (Bt[1], Bt[0])
rows_s = cf.compare_fields(swapped, (Bre_ref, Bim_ref), body, x, y, depth)
check(abs(row(rows_s, "Platte", "Innen", "Re")["proj"]) < 0.3,
      "vertauschte Phasen: Projektion bricht ein (erkannt)")
perm = np.concatenate([np.where(body == 2)[0], np.where(body == 3)[0], np.where(body == 1)[0]])
Bt_perm = tuple(B[perm] for B in Bt)    # Koerper 1 <- Feld von 2, usw.
rows_p = cf.compare_fields(Bt_perm, (Bre_ref, Bim_ref), body, x, y, depth)
check(abs(row(rows_p, "Platte", "alle", "Re")["proj"]) < 0.3,
      "vertauschte Koerper: Projektion bricht ein (erkannt)")

print("Test 4: Kommandozeile")
with tempfile.TemporaryDirectory() as d:
    def write(path, Bre, Bim, bodies=body, xx=x):
        with open(path, "w") as f:
            f.write("elem,body,x,y,z,Arx,Ary,Arz,Aix,Aiy,Aiz,Brx,Bry,Brz,Bix,Biy,Biz\n")
            for i in range(len(xx)):
                f.write(f"{i},{bodies[i]},{xx[i]*1e-3},{y[i]*1e-3},{cen[i,2]*1e-3},"
                        + ",".join(["0"] * 6) + ","
                        + ",".join(f"{v}" for v in list(Bre[i]) + list(Bim[i])) + "\n")
    write(os.path.join(d, "t.csv"), *Bt)
    write(os.path.join(d, "r.csv"), Bre_ref, Bim_ref)
    write(os.path.join(d, "o.csv"), Bre_ref, Bim_ref, xx=x + 5.0)
    ok = subprocess.run([sys.executable, os.path.join(TOOLS, "compare_fields.py"),
                         os.path.join(d, "t.csv"), os.path.join(d, "r.csv")],
                        capture_output=True, text=True)
    check(ok.returncode == 0 and "Platte" in ok.stdout, "Aufruf ohne Netz laeuft")
    bad = subprocess.run([sys.executable, os.path.join(TOOLS, "compare_fields.py"),
                          os.path.join(d, "t.csv"), os.path.join(d, "o.csv")],
                         capture_output=True, text=True)
    check(bad.returncode != 0, "Dumps verschiedener Netze werden abgewiesen")

print("\n" + ("ALLE TESTS BESTANDEN" if fails == 0 else f"{fails} FEHLER"))
sys.exit(0 if fails == 0 else 1)
