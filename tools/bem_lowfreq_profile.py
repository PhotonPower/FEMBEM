#!/usr/bin/env python3
"""Plattenfeld B_z (Realteil) je Radius gegen Biot-Savart bei sehr kleiner Frequenz.

Bei f -> 0 ist die Platte magnetisch transparent (Wirbelstromreaktion ~ omega); das Biot-Savart-Feld
der Spulenfaeden ist dort die Referenz. Je Radiusbin: Mittel von B_z in den Plattenelementen fuer
Biot-Savart, Referenzpfad und BEM-FEM, und die Verhaeltnisse zu Biot-Savart. Dazu |B_i|/|B_r| je Koerper
(Mass fuer den Anteil der Plattenreaktion; Dump-Codes 1 Platte, 2 Innenspule, 3 Aussenspule).

    python3 tools/bem_lowfreq_profile.py BF.csv REF.csv
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_fields as c  # noqa: E402

BINS = [(0.0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10)]


def main():
    b, r = c.read_dump(sys.argv[1]), c.read_dump(sys.argv[2])
    bs = c.biot_savart_B(b)[0]
    Bb, Bbi = c.b_vectors(b)
    Br, Bri = c.b_vectors(r)
    m = b["body"] == 1
    rad = np.hypot(b["x"][m], b["y"][m])
    print("| Radius [m] | n | B_z Biot-Savart | B_z Referenz | B_z BEM-FEM | Ref / BS | BEM-FEM / BS |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for lo, hi in BINS:
        k = (rad >= lo) & (rad < hi)
        f = lambda B: B[m][k][:, 2].mean()
        print(f"| {lo:.2f} bis {hi:.2f} | {k.sum()} | {f(bs):.4g} | {f(Br):.4g} | {f(Bb):.4g} | "
              f"{f(Br) / f(bs):.3f} | {f(Bb) / f(bs):.3f} |")
    print("\n| Koerper | \\|B_i\\|/\\|B_r\\| BEM-FEM | \\|B_i\\|/\\|B_r\\| Referenz |")
    print("|---|---:|---:|")
    for code, name in ((1, "Platte"), (2, "Innenspule"), (3, "Aussenspule")):
        k = b["body"] == code
        print(f"| {name} | {np.linalg.norm(Bbi[k]) / np.linalg.norm(Bb[k]):.2e} | "
              f"{np.linalg.norm(Bri[k]) / np.linalg.norm(Br[k]):.2e} |")


if __name__ == "__main__":
    main()
