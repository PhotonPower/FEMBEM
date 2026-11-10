#!/usr/bin/env python3
"""Vergleicht zwei Felddumps von trace_magnetostatics (gleiches Netz, anderer Baum/andere Option) je Koerper:
relative L2-Abweichung von B und von A (Dumpformat wie tools/compare_fields.py).

    python3 tools/trace_dump_compare.py A.csv B.csv
"""
import csv
import sys

import numpy as np

NAMES = {1: "Platte", 2: "Innenspule", 3: "Aussenspule"}


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    for body, name in NAMES.items():
        m = a["body"] == body
        Ba = np.stack([a["Brx"][m], a["Bry"][m], a["Brz"][m]], 1)
        Bb = np.stack([b["Brx"][m], b["Bry"][m], b["Brz"][m]], 1)
        Aa = np.stack([a["Arx"][m], a["Ary"][m], a["Arz"][m]], 1)
        Ab = np.stack([b["Arx"][m], b["Ary"][m], b["Arz"][m]], 1)
        print(f"{name:12s} n={m.sum():5d}  |B_A - B_B|/|B_A| = {np.linalg.norm(Ba - Bb) / np.linalg.norm(Ba):.3e}   "
              f"|A_A - A_B|/|A_A| = {np.linalg.norm(Aa - Ab) / np.linalg.norm(Aa):.3e}")


if __name__ == "__main__":
    main()
