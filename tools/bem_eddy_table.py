#!/usr/bin/env python3
"""Wirbelstromgroessen und Kraft beider Pfade nebeneinander (docs/design/force-excess-localization.md).

    python3 tools/bem_eddy_table.py DIR SKALIERUNG F1 F2 ...      Praefixe bf_f<F>_s<SC>.log / ref_f<F>_s<SC>.log

Liest je Frequenz aus dem BEM-FEM-Log (Zeilen '[Harmonisch] Wirbelstrom', 'Verlustleistung', 'Q_eff',
'LORENTZ (Volumen)') und dem Referenzlog ('Wirbelstrom', 'Verlustleistung', 'Q_eff', 'F_z =') und
schreibt die Tabelle |m_z|(Max), |m_z|(Null), P, Q_eff, F_z Lorentz mit Verhaeltnissen BEM-FEM / Referenz.
Optional (--fields): Verhaeltnis |B_i|/|B_r| der Platte aus den Felddumps bf_f<F>_s<SC>.csv.
"""
import os
import re
import sys

import numpy as np


def num(m):
    return float(m.group(1))


def parse(path, bf):
    t = open(path, encoding="utf-8", errors="replace").read()
    pre = r"(?:\[Harmonisch\]|\[Ref-Harm\])?\s*"
    d = {}
    m = re.search(r"Wirbelstrom: \|m_z\| im Strommaximum ([-+0-9.eE]+) A m\^2, im Nulldurchgang ([-+0-9.eE]+)", t)
    d["mzmax"], d["mznull"] = (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)
    m = re.search(r"Verlustleistung \(Zyklusmittel\) = ([-+0-9.eE]+) W", t)
    d["P"] = num(m) if m else np.nan
    m = re.search(r"Q_eff = \|m_z\(Max\)\|/\|m_z\(Null\)\| = ([-+0-9.eE]+)", t)
    d["Q"] = num(m) if m else np.nan
    if bf:
        m = re.search(r"LORENTZ \(Volumen\)\s*: \(([-+0-9.eE]+), ([-+0-9.eE]+), ([-+0-9.eE]+)\) N", t)
        d["F"] = float(m.group(3)) if m else np.nan
    else:
        m = re.search(r"F_z = ([-+0-9.eE]+) N", t)
        d["F"] = num(m) if m else np.nan
    return d


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    d, sc, fs = args[0], args[1], args[2:]
    print("| f [Hz] | Groesse | BEM-FEM | Referenz | BEM-FEM / Ref |")
    print("|---:|---|---:|---:|---:|")
    for f in fs:
        b = parse(os.path.join(d, f"bf_f{f}_s{sc}.log"), True)
        r = parse(os.path.join(d, f"ref_f{f}_s{sc}.log"), False)
        for k, lab in (("mzmax", "|m_z| Strommaximum [A m^2]"), ("mznull", "|m_z| Nulldurchgang [A m^2]"),
                       ("P", "P [W]"), ("Q", "Q_eff"), ("F", "F_z Lorentz [N]")):
            print(f"| {f} | {lab} | {b[k]:.4g} | {r[k]:.4g} | {b[k] / r[k]:.3f} |")
    if "--fields" in sys.argv:
        print("\n| f [Hz] | Pfad | Platte |B_i|/|B_r| | Platte |B_r| |")
        print("|---:|---|---:|---:|")
        for f in fs:
            for n, lab in (("bf", "BEM-FEM"), ("ref", "Referenz")):
                x = np.genfromtxt(os.path.join(d, f"{n}_f{f}_s{sc}.csv"), delimiter=",", names=True)
                m = x["body"] == 1   # Dump-Code: 1 Platte, 2 Innenspule, 3 Aussenspule
                Br = np.c_[x["Brx"][m], x["Bry"][m], x["Brz"][m]]
                Bi = np.c_[x["Bix"][m], x["Biy"][m], x["Biz"][m]]
                print(f"| {f} | {lab} | {np.linalg.norm(Bi) / np.linalg.norm(Br):.3e} | {np.linalg.norm(Br):.4g} |")


if __name__ == "__main__":
    main()
