#!/usr/bin/env python3
"""Zeilenresiduen aus --bem-row-check-Logs (V0) je Koerper und Klasse, Skalierungen nebeneinander.

    python3 tools/bem_row_compare.py LOG1 LOG2 ...      (z. B. row_s1.0.log row_s1.25.log)

Ausgabe: Kopfzeilen (Quads, Normalenorientierung, Blockpaare) und je Zeile 1 (+K2) bzw.
Zeile 2 (+K1) eine Tabelle Koerper x Klasse mit einer Spalte je Log.
"""
import re
import sys


def parse(path):
    log = open(path, encoding="utf-8", errors="replace").read()
    head = {}
    m = re.search(r"(\d+) Randquads, (\d+) FEM-Kanten, (\d+) BEM-Freiheitsgrade", log)
    head["Quads / FEM-Kanten / BEM-DoF"] = f"{m.group(1)} / {m.group(2)} / {m.group(3)}" if m else "?"
    m = re.search(r"Normalenorientierung: (\d+) von (\d+) Quads haben eine nach INNEN", log)
    head["Normalen nach innen"] = f"{m.group(1)} von {m.group(2)}" if m else "?"
    m = re.search(r"(\d+) Blockpaare \((\d+) ACA, (\d+) dense\)", log)
    head["Blockpaare (ACA / dicht)"] = f"{m.group(1)} ({m.group(2)} / {m.group(3)})" if m else "?"
    m = re.search(r"cluster tree with (\d+) nodes", log)
    head["Clusterbaum-Knoten"] = m.group(1) if m else "?"
    for body in ("Platte", "Innenspule", "Aussenspule"):
        m = re.search(rf"\[RowCheck\]\s+{body}\s+Quads: eben (\d+), Seite (\d+), Kante (\d+)", log)
        head[f"{body}: Quads eben / Seite / Kante"] = " / ".join(m.groups()) if m else "?"
    tabs = {}
    key = None
    for l in log.split("\n"):
        if l.startswith("[RowCheck] ----------") or l.startswith("[RowCheck] Normen"):
            if "V0" not in l and l.startswith("[RowCheck] ----------"):
                key = None
            continue
        if l.startswith("Zeile 1 (FEM auf Randkanten), V0"):
            key = ("Zeile 1", 5)   # +K2
        elif l.startswith("Zeile 2 (BEM), V0"):
            key = ("Zeile 2", 3)   # +K1
        elif l.startswith("Zeile"):
            key = None
        elif key and l.startswith("|"):
            c = [x.strip() for x in l.strip("|").split("|")]
            if len(c) >= 7 and c[0] in ("Platte", "Innenspule", "Aussenspule", "alle") and c[1] in ("eben", "Seite", "Kante", "alle"):
                tabs.setdefault(key[0], {}).setdefault((c[0], c[1]), c[key[1]])
    return head, tabs


def main():
    paths = sys.argv[1:]
    data = [parse(p) for p in paths]
    names = [p.split("/")[-1] for p in paths]
    print("| Groesse | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for k in data[0][0]:
        print(f"| {k} | " + " | ".join(d[0][k] for d in data) + " |")
    for z, lab in (("Zeile 1", "+K2"), ("Zeile 2", "+K1")):
        print(f"\n{z} ({lab}), V0:\n")
        print("| Koerper | Klasse | " + " | ".join(names) + " |")
        print("|---|---|" + "---:|" * len(names))
        for k in data[0][1][z]:
            print(f"| {k[0]} | {k[1]} | " + " | ".join(d[1][z].get(k, "?") for d in data) + " |")


if __name__ == "__main__":
    main()
