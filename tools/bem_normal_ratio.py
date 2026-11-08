#!/usr/bin/env python3
"""Normal- gegen Tangentialanteil des geloesten A auf den Randquads (bem-normal-components, Schritt 3).

Liest die Datei <field-dump>.bquads.csv, die `fembem --harmonic-field-dump=...` im BEM-FEM-Pfad schreibt (je
Randquad Knoten, Flaeche, ||A_tan||^2 und ||A.n||^2 aus dem Nedelec-Interpolanten des angrenzenden
Hexaeders, getrennt fuer Real- und Imaginaerteil). Klassifiziert die Quads wie `--bem-row-check`
(eben: |n_z| > 0.99; Kante: nicht eben und eine Quadkante wird mit einem Quad geteilt, dessen Normale um
mehr als ~26 Grad abweicht, |n.n'| < 0.9; Seite: Rest) und gibt je Koerper und Klasse
    ||A.n|| / ||A_tan||
aus (Flaechen-L2, Real- und Imaginaerteil zusammen: sqrt(sum_r + sum_i)).

Aufruf: bem_normal_ratio.py datei.bquads.csv [weitere ...]
"""
import csv
import math
import sys
from collections import defaultdict

BODY = {1: "Platte", 2: "Innenspule", 3: "Aussenspule"}
CLASSES = ["eben", "Seite", "Kante"]


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def key(p):
    return tuple(round(float(c) * 1e9) for c in p)


def normal(nodes):
    d1 = [nodes[2][i] - nodes[0][i] for i in range(3)]
    d2 = [nodes[3][i] - nodes[1][i] for i in range(3)]
    n = [d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]]
    l = math.sqrt(sum(c * c for c in n))
    return [c / l for c in n]


def classify(rows):
    nodes, normals, edges = [], [], defaultdict(list)
    for i, r in enumerate(rows):
        nd = [(float(r[f"x{k}"]), float(r[f"y{k}"]), float(r[f"z{k}"])) for k in range(4)]
        nodes.append(nd)
        normals.append(normal(nd))
        for k in range(4):
            a, b = key(nd[k]), key(nd[(k + 1) % 4])
            edges[(min(a, b), max(a, b))].append(i)
    cls = []
    for i, r in enumerate(rows):
        n = normals[i]
        if abs(n[2]) > 0.99:
            cls.append(0)
            continue
        sharp = False
        nd = nodes[i]
        for k in range(4):
            a, b = key(nd[k]), key(nd[(k + 1) % 4])
            for j in edges[(min(a, b), max(a, b))]:
                if j != i and abs(sum(n[c] * normals[j][c] for c in range(3))) < 0.9:
                    sharp = True
        cls.append(2 if sharp else 1)
    return cls


def table(path):
    rows = load(path)
    cls = classify(rows)
    acc = defaultdict(lambda: [0.0, 0.0, 0])
    for r, c in zip(rows, cls):
        b = int(r["body"])
        nn = float(r["nrm_r2"]) + float(r["nrm_i2"])
        tt = float(r["tan_r2"]) + float(r["tan_i2"])
        for key_ in ((b, c), (b, -1), (0, -1)):
            a = acc[key_]
            a[0] += nn
            a[1] += tt
            a[2] += 1
    print(f"\n{path}\n")
    print("| Koerper | Klasse | Quads | ||A.n|| / ||A_tan|| |")
    print("|---|---|---:|---:|")
    for b in (1, 2, 3):
        for c in (0, 1, 2, -1):
            if (b, c) in acc and acc[(b, c)][1] > 0:
                a = acc[(b, c)]
                print(f"| {BODY[b]} | {CLASSES[c] if c >= 0 else 'alle'} | {a[2]} | {math.sqrt(a[0] / a[1]):.4f} |")
    a = acc[(0, -1)]
    print(f"| alle | alle | {a[2]} | {math.sqrt(a[0] / a[1]):.4f} |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for p in sys.argv[1:]:
        table(p)
