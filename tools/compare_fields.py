#!/usr/bin/env python3
"""Vergleicht zwei Felddumps (--harmonic-field-dump) desselben Netzes je
Koerper, Phasenanteil und Randtiefe: BEM-FEM-Pfad gegen Referenzpfad, auf
Wunsch dazu Biot-Savart (Kreisstromfaeden) und Ringmodell.

Aufruf (TEST wird an REF gemessen):

    python3 tools/compare_fields.py TEST.csv REF.csv --mesh netz.vtk \\
        [--biot-savart] [--freq HZ --plate-z0 MM] [--markdown]

Dumps: CSV mit elem,body,x,y,z,Ar*,Ai*,Br*,Bi* (SI), wie sie der BEM-FEM-
Pfad und der Referenzpfad schreiben. Das Netz (legacy VTK, mm) liefert die
Randtiefe; ohne --mesh gibt es nur die Zeilen "alle".

Spalten je (Koerper, Phase, Schicht):
  n         Elemente
  proj      Projektion <B_test, B_ref> / <B_ref, B_ref> ueber alle drei
            Komponenten und alle Elemente der Gruppe (Soll 1)
  L2        relativer L2-Fehler ||B_test - B_ref|| / ||B_ref||
  |B|/|Bref|  Verhaeltnis der L2-Normen
  Bphi      Anteil ||B_phi|| / ||B|| von TEST und von REF (Soll 0, die
            Anordnung ist rotationssymmetrisch)

Phase: "Re" und "Im" sind Real- und Imaginaerteil des Dumps, getrennt
ausgewertet. Schicht: Randtiefe eines Elements = kleinste Zahl von
Flaechenuebergaengen innerhalb seines Koerpers bis zu einem Element mit
einer freien Flaeche (Tiefe 0 = Randschicht, 1 = zweite Lage, ...).
Gruppen: "Rand" (Tiefe 0), "Tiefe 1", "Innen" (Tiefe >= 1), "alle".
"""
import argparse
import csv
import os
import sys

import numpy as np

BODIES = {1: "Platte", 2: "Innenspule", 3: "Aussenspule"}
FIELDS = ("Arx", "Ary", "Arz", "Aix", "Aiy", "Aiz",
          "Brx", "Bry", "Brz", "Bix", "Biy", "Biz")


def read_dump(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: leer")
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    for k in ("body", "x", "y", "z") + FIELDS:
        if k not in cols:
            raise ValueError(f"{path}: Spalte {k} fehlt")
    return cols


def b_vectors(d):
    """(n, 3)-Arrays Re und Im von B."""
    return (np.stack([d["Brx"], d["Bry"], d["Brz"]], axis=1),
            np.stack([d["Bix"], d["Biy"], d["Biz"]], axis=1))


def element_depths(cells):
    """Randtiefe je Hexaeder. cells: (n, 8) Knotenindizes (VTK-Reihenfolge).
    Tiefe 0: mindestens eine Flaeche gehoert keinem zweiten Element."""
    faces_of = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    owner = {}
    for e, c in enumerate(cells):
        for f in faces_of:
            owner.setdefault(tuple(sorted(int(c[i]) for i in f)), []).append(e)
    nbrs = [[] for _ in range(len(cells))]
    depth = np.full(len(cells), -1, dtype=int)
    frontier = []
    for fs in owner.values():
        if len(fs) == 1:
            if depth[fs[0]] < 0:
                depth[fs[0]] = 0
                frontier.append(fs[0])
        else:
            for a in fs:
                nbrs[a].extend(b for b in fs if b != a)
    d = 0
    while frontier:
        d += 1
        nxt = []
        for e in frontier:
            for nb in nbrs[e]:
                if depth[nb] < 0:
                    depth[nb] = d
                    nxt.append(nb)
        frontier = nxt
    return depth


def read_mesh_hexes(path):
    """Hexaeder und Knoten (m) aus einem legacy-VTK in mm, in Dateireihenfolge."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import meshcheck as mc
    m = mc.read_vtk(path)
    idx = [i for i, t in enumerate(m.types) if t == mc.VTK_HEX]
    cells = np.array([m.cells[i] for i in idx])
    return cells, m.points * 1e-3


def depths_from_mesh(path, dump):
    """Randtiefe je Dump-Zeile; prueft, dass Netz und Dump zusammenpassen
    (gleiche Elementzahl, gleiche Schwerpunkte in x,y; z der Platte darf
    durch --plate-z verschoben sein)."""
    cells, pts = read_mesh_hexes(path)
    if len(cells) != len(dump["x"]):
        raise ValueError(f"{path}: {len(cells)} Hexaeder, Dump {len(dump['x'])} Zeilen")
    c = pts[cells].mean(axis=1)
    dxy = np.hypot(c[:, 0] - dump["x"], c[:, 1] - dump["y"]).max()
    nonplate = dump["body"] != 1
    dz = np.abs(c[nonplate, 2] - dump["z"][nonplate]).max()
    if dxy > 1e-7 or dz > 1e-7:
        raise ValueError(f"{path} passt nicht zum Dump (max. Abweichung der "
                         f"Schwerpunkte xy {dxy:.2e} m, z {dz:.2e} m)")
    return element_depths(cells)


def group_masks(body, depth):
    """Liste (Koerpername, Schichtname, Maske)."""
    out = []
    for b, name in BODIES.items():
        sb = body == b
        if not sb.any():
            continue
        if depth is None:
            out.append((name, "alle", sb))
            continue
        out.append((name, "Rand", sb & (depth == 0)))
        out.append((name, "Tiefe 1", sb & (depth == 1)))
        out.append((name, "Innen", sb & (depth >= 1)))
        out.append((name, "alle", sb))
    return out


def phi_fraction(B, cphi, sphi):
    bphi = -B[:, 0] * sphi + B[:, 1] * cphi
    nb = np.linalg.norm(B)
    return np.linalg.norm(bphi) / nb if nb > 0 else float("nan")


def metrics(Bt, Br, cphi, sphi):
    """Bt, Br: (n, 3) reell, je Gruppe."""
    nr = np.linalg.norm(Br)
    nt = np.linalg.norm(Bt)
    if Bt.shape[0] == 0 or nr == 0.0:
        return dict(n=Bt.shape[0], proj=float("nan"), l2=float("nan"),
                    ratio=float("nan"), bphi_t=float("nan"), bphi_r=float("nan"))
    return dict(n=Bt.shape[0],
                proj=float(np.sum(Bt * Br) / np.sum(Br * Br)),
                l2=float(np.linalg.norm(Bt - Br) / nr),
                ratio=float(nt / nr),
                bphi_t=float(phi_fraction(Bt, cphi, sphi)),
                bphi_r=float(phi_fraction(Br, cphi, sphi)))


def compare_fields(Bt, Brf, body, x, y, depth=None):
    """Bt, Brf: Tupel (Re, Im), je (n, 3). Rueckgabe Zeilenliste."""
    r = np.maximum(np.hypot(x, y), 1e-12)
    cphi, sphi = x / r, y / r
    rows = []
    for name, layer, m in group_masks(body, depth):
        for pi, part in enumerate(("Re", "Im")):
            k = metrics(Bt[pi][m], Brf[pi][m], cphi[m], sphi[m])
            rows.append(dict(body=name, layer=layer, part=part, **k))
    return rows


def biot_savart_B(d, freq=None, plate_z0_mm=None):
    """Referenz (Re, Im) als (n, 3) in kartesischen Komponenten: Spulen
    als Kreisstromfaeden, mit freq und plate_z0_mm plus Wirbelstroeme aus
    dem Ringmodell (wie tools/team28_field_check.py)."""
    import team28_field_check as fc
    r = np.hypot(d["x"], d["y"])
    cphi, sphi = d["x"] / np.maximum(r, 1e-12), d["y"] / np.maximum(r, 1e-12)
    Br, Bz, _ = fc.reference(r, d["z"], freq,
                             plate_z0_mm * 1e-3 if freq else None)
    Bx, By = Br * cphi, Br * sphi
    full = np.stack([Bx, By, Bz], axis=1)
    return full.real, full.imag


def fmt_rows(rows, markdown=False, title=""):
    hdr = ("Körper", "Schicht", "Phase", "n", "proj", "L2", "‖B‖/‖Bref‖",
           "Bphi test", "Bphi ref")
    lines = []
    if title:
        lines.append(title)
    def f(v, p=4):
        return "-" if v != v else f"{v:.{p}f}"
    body = [(r["body"], r["layer"], r["part"], str(r["n"]), f(r["proj"]), f(r["l2"]),
             f(r["ratio"]), f(r["bphi_t"]), f(r["bphi_r"]))
            for r in rows if r["n"] > 0]
    if markdown:
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "---|" * len(hdr))
        lines += ["| " + " | ".join(b) + " |" for b in body]
    else:
        w = [12, 8, 5, 5, 9, 9, 11, 9, 9]
        lines.append(" ".join(h.ljust(k) for h, k in zip(hdr, w)))
        lines += [" ".join(c.ljust(k) for c, k in zip(b, w)) for b in body]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("test", help="Dump, der gemessen wird (BEM-FEM)")
    ap.add_argument("ref", help="Dump, gegen den gemessen wird (Referenzpfad)")
    ap.add_argument("--mesh", help="Netz (legacy VTK, mm) fuer die Randtiefe")
    ap.add_argument("--biot-savart", action="store_true",
                    help="zusaetzlich beide Dumps gegen Biot-Savart (Spulen, "
                         "mit --freq plus Ringmodell)")
    ap.add_argument("--freq", type=float, default=None,
                    help="Frequenz fuer die Wirbelstroeme der analytischen Referenz")
    ap.add_argument("--plate-z0", type=float, default=11.3,
                    help="Plattenunterkante in mm (Ringmodell)")
    ap.add_argument("--markdown", action="store_true", help="Tabellen als Markdown")
    a = ap.parse_args(argv)

    t, r = read_dump(a.test), read_dump(a.ref)
    if len(t["x"]) != len(r["x"]):
        sys.exit(f"Elementzahlen verschieden: {len(t['x'])} gegen {len(r['x'])}")
    if not (np.array_equal(t["body"], r["body"]) and
            np.abs(t["x"] - r["x"]).max() < 1e-7 and np.abs(t["y"] - r["y"]).max() < 1e-7
            and np.abs(t["z"] - r["z"]).max() < 1e-7):
        sys.exit("Die Dumps gehoeren nicht zum selben Netz (Koerperkennung oder "
                 "Schwerpunkte verschieden).")
    depth = depths_from_mesh(a.mesh, r) if a.mesh else None
    Bt, Brf = b_vectors(t), b_vectors(r)
    body, x, y = r["body"].astype(int), r["x"], r["y"]

    print(fmt_rows(compare_fields(Bt, Brf, body, x, y, depth), a.markdown,
                   f"\n== {os.path.basename(a.test)} gegen {os.path.basename(a.ref)} =="))
    if a.biot_savart:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        Bbs = biot_savart_B(r, a.freq, a.plate_z0)
        lab = "Biot-Savart" + (f" + Ringmodell {a.freq:g} Hz" if a.freq else "")
        print(fmt_rows(compare_fields(Brf, Bbs, body, x, y, depth), a.markdown,
                       f"\n== {os.path.basename(a.ref)} gegen {lab} =="))
        print(fmt_rows(compare_fields(Bt, Bbs, body, x, y, depth), a.markdown,
                       f"\n== {os.path.basename(a.test)} gegen {lab} =="))


if __name__ == "__main__":
    main()
