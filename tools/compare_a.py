#!/usr/bin/env python3
"""
Vergleich von A an den Leiterknoten zweier Laeufe des harmonischen Referenzpfads.

    python tools/compare_a.py REF.csv STUFE.csv [--markdown]

Beide Dateien kommen von `fembem --reference --harmonic --fixed-plate
--reference-dump-a=DATEI` (Spalten x_m,y_m,z_m,comp,Ar_x,Ar_y,Ar_z,Ai_x,Ai_y,Ai_z).
Der Vergleich setzt IDENTISCHE Leiterknoten voraus -- das garantiert
tools/strip_air.py, das nie neu vernetzt. Die Knoten werden ueber die
Koordinaten zugeordnet (Toleranz 1e-12 m, eindeutig, sonst Fehler); nicht
zugeordnete Knoten werden gemeldet und zaehlen als Fehler (Exit-Code 1).

Je Koerper (comp 3 Platte, 2 Innenspule, 1 Aussenspule), mit REF als Bezug:
  * ||A_r - A_r,ref|| / ||A_r,ref|| und dasselbe fuer A_i, je L2 ueber die
    Knoten und die drei Komponenten
  * das Maximum der Knotenabweichung |A - A_ref| (komplex, ueber die drei
    Komponenten) relativ zu max |A_ref| im Koerper, mit Ort in mm
  * die Phasenlage arg(S A_ref* . A) als globale Kontrolle (Grad) und das
    Amplitudenverhaeltnis |S| / S(A_ref* . A_ref)
"""
import argparse
import sys

import numpy as np

TOL = 1e-12                       # m
NAMES = {3: "Platte", 2: "Innenspule", 1: "Aussenspule"}
COLUMNS = ["x_m", "y_m", "z_m", "comp", "Ar_x", "Ar_y", "Ar_z", "Ai_x", "Ai_y", "Ai_z"]


class CompareError(Exception):
    pass


def read_dump(path):
    """Gibt (xyz[N,3], comp[N], A[N,3] komplex) zurueck."""
    with open(path) as f:
        head = f.readline().strip().split(",")
        if head != COLUMNS:
            raise CompareError(f"{path}: Kopfzeile {head} statt {COLUMNS}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != len(COLUMNS):
        raise CompareError(f"{path}: {data.shape[1]} Spalten")
    xyz = data[:, :3]
    comp = data[:, 3].astype(int)
    A = data[:, 4:7] + 1j * data[:, 7:10]
    return xyz, comp, A


def keys(xyz, tol=TOL):
    return np.round(xyz / tol).astype(np.int64)


def match(xyz_a, xyz_b, tol=TOL):
    """Zuordnung a -> b ueber die Koordinaten. Gibt (ia, ib, nur_a, nur_b) zurueck.

    Mehrfache Koordinaten in einer Datei sind ein Fehler.
    """
    ka, kb = keys(xyz_a, tol), keys(xyz_b, tol)
    for name, k in (("REF", ka), ("STUFE", kb)):
        if len(np.unique(k, axis=0)) != len(k):
            raise CompareError(f"{name}: doppelte Koordinaten (Toleranz {tol:g} m) -- "
                               "die Zuordnung waere nicht eindeutig")
    index_b = {tuple(k): i for i, k in enumerate(kb)}
    ia, ib, only_a = [], [], []
    for i, k in enumerate(ka):
        j = index_b.pop(tuple(k), None)
        if j is None:
            only_a.append(i)
        else:
            ia.append(i)
            ib.append(j)
    only_b = sorted(index_b.values())
    return np.array(ia, dtype=int), np.array(ib, dtype=int), only_a, only_b


def compare(ref_path, test_path):
    """Liefert dict: rows je Koerper, unmatched = (nur REF, nur STUFE)."""
    xr, cr, Ar = read_dump(ref_path)
    xt, ct, At = read_dump(test_path)
    ia, ib, only_ref, only_test = match(xr, xt)
    out = dict(n_ref=len(xr), n_test=len(xt), n_matched=len(ia),
               only_ref=len(only_ref), only_test=len(only_test), bodies={})
    if len(ia) and np.any(cr[ia] != ct[ib]):
        raise CompareError(f"{int(np.sum(cr[ia] != ct[ib]))} zugeordnete Knoten mit "
                           "verschiedener Komponentenkennung")
    for c, name in NAMES.items():
        m = cr[ia] == c
        if not m.any():
            continue
        R, T = Ar[ia][m], At[ib][m]
        P = xr[ia][m]
        nr = lambda v: float(np.sqrt(np.sum(np.abs(v) ** 2)))
        rel_r = nr(T.real - R.real) / nr(R.real) if nr(R.real) > 0 else float("nan")
        rel_i = nr(T.imag - R.imag) / nr(R.imag) if nr(R.imag) > 0 else float("nan")
        dev = np.sqrt(np.sum(np.abs(T - R) ** 2, axis=1))
        mag = np.sqrt(np.sum(np.abs(R) ** 2, axis=1))
        k = int(np.argmax(dev))
        S = np.sum(np.conj(R) * T)
        out["bodies"][c] = dict(
            name=name, n=int(m.sum()), rel_r=rel_r, rel_i=rel_i,
            max_dev_rel=float(dev[k] / mag.max()), where_mm=tuple(float(v) * 1e3 for v in P[k]),
            phase_deg=float(np.degrees(np.angle(S))),
            amp_ratio=float(abs(S) / np.sum(np.abs(R) ** 2)))
    return out


def format_result(res, markdown=False):
    lines = []
    if markdown:
        lines.append("| Koerper | Knoten | rel. dA_r | rel. dA_i | max. Knotenabw. | Ort (mm) | Phase (Grad) | Amplitude |")
        lines.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
    for c, b in res["bodies"].items():
        loc = "({:.1f}, {:.1f}, {:.1f})".format(*b["where_mm"])
        if markdown:
            lines.append(f"| {b['name']} | {b['n']} | {b['rel_r']:.3e} | {b['rel_i']:.3e} | "
                         f"{b['max_dev_rel']:.3e} | {loc} | {b['phase_deg']:+.4f} | {b['amp_ratio']:.6f} |")
        else:
            lines.append(f"{b['name']:<12s} n={b['n']:<6d} rel dA_r {b['rel_r']:.3e}  rel dA_i "
                         f"{b['rel_i']:.3e}  max {b['max_dev_rel']:.3e} bei {loc}  "
                         f"Phase {b['phase_deg']:+.4f} Grad  Amplitude {b['amp_ratio']:.6f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="A an den Leiterknoten zweier Referenzlaeufe vergleichen.")
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = compare(a.ref, a.test)
    except (CompareError, OSError, ValueError) as e:
        print(f"FEHLER: {e}")
        return 1
    print(f"{a.test} gegen {a.ref}: {res['n_matched']} Knoten zugeordnet "
          f"({res['n_ref']} in REF, {res['n_test']} in STUFE; nur REF {res['only_ref']}, "
          f"nur STUFE {res['only_test']})")
    print(format_result(res, a.markdown))
    if res["only_ref"] or res["only_test"]:
        print("FEHLER: nicht zugeordnete Knoten -- der Vergleich setzt identische Leiterknoten voraus")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
