#!/usr/bin/env python3
"""Vergleicht das Feld des harmonischen Pfads (--harmonic-field-dump) je
Element mit Biot-Savart.

Die Referenz ist das Feld der Spulen als Kreisstromfaeden (exakt ueber
elliptische Integrale), auf Wunsch plus das Feld der Wirbelstroeme aus dem
Ringmodell (tools/team28_axisymmetric.py). Bei kleiner Frequenz (etwa
--harmonic=2) ist die Rueckwirkung der Platte unter 2 Prozent, und der
Realteil muss das reine Spulenfeld sein -- damit trennt der Vergleich das
Quellfeld und die Aussenraumkopplung vom Wirbelstrom- und Kraftteil.

Ausgabe je Koerper (Platte, Innen-, Aussenspule): Verhaeltnis der Betraege,
Projektion auf die Referenz und relativer L2-Fehler, fuer B und fuer A_phi.

    python3 tools/team28_field_check.py dump.csv
    python3 tools/team28_field_check.py dump.csv --plate-z0 11.3 --freq 50
"""
import argparse
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import team28_axisymmetric as ax  # noqa: E402  (setzt OPENBLAS_NUM_THREADS vor numpy)

import numpy as np  # noqa: E402

BODIES = {1: "Platte", 2: "Innenspule", 3: "Aussenspule"}


def loop_field(a, I, r, z):
    """B_r, B_z, A_phi eines Kreisstroms (Radius a, Strom I, in der Ebene z=0)
    am Punkt (r, z). Arrays: a, I je Faden (Spalten), r, z je Punkt (Zeilen)."""
    r = np.maximum(r, 1e-9)                    # auf der Achse ist B_r = 0
    s2 = (a + r) ** 2 + z ** 2
    m = np.clip(4.0 * a * r / s2, 0.0, 1.0 - 1e-15)
    K, E = ax.ellipke(m)
    d2 = (a - r) ** 2 + z ** 2
    c = ax.MU0 * I / (2.0 * math.pi * np.sqrt(s2))
    Bz = c * (K + (a * a - r * r - z * z) / d2 * E)
    Br = c * z / r * (-K + (a * a + r * r + z * z) / d2 * E)
    k = np.sqrt(m)
    Aphi = ax.MU0 * I / (math.pi * np.maximum(k, 1e-12)) * np.sqrt(a / r) * ((1.0 - 0.5 * m) * K - E)
    return Br, Bz, Aphi


def reference(r, z, freq, plate_z0, ncr=(14, 8), ncz=26):
    """Komplexe Referenz (B_r, B_z, A_phi) an den Punkten (r, z)."""
    rs, zs, Is = ax.coil_filaments(ax.I_MAX, ncr, ncz)
    srcs = [(rs, zs, Is.astype(complex))]
    if freq is not None and plate_z0 is not None:
        R, Z, dr, dz = ax.plate_rings(130, 6, plate_z0)
        Ip = ax.ring_currents(R, Z, dr, dz, rs, zs, Is, ax.SIGMA, freq)
        srcs.append((R, Z, Ip))
    Br = np.zeros(r.size, complex)
    Bz = np.zeros(r.size, complex)
    Ap = np.zeros(r.size, complex)
    for a, za, I in srcs:
        # in Bloecken, damit die Matrix Punkte x Faeden nicht zu gross wird
        for i0 in range(0, r.size, 2000):
            sl = slice(i0, i0 + 2000)
            br, bz, ap = loop_field(a[None, :], 1.0, r[sl, None], z[sl, None] - za[None, :])
            Br[sl] += br @ I
            Bz[sl] += bz @ I
            Ap[sl] += ap @ I
    return Br, Bz, Ap


def read_dump(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    cols = {k: np.array([float(row[k]) for row in rows]) for k in rows[0]}
    return cols


def stats(num, ref):
    """num, ref: (n, k) komplex. Betragsverhaeltnis (Median), Projektion
    <num, ref>/<ref, ref> und relativer L2-Fehler."""
    nn = np.linalg.norm(num, axis=1)
    nr = np.linalg.norm(ref, axis=1)
    ok = nr > 0
    ratio = np.median(nn[ok] / nr[ok])
    proj = np.vdot(ref, num) / np.vdot(ref, ref)
    err = np.linalg.norm(num - ref) / np.linalg.norm(ref)
    return ratio, proj, err


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dump", help="CSV aus --harmonic-field-dump")
    ap.add_argument("--freq", type=float, default=None,
                    help="Frequenz fuer die Wirbelstroeme der Referenz (ohne: nur Spulenfeld)")
    ap.add_argument("--plate-z0", type=float, default=11.3,
                    help="Unterkante der Platte in mm (fuer die Wirbelstroeme)")
    ap.add_argument("--samples", type=int, default=4, help="Beispielelemente je Koerper")
    a = ap.parse_args()

    d = read_dump(a.dump)
    x, y, z = d["x"], d["y"], d["z"]
    r = np.hypot(x, y)
    cphi, sphi = x / np.maximum(r, 1e-12), y / np.maximum(r, 1e-12)
    Bnum = {}
    for part, sgn in (("r", 1.0), ("i", 1.0)):
        bx, by, bz = d[f"B{part}x"], d[f"B{part}y"], d[f"B{part}z"]
        Bnum[part] = (bx * cphi + by * sphi, bz)
    Ar_phi = -d["Arx"] * sphi + d["Ary"] * cphi
    Ai_phi = -d["Aix"] * sphi + d["Aiy"] * cphi
    Br_n = Bnum["r"][0] + 1j * Bnum["i"][0]
    Bz_n = Bnum["r"][1] + 1j * Bnum["i"][1]
    Ap_n = Ar_phi + 1j * Ai_phi
    # Anteil ausserhalb der r-z-Ebene: bei Rotationssymmetrie ist B_phi = 0
    Bphi_n = (-d["Brx"] * sphi + d["Bry"] * cphi) + 1j * (-d["Bix"] * sphi + d["Biy"] * cphi)

    Br_ref, Bz_ref, Ap_ref = reference(r, z, a.freq, a.plate_z0 * 1e-3 if a.freq else None)

    print(f"Referenz: Spulen 20 A" + (f" + Wirbelstroeme bei {a.freq} Hz, Platte ab {a.plate_z0} mm"
                                     if a.freq else " (ohne Wirbelstroeme)"))
    print(f"Koordinaten: z von {z.min()*1e3:.1f} bis {z.max()*1e3:.1f} mm, r bis {r.max()*1e3:.1f} mm\n")
    print(f"{'Koerper':12s} {'n':>5s}  {'|B|/|B_ref|':>11s}  {'Projektion B':>22s}  {'L2 B':>7s}"
          f"  {'|B_phi|/|B|':>11s}  {'Projektion A_phi':>22s}")
    body = d["body"].astype(int)
    for b, name in BODIES.items():
        sel = body == b
        if not sel.any():
            continue
        num = np.stack([Br_n[sel], Bz_n[sel]], axis=1)
        ref = np.stack([Br_ref[sel], Bz_ref[sel]], axis=1)
        ratio, proj, err = stats(num, ref)
        _, projA, _ = stats(Ap_n[sel, None], Ap_ref[sel, None])
        bphi = np.linalg.norm(Bphi_n[sel]) / np.linalg.norm(num)
        print(f"{name:12s} {sel.sum():5d}  {ratio:11.4f}  {proj.real:+10.4f}{proj.imag:+10.4f}j"
              f"  {err:7.3f}  {bphi:11.4f}  {projA.real:+10.4f}{projA.imag:+10.4f}j")

    print("\nBeispiele (Realteile, mT):")
    print(f"{'Koerper':12s} {'r mm':>6s} {'z mm':>6s}  {'B_r':>9s} {'B_r ref':>9s}  {'B_z':>9s} {'B_z ref':>9s}")
    for b, name in BODIES.items():
        idx = np.flatnonzero(body == b)
        if idx.size == 0:
            continue
        for i in idx[np.linspace(0, idx.size - 1, a.samples).astype(int)]:
            print(f"{name:12s} {r[i]*1e3:6.1f} {z[i]*1e3:6.1f}  {Br_n[i].real*1e3:9.3f} {Br_ref[i].real*1e3:9.3f}"
                  f"  {Bz_n[i].real*1e3:9.3f} {Bz_ref[i].real*1e3:9.3f}")


if __name__ == "__main__":
    main()
