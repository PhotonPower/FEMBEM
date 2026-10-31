#!/usr/bin/env python3
"""Unabhaengige Referenz fuer die Hubkraft in TEAM 28: rotationssymmetrisches
Ringmodell (PEEC) im freien Raum.

TEAM 28 ist rotationssymmetrisch, und alle Stroeme fliessen in Umfangsrichtung.
Die Platte wird deshalb in koaxiale Ringe mit Rechteckquerschnitt zerlegt, die
Spulen in Stromfaeden. Die Gegeninduktivitaet zweier koaxialer Kreisringe ist
exakt bekannt (elliptische Integrale); die Ringstroeme der Platte folgen aus

    (R + j w L) I_p = -j w M_ps I_s,

die zeitgemittelte Kraft auf die Platte aus

    F_z = 1/2 Re( sum I_p conj(I_s) dM_ps/dz ).

Das Modell hat weder ein abgeschnittenes Luftgebiet noch Elemente; es teilt mit
dem Loeser nur die Eingabedaten. Es ist damit eine Gegenprobe fuer die Kraft,
die der Maxwell-Flaechenpfad nicht leisten kann (er verfehlt einen bekannten
Nullwert um mehr als das Ergebnis, siehe docs/design/gmsh-phase8-report.md).

Daten wie in src/fem.cu (TEAM 28, Karl et al.): sigma = 3.40e7 S/m, 20 A
Scheitelwert, 50 Hz, 960 / 576 Windungen gegensinnig, Spulen r = 27..55 und
80..95 mm, z = -52..0 mm, Platte r = 65 mm, 3 mm dick, Gewicht 0.107 kg * 9.81.

    python3 tools/team28_axisymmetric.py                  # F_z bei 11.3 mm
    python3 tools/team28_axisymmetric.py --plate-z0 3.8
    python3 tools/team28_axisymmetric.py --equilibrium    # Hoehe mit F_z = m g
"""
import argparse
import math
import os

# Die Systeme sind klein (einige tausend Unbekannte). Unter Windows mit
# OpenBLAS bremsen sich die Threads dabei gegenseitig aus: ein komplexes
# 780er-System brauchte 15 s statt 0.3 s mit einem Thread. Vor dem Import
# setzen; eine Vorgabe von aussen bleibt erhalten.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402

MU0 = 4e-7 * math.pi
SIGMA = 3.40e7
I_MAX = 20.0
FREQ = 50.0
COILS = ((27e-3, 55e-3, 960, +1),      # Innenspule, Strom +J (wie J_INNER in fem.cu)
         (80e-3, 95e-3, 576, -1))      # Aussenspule, gegensinnig
COIL_Z = (-52e-3, 0.0)
PLATE_R, PLATE_T = 65e-3, 3e-3
WEIGHT = 0.107 * 9.81                  # N


def ellipke(m):
    """Vollstaendige elliptische Integrale K(m), E(m) (Parameter m = k^2)
    ueber das arithmetisch-geometrische Mittel; numpy statt scipy, damit die
    CI keine weitere Abhaengigkeit braucht. Konvergiert quadratisch, 30
    Schritte reichen fuer m bis 1 - 1e-15."""
    m = np.asarray(m, dtype=float)
    a = np.ones_like(m)
    b = np.sqrt(1.0 - m)
    c2_sum = 0.5 * m              # sum 2^(n-1) c_n^2 mit c_0^2 = m
    p = 0.5
    for _ in range(30):
        a, b, c = 0.5 * (a + b), np.sqrt(a * b), 0.5 * (a - b)
        p *= 2.0
        c2_sum = c2_sum + p * c * c
    K = math.pi / (2.0 * a)
    return K, K * (1.0 - c2_sum)


def mutual(a, b, d):
    """Gegeninduktivitaet zweier koaxialer Kreisringe (Radien a, b, Abstand d)."""
    m = 4.0 * a * b / ((a + b) ** 2 + d ** 2)
    k = np.sqrt(m)
    K, E = ellipke(m)
    return MU0 * np.sqrt(a * b) * ((2.0 / k - k) * K - 2.0 / k * E)


def dmutual_dz(a, b, d, h=1e-7):
    """dM/dd mit zentraler Differenz; h = 0.1 um liegt weit unter jedem
    Ringabstand und weit ueber der Rundung von M."""
    return (mutual(a, b, d + h) - mutual(a, b, d - h)) / (2.0 * h)


def plate_rings(nr, nz, z0):
    dr, dz = PLATE_R / nr, PLATE_T / nz
    r = (np.arange(nr) + 0.5) * dr
    z = z0 + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(r, z, indexing="ij")
    return R.ravel(), Z.ravel(), dr, dz


def coil_filaments(I, ncr, ncz, coils=COILS):
    rs, zs, cur = [], [], []
    h = COIL_Z[1] - COIL_Z[0]
    for (r0, r1, turns, sign), n in zip(coils, ncr):
        # Amperewindungen gleichmaessig auf den r-z-Querschnitt verteilt (J = w I / A)
        cr = r0 + (np.arange(n) + 0.5) * (r1 - r0) / n
        cz = COIL_Z[0] + (np.arange(ncz) + 0.5) * h / ncz
        CR, CZ = np.meshgrid(cr, cz, indexing="ij")
        rs.append(CR.ravel())
        zs.append(CZ.ravel())
        cur.append(np.full(CR.size, sign * turns * I / (n * ncz)))
    return np.concatenate(rs), np.concatenate(zs), np.concatenate(cur)


def ring_currents(R, Z, dr, dz, rs, zs, Is, sigma, freq):
    """Komplexe Ringstroeme der Platte zu den Quellstroemen Is."""
    w = 2.0 * math.pi * freq
    D = Z[:, None] - Z[None, :]
    same = np.eye(R.size, dtype=bool)
    with np.errstate(all="ignore"):
        L = mutual(R[:, None], R[None, :], np.where(same, 1.0, D))
    # Selbstinduktivitaet eines duennen Rings mit Rechteckquerschnitt ueber den
    # mittleren geometrischen Abstand g = 0.2235 (dr + dz). Der Wert geht nur
    # schwach ein (+-30 % in g aendern F_z bei 260 x 12 Ringen um 1e-4).
    g = 0.2235 * (dr + dz)
    L[same] = MU0 * R * (np.log(8.0 * R / g) - 2.0)
    res = 2.0 * math.pi * R / (sigma * dr * dz)
    M = mutual(R[:, None], rs[None, :], Z[:, None] - zs[None, :])
    return np.linalg.solve(np.diag(res) + 1j * w * L, -1j * w * (M @ Is))


def lift_force(z0, I=I_MAX, sigma=SIGMA, freq=FREQ, nr=260, nz=12, ncr=(14, 8), ncz=26):
    """Zeitgemittelte Hubkraft in N; z0 = Unterkante der Platte ueber der
    Spulenoberkante in m."""
    R, Z, dr, dz = plate_rings(nr, nz, z0)
    rs, zs, Is = coil_filaments(I, ncr, ncz)
    Ip = ring_currents(R, Z, dr, dz, rs, zs, Is, sigma, freq)
    dM = dmutual_dz(R[:, None], rs[None, :], Z[:, None] - zs[None, :])
    return 0.5 * float(np.real(np.sum(Ip[:, None] * np.conj(Is)[None, :] * dM)))


def equilibrium(weight=WEIGHT, lo=5e-3, hi=20e-3, tol=1e-6, **kw):
    """Hoehe, bei der F_z = weight (F_z faellt monoton mit der Hoehe)."""
    f_lo, f_hi = lift_force(lo, **kw) - weight, lift_force(hi, **kw) - weight
    if f_lo * f_hi > 0:
        raise ValueError("Gewicht liegt nicht zwischen den Kraeften an den Intervallgrenzen")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        f_mid = lift_force(mid, **kw) - weight
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--plate-z0", type=float, default=11.3, help="Unterkante der Platte in mm")
    ap.add_argument("--sigma", type=float, default=SIGMA, help="Leitfaehigkeit der Platte, S/m")
    ap.add_argument("--current", type=float, default=I_MAX, help="Scheitelwert des Stroms, A")
    ap.add_argument("--nr", type=int, default=260, help="Ringe radial in der Platte")
    ap.add_argument("--nz", type=int, default=12, help="Ringe ueber die Plattendicke")
    ap.add_argument("--equilibrium", action="store_true",
                    help="Schwebehoehe suchen, bei der F_z = 0.107 kg * 9.81")
    a = ap.parse_args()
    kw = dict(I=a.current, sigma=a.sigma, nr=a.nr, nz=a.nz)
    if a.equilibrium:
        z = equilibrium(**kw)
        print(f"Schwebehoehe (F_z = {WEIGHT:.4f} N): {z * 1e3:.3f} mm")
    else:
        F = lift_force(a.plate_z0 * 1e-3, **kw)
        print(f"F_z = {F:.5f} N bei {a.plate_z0} mm "
              f"({(F / WEIGHT - 1) * 100:+.2f} % gegen das Gewicht {WEIGHT:.4f} N)")


if __name__ == "__main__":
    main()
