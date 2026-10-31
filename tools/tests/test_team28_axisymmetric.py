#!/usr/bin/env python3
"""
Tests fuer das Ringmodell in tools/team28_axisymmetric.py.

Jede Stufe wird gegen eine unabhaengige Groesse geprueft: die elliptischen
Integrale gegen Tabellenwerte, die Gegeninduktivitaet gegen das Neumann-
Integral, Zusammenbau und Zeitmittelung gegen die geschlossene Loesung fuer
einen Plattenring. Die Gegenprobe mit dem Faktor 1 statt 1/2 in der
Zeitmittelung muss am geschlossenen Fall scheitern.

    python3 tools/tests/test_team28_axisymmetric.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import team28_axisymmetric as ax  # noqa: E402  (setzt OPENBLAS_NUM_THREADS vor numpy)

import numpy as np  # noqa: E402

FAILS = []


def expect(name, cond):
    print(f"  {'ok' if cond else 'FEHLGESCHLAGEN':14s} {name}")
    if not cond:
        FAILS.append(name)


def neumann(a, b, d, n=4000):
    """M = mu0 a b / 2 * int_0^2pi cos(phi) / |r1 - r2| dphi (Mittelpunktsregel,
    fuer den periodischen Integranden spektral genau)."""
    phi = (np.arange(n) + 0.5) * 2.0 * math.pi / n
    dist = np.sqrt(a * a + b * b - 2.0 * a * b * np.cos(phi) + d * d)
    return ax.MU0 * a * b / 2.0 * np.sum(np.cos(phi) / dist) * (2.0 * math.pi / n)


def single_ring_force(z_p, z_s, r_p, r_s, dr, dz, I, sigma, freq, factor):
    """Geschlossene Loesung fuer einen Plattenring und einen Quellfaden:
    I_p = -j w M I / (R + j w L), F = factor * Re(I_p conj(I) M')."""
    w = 2.0 * math.pi * freq
    M = ax.mutual(r_p, r_s, z_p - z_s)
    dM = ax.dmutual_dz(r_p, r_s, z_p - z_s)
    L = ax.MU0 * r_p * (math.log(8.0 * r_p / (0.2235 * (dr + dz))) - 2.0)
    R = 2.0 * math.pi * r_p / (sigma * dr * dz)
    return factor * (-(I * I) * M * dM * w * w * L / (R * R + w * w * L * L))


def main():
    print("Elliptische Integrale (Tabellenwerte)")
    K, E = ax.ellipke(np.array([0.0, 0.5, 0.9]))
    expect("K(0) = E(0) = pi/2", abs(K[0] - math.pi / 2) < 1e-15 and abs(E[0] - math.pi / 2) < 1e-15)
    expect("K(0.5) = 1.854074677301372", abs(K[1] - 1.854074677301372) < 1e-14)
    expect("E(0.5) = 1.350643881047675", abs(E[1] - 1.350643881047675) < 1e-14)
    expect("K(0.9) = 2.578092113348173", abs(K[2] - 2.578092113348173) < 1e-14)
    expect("E(0.9) = 1.104774732704073", abs(E[2] - 1.104774732704073) < 1e-14)

    print("Gegeninduktivitaet gegen das Neumann-Integral")
    worst = 0.0
    for a, b, d in [(0.04, 0.065, 0.02), (0.03, 0.031, 0.0005), (0.09, 0.01, 0.05)]:
        worst = max(worst, abs(ax.mutual(a, b, d) / neumann(a, b, d) - 1.0))
    expect(f"relative Abweichung {worst:.1e} < 1e-10", worst < 1e-10)

    print("Ein Plattenring, ein Quellfaden: Zusammenbau und Zeitmittelung")
    z_p, z_s, r_p, r_s, dr, dz = 0.012, -0.02, 0.05, 0.04, 2e-3, 3e-3
    R, Z = np.array([r_p]), np.array([z_p])
    rs, zs, Is = np.array([r_s]), np.array([z_s]), np.array([20.0])
    Ip = ax.ring_currents(R, Z, dr, dz, rs, zs, Is, ax.SIGMA, ax.FREQ)
    F_model = 0.5 * float(np.real(Ip[0] * np.conj(Is[0]) * ax.dmutual_dz(r_p, r_s, z_p - z_s)))
    F_exact = single_ring_force(z_p, z_s, r_p, r_s, dr, dz, 20.0, ax.SIGMA, ax.FREQ, 0.5)
    expect(f"F = {F_model:.6e} N gleich geschlossener Loesung",
           abs(F_model / F_exact - 1.0) < 1e-12)
    expect("abstossend (F_z > 0)", F_model > 0)
    F_wrong = single_ring_force(z_p, z_s, r_p, r_s, dr, dz, 20.0, ax.SIGMA, ax.FREQ, 1.0)
    expect("Gegenprobe: Faktor 1 statt 1/2 wird erkannt", abs(F_model / F_wrong - 1.0) > 0.4)

    print("TEAM 28")
    lo = ax.lift_force(11.3e-3, sigma=ax.SIGMA / 100.0, nr=65, nz=3, ncr=(7, 4), ncz=13)
    hi = ax.lift_force(11.3e-3, sigma=ax.SIGMA / 10.0, nr=65, nz=3, ncr=(7, 4), ncz=13)
    expect(f"kleine Leitfaehigkeit: F ~ sigma^2 (Verhaeltnis {hi / lo:.2f})", abs(hi / lo - 100.0) < 1.0)
    # Mittleres Plattengitter und grobe Spulenfaeden (14x8x26), damit der Test
    # in Sekunden laeuft. Plattengitter 65x3 / 130x6 / 260x12: 1.00349 /
    # 1.00602 / 1.00611 N; die Vorgabe des Werkzeugs (260x12, 28x16x52) gibt
    # 1.00665 N (docs/design/gmsh-phase8-report.md, gmsh-phase9-report.md).
    grid = dict(nr=130, nz=6, ncr=(14, 8), ncz=26)
    F = ax.lift_force(11.3e-3, **grid)
    expect(f"F_z(11.3 mm) = {F:.5f} N, Bezug 1.00602 N", abs(F - 1.00602) < 2e-5)
    F_low = ax.lift_force(11.0e-3, **grid)
    expect(f"F_z(11.0 mm) = {F_low:.5f} N > Gewicht {ax.WEIGHT:.4f} N > F_z(11.3 mm)",
           F_low > ax.WEIGHT > F)

    if FAILS:
        print(f"\n{len(FAILS)} Pruefung(en) fehlgeschlagen")
        sys.exit(1)
    print("\nalle Pruefungen bestanden")


if __name__ == "__main__":
    main()
