#!/usr/bin/env python3
"""Volumenstrom-Biot-Savart fuer Ringspulen mit Rechteckquerschnitt: Verfahren und Pruefung (maxwell-trace-coupling.md, 7.5).

Verfahren: Ringspule = Integral gleichmaessig verteilter Kreisschleifen ueber den Querschnitt [r0,r1] x [z0,z1] (Windungszahl N, Strom I,
Stromdichte J = N I /((r1-r0)(z1-z0)) in Richtung e_phi). Je Kreisschleife (Radius a, Hoehe z', Strom 1):
   B_rho, B_z, A_phi in geschlossener Form mit vollstaendigen elliptischen Integralen K(m), E(m), m = k^2 = 4 a rho/((a+rho)^2 + dz^2)
(wie BiotSavart.cpp / BemRowCheck.h). Das Querschnittsintegral: Produkt-Gauss (8x8) auf geometrisch zum Aufpunkt hin verdichteten Panels
(Faktor 0.3, 11 Stufen). Der Integrand hat bei Aufpunkten im oder am Querschnitt eine integrable Singularitaet ~1/dist;
die Verdichtung legt sie auf eine Panelecke.
Pruefungen:
 (B1) Schleifenformel gegen direktes Linienintegral (Biot-Savart, 4000 Segmente) an Aufpunkten ausserhalb der Achse
 (B2) Konsistenz  curl A = B  (Zentraldifferenzen) der Schleifenformeln
 (B3) Rechteckquerschnitt: Achsenfeld gegen unabhaengige geschlossene 1D-Quadratur (scipy.integrate.quad der Achsenformel)
 (B4) Rechteckquerschnitt ausserhalb der Achse: Verfahren gegen Selbstkonvergenz und unabhaengige 900x900-Fadensumme in Abstaenden 20 mm bis 0 mm
      zum Querschnitt (auch auf dem Rand und im Innern der Spule); Konvergenz in tol
 (B5) Fadenmodell (TEAM-28-Fadenraster, 0.5 mm) gegen Volumenstrom an Randpunkten der Innenspule: Abweichung (zeigt, warum das
      Fadenmodell als Referenz nicht genuegt)
Gegenprobe: falsches Vorzeichen von Br (B2) bzw. Fadenabstand 3 mm (B5) liefert Abweichung >> Toleranz.
"""
import numpy as np
from scipy.special import ellipkm1, ellipe
from scipy import integrate

mu0 = 4e-7 * np.pi

def loop_fields(a, zp, rho, z):
    """(A_phi, B_rho, B_z) einer Kreisschleife (Radius a, Hoehe zp, Strom 1 A) am Aufpunkt (rho, z); rho > 0."""
    dz = z - zp
    s2 = (a + rho)**2 + dz**2
    d2 = (a - rho)**2 + dz**2
    p = d2 / s2; m = 1.0 - p                          # m = 4 a rho / s2 = 1 - p, numerisch <= 1
    K = ellipkm1(p); E = ellipe(m)             # K ueber 1-m = d2/s2: auch bei m -> 1 ohne Ausloeschung
    Bz = mu0 / (2 * np.pi) / np.sqrt(s2) * (K + (a * a - rho * rho - dz * dz) / d2 * E)
    Br = mu0 / (2 * np.pi) * dz / (rho * np.sqrt(s2)) * (-K + (a * a + rho * rho + dz * dz) / d2 * E)
    k = np.sqrt(m)
    Aphi = mu0 / (np.pi * k) * np.sqrt(a / rho) * ((1 - m / 2) * K - E)
    return Aphi, Br, Bz

def loop_direct(a, zp, rho, z, n=4000):
    ph = (np.arange(n) + 0.5) * 2 * np.pi / n
    src = np.stack([a * np.cos(ph), a * np.sin(ph), np.full(n, zp)], -1)
    dl = np.stack([-np.sin(ph), np.cos(ph), np.zeros(n)], -1) * a * 2 * np.pi / n
    x = np.array([rho, 0.0, z])
    R = x - src
    B = mu0 / (4 * np.pi) * np.sum(np.cross(dl, R) / np.linalg.norm(R, axis=1)[:, None]**3, axis=0)
    # A = mu0/(4 pi) Int dl / |R|
    A = mu0 / (4 * np.pi) * np.sum(dl / np.linalg.norm(R, axis=1)[:, None], axis=0)
    return A[1], B[0], B[2]

def _graded(lo, hi, anchor, q, levels):
    """Stuetzstellen in [lo,hi], geometrisch zum Rand `anchor` (lo oder hi) verdichtet: Abstaende L q^k."""
    L = hi - lo
    d = [L * q**k for k in range(levels + 1)] + [0.0]
    pts = sorted({lo + x for x in d} if anchor == "lo" else {hi - x for x in d})
    return np.array(pts)

def _panel_sum(rr, zz, f, ng):
    """Produkt-Gauss (ng x ng) auf allen Panels rr x zz (Stuetzstellenlisten); f vektorisiert -> (3, ...)."""
    x, w = np.polynomial.legendre.leggauss(ng)
    ra, rb = rr[:-1], rr[1:]; za, zb = zz[:-1], zz[1:]
    R = (0.5 * (rb - ra)[:, None] * x[None] + 0.5 * (rb + ra)[:, None]).reshape(-1)
    WR = (0.5 * (rb - ra)[:, None] * w[None]).reshape(-1)
    Z = (0.5 * (zb - za)[:, None] * x[None] + 0.5 * (zb + za)[:, None]).reshape(-1)
    WZ = (0.5 * (zb - za)[:, None] * w[None]).reshape(-1)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    val = f(RR, ZZ)                                          # (3, nR, nZ)
    return np.einsum("kij,i,j->k", val, WR, WZ)

def rect_ring_fields(r0, r1, z0, z1, turns, rho, z, q=0.3, levels=11, ng=8):
    """(A_phi, B_rho, B_z) der Ringspule mit Rechteckquerschnitt (Strom 1 A je Windung).
    Aufpunkt-Projektion teilt den Querschnitt in bis zu 4 Teilrechtecke; in jedem werden die Panels geometrisch (Faktor q)
    zur dem Aufpunkt naechsten Ecke verdichtet (die integrable 1/dist-Singularitaet liegt dann an einer Panelecke)."""
    dens = turns / ((r1 - r0) * (z1 - z0))
    f = lambda a, zp: np.array(loop_fields(a, zp, rho, z)) * dens
    rc = min(max(rho, r0), r1); zc = min(max(z, z0), z1)
    rcuts = sorted({r0, r1, rc}); zcuts = sorted({z0, z1, zc})
    tot = np.zeros(3)
    for ra, rb in zip(rcuts[:-1], rcuts[1:]):
        for za, zb in zip(zcuts[:-1], zcuts[1:]):
            ar = "lo" if abs(rho - ra) <= abs(rho - rb) else "hi"
            az = "lo" if abs(z - za) <= abs(z - zb) else "hi"
            tot += _panel_sum(_graded(ra, rb, ar, q, levels), _graded(za, zb, az, q, levels), f, ng)
    return tot

def axis_bz_closed(r0, r1, z0, z1, turns, z):
    dens = turns / ((r1 - r0) * (z1 - z0))
    g = lambda zp, a: mu0 / 2 * a * a / (a * a + (z - zp)**2)**1.5
    val, _ = integrate.dblquad(lambda a, zp: g(zp, a), z0, z1, r0, r1, epsabs=1e-14, epsrel=1e-12)
    return dens * val

if __name__ == "__main__":
    ok = True
    # (B1)
    for (a, zp, rho, z) in [(0.04, -0.02, 0.07, 0.01), (0.09, -0.05, 0.03, 0.04), (0.05, 0.0, 0.05, 0.015)]:
        A1, Br1, Bz1 = loop_fields(a, zp, rho, z); A2, Br2, Bz2 = loop_direct(a, zp, rho, z)
        e = max(abs(A1 - A2) / abs(A2), abs(Br1 - Br2) / abs(Br2), abs(Bz1 - Bz2) / abs(Bz2))
        print(f"(B1) Schleife a={a} zp={zp}  Aufpunkt (rho={rho}, z={z}): max. rel. Abw. Formel/Linienintegral {e:.2e}"); ok &= e < 1e-5
    # (B2) curl A = B in Zylinderkoordinaten: B_r = -dA/dz, B_z = (1/rho) d(rho A)/drho
    a, zp, rho, z, h = 0.04, -0.02, 0.07, 0.01, 1e-6
    A0, Br0, Bz0 = loop_fields(a, zp, rho, z)
    Br_fd = -(loop_fields(a, zp, rho, z + h)[0] - loop_fields(a, zp, rho, z - h)[0]) / (2 * h)
    Bz_fd = ((rho + h) * loop_fields(a, zp, rho + h, z)[0] - (rho - h) * loop_fields(a, zp, rho - h, z)[0]) / (2 * h) / rho
    e = max(abs(Br_fd - Br0) / abs(Br0), abs(Bz_fd - Bz0) / abs(Bz0)); ew = abs(-Br_fd - Br0) / abs(Br0)
    print(f"(B2) curl A = B: Abw. {e:.2e}  (Gegenprobe Vorzeichen Br: {ew:.2f})"); ok &= e < 1e-6 and ew > 0.5
    # (B3) Achse, Innenspule TEAM 28 (r 27..55 mm, z -52..0 mm, 960 Windungen)
    for z in (0.02, -0.026, -0.06):
        v = rect_ring_fields(27e-3, 55e-3, -52e-3, 0.0, 960.0, 1e-9, z)   # rho = 1 nm (Achsennaehe, Formel braucht rho>0)
        ref = axis_bz_closed(27e-3, 55e-3, -52e-3, 0.0, 960.0, z)
        e = abs(v[2] - ref) / abs(ref)
        print(f"(B3) Achse z={z * 1e3:6.1f} mm: B_z adaptiv {v[2]:.8e} T/A, Referenz {ref:.8e},  rel. Abw. {e:.1e}"); ok &= e < 1e-6
    # (B4) Abstaende zum Querschnitt: Aufpunkt (rho, z), rho = 55 mm + d (aussen), d = 20, 5, 1, 0.1 mm, 0 (Rand), -1 mm (innen)
    z = -0.026
    def filaments_fine(rho, z, n=900):                      # unabhaengige Referenz: n x n Faeden (Zellmitten), vektorisiert
        a_ = 27e-3 + (np.arange(n) + 0.5) * 28e-3 / n; zp_ = -52e-3 + (np.arange(n) + 0.5) * 52e-3 / n
        A_, Z_ = np.meshgrid(a_, zp_, indexing="ij")
        v = np.array(loop_fields(A_, Z_, rho, z)) * (960.0 / (n * n))
        return v.sum(axis=(1, 2))
    for d in (20e-3, 5e-3, 1e-3, 1e-4, 0.0, -1e-3):
        rho = 55e-3 + d
        t = rect_ring_fields(27e-3, 55e-3, -52e-3, 0.0, 960.0, rho, z)
        t_fine = rect_ring_fields(27e-3, 55e-3, -52e-3, 0.0, 960.0, rho, z, q=0.2, levels=16, ng=10)
        ref = filaments_fine(rho + (1e-12 if d == 0.0 else 0.0), z)
        e_self = np.linalg.norm(t - t_fine) / np.linalg.norm(t_fine)
        e_ref = np.linalg.norm(t - ref) / np.linalg.norm(ref)
        print(f"(B4) Abstand d={d * 1e3:7.2f} mm vom Spulenrand: B_z = {t[2]:.8e} T/A;  Selbstkonvergenz (q=0.3/8 gegen q=0.2/10 Gauss) {e_self:.1e};  gegen 900x900-Faeden {e_ref:.1e}")
        ok &= e_self < 1e-8 and e_ref < 2e-3 if d > 0 else (e_self < 1e-6 and e_ref < 5e-3)
    # (B5) Fadenmodell gegen Volumenstrom, Innenspule: Randpunkte (rho, z)
    import itertools
    def filament(spacing, rho, z):
        nr = int(np.ceil((55e-3 - 27e-3) / spacing)); nz = int(np.ceil(52e-3 / spacing))
        a_ = 27e-3 + (np.arange(nr) + 0.5) * (28e-3 / nr); zp_ = -52e-3 + (np.arange(nz) + 0.5) * (52e-3 / nz)
        A_, Z_ = np.meshgrid(a_, zp_, indexing="ij")
        return (np.array(loop_fields(A_, Z_, rho, z)) * (960.0 / (nr * nz))).sum(axis=(1, 2))
    for (rho, z) in [(55e-3, -26e-3), (27e-3, -26e-3), (41e-3, 0.0), (41e-3, -52e-3)]:
        vol = rect_ring_fields(27e-3, 55e-3, -52e-3, 0.0, 960.0, rho, z)
        f05 = filament(0.5e-3, rho, z); f3 = filament(3e-3, rho, z)
        print(f"(B5) Randpunkt rho={rho * 1e3:.0f} mm z={z * 1e3:.0f} mm: |B| Volumen {np.linalg.norm(vol[1:]):.5e}; Faden 0.5 mm {np.linalg.norm(f05[1:] - vol[1:]) / np.linalg.norm(vol[1:]):.1e}, Faden 3 mm {np.linalg.norm(f3[1:] - vol[1:]) / np.linalg.norm(vol[1:]):.1e} relative Abweichung")
    print("ERGEBNIS:", "OK" if ok else "FEHLER"); raise SystemExit(0 if ok else 1)
