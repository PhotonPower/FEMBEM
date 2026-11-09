#!/usr/bin/env python3
"""Prueft die Calderon-Zeilen des Aussenraums bei kappa = 0 (maxwell-trace-coupling.md, Abschnitt 2.3).

Mit n nach aussen, lam = n x H, g = n.B, V g = Int G g, C lam = PV Int grad_x G x lam, K' g = n_x . PV Int grad_x G g:
  (R-T)  1/2 lam - n x C lam + (1/mu0) n x grad_Gamma V g = 0        (tangential, vektorwertig)
  (R-N)  1/2 g + K' g - mu0 n . C lam = 0                             (skalar)
Hauptwerte per Polarkoordinaten um den Aufpunkt (Winkelsumme zuerst: die 1/rho^2-Anteile heben sich exakt).
Testfeld wie in check_representation.py (Dipol im Innern); Kugel (analytische Parametrisierung).
Gegenproben: Vorzeichen jedes Summanden umgekehrt.
"""
import numpy as np
mu0 = 1.0
def dipole_B(r, m=np.array([0, 0, 1.0])):
    rr = np.linalg.norm(r, axis=-1, keepdims=True); rh = r / rr
    return mu0 / (4 * np.pi) * (3 * np.einsum("...k,k->...", rh, m)[..., None] * rh - m) / rr**3

def pv_integrals(x0, nrho=200, nphi=128):
    x0 = x0 / np.linalg.norm(x0)
    """liefert (C lam, grad-Anteil von Int grad_x G g, n_x . dasselbe) am Aufpunkt x0 der Einheitskugel."""
    e0 = x0 / np.linalg.norm(x0)
    t = np.array([1.0, 0, 0]) if abs(e0[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(e0, t); e1 /= np.linalg.norm(e1); e2 = np.cross(e0, e1)
    xr, wr = np.polynomial.legendre.leggauss(nrho)
    rho = (xr + 1) * np.pi / 2; wrho = wr * np.pi / 2
    phi = np.arange(nphi) * 2 * np.pi / nphi
    RHO, PHI = np.meshgrid(rho, phi, indexing="ij")
    y = (np.cos(RHO)[..., None] * e0 + np.sin(RHO)[..., None] * (np.cos(PHI)[..., None] * e1 + np.sin(PHI)[..., None] * e2))
    dS = np.sin(RHO) * wrho[:, None] * (2 * np.pi / nphi)
    n = y
    B = dipole_B(y)
    lam = np.cross(n, B) / mu0
    g = np.einsum("ijk,ijk->ij", n, B)
    R = x0 - y
    r = np.linalg.norm(R, axis=-1)
    mask = r > 1e-12
    gradG = np.zeros_like(R)
    gradG[mask] = -R[mask] / (4 * np.pi * r[mask][..., None]**3)
    # Winkelsumme zuerst (Hauptwert), dann rho
    C_lam = np.einsum("ij,ijk->k", dS, np.cross(gradG, lam))
    gradVg = np.einsum("ij,ijk,ij->k", dS, gradG, g)
    return C_lam, gradVg, e0

def residuals(x0):
    C_lam, gradVg, n0 = pv_integrals(x0)
    B0 = dipole_B(n0)
    lam0 = np.cross(n0, B0) / mu0
    g0 = n0 @ B0
    Kp_g = n0 @ gradVg
    gradGamma_Vg = gradVg - n0 * (n0 @ gradVg)
    terms_T = [0.5 * lam0, -np.cross(n0, C_lam), np.cross(n0, gradGamma_Vg) / mu0]
    terms_N = [0.5 * g0, Kp_g, -mu0 * (n0 @ C_lam)]
    return terms_T, terms_N, lam0, g0

if __name__ == "__main__":
    ok = True
    rng = np.random.default_rng(1)
    for x0 in [np.array([1.0, 0.0, 0.2]), np.array([0.3, 0.5, 0.8]), np.array([-0.6, 0.1, -0.4])]:
        tT, tN, lam0, g0 = residuals(x0)
        rT = np.linalg.norm(sum(tT)) / np.linalg.norm(lam0)
        rN = abs(sum(tN)) / abs(g0)
        # Gegenproben: jeweils ein Summand mit umgekehrtem Vorzeichen
        cT = [np.linalg.norm(sum(tT) - 2 * tT[i]) / np.linalg.norm(lam0) for i in range(3)]
        cN = [abs(sum(tN) - 2 * tN[i]) / abs(g0) for i in range(3)]
        print(f"x0 = {np.round(x0 / np.linalg.norm(x0), 3)}:  (R-T) {rT:.2e}   (R-N) {rN:.2e}   Gegenproben T {['%.2f' % c for c in cT]} N {['%.2f' % c for c in cN]}")
        ok &= rT < 1e-6 and rN < 1e-6 and min(cT) > 0.05 and min(cN) > 0.05
    print("ERGEBNIS:", "OK" if ok else "FEHLER")
    raise SystemExit(0 if ok else 1)
