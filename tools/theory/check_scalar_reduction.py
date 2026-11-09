#!/usr/bin/env python3
"""Prueft die Rueckfuehrung der Vektoroperatoren auf skalare Laplace-Operatoren (maxwell-trace-coupling.md, 3.3).

Konvention: curl_Gamma psi := grad_Gamma psi x n,  fuer tangentiale v: curl_Gamma v := n.curl v,  div_Gamma(n x v) = - n.curl v.
 (I1) Potentialidentitaet   Psi(curl_Gamma psi) = curl Psi(psi n)   und   curl Psi(curl_Gamma psi) = - grad DLP(psi),
      DLP(psi)(x) = Int d_{n_y}G psi dS_y     (regulaer ausgewertet in Aussen- und Innenpunkten)
 (I2) Nedelec-Identitaet    <V curl_Gamma psi', curl_Gamma psi> = <W psi', psi>,  W = - d_n DLP,
      Kugel R=1, psi = Y_l: Wert l(l+1)/(2l+1) * ||Y_l||^2 (aus den Sprungrelationen, unabhaengig hergeleitet)
 (I3) Doppelschicht-Identitaet  <curl_Gamma psi', C m> = <psi', K' div_Gamma m>,  C m = PV Int grad_x G x m, K' = n_x . PV Int grad_x G (.)
      mit m = n x A_t des Dipolfelds (div_Gamma m = - n.B).
Gegenproben: falsches Vorzeichen in (I1)/(I3) bzw. falscher Faktor in (I2).
"""
import numpy as np
from check_representation import surface, dipole_A, dipole_B

def sph_Y(l, pts):
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    if l == 1: return z
    if l == 2: return 0.5 * (3 * z**2 - 1)
    if l == 3: return 0.5 * (5 * z**3 - 3 * z)
    raise ValueError

def grad_Gamma(l, pts):
    """tangentialer Gradient von Y_l(z) auf der Einheitskugel."""
    z = pts[..., 2]
    if l == 1: g = np.array([0, 0, 1.0]) * np.ones_like(pts)
    elif l == 2: g = np.stack([0 * z, 0 * z, 3 * z], -1)
    else: g = np.stack([0 * z, 0 * z, 0.5 * (15 * z**2 - 3)], -1)
    return g - np.einsum("...k,...k->...", g, pts)[..., None] * pts

def check_I1():
    y, n, w = surface("sphere", 90, 180)
    l = 2
    psi = sph_Y(l, y)
    curl_psi = np.cross(grad_Gamma(l, y), n)
    xs = np.array([[2.0, 0.3, 0.5], [0.4, -2.5, 1.5], [0.2, 0.1, 0.3], [-0.3, 0.2, -0.4]])
    R = xs[:, None, :] - y[None]; r = np.linalg.norm(R, axis=-1)
    gradG = -R / (4 * np.pi * r[..., None]**3)
    lhs = np.einsum("j,ijk->ik", w, np.cross(gradG, np.broadcast_to(curl_psi[None], gradG.shape)))       # curl Psi(curl_Gamma psi)
    rhs_curlPsi_psin = np.einsum("j,ijk->ik", w, np.cross(gradG, (psi[:, None] * n)[None]))              # curl Psi(psi n) (innen/aussen)
    # Psi(curl psi) = curl Psi(psi n): Vergleich der Potentiale
    G = 1 / (4 * np.pi * r)
    Psi_curl = np.einsum("ij,jk,j->ik", G, curl_psi, w)
    err_a = np.linalg.norm(Psi_curl - rhs_curlPsi_psin) / np.linalg.norm(Psi_curl)
    # curl curl Psi(psi n) = - grad DLP(psi):   DLP-Gradient numerisch durch zentrale Differenzen (h klein, glatt)
    def dlp(x):
        R_ = x[None, :] - y; r_ = np.linalg.norm(R_, axis=-1)
        return np.sum(w * psi * np.einsum("jk,jk->j", R_, n) / (4 * np.pi * r_**3))
    grad_dlp = np.zeros((len(xs), 3)); h = 1e-4
    for i, x in enumerate(xs):
        for k in range(3):
            e = np.zeros(3); e[k] = h
            grad_dlp[i, k] = (dlp(x + e) - dlp(x - e)) / (2 * h)
    err_b = np.linalg.norm(lhs + grad_dlp) / np.linalg.norm(lhs)
    err_b_wrong = np.linalg.norm(lhs - grad_dlp) / np.linalg.norm(lhs)
    print(f"(I1) Psi(curl psi) = curl Psi(psi n): {err_a:.2e};  curl Psi(curl psi) = -grad DLP(psi): {err_b:.2e}  (Gegenprobe +grad: {err_b_wrong:.2f})")
    return err_a < 1e-9 and err_b < 1e-6 and err_b_wrong > 0.5

def polar_frame(x0):
    e0 = x0 / np.linalg.norm(x0)
    t = np.array([1.0, 0, 0]) if abs(e0[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(e0, t); e1 /= np.linalg.norm(e1); e2 = np.cross(e0, e1)
    return e0, e1, e2

def inner_points(x0, nrho, nphi):
    e0, e1, e2 = polar_frame(x0)
    xr, wr = np.polynomial.legendre.leggauss(nrho)
    rho = (xr + 1) * np.pi / 2; wrho = wr * np.pi / 2
    phi = np.arange(nphi) * 2 * np.pi / nphi
    RHO, PHI = np.meshgrid(rho, phi, indexing="ij")
    y = (np.cos(RHO)[..., None] * e0 + np.sin(RHO)[..., None] * (np.cos(PHI)[..., None] * e1 + np.sin(PHI)[..., None] * e2))
    dS = np.sin(RHO) * wrho[:, None] * (2 * np.pi / nphi)
    return e0, y, dS

MD = np.array([1.0, 0.5, 0.3])        # Dipolrichtung (nicht achsensymmetrisch)
CL = np.array([0.7, -0.2, 0.4])       # psi' = CL . x  (Grad l = 1)

def check_I2_I3():
    l = 2
    ok = True
    # aeussere Quadratur ueber die Kugel
    xo, no, wo = surface("sphere", 14, 28)
    nrho, nphi = 90, 64
    sum_V = 0.0; sum_Ctest = 0.0; sum_Ktest = 0.0
    for x0, n0, w0 in zip(xo, no, wo):
        e0, y, dS = inner_points(x0, nrho, nphi)
        # (I2) innere Einfachschicht von curl_Gamma psi (komponentenweise, schwach singulaer, Polarkoordinaten)
        cy = np.cross(grad_Gamma(l, y), y)
        R = x0 - y; r = np.linalg.norm(R, axis=-1); r[r < 1e-14] = 1e-14
        Vc = np.einsum("ij,ijk->k", dS / (4 * np.pi * r), cy)
        cx = np.cross(grad_Gamma(l, x0[None])[0], n0)
        sum_V += w0 * cx @ Vc
        # (I3) m = n x A_t des Dipolfelds; div_Gamma m = - n.B
        A = dipole_A(y, MD); B = dipole_B(y, MD)
        m = np.cross(y, A)
        gradG = -R / (4 * np.pi * r[..., None]**3)
        Cm = np.einsum("ij,ijk->k", dS, np.cross(gradG, m))
        g = np.einsum("ijk,ijk->ij", y, B)
        Kp = e0 @ np.einsum("ij,ijk,ij->k", dS, gradG, -g)             # K' (div_Gamma m) mit div_Gamma m = - g
        psi_x = CL @ x0
        gl = CL - (CL @ x0) * x0
        sum_Ctest += w0 * (np.cross(gl, n0) @ Cm)
        sum_Ktest += w0 * psi_x * Kp
    Y = sph_Y(l, xo)
    normY2 = np.sum(wo * Y**2)
    W_exact = l * (l + 1) / (2 * l + 1) * normY2
    print(f"(I2) <V curl psi, curl psi> = {sum_V:.8f},  l(l+1)/(2l+1)*||Y||^2 = {W_exact:.8f},  rel. Abw. {abs(sum_V - W_exact) / W_exact:.2e}  (Gegenprobe Faktor 2: {abs(2 * sum_V - W_exact) / W_exact:.2f})")
    ok &= abs(sum_V - W_exact) / W_exact < 1e-4 and abs(2 * sum_V - W_exact) / W_exact > 0.5
    print(f"(I3) <curl psi', C m> = {sum_Ctest:.8f},  <psi', K' div m> = {sum_Ktest:.8f},  Abw. {abs(sum_Ctest - sum_Ktest) / abs(sum_Ktest):.2e}  (Gegenprobe Vorzeichen: {abs(sum_Ctest + sum_Ktest) / abs(sum_Ktest):.2f})")
    ok &= abs(sum_Ctest - sum_Ktest) / abs(sum_Ktest) < 1e-4 and abs(sum_Ctest + sum_Ktest) / abs(sum_Ktest) > 0.5
    return ok

if __name__ == "__main__":
    ok = check_I1()
    ok &= check_I2_I3()
    print("ERGEBNIS:", "OK" if ok else "FEHLER")
    raise SystemExit(0 if ok else 1)
