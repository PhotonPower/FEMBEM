#!/usr/bin/env python3
"""Prueft die Darstellungsformel des Aussenfelds in den Spuren (maxwell-trace-coupling.md, Abschnitt 2.2).

Identitaeten (n zeigt in den Aussenraum, G = 1/(4 pi |x-y|), Psi(v)(x) = Int_Gamma G(x,y) v(y) dS_y):
  (R-A) A(x) = mu0 Psi(lam) + curl Psi(n x A) - grad Psi(n.A)      x aussen, lam = n x H
  (R-B) B(x) = mu0 curl Psi(lam) - grad Psi(n.B)                   x aussen
  beide ergeben 0 fuer x im Innern.
Gegenproben: jedes einzelne Vorzeichen umgekehrt muss durchfallen.
Testfeld: Punktdipol m = z im Ursprung (A = mu0 m x r / (4 pi r^3), auch mit Eichverschiebung grad chi,
chi = Dipolpotential, aussen harmonisch und abklingend), Kugel Radius 1 und ein verschobener Zylinder-aehnlicher
Rand (Ellipsoid) zur Kontrolle, dass nichts an der Kugel haengt.
"""
import numpy as np

mu0 = 1.0

def surface(kind, nt=80, np_=160):
    x, w = np.polynomial.legendre.leggauss(nt)          # x = cos(theta)
    ph = (np.arange(np_) + 0.5) * 2 * np.pi / np_
    X, P = np.meshgrid(x, ph, indexing="ij")
    W = np.outer(w, np.full(np_, 2 * np.pi / np_))
    s = np.sqrt(1 - X**2)
    u = np.stack([s * np.cos(P), s * np.sin(P), X], -1)   # Einheitskugel
    if kind == "sphere":
        a = np.array([1.0, 1.0, 1.0])
    else:
        a = np.array([1.0, 0.8, 1.3])                      # Ellipsoid (Halbachsen)
    pts = u * a
    # Normale n ~ u / a, Flaechenelement |dr/dth x dr/dph| = |grad F|^-1 ... ueber Vektorprodukt der Tangenten
    # F = sum (x/a)^2: n = grad F/|grad F|, dS = (1/(n_u.?)) -> direkt ueber Tangenten:
    th = np.arccos(X)
    dr_dth = np.stack([np.cos(th) * np.cos(P) * a[0], np.cos(th) * np.sin(P) * a[1], -np.sin(th) * a[2]], -1)
    dr_dph = np.stack([-np.sin(th) * np.sin(P) * a[0], np.sin(th) * np.cos(P) * a[1], np.zeros_like(th)], -1)
    nn = np.cross(dr_dth, dr_dph)
    dS = np.linalg.norm(nn, axis=-1)                       # dS = |.| dth dph ; Gauss: dcos = sin dth
    n = nn / dS[..., None]
    wt = W * dS / np.sin(th)                               # w_cos * dphi = sin(th) dth dphi
    # Aussennormale pruefen
    assert np.all(np.einsum("ijk,ijk->ij", n, pts) > 0)
    return pts.reshape(-1, 3), n.reshape(-1, 3), wt.reshape(-1)

def dipole_B(r, m=np.array([0, 0, 1.0])):
    rr = np.linalg.norm(r, axis=-1, keepdims=True)
    rh = r / rr
    return mu0 / (4 * np.pi) * (3 * np.einsum("...k,k->...", rh, m)[..., None] * rh - m) / rr**3

def dipole_A(r, m=np.array([0, 0, 1.0])):
    rr = np.linalg.norm(r, axis=-1, keepdims=True)
    return mu0 / (4 * np.pi) * np.cross(m, r) / rr**3

def chi_grad(r, c, d=np.array([0.3, -0.2, 1.0])):
    """grad chi mit chi = c d.r / |r|^3 (aussen harmonisch, abklingend; Eichverschiebung)."""
    rr = np.linalg.norm(r, axis=-1, keepdims=True)
    return c * (d / rr**3 - 3 * np.einsum("...k,k->...", r, d)[..., None] * r / rr**5)

def potentials(x, y, n, w, lam, mA, nA, nB):
    """liefert Psi(lam), curl Psi(mA), grad Psi(nA), grad Psi(nB) an den Punkten x (Summe ueber y)."""
    R = x[:, None, :] - y[None, :, :]
    r = np.linalg.norm(R, axis=-1)
    G = 1 / (4 * np.pi * r)
    gradG = -R / (4 * np.pi * r[..., None]**3)             # grad_x G
    ww = w[None, :]
    Psi_l = np.einsum("ij,ijk,j->ik", G, np.broadcast_to(lam[None], (len(x),) + lam.shape), w)
    curlPsi = np.einsum("j,ijk->ik", w, np.cross(gradG, np.broadcast_to(mA[None], gradG.shape)))
    gradPsi_nA = np.einsum("ijk,j,j->ik", gradG, w, nA)
    gradPsi_nB = np.einsum("ijk,j,j->ik", gradG, w, nB)
    return Psi_l, curlPsi, gradPsi_nA, gradPsi_nB

def run(kind, c_gauge, label):
    y, n, w = surface(kind)
    B = dipole_B(y)
    A = dipole_A(y) + chi_grad(y, c_gauge)
    lam = np.cross(n, B) / mu0
    mA = np.cross(n, A)
    nA = np.einsum("ik,ik->i", n, A)
    nB = np.einsum("ik,ik->i", n, B)
    xs_out = np.array([[2.0, 0.3, 0.5], [0.4, -2.5, 1.5], [-1.5, 1.0, -3.0]])
    xs_in = np.array([[0.2, 0.1, 0.1], [-0.3, 0.2, 0.0]])
    Psi_l, curlPsi, gnA, gnB = potentials(np.vstack([xs_out, xs_in]), y, n, w, lam, mA, nA, nB)
    A_ex = dipole_A(xs_out) + chi_grad(xs_out, c_gauge)
    B_ex = dipole_B(xs_out)
    o = slice(0, 3); i = slice(3, 5)
    scale = np.linalg.norm(A_ex)
    errA = np.linalg.norm(mu0 * Psi_l[o] + curlPsi[o] - gnA[o] - A_ex) / scale
    errB = np.linalg.norm(mu0 * np.stack([np.zeros(3)] * 3, -1) + 0 - 0) if False else None
    # curl Psi(lam) braucht eigenes Potential
    R = xs_out[:, None, :] - y[None]; r = np.linalg.norm(R, axis=-1)
    gradG = -R / (4 * np.pi * r[..., None]**3)
    curl_lam = np.einsum("j,ijk->ik", w, np.cross(gradG, np.broadcast_to(lam[None], gradG.shape)))
    errB = np.linalg.norm(mu0 * curl_lam - gnB[o] - B_ex) / np.linalg.norm(B_ex)
    Rin = xs_in[:, None, :] - y[None]; rin = np.linalg.norm(Rin, axis=-1)
    gradGin = -Rin / (4 * np.pi * rin[..., None]**3)
    curl_lam_in = np.einsum("j,ijk->ik", w, np.cross(gradGin, np.broadcast_to(lam[None], gradGin.shape)))
    inside_A = np.linalg.norm(mu0 * Psi_l[i] + curlPsi[i] - gnA[i]) / scale
    inside_B = np.linalg.norm(mu0 * curl_lam_in - gnB[i]) / np.linalg.norm(B_ex)
    print(f"{label:28s} (R-A) aussen {errA:9.2e}   (R-B) aussen {errB:9.2e}   innen (R-A) {inside_A:9.2e}  innen (R-B) {inside_B:9.2e}")
    # Gegenproben: Vorzeichen einzeln umkehren
    res = {}
    for name, (s1, s2, s3) in {"lam": (-1, 1, 1), "n x A": (1, -1, 1), "n.A": (1, 1, -1)}.items():
        e = np.linalg.norm(s1 * mu0 * Psi_l[o] + s2 * curlPsi[o] - s3 * gnA[o] - A_ex) / scale
        res[name] = e
    resB = {}
    for name, (s1, s2) in {"lam": (-1, 1), "n.B": (1, -1)}.items():
        resB[name] = np.linalg.norm(s1 * mu0 * curl_lam - s2 * gnB[o] - B_ex) / np.linalg.norm(B_ex)
    print("   Gegenproben A (Vorzeichen von ...):", {k: f"{v:.2e}" for k, v in res.items()}, " B:", {k: f"{v:.2e}" for k, v in resB.items()})
    return errA, errB, inside_A, inside_B, res, resB

if __name__ == "__main__":
    ok = True
    for kind in ("sphere", "ellipsoid"):
        for c in (0.0, 0.5):
            lab = f"{kind}, Eichverschiebung c={c}"
            eA, eB, iA, iB, res, resB = run(kind, c, lab)
            ok &= eA < 1e-8 and eB < 1e-8 and iA < 1e-8 and iB < 1e-8
            ok &= res["lam"] > 1e-2 and res["n x A"] > 1e-2 and resB["lam"] > 1e-2 and resB["n.B"] > 1e-2
            if c != 0.0:
                ok &= res["n.A"] > 1e-2      # n.A-Term ist nur bei Eichverschiebung nicht null
    print("ERGEBNIS:", "OK" if ok else "FEHLER")
    raise SystemExit(0 if ok else 1)
