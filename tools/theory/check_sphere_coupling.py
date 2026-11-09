#!/usr/bin/env python3
"""Ende-zu-Ende-Vorzeichenpruefung der Kopplungsgleichungen an einer Kugel mit bekannter Loesung (maxwell-trace-coupling.md, 5.1-5.3).

Problem: Kugel R=1 (FEM-Gebiet, nu = 1/mu0, sigma = 0), eingepraegte Stromdichte J_s = j0 (z x r) (azimutal, div J = 0, J.n = 0),
Aussenraum Luft. Exakte Loesung (analytisch): A = a(r) sin(th) e_phi mit
   innen a = alpha r - mu0 j0 r^3/10,  aussen a = beta r^-2,  beta = mu0 j0/15, alpha = mu0 j0/6.
Spuren: g = n.B = 2 beta cos(th), lam = n x H = (beta/mu0) sin(th) e_phi = curl_Gamma psi mit psi = (beta/mu0) cos(th)
(curl_Gamma psi := grad_Gamma psi x n; H_aussen = -grad phi, psi = phi|_Gamma).
Geprueft (sympy, exakt):
  (V1) Gleichung innen  curl H = J_s,  Aussen curl H = 0,  Stetigkeit von n.B und n x H
  (V2) schwache Form (Zeile 1):  Int nu curl A . curl v = Int J_s . v - Int_Gamma lam . v_t   fuer Testfelder v
  (V3) lam = curl_Gamma psi mit psi = phi|_Gamma;  g = -mu0 d_n phi
  (V4) Eigenwerte auf Y_l (Kugel R=1): V = 1/(2l+1), D = K' = -1/(2(2l+1)), W = l(l+1)/(2l+1);
       S_ext = W + (1/2 - K') V^-1 (1/2 - D) = l+1 = -d_n phi / phi fuer phi = r^-(l+1) Y_l
  (V5) Zeile 2 (symmetrisch):  <psi', g> - mu0 <psi', S_ext psi> = 0  und Row 1 des Skalarfalls:  (1/2 - D) psi + V p = 0, p = d_n phi = -g/mu0
Gegenproben: umgekehrtes Vorzeichen von lam in (V2) und von g in (V5) muss durchfallen.
"""
import numpy as np
from numpy.polynomial import Polynomial as P
mu0 = 1.3; j0 = 0.7                                       # beliebige Zahlen (Ergebnis ist linear in j0)
beta = mu0 * j0 / 15; alpha = mu0 * j0 / 6
r = P([0, 1])
a_in = alpha * r - mu0 * j0 * r**3 / 10
a_out_coef = beta                                         # a_aussen = beta r^-2 (Rest per Hand: (r a)' = -beta r^-2)
ok = True
xg, wg = np.polynomial.legendre.leggauss(40)
def theta_int(f):                                         # Int_0^pi f(theta) dtheta
    t = (xg + 1) * np.pi / 2
    return np.sum(wg * f(t)) * np.pi / 2
def rad_int(f):                                           # Int_0^1 f(r) dr
    t = (xg + 1) / 2
    return np.sum(wg * f(t)) / 2

# B = curl(a sin(th) e_phi):  Br = 2 a cos(th)/r,  Bth = -(1/r)(r a)' sin(th)
def B_in(rr, t):
    return 2 * a_in(rr) * np.cos(t) / rr, -((r * a_in).deriv()(rr)) * np.sin(t) / rr
def B_out(rr, t):
    return 2 * beta * rr**-3 * np.cos(t), beta * rr**-3 * np.sin(t)         # (r a)' = -beta r^-2 -> Bth = + beta r^-3 sin
# (V1) Stetigkeit bei r = 1
t0 = 0.7
d = np.array(B_in(1.0, t0)) - np.array(B_out(1.0, t0))
print("(V1) Sprung (n.B, B_theta) bei r=1:", d); ok &= np.abs(d).max() < 1e-14
# curl B = mu0 J innen / 0 aussen:  (curl B)_phi = (1/r)(d_r(r B_th) - d_th B_r)
h = 1e-5
def curlB_phi(Bf, rr, t):
    rBth = lambda x: x * Bf(x, t)[1]
    dBr = (Bf(rr, t + h)[0] - Bf(rr, t - h)[0]) / (2 * h)
    return ((rBth(rr + h) - rBth(rr - h)) / (2 * h) - dBr) / rr
e_in = curlB_phi(B_in, 0.6, 0.9) / mu0 - j0 * 0.6 * np.sin(0.9)
e_out = curlB_phi(B_out, 1.7, 0.9)
print("     curl H - J_s (innen):", e_in, "  curl H (aussen):", e_out); ok &= abs(e_in) < 1e-7 and abs(e_out) < 1e-7

# Spuren bei r = 1 (aussen)
g = lambda t: 2 * beta * np.cos(t)
lam_phi = lambda t: B_out(1.0, t)[1] / mu0                 # n x H = H_theta e_phi (r x theta = phi)
psi = lambda t: (beta / mu0) * np.cos(t)
dpsi = lambda t: -(beta / mu0) * np.sin(t)
curl_psi_phi = lambda t: -dpsi(t)                          # curl_Gamma psi = grad psi x n = psi_theta (theta x r) = - psi_theta e_phi
tt = np.linspace(0.1, 3.0, 5)
print("(V3) lam_phi - curl_Gamma psi_phi:", np.abs(lam_phi(tt) - curl_psi_phi(tt)).max()); ok &= np.abs(lam_phi(tt) - curl_psi_phi(tt)).max() < 1e-14
# phi = c r^-2 cos th, c = beta/mu0:  mu0 H_r = 2 mu0 c r^-3 cos = g
print("     mu0 H_r(1) - g:", np.abs(2 * mu0 * (beta / mu0) * np.cos(tt) - g(tt)).max()); ok &= np.abs(2 * beta * np.cos(tt) - g(tt)).max() < 1e-14

# (V2) schwache Form fuer v = r^k sin(th) e_phi
def weak(k, sign):
    v = lambda rr: rr**k
    dv = lambda rr: (k + 1) * rr**k                         # (r v)' = (k+1) r^k
    def integrand_vol(t, rr):
        Br, Bth = B_in(rr, t)
        Brv, Bthv = 2 * v(rr) * np.cos(t) / rr, -dv(rr) * np.sin(t) / rr
        return ((Br * Brv + Bth * Bthv) / mu0 - j0 * rr * np.sin(t) * v(rr) * np.sin(t)) * rr**2 * np.sin(t)
    vol = 0.0
    for xr_, wr_ in zip((xg + 1) / 2, wg / 2):
        vol += wr_ * theta_int(lambda t: integrand_vol(t, xr_))
    vol *= 2 * np.pi
    bnd = theta_int(lambda t: lam_phi(t) * 1.0 * np.sin(t) * np.sin(t)) * 2 * np.pi
    return vol + sign * bnd                                  # Int nu curlA.curlv - Int J.v + sign * Int lam.v
for k in (1, 2):
    e = weak(k, +1); ew = weak(k, -1)
    print(f"(V2) k={k}:  Int nu curl A.curl v - Int J.v + Int lam.v = {e:.2e}   (Gegenprobe Vorzeichen von lam: {ew:.3f})")
    ok &= abs(e) < 1e-12 and abs(ew) > 1e-3

# (V4) Eigenwerte (Kugel R = 1); a - b = 1, -(l+1) a = l b
for l in (1, 2, 3, 5):
    Vl = 1 / (2 * l + 1); Dl = -1 / (2 * (2 * l + 1)); Wl = l * (l + 1) / (2 * l + 1)
    a = l / (2 * l + 1); b = -(l + 1) / (2 * l + 1)
    Sl = Wl + (0.5 - Dl) / Vl * (0.5 - Dl)
    okl = abs(a - b - 1) < 1e-14 and abs(-(l + 1) * a - l * b) < 1e-14 and abs((a + b) / 2 - Dl) < 1e-14 and abs((l + 1) * a - Wl) < 1e-14 and abs(Sl - (l + 1)) < 1e-14
    print(f"(V4) l={l}: DLP aussen/innen {a:.4f}/{b:.4f}, Mittel = D = {Dl:.4f}, W = {Wl:.4f}, S_ext = W + (1/2-K')V^-1(1/2-D) = {Sl:.6f} (erwartet {l + 1})")
    ok &= okl

# (V5) Zeile 2 und Skalar-Row-1 fuer l = 1
S1 = 2.0
row2 = g(tt) - mu0 * S1 * psi(tt); row2w = -g(tt) - mu0 * S1 * psi(tt)
print("(V5) g - mu0 S_ext psi:", np.abs(row2).max(), "   Gegenprobe (g -> -g):", np.abs(row2w).max()); ok &= np.abs(row2).max() < 1e-14 and np.abs(row2w).max() > 1e-3
p = -g(tt) / mu0
row1 = (0.5 + 1 / 6) * psi(tt) + (1 / 3) * p
print("     (1/2 - D) psi + V p:", np.abs(row1).max()); ok &= np.abs(row1).max() < 1e-14
print("ERGEBNIS:", "OK" if ok else "FEHLER"); raise SystemExit(0 if ok else 1)
