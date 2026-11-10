#!/usr/bin/env python3
"""Adjungiertheit der Maxwell-Doppelschicht bei kappa = 0 (Grundlage der Symmetrie der Kopplung, maxwell-trace-coupling.md 5.2):
    <v, n x C eta> = - <eta, C (n x v)>     fuer tangentiale Felder v, eta auf der Kugel
mit C u(x) = PV Int grad_x G(x,y) x u(y) dS_y. Gegenprobe: Vorzeichen +.
Polarkoordinaten-Hauptwert wie in check_scalar_reduction.py.
"""
import numpy as np
from check_representation import surface
from check_scalar_reduction import inner_points

W0 = np.array([0.3, -0.5, 0.8]); M0 = np.array([[0.2, 0.7, -0.1], [0.4, -0.3, 0.5], [0.1, 0.2, 0.6]])
def tang(y, F):
    return F - np.einsum("...k,...k->...", F, y)[..., None] * y
v_field = lambda y: tang(y, W0 + y @ M0.T)                         # beliebiges tangentiales Feld
eta_field = lambda y: tang(y, np.cross(np.array([0.5, 0.2, -0.7]), y) + 0.3 * (y @ M0.T) ** 2)   # beliebiges tangentiales Feld (nicht div-frei)

def C_apply(x0, field, nrho=90, nphi=64):
    e0, y, dS = inner_points(x0, nrho, nphi)
    R = x0 - y; r = np.linalg.norm(R, axis=-1); r[r < 1e-14] = 1e-14
    gradG = -R / (4 * np.pi * r[..., None]**3)
    return np.einsum("ij,ijk->k", dS, np.cross(gradG, field(y)))

if __name__ == "__main__":
    xo, no, wo = surface("sphere", 14, 28)
    L = 0.0; Rr = 0.0
    for x0, n0, w0 in zip(xo, no, wo):
        Ce = C_apply(x0, eta_field)
        L += w0 * v_field(x0[None])[0] @ np.cross(n0, Ce)                           # <v, n x C eta>
        Cnv = C_apply(x0, lambda y: np.cross(y, v_field(y)))
        Rr += w0 * eta_field(x0[None])[0] @ Cnv                                      # <eta, C(n x v)>
    print(f"<v, n x C eta> = {L:.8f},  <eta, C(n x v)> = {Rr:.8f},  Summe (erwartet 0): {abs(L + Rr) / abs(L):.2e}  (Gegenprobe Differenz: {abs(L - Rr) / abs(L):.2f})")
    ok = abs(L + Rr) / abs(L) < 1e-4 and abs(L - Rr) / abs(L) > 0.5
    print("ERGEBNIS:", "OK" if ok else "FEHLER"); raise SystemExit(0 if ok else 1)
