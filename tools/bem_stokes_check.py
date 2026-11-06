#!/usr/bin/env python3
"""m_z(Nulldurchgang) der Platte aus dem B_z-Dump allein (Stokes), gegen die berichtete Diagnose.

Axialsymmetrie: 2 pi r A_phi(r) = Phi(r) = Fluss durch den Kreis r. Mit J = -sigma omega A_r (Nulldurchgang)
folgt m_z = 1/2 sigma omega Int r A_phi dV = (sigma omega / 4 pi) Int Phi(r) dV. Phi(r) wird aus den
Plattenelementen (B_z mal Elementflaeche V/t, nach Radius des Mittelpunkts sortiert) summiert. Das ist
unabhaengig von beiden Lorentz- und Wirbelstromkernen: stimmt der berichtete Wert, ist A_r der Platte mit
dem B_z des Pfads konsistent, und ein Unterschied zwischen den Pfaden liegt im Feld, nicht im Kernel.

    python3 tools/bem_stokes_check.py NETZ.vtk FREQ_HZ DUMP.csv [DUMP2.csv ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import meshcheck  # noqa: E402

SIGMA = 3.4e7      # S/m, Aluminium
THICK = 3e-3       # Plattendicke [m]
TETS = [(0, 1, 3, 4), (1, 2, 3, 6), (1, 4, 5, 6), (3, 4, 6, 7), (1, 3, 4, 6)]


def main():
    mesh = meshcheck.read_vtk(sys.argv[1])
    omega = 2 * np.pi * float(sys.argv[2])
    pts = mesh.points * 1e-3
    vol = np.array([sum(abs(np.dot(np.cross(pts[c][b] - pts[c][a], pts[c][cc] - pts[c][a]),
                                    pts[c][d] - pts[c][a])) / 6 for a, b, cc, d in TETS)
                    for c in mesh.cells])
    for path in sys.argv[3:]:
        d = np.genfromtxt(path, delimiter=",", names=True)
        m = d["body"] == 1                       # Dump-Code 1 = Platte
        idx = d["elem"][m].astype(int)
        V = vol[idx]
        flux_el = d["Brz"][m] * V / THICK
        o = np.argsort(np.hypot(d["x"][m], d["y"][m]))
        cum = np.cumsum(flux_el[o])
        phi = np.empty_like(cum)
        phi[o] = cum - 0.5 * flux_el[o]
        mz = SIGMA * omega / (4 * np.pi) * np.sum(phi * V)
        print(f"{os.path.basename(path)}: |m_z(Null)| aus B_z (Stokes) = {abs(mz):.5g} A m^2")


if __name__ == "__main__":
    main()
