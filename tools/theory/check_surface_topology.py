#!/usr/bin/env python3
"""Kombinatorische Pruefung der Oberflaechen-Topologie auf Quadnetzen (maxwell-trace-coupling.md, Abschnitt 4).

Kantenkette a (Zirkulationen, Nedelec-Spur), G: Knoten -> Kante (diskreter Oberflaechengradient), R: Kante -> Quad
(Zirkulation um das Quad = Fluss n.B, diskretes curl_Gamma). Rotation n x bildet Nedelec-Spur auf RT-Kantenfluesse ab,
div_Gamma(n x a) = - curl_Gamma a; daher ist  ker(div_Gamma auf RT) = n x ker(R).
Geprueft fuer Wuerfeloberflaeche (g=0), Torus (g=1), Doppeltorus-Zusammenhang durch zwei getrennte Tori (2 Komponenten),
Brezel g=2 (aus zwei verklebten Tori nicht trivial, daher ersatzweise: Torus mit 2 Loechern per Euler-Rechnung nicht gebaut):
  (T1) R G = 0 exakt (fuer beliebige, auch nichtharmonische Knotenwerte)           -> Eichinvarianz der Kopplungsblocke
  (T2) dim ker R = dim range G + 2g_gesamt,  dim ker(G^T G) = Zahl der Komponenten
  (T3) harmonische Basis (ker R  orthogonal zu range G) hat Dimension 2g; Zirkulationen auf Meridian und Laengskreis bilden
       eine regulaere 2x2-Matrix je Torus (d.h. die 2g Freiheitsgrade sind genau die Zirkulationen auf den Erzeugern)
  (T4) RT-Zaehlung: dim ker(div) = V - K + 2g (K Komponenten), F+1-... Euler V - E + F = 2K - 2g_gesamt, E = 2F
Gegenprobe zu (T3): Gradienten haben auf jedem Zyklus Zirkulation 0.
"""
import numpy as np

class Surf:
    def __init__(self, quads, nnodes):
        self.quads = quads; self.V = nnodes
        edges = {}; elist = []
        self.qe = []                      # je Quad: [(edge_id, sign)]*4
        for q in quads:
            row = []
            for k in range(4):
                a, b = q[k], q[(k + 1) % 4]
                key = (min(a, b), max(a, b))
                if key not in edges: edges[key] = len(elist); elist.append(key)
                row.append((edges[key], 1 if a < b else -1))
            self.qe.append(row)
        self.edges = elist; self.E = len(elist); self.F = len(quads)
        self.G = np.zeros((self.E, self.V))
        for e, (a, b) in enumerate(elist): self.G[e, a] = -1; self.G[e, b] = 1
        self.R = np.zeros((self.F, self.E))
        for f, row in enumerate(self.qe):
            for e, s in row: self.R[f, e] += s
        self.eid = edges
    def chain(self, path):
        """geschlossener Knotenweg -> Kantenkette"""
        c = np.zeros(self.E)
        for a, b in zip(path, path[1:] + path[:1]):
            key = (min(a, b), max(a, b)); c[self.eid[key]] += 1 if a < b else -1
        return c

def torus(N, M, shift=0):
    idx = lambda i, j: (i % N) * M + (j % M) + shift
    quads = [[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(N) for j in range(M)]
    return quads, N * M, idx

def cube(n):
    pts = {}; quads = []
    def node(p):
        key = tuple(np.round(p, 9)); 
        if key not in pts: pts[key] = len(pts)
        return pts[key]
    u = np.linspace(-1, 1, n + 1)
    for axis in range(3):
        for sgn in (-1, 1):
            for i in range(n):
                for j in range(n):
                    def P(a, b):
                        v = [0, 0, 0]; v[axis] = sgn; v[(axis + 1) % 3] = u[a]; v[(axis + 2) % 3] = u[b]; return np.array(v)
                    q = [node(P(i, j)), node(P(i + 1, j)), node(P(i + 1, j + 1)), node(P(i, j + 1))]
                    quads.append(q)
    return quads, len(pts)

def analyse(name, S, genus_total, ncomp, cycles=None, tol=1e-9):
    ok = True
    RG = np.abs(S.R @ S.G).max()
    # nichtharmonische Knotenwerte: R G psi = 0 exakt
    psi = np.random.default_rng(0).normal(size=S.V) ** 3
    RGpsi = np.abs(S.R @ (S.G @ psi)).max()
    rankR = np.linalg.matrix_rank(S.R, tol=1e-9); rankG = np.linalg.matrix_rank(S.G, tol=1e-9)
    dimkerR = S.E - rankR
    # Nullraum von G^T G = Konstanten je Komponente
    kerGtG = S.V - rankG
    euler = S.V - S.E + S.F
    # harmonische Basis
    u, s, vt = np.linalg.svd(S.R)
    kerR = vt[np.sum(s > 1e-9):].T                       # Basis von ker R
    qG, _ = np.linalg.qr(S.G[:, :], mode="reduced")
    uG, sG, _ = np.linalg.svd(S.G, full_matrices=False)
    QG = uG[:, sG > 1e-9]                                # orthonormale Basis range G
    Hproj = kerR - QG @ (QG.T @ kerR)
    uh, sh, _ = np.linalg.svd(Hproj, full_matrices=False)
    H = uh[:, sh > 1e-8]
    dimH = H.shape[1]
    print(f"{name}: V={S.V} E={S.E} F={S.F}  Euler={euler} (erwartet {2 * ncomp - 2 * genus_total})  rank R={rankR} (F-K={S.F - ncomp})  "
          f"dim ker R={dimkerR}  dim range G={rankG}  dim H={dimH} (erwartet {2 * genus_total})  Komponenten (ker G^T G)={kerGtG}")
    print(f"    max|R G|={RG:.1e}   max|R G psi| mit nichtharmonischem psi={RGpsi:.1e}   RT-Zaehlung dim ker(div) = E - (F-K) = {S.E - (S.F - ncomp)} = V - K + 2g = {S.V - ncomp + 2 * genus_total}")
    ok &= RG == 0 and RGpsi < 1e-9 and euler == 2 * ncomp - 2 * genus_total and dimH == 2 * genus_total and kerGtG == ncomp
    ok &= S.E - (S.F - ncomp) == S.V - ncomp + 2 * genus_total
    if dimH: ok &= np.abs(S.R @ H).max() < 1e-9 and np.abs(QG.T @ H).max() < 1e-9
    if cycles:
        C = np.array([S.chain(c) for c in cycles])           # Zyklen x Kanten
        P = C @ H                                             # Zirkulationen der Harmonischen
        Cg = np.abs(C @ S.G).max()                            # Gradienten: Zirkulation auf geschlossenen Wegen 0
        sv = np.linalg.svd(P, compute_uv=False)
        print(f"    Zirkulationsmatrix (Zyklen x harmonische Basis) Singularwerte {np.round(sv, 4)}; max|Zirkulation eines Gradienten| = {Cg:.1e}")
        ok &= Cg < 1e-12 and sv.min() > 1e-3 and P.shape[0] == dimH
    return ok

if __name__ == "__main__":
    ok = True
    q, V = cube(4); ok &= analyse("Wuerfel 4x4 je Seite (g=0)", Surf(q, V), 0, 1)
    N, M = 12, 6
    q, V, idx = torus(N, M)
    cyc = [[idx(0, j) for j in range(M)], [idx(i, 0) for i in range(N)]]       # Meridian (klein), Laengskreis (gross)
    ok &= analyse("Torus 12x6 (g=1)", Surf(q, V), 1, 1, cyc)
    q1, V1, idx1 = torus(N, M)
    q2, V2, idx2 = torus(8, 5, shift=V1)
    cyc2 = cyc + [[idx2(0, j) for j in range(5)], [idx2(i, 0) for i in range(8)]]
    ok &= analyse("zwei getrennte Tori (g=1+1, K=2)", Surf(q1 + q2, V1 + V2), 2, 2, cyc2)
    # Platte (Vollscheibe/Quader, g=0) und Torus zusammen: 2 Komponenten, g=1
    qc, Vc = cube(3)
    qt, Vt, idxt = torus(10, 5, shift=Vc)
    ok &= analyse("Quader + Torus (K=2, g=1)", Surf(qc + qt, Vc + Vt), 1, 2, [[idxt(0, j) for j in range(5)], [idxt(i, 0) for i in range(10)]])
    print("ERGEBNIS:", "OK" if ok else "FEHLER")
    raise SystemExit(0 if ok else 1)
