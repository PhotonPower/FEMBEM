# =============================================================================
# Modellrechnung: woher die Vorgaben fuer --pc-inner-iters / --pc-inner-tol
# stammen
# =============================================================================
#
# Laufen lassen:  python3 tests/model_block_triangular_pc.py
# Braucht nur numpy und scipy, keine GPU, kein Netz, keinen Build.
#
# Diese Datei ist KEIN Abnahmetest des C++-Codes -- sie prueft das VERFAHREN,
# nicht die Implementierung. Sie baut ein kleines Modellproblem mit derselben
# Blockstruktur wie CoupledMVOperator::apply_physical()
#
#     M = (  A                -T      )
#         ( -(0.5*T + D_BEM)  +mu_0*G )
#
# skaliert es symmetrisch diagonal (wie setup_equilibration()) und laesst
# darauf ein FGMRES in genau der Bauart von GMRESSolver::solve() laufen --
# rechts vorkonditioniert, mit mitgespeicherter Z-Basis, weshalb ein von
# Aufruf zu Aufruf leicht verschiedener (innerer) Vorkonditionierer zulaessig
# ist.
#
# Verglichen werden der blockdiagonale Vorkonditionierer diag(A~, I) und der
# blocktriangulare mit S~ = mu_0*G_BEM, angewendet ueber ein inneres CG.
#
# Gemessenes Ergebnis (Stand der Einfuehrung von --blocktri):
#
#     blockdiagonal (Ist)             :   62 FGMRES-Iterationen
#     blocktriangulaer, inner_it= 2   :   47
#     blocktriangulaer, inner_it= 5   :   22
#     blocktriangulaer, inner_it=10   :   20     <- Vorgabe
#     blocktriangulaer, inner_it=20   :   20
#
# Daraus die Vorgaben in Backend.cu: 10 innere Iterationen, Toleranz 1e-2.
# Oberhalb von ~10 kostet das innere CG Aufwand, ohne die aeussere
# Iterationszahl weiter zu senken -- ein Vorkonditionierer muss billig
# annaehern, nicht genau loesen.
#
# Wer die Vorgaben aendern will, sollte diese Rechnung mit den eigenen
# Groessenordnungen (Kopplungsstaerke, kappa(G)) wiederholen, statt zu raten.
# =============================================================================

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

rng = np.random.default_rng(7)
MU0 = 4e-7 * np.pi

n_fem, n_bem = 1200, 300

# --- A: 3D-Laplace-artig, schlecht konditioniert -----------------------------
main = 6.0 + 0.5 * rng.random(n_fem)
A = sp.diags([main, -np.ones(n_fem-1), -np.ones(n_fem-1),
              -np.ones(n_fem-20), -np.ones(n_fem-20)],
             [0, 1, -1, 20, -20], format='csr') * 8.4e5   # Reluktivitaets-Skala

# --- T: Rand-Massenmatrix, koppelt die letzten n_bem FEM-DOFs ----------------
Tsq = sp.diags([4.0*np.ones(n_bem), np.ones(n_bem-1), np.ones(n_bem-1)],
               [0, 1, -1], format='csr') * 2.5e-5
T = sp.lil_matrix((n_fem, n_bem))
T[n_fem-n_bem:, :] = Tsq
T = T.tocsr() * 3.0e4

# --- G: Single-Layer, dicht, SPD, 1/(4 pi r) ---------------------------------
x = np.linspace(0.0, 1.0, n_bem)
R = np.abs(x[:, None] - x[None, :]) + 1e-2
G = 1.0 / (4.0 * np.pi * R)
G = 0.5 * (G + G.T)

# --- D: Double-Layer, unsymmetrisch ------------------------------------------
D = (x[:, None] - x[None, :]) / (4.0 * np.pi * R**2)
D = D * 1e-6

Dfull = sp.lil_matrix((n_bem, n_fem))
Dfull[:, n_fem-n_bem:] = D
M = sp.bmat([[A,                        -T],
             [-(0.5*T.T + Dfull.tocsr()), MU0*G]], format='csr').toarray()

N = n_fem + n_bem

# --- SDE: D_scale = 1/sqrt(|M_ii|) -------------------------------------------
d = 1.0 / np.sqrt(np.abs(np.diag(M)))
Mt = d[:, None] * M * d[None, :]

At = Mt[:n_fem, :n_fem]
Tt = -Mt[:n_fem, n_fem:]          # Block (1,2) ist -T
St = Mt[n_fem:, n_fem:]           # ~ D_K mu0 G D_K, Diagonale exakt 1

C1 =  Mt[:n_fem, n_fem:]
C2 =  Mt[n_fem:, :n_fem]
asym = np.linalg.norm(C1 - C2.T, 'fro') / np.linalg.norm(C1 + C2.T, 'fro')
print(f"Unsymmetrie der Kopplungsbloecke : {100*asym:.1f} %")
print(f"kappa(S_tilde)                   : {np.linalg.cond(St):.3e}")
print(f"kappa(M_tilde)                   : {np.linalg.cond(Mt):.3e}")

# ILU(0) auf dem FEM-Block
ilu = spla.spilu(sp.csc_matrix(At), fill_factor=1.0, drop_tol=0.0)

def inner_cg(Sop, b, max_it, tol):
    xk = np.zeros_like(b)
    r = b.copy()
    rho = r @ r
    if rho == 0.0:
        return xk, 0
    nb = np.sqrt(rho)
    p = r.copy()
    for it in range(max_it):
        Ap = Sop @ p
        pAp = p @ Ap
        if pAp <= 0.0:
            return xk, it
        al = rho / pAp
        xk += al * p
        r -= al * Ap
        rho_new = r @ r
        if np.sqrt(rho_new) <= tol * nb:
            return xk, it + 1
        p = r + (rho_new / rho) * p
        rho = rho_new
    return xk, max_it

def P_blockdiag(v):
    out = np.empty_like(v)
    out[:n_fem] = ilu.solve(v[:n_fem])
    out[n_fem:] = v[n_fem:]
    return out

inner_stats = []
def P_blocktri(v, inner_it=10, inner_tol=1e-2):
    zK, k = inner_cg(St, v[n_fem:], inner_it, inner_tol)
    inner_stats.append(k)
    rA = v[:n_fem] + Tt @ zK
    out = np.empty_like(v)
    out[:n_fem] = ilu.solve(rA)
    out[n_fem:] = zK
    return out

b = np.zeros(N)
b[:n_fem] = d[:n_fem] * rng.standard_normal(n_fem) * 1e3


def fgmres(Aop, b, Pfun, m=50, tol=1e-8, maxiter=1000):
    """Flexible GMRES(m), rechts-vorkonditioniert -- exakt die Bauart des
    GMRESSolver in src/CoupledSolver.cu (Z-Basis wird mitgespeichert)."""
    n = b.size
    x = np.zeros(n)
    bn = np.linalg.norm(b)
    it = 0
    while it < maxiter:
        r = b - Aop @ x
        rn = np.linalg.norm(r)
        if rn / bn <= tol:
            return x, it, rn / bn
        V = np.zeros((n, m + 1)); Z = np.zeros((n, m))
        H = np.zeros((m + 1, m)); g = np.zeros(m + 1)
        V[:, 0] = r / rn; g[0] = rn
        cs = np.zeros(m); sn = np.zeros(m)
        j = 0
        while j < m and it < maxiter:
            Z[:, j] = Pfun(V[:, j])
            w = Aop @ Z[:, j]
            for i in range(j + 1):
                H[i, j] = w @ V[:, i]
                w -= H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 1e-14:
                V[:, j + 1] = w / H[j + 1, j]
            for i in range(j):
                t = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = t
            den = np.hypot(H[j, j], H[j + 1, j])
            cs[j], sn[j] = H[j, j] / den, H[j + 1, j] / den
            H[j, j] = den; H[j + 1, j] = 0.0
            g[j + 1] = -sn[j] * g[j]; g[j] = cs[j] * g[j]
            it += 1; j += 1
            if abs(g[j]) / bn <= tol:
                break
        y = np.linalg.solve(H[:j, :j], g[:j])
        x += Z[:, :j] @ y
    return x, it, np.linalg.norm(b - Aop @ x) / bn


inner_stats.clear()
_, it, rel = fgmres(Mt, b, P_blockdiag, m=50, tol=1e-8, maxiter=1000)
print(f"{'blockdiagonal (Ist)':32s}: {it:4d} FGMRES-Iter., Rel.Res = {rel:.2e}")
for ni in (2, 5, 10, 20):
    inner_stats.clear()
    _, it, rel = fgmres(Mt, b, lambda v, k=ni: P_blocktri(v, k, 1e-2),
                        m=50, tol=1e-8, maxiter=1000)
    print(f"{'blocktriangulaer, inner_it=' + str(ni):32s}: {it:4d} FGMRES-Iter., "
          f"Rel.Res = {rel:.2e}  (CG im Mittel {np.mean(inner_stats):.1f})")
