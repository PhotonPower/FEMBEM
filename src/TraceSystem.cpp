#include "TraceSystem.h"

namespace trace {

Mat as_matrix(const std::vector<double>& d, int rows, int cols)
{
    Mat M(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) M(i, j) = d[static_cast<size_t>(i) * cols + j];
    return M;
}

DerivedOps derive_ops(const TraceSurface& S, const BoundaryOps& O, bool flat)
{
    DerivedOps D;
    D.G = as_matrix(S.G_dense(), O.E, O.V);
    D.R = as_matrix(S.R_dense(), O.F, O.E);
    D.H = as_matrix(S.H, O.E, S.nH);
    const Mat Vr = as_matrix(flat ? O.VrotFlat : O.Vrot, O.E, O.E);
    const Mat Cr = as_matrix(O.Crot, O.E, O.E), Mr = as_matrix(O.Mrot, O.E, O.E);
    D.W = D.G.transpose() * Vr * D.G;
    // Z = (-G)^T Vrot H: lambda = n x sum c_e W_e mit c = -G psi + H c_h, also Kreuzblock der Gram-Matrix [G^T V G, -G^T V H; ., H^T V H]
    D.Z = -D.G.transpose() * Vr * D.H;
    D.Y = D.H.transpose() * Vr * D.H;
    D.Q = (0.5 * Mr + Cr) * D.H;      // <W_e, (1/2 + n x C) h_l>, h_l = n x (H_l-Feld)
    D.P = Mr * D.H;                    // P_{e,k} = <h_k, W_e> = int (n x H_k-Feld).W_e = (Mrot H)_{ek}   (Variante (J))
    D.N = 0.5 * as_matrix(O.Mt, O.F, O.V) + as_matrix(O.Dt, O.F, O.V);
    D.V0 = as_matrix(O.V0, O.F, O.F);
    return D;
}

Mat dense_K(const FemProblem& F, double sigma_factor)
{
    const int E = F.num_edges;
    Mat K = as_matrix(F.K, E, E);
    if (sigma_factor != 0.0) K += sigma_factor * as_matrix(F.Msig, E, E);
    return K;
}

Mat Coupled::reduced() const
{
    const int n = static_cast<int>(keep.size());
    Mat R(n, n);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) R(i, j) = A(keep[i], keep[j]);
    return R;
}

Vec Coupled::solve() const
{
    const int n = static_cast<int>(keep.size());
    Vec rb(n);
    for (int i = 0; i < n; ++i) rb[i] = b[keep[i]];
    const Vec xr = reduced().partialPivLu().solve(rb);
    Vec x = Vec::Zero(A.rows());
    for (int i = 0; i < n; ++i) x[keep[i]] = xr[i];
    return x;
}

Coupled build_coupled(const FemProblem& F, const BoundaryOps& O, const DerivedOps& D, const Vec& f, const CoupledOptions& opt)
{
    const TraceSurface& S = F.surface;
    Coupled C;
    C.nE = F.num_edges; C.nV = S.num_nodes(); C.nH = S.nH;
    const int nE = C.nE, nV = C.nV, nH = C.nH, N = nE + nV + nH;
    C.A = Mat::Zero(N, N);
    C.b = Vec::Zero(N);
    C.b.head(nE) = f;
    C.A.topLeftCorner(nE, nE) = dense_K(F, opt.sigma_factor);

    if (opt.include_bem) {
        const int Es = S.num_edges();
        const Mat Mt = as_matrix(O.Mt, O.F, O.V);
        const Mat& R = D.R;
        // Zeilen/Spalten der Flaechenkanten in den FEM-Raum einbetten: Bloecke M (Es x X) -> FEM-Zeilen surf_edge_to_fem
        auto add_rows = [&](int col0, const Mat& M) {          // A(fem_edge(e), col0 + j) += M(e, j)
            for (int e = 0; e < Es; ++e) for (int j = 0; j < M.cols(); ++j) C.A(F.surf_edge_to_fem[e], col0 + j) += M(e, j);
        };
        auto add_cols = [&](int row0, const Mat& M) {          // A(row0 + i, fem_edge(e)) += M(i, e)
            for (int e = 0; e < Es; ++e) for (int i = 0; i < M.rows(); ++i) C.A(row0 + i, F.surf_edge_to_fem[e]) += M(i, e);
        };
        const Mat Nt = D.N.transpose();                           // V x F
        if (!opt.unsymmetric) {
            // (1,1): (1/mu0) R^T V0 R eingebettet
            const Mat RtV0R = (1.0 / F.mu0) * R.transpose() * D.V0 * R;   // Es x Es
            for (int e = 0; e < Es; ++e) for (int g = 0; g < Es; ++g) C.A(F.surf_edge_to_fem[e], F.surf_edge_to_fem[g]) += RtV0R(e, g);
            add_rows(nE, R.transpose() * D.N);                    // (1,2)
            add_rows(nE + nV, D.Q);                               // (1,3)
        } else {
            add_rows(nE, R.transpose() * Mt);                     // (J1,2): <psi, g_v> = (R^T Mt psi)
            add_rows(nE + nV, D.P);                               // (J1,3): <h_k, W_e>
        }
        add_cols(nE, Nt * R);                                     // (2,1)
        add_cols(nE + nV, D.Q.transpose());                       // (3,1)
        C.A.block(nE, nE, nV, nV) = -F.mu0 * D.W;                 // (2,2)
        C.A.block(nE, nE + nV, nV, nH) = -F.mu0 * D.Z;            // (2,3)
        C.A.block(nE + nV, nE, nH, nV) = -F.mu0 * D.Z.transpose();// (3,2)
        C.A.block(nE + nV, nE + nV, nH, nH) = -F.mu0 * D.Y;       // (3,3)
    }

    if (opt.flip_lambda && opt.include_bem) {
        C.A.block(0, nE, nE, nV + nH) *= -1.0;
    }

    // Freiheitsgrade
    std::vector<char> drop(N, 0);
    if (opt.gauge_tree) for (int e = 0; e < nE; ++e) if (F.gauged[e]) drop[e] = 1;
    if (opt.include_bem && opt.pin_psi) {
        std::vector<char> seen(S.num_components, 0);
        for (int v = 0; v < nV; ++v) if (!seen[S.node_comp[v]]) { seen[S.node_comp[v]] = 1; drop[nE + v] = 1; }
    }
    if (!opt.include_bem) for (int i = nE; i < N; ++i) drop[i] = 1;
    for (int i = 0; i < N; ++i) if (!drop[i]) C.keep.push_back(i);
    return C;
}

} // namespace trace
