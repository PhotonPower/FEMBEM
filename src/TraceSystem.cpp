#include "TraceSystem.h"

#include <Eigen/Sparse>
#include <chrono>

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
    Mat K = Mat::Zero(E, E);
    for (const Triplet& t : F.K) K(t.row, t.col) += t.val;
    if (sigma_factor != 0.0) for (const Triplet& t : F.Msig) K(t.row, t.col) += sigma_factor * t.val;
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
    if (scaled) x.tail(nV + nH) /= mu0;                // psi = psi~ / mu0, c = c~ / mu0
    return x;
}

Coupled build_coupled(const FemProblem& F, const BoundaryOps& O, const DerivedOps& D, const Vec& f, const CoupledOptions& opt)
{
    const TraceSurface& S = F.surface;
    Coupled C;
    C.nE = F.num_edges; C.nV = S.num_nodes(); C.nH = S.nH;
    C.scaled = opt.scaled; C.mu0 = F.mu0;
    const double kfac = opt.scaled ? F.mu0 : 1.0;          // Faktor vor K (und vor der rechten Seite)
    const double v0fac = opt.scaled ? 1.0 : 1.0 / F.mu0;   // Faktor vor R^T V0 R
    const double wfac = opt.scaled ? 1.0 : F.mu0;          // Faktor vor W, Z, Y
    const int nE = C.nE, nV = C.nV, nH = C.nH, N = nE + nV + nH;
    C.A = Mat::Zero(N, N);
    C.b = Vec::Zero(N);
    C.b.head(nE) = kfac * f;
    C.A.topLeftCorner(nE, nE) = kfac * dense_K(F, opt.sigma_factor);

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
            const Mat RtV0R = v0fac * R.transpose() * D.V0 * R;   // Es x Es
            for (int e = 0; e < Es; ++e) for (int g = 0; g < Es; ++g) C.A(F.surf_edge_to_fem[e], F.surf_edge_to_fem[g]) += RtV0R(e, g);
            add_rows(nE, R.transpose() * D.N);                    // (1,2)
            add_rows(nE + nV, D.Q);                               // (1,3)
        } else {
            add_rows(nE, R.transpose() * Mt);                     // (J1,2): <psi, g_v> = (R^T Mt psi)
            add_rows(nE + nV, D.P);                               // (J1,3): <h_k, W_e>
        }
        add_cols(nE, Nt * R);                                     // (2,1)
        add_cols(nE + nV, D.Q.transpose());                       // (3,1)
        C.A.block(nE, nE, nV, nV) = -wfac * D.W;                  // (2,2)
        C.A.block(nE, nE + nV, nV, nH) = -wfac * D.Z;             // (2,3)
        C.A.block(nE + nV, nE, nH, nV) = -wfac * D.Z.transpose(); // (3,2)
        C.A.block(nE + nV, nE + nV, nH, nH) = -wfac * D.Y;        // (3,3)
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

SchurResult solve_schur(const FemProblem& F, const BoundaryOps& O, const DerivedOps& D, const Vec& f, const SchurOptions& opt)
{
    using Clock = std::chrono::steady_clock;
    auto secs = [](Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); };
    const TraceSurface& S = F.surface;
    const int nE = F.num_edges, nV = S.num_nodes(), nHall = S.nH, Es = S.num_edges();
    std::vector<int> listC;
    for (int k = 0; k < nHall; ++k) if (opt.drop_c.empty() || !opt.drop_c[k]) listC.push_back(k);
    const int nH = static_cast<int>(listC.size());
    const std::vector<char>& gauged = opt.gauged.empty() ? F.gauged : opt.gauged;
    const double mu0 = F.mu0;

    std::vector<int> surf_of_fem(nE, -1);
    for (int e = 0; e < Es; ++e) surf_of_fem[F.surf_edge_to_fem[e]] = e;
    // Indexmengen
    std::vector<int> posI(nE, -1), posGa(nE, -1);
    std::vector<int> listI, listGa;
    for (int e = 0; e < nE; ++e) {
        if (gauged[e]) continue;
        if (surf_of_fem[e] >= 0) { posGa[e] = static_cast<int>(listGa.size()); listGa.push_back(e); }
        else { posI[e] = static_cast<int>(listI.size()); listI.push_back(e); }
    }
    std::vector<int> listPsi;
    {
        std::vector<char> seen(S.num_components, 0);
        for (int v = 0; v < nV; ++v) {
            if (opt.pin_psi && !seen[S.node_comp[v]]) { seen[S.node_comp[v]] = 1; continue; }
            listPsi.push_back(v);
        }
    }
    const int nI = static_cast<int>(listI.size()), nGa = static_cast<int>(listGa.size()), nP = static_cast<int>(listPsi.size()), nG = nGa + nP + nH;

    SchurResult R;
    R.nI = nI; R.nGa = nGa; R.nPsi = nP; R.nC = nH;
    R.schur_gb = 8.0 * static_cast<double>(nG) * nG / 1e9;

    // ---- duenne Bloecke aus K (mal mu0) ----
    using SpMat = Eigen::SparseMatrix<double>;
    std::vector<Eigen::Triplet<double>> tII, tIG, tGG;
    for (const Triplet& t : F.K) {
        if (gauged[t.row] || gauged[t.col]) continue;
        const double v = mu0 * t.val;
        const int ri = posI[t.row], ci = posI[t.col], rg = posGa[t.row], cg = posGa[t.col];
        if (ri >= 0 && ci >= 0) tII.emplace_back(ri, ci, v);
        else if (ri >= 0 && cg >= 0) tIG.emplace_back(ri, cg, v);
        else if (rg >= 0 && cg >= 0) tGG.emplace_back(rg, cg, v);
    }
    SpMat AII(nI, nI), AIG(nI, nGa);
    AII.setFromTriplets(tII.begin(), tII.end());
    AIG.setFromTriplets(tIG.begin(), tIG.end());
    Vec bI(nI), bG = Vec::Zero(nG);
    for (int i = 0; i < nI; ++i) bI[i] = mu0 * f[listI[i]];
    for (int i = 0; i < nGa; ++i) bG[i] = mu0 * f[listGa[i]];

    // ---- dichter Block A_GG ----
    Mat AGG = Mat::Zero(nG, nG);
    for (const auto& t : tGG) AGG(t.row(), t.col()) += t.value();
    {   // R^T V0 R, R^T N, Q auf den Flaechenkanten der erhaltenen Randfreiheitsgrade
        std::vector<int> se(nGa);
        for (int i = 0; i < nGa; ++i) se[i] = surf_of_fem[listGa[i]];
        // V0 R (F x Es) nur fuer die benoetigten Spalten
        Mat R_sel(D.R.rows(), nGa);
        for (int i = 0; i < nGa; ++i) R_sel.col(i) = D.R.col(se[i]);
        AGG.topLeftCorner(nGa, nGa) += R_sel.transpose() * (D.V0 * R_sel);
        const Mat RtN = R_sel.transpose() * D.N;                 // nGa x nV
        const double fl = opt.flip_lambda ? -1.0 : 1.0;   // Gegenprobe: Vorzeichen der Kopplung in der FEM-Zeile
        for (int i = 0; i < nGa; ++i) for (int j = 0; j < nP; ++j) { AGG(i, nGa + j) = fl * RtN(i, listPsi[j]); AGG(nGa + j, i) = RtN(i, listPsi[j]); }
        for (int i = 0; i < nGa; ++i) for (int k = 0; k < nH; ++k) { AGG(i, nGa + nP + k) = fl * D.Q(se[i], listC[k]); AGG(nGa + nP + k, i) = D.Q(se[i], listC[k]); }
    }
    for (int i = 0; i < nP; ++i) for (int j = 0; j < nP; ++j) AGG(nGa + i, nGa + j) = -D.W(listPsi[i], listPsi[j]);
    for (int i = 0; i < nP; ++i) for (int k = 0; k < nH; ++k) { AGG(nGa + i, nGa + nP + k) = -D.Z(listPsi[i], listC[k]); AGG(nGa + nP + k, nGa + i) = -D.Z(listPsi[i], listC[k]); }
    for (int k = 0; k < nH; ++k) for (int l = 0; l < nH; ++l) AGG(nGa + nP + k, nGa + nP + l) = -D.Y(listC[k], listC[l]);

    // ---- Elimination der inneren Kanten ----
    const auto t0 = Clock::now();
    Eigen::SimplicialLDLT<SpMat> ldlt;
    Eigen::SparseLU<SpMat> splu;
    ldlt.compute(AII);
    R.used_ldlt = (ldlt.info() == Eigen::Success);
    if (!R.used_ldlt) splu.compute(AII);
    auto solveI = [&](const Mat& rhs) -> Mat { return R.used_ldlt ? Mat(ldlt.solve(rhs)) : Mat(splu.solve(rhs)); };
    const Mat Xdense = solveI(Mat(AIG));                         // A_II^{-1} A_IG (nI x nGa)
    const Vec yI = solveI(bI);
    const auto t1 = Clock::now();
    R.t_eliminate = secs(t0, t1);

    // ---- Schurkomplement S = A_GG - A_GI X auf allen Randfreiheitsgraden; A_GI = (A_IG, 0)^T ----
    Mat Sc = AGG;
    {
        const Mat AGI_X = Mat(AIG.transpose()) * Xdense;          // nGa x nGa
        Sc.topLeftCorner(nGa, nGa) -= AGI_X;
    }
    Vec rhsG = bG;
    rhsG.head(nGa) -= AIG.transpose() * yI;
    const auto t2 = Clock::now();
    R.t_schur = secs(t1, t2);
    const Vec xG = Sc.partialPivLu().solve(rhsG);
    const Vec xI = yI - Xdense * xG.head(nGa);
    R.t_solve = secs(t2, Clock::now());

    // ---- Rueckwaertsfehler im (nicht eliminierten) skalierten System ----
    {
        Vec rI = AII * xI + AIG * xG.head(nGa) - bI;
        Vec rG = AGG * xG - bG;
        rG.head(nGa) += AIG.transpose() * xI;
        const double bn = std::sqrt(bI.squaredNorm() + bG.squaredNorm());
        R.residual_rel = std::sqrt(rI.squaredNorm() + rG.squaredNorm()) / bn;
    }

    R.a = Vec::Zero(nE); R.psi = Vec::Zero(nV); R.c = Vec::Zero(nHall);
    for (int k = 0; k < nH; ++k) R.c[listC[k]] = xG[nGa + nP + k] / mu0;
    for (int i = 0; i < nI; ++i) R.a[listI[i]] = xI[i];
    for (int i = 0; i < nGa; ++i) R.a[listGa[i]] = xG[i];
    for (int j = 0; j < nP; ++j) R.psi[listPsi[j]] = xG[nGa + j] / mu0;
    return R;
}

} // namespace trace
