// Oberflaechenstrukturen der Maxwell-Spur-Kopplung, siehe include/TraceSurface.h
#include "TraceSurface.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace trace {

TraceSurface build_trace_surface(const std::vector<Vec3>& X, const std::vector<std::array<int, 4>>& quads)
{
    TraceSurface S;
    S.X = X;
    S.quads = quads;
    const int F = static_cast<int>(quads.size());
    const int V = static_cast<int>(X.size());
    S.quad_edges.assign(F, {-1, -1, -1, -1});
    S.quad_sign.assign(F, {0, 0, 0, 0});

    std::map<std::pair<int, int>, int> emap;
    std::vector<int> ecount;
    std::vector<int> esum;   // Summe der Vorzeichen (0 bei konsistenter Orientierung)
    for (int q = 0; q < F; ++q) {
        for (int i = 0; i < 4; ++i) {
            const int a = quads[q][i], b = quads[q][(i + 1) % 4];
            const std::pair<int, int> key{std::min(a, b), std::max(a, b)};
            auto it = emap.find(key);
            int id;
            if (it == emap.end()) {
                id = static_cast<int>(S.edges.size());
                emap[key] = id;
                S.edges.push_back({key.first, key.second});
                ecount.push_back(0);
                esum.push_back(0);
            } else id = it->second;
            const int s = (a < b) ? +1 : -1;
            S.quad_edges[q][i] = id;
            S.quad_sign[q][i] = s;
            ++ecount[id];
            esum[id] += s;
        }
    }
    const int E = static_cast<int>(S.edges.size());

    std::ostringstream diag;
    int bad_count = 0, bad_orient = 0;
    for (int e = 0; e < E; ++e) {
        if (ecount[e] != 2) ++bad_count;
        else if (esum[e] != 0) ++bad_orient;
    }
    S.manifold_oriented = (bad_count == 0 && bad_orient == 0);
    if (!S.manifold_oriented)
        diag << bad_count << " Kanten mit Quadzahl != 2, " << bad_orient << " Kanten mit gleichsinnig durchlaufenden Quads";
    S.diagnostics = diag.str();

    // Zusammenhang ueber Kanten (Union-Find ueber Quads)
    std::vector<int> parent(F);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    std::vector<int> first_quad_of_edge(E, -1);
    for (int q = 0; q < F; ++q)
        for (int i = 0; i < 4; ++i) {
            const int e = S.quad_edges[q][i];
            if (first_quad_of_edge[e] < 0) first_quad_of_edge[e] = q;
            else parent[find(q)] = find(first_quad_of_edge[e]);
        }
    std::map<int, int> root_to_comp;
    S.quad_comp.assign(F, -1);
    for (int q = 0; q < F; ++q) {
        const int r = find(q);
        auto it = root_to_comp.find(r);
        if (it == root_to_comp.end()) it = root_to_comp.emplace(r, static_cast<int>(root_to_comp.size())).first;
        S.quad_comp[q] = it->second;
    }
    S.num_components = static_cast<int>(root_to_comp.size());
    S.node_comp.assign(V, -1);
    S.edge_comp.assign(E, -1);
    for (int q = 0; q < F; ++q)
        for (int i = 0; i < 4; ++i) {
            S.node_comp[quads[q][i]] = S.quad_comp[q];
            S.edge_comp[S.quad_edges[q][i]] = S.quad_comp[q];
        }
    S.comp_V.assign(S.num_components, 0);
    S.comp_E.assign(S.num_components, 0);
    S.comp_F.assign(S.num_components, 0);
    for (int v = 0; v < V; ++v) if (S.node_comp[v] >= 0) ++S.comp_V[S.node_comp[v]];
    for (int e = 0; e < E; ++e) ++S.comp_E[S.edge_comp[e]];
    for (int q = 0; q < F; ++q) ++S.comp_F[S.quad_comp[q]];
    S.comp_genus.assign(S.num_components, 0);
    for (int c = 0; c < S.num_components; ++c) {
        const int chi = S.comp_V[c] - S.comp_E[c] + S.comp_F[c];
        S.comp_genus[c] = (2 - chi) / 2;
    }
    return S;
}

std::vector<Triplet> TraceSurface::G_triplets() const
{
    std::vector<Triplet> t;
    for (int e = 0; e < num_edges(); ++e) {
        t.push_back({e, edges[e][1], +1.0});
        t.push_back({e, edges[e][0], -1.0});
    }
    return t;
}

std::vector<Triplet> TraceSurface::R_triplets() const
{
    std::vector<Triplet> t;
    for (int q = 0; q < num_quads(); ++q)
        for (int i = 0; i < 4; ++i) t.push_back({q, quad_edges[q][i], static_cast<double>(quad_sign[q][i])});
    return t;
}

std::vector<double> TraceSurface::G_dense() const
{
    std::vector<double> M(static_cast<size_t>(num_edges()) * num_nodes(), 0.0);
    for (const Triplet& t : G_triplets()) M[static_cast<size_t>(t.row) * num_nodes() + t.col] += t.val;
    return M;
}

std::vector<double> TraceSurface::R_dense() const
{
    std::vector<double> M(static_cast<size_t>(num_quads()) * num_edges(), 0.0);
    for (const Triplet& t : R_triplets()) M[static_cast<size_t>(t.row) * num_edges() + t.col] += t.val;
    return M;
}

using SpMat = Eigen::SparseMatrix<double>;

static SpMat to_sparse(const std::vector<Triplet>& t, int rows, int cols)
{
    std::vector<Eigen::Triplet<double>> et;
    et.reserve(t.size());
    for (const Triplet& x : t) et.emplace_back(x.row, x.col, x.val);
    SpMat M(rows, cols);
    M.setFromTriplets(et.begin(), et.end());
    return M;
}

// Loest A x = b fuer symmetrisch positiv semidefinites A mit konsistenter rechter Seite (CG).
static Eigen::VectorXd cg_solve(const SpMat& A, const Eigen::VectorXd& b)
{
    Eigen::ConjugateGradient<SpMat, Eigen::Lower | Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.setTolerance(1e-14);
    cg.setMaxIterations(20 * static_cast<int>(A.rows()) + 1000);
    cg.compute(A);
    return cg.solve(b);
}

void compute_harmonic_fields(TraceSurface& S, unsigned seed)
{
    if (!S.manifold_oriented) throw std::runtime_error("compute_harmonic_fields: Flaeche nicht orientiert/geschlossen: " + S.diagnostics);
    const int E = S.num_edges(), V = S.num_nodes(), F = S.num_quads();
    SpMat G = to_sparse(S.G_triplets(), E, V);
    SpMat R = to_sparse(S.R_triplets(), F, E);
    SpMat RRt = R * R.transpose();
    SpMat GtG = G.transpose() * G;

    // Projektor auf (ker R) ∩ (range G)^perp:  c -> c - R^T y - G phi,  mit R R^T y = R c  und  G^T G phi = G^T (c - R^T y).
    // Reihenfolge: range(R^T) und range(G) sind orthogonal (R G = 0), die beiden Projektionen vertauschen.
    auto project = [&](Eigen::VectorXd c) {
        Eigen::VectorXd y = cg_solve(RRt, R * c);
        Eigen::VectorXd c1 = c - R.transpose() * y;
        Eigen::VectorXd phi = cg_solve(GtG, G.transpose() * c1);
        return Eigen::VectorXd(c1 - G * phi);
    };

    const int expected = 2 * S.total_genus();
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Eigen::VectorXd> basis;
    std::vector<int> bcomp;
    for (int comp = 0; comp < S.num_components; ++comp) {
        const int want = 2 * S.comp_genus[comp];
        std::vector<int> eidx;
        for (int e = 0; e < E; ++e) if (S.edge_comp[e] == comp) eidx.push_back(e);
        int found = 0, tries = 0;
        while (found < want && tries < 20 * want + 20) {
            ++tries;
            Eigen::VectorXd c = Eigen::VectorXd::Zero(E);
            for (int e : eidx) c[e] = nd(rng);
            Eigen::VectorXd h = project(c);
            // mit bereits gefundenen Feldern dieser Komponente orthogonalisieren (zweimal)
            for (int pass = 0; pass < 2; ++pass)
                for (size_t b = 0; b < basis.size(); ++b)
                    if (bcomp[b] == comp) h -= basis[b].dot(h) * basis[b];
            const double nrm = h.norm();
            if (nrm < 1e-6 * std::sqrt(static_cast<double>(eidx.size()))) continue;   // kein neuer Vektor
            basis.push_back(h / nrm);
            bcomp.push_back(comp);
            ++found;
        }
    }
    if (static_cast<int>(basis.size()) != expected) {
        std::ostringstream m;
        m << "compute_harmonic_fields: " << basis.size() << " statt " << expected << " harmonische Felder gefunden";
        throw std::runtime_error(m.str());
    }
    S.nH = expected;
    S.H.assign(static_cast<size_t>(E) * expected, 0.0);
    S.H_comp = bcomp;
    S.harmonic_residual_R = 0.0;
    S.harmonic_residual_G = 0.0;
    for (int k = 0; k < expected; ++k) {
        for (int e = 0; e < E; ++e) S.H[static_cast<size_t>(e) * expected + k] = basis[k][e];
        S.harmonic_residual_R = std::max(S.harmonic_residual_R, (R * basis[k]).norm());
        S.harmonic_residual_G = std::max(S.harmonic_residual_G, (G.transpose() * basis[k]).norm());
    }
}

double cycle_circulation(const std::vector<std::pair<int, int>>& cycle, const double* c)
{
    double s = 0.0;
    for (const auto& p : cycle) s += p.second * c[p.first];
    return s;
}

double normalize_harmonic_to_cycles(TraceSurface& S, int comp, const std::vector<std::vector<std::pair<int, int>>>& cycles)
{
    std::vector<int> cols;
    for (int k = 0; k < S.nH; ++k) if (S.H_comp[k] == comp) cols.push_back(k);
    const int m = static_cast<int>(cols.size());
    if (static_cast<int>(cycles.size()) != m) throw std::runtime_error("normalize_harmonic_to_cycles: Zyklenzahl != 2g");
    const int E = S.num_edges();
    Eigen::MatrixXd C(m, m);
    for (int i = 0; i < m; ++i)
        for (int k = 0; k < m; ++k) {
            double s = 0.0;
            for (const auto& p : cycles[i]) s += p.second * S.H[static_cast<size_t>(p.first) * S.nH + cols[k]];
            C(i, k) = s;
        }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C);
    const double smin = svd.singularValues().minCoeff();
    Eigen::MatrixXd Cinv = C.inverse();     // H_new = H * C^{-1}  =>  circ(cycle_i, H_new_k) = delta_ik
    Eigen::MatrixXd Hold(E, m);
    for (int e = 0; e < E; ++e) for (int k = 0; k < m; ++k) Hold(e, k) = S.H[static_cast<size_t>(e) * S.nH + cols[k]];
    Eigen::MatrixXd Hnew = Hold * Cinv;
    for (int e = 0; e < E; ++e) for (int k = 0; k < m; ++k) S.H[static_cast<size_t>(e) * S.nH + cols[k]] = Hnew(e, k);
    return smin;
}

} // namespace trace
