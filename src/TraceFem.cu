// FEM-Seite der Maxwell-Spur-Kopplung, siehe include/TraceFem.h
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "FemDirectBC.h"
#include "Mesh.cuh"
#include "MaterialTypes.cuh"
#include "NedelecDispatch.cuh"
#include "NedelecHexahedron.cuh"
#include "TopologyExtractor.h"
#include "TraceFem.h"
#include "TreeCotreeGauge.h"
#include "VtkReader.h"

namespace trace {
namespace {

struct Loaded {
    Mesh mesh;
};

void load_mesh(const std::string& path, Mesh& mesh)
{
    // Meldungen des Bestands (VtkReader, TopologyExtractor) auf cout unterdruecken
    std::streambuf* old = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    try {
        VtkReader::read_hexahedral_mesh(path, mesh);
        TopologyExtractor::extract_edges(mesh);
        TopologyExtractor::extract_boundary_faces(mesh);
    } catch (...) { std::cout.rdbuf(old); throw; }
    std::cout.rdbuf(old);
}

void hex_geometry(const Mesh& mesh, int h, double3 nd[8])
{
    for (int k = 0; k < 8; ++k) nd[k] = mesh.nodes_current[mesh.hex_nodes[h].v[k]];
}

} // namespace

Vec3 FemProblem::center(int h) const
{
    Vec3 c;
    for (int k = 0; k < 8; ++k) { c.x += X[hex[h][k]].x / 8; c.y += X[hex[h][k]].y / 8; c.z += X[hex[h][k]].z / 8; }
    return c;
}

Vec3 FemProblem::curl_at_center(int h, const std::vector<double>& a) const
{
    double3 nd[8];
    for (int k = 0; k < 8; ++k) nd[k] = make_double3(X[hex[h][k]].x, X[hex[h][k]].y, X[hex[h][k]].z);
    double J[3][3], iJ[3][3], det;
    compute_jacobian(nd, 0, 0, 0, J, iJ, det);
    double3 C[12];
    evaluate_nedelec_curl(0, 0, 0, C);
    Vec3 r;
    for (int i = 0; i < 12; ++i) {
        const double ai = a[hex_edge[h][i]] * hex_sign[h][i];
        const double3 c = piola_contravariant(J, det, C[i]);
        r.x += ai * c.x; r.y += ai * c.y; r.z += ai * c.z;
    }
    return r;
}

Vec3 FemProblem::a_at_center(int h, const std::vector<double>& a) const
{
    double3 nd[8];
    for (int k = 0; k < 8; ++k) nd[k] = make_double3(X[hex[h][k]].x, X[hex[h][k]].y, X[hex[h][k]].z);
    double J[3][3], iJ[3][3], det;
    compute_jacobian(nd, 0, 0, 0, J, iJ, det);
    double3 W[12];
    evaluate_nedelec_basis(0, 0, 0, W);
    Vec3 r;
    for (int i = 0; i < 12; ++i) {
        const double ai = a[hex_edge[h][i]] * hex_sign[h][i];
        const double3 w = piola_covariant(iJ, W[i]);
        r.x += ai * w.x; r.y += ai * w.y; r.z += ai * w.z;
    }
    return r;
}

FemProblem build_fem_problem(const std::string& vtk_path, const FemMaterials& mats, double mu0,
                             const std::function<Vec3(const Vec3&, int)>& Js, double length_scale,
                             const std::vector<Vec3>* comp_shift)
{
    Mesh mesh;
    load_mesh(vtk_path, mesh);
    if (length_scale != 1.0)
        for (size_t i = 0; i < mesh.num_nodes; ++i) {
            double3 p = mesh.nodes_current[i];
            p.x *= length_scale; p.y *= length_scale; p.z *= length_scale;
            mesh.nodes_current[i] = p;
            mesh.nodes_initial[i] = p;
        }
    if (comp_shift) {
        std::vector<char> shifted(mesh.num_nodes, 0);
        for (size_t h = 0; h < mesh.num_hexes; ++h) {
            const int c = mesh.hex_component_ids[h];
            if (c >= static_cast<int>(comp_shift->size())) continue;
            const Vec3& d = (*comp_shift)[c];
            if (d.x == 0 && d.y == 0 && d.z == 0) continue;
            for (int k = 0; k < 8; ++k) {
                const int v = mesh.hex_nodes[h].v[k];
                if (shifted[v]) continue;
                shifted[v] = 1;
                double3 p = mesh.nodes_current[v];
                p.x += d.x; p.y += d.y; p.z += d.z;
                mesh.nodes_current[v] = p;
                mesh.nodes_initial[v] = p;
            }
        }
    }
    FemProblem P;
    P.mu0 = mu0;
    P.num_nodes = static_cast<int>(mesh.num_nodes);
    P.num_hexes = static_cast<int>(mesh.num_hexes);
    P.num_edges = static_cast<int>(mesh.num_edges);
    const int E = P.num_edges;
    P.X.resize(P.num_nodes);
    for (int i = 0; i < P.num_nodes; ++i) { const double3 p = mesh.nodes_current[i]; P.X[i].x = p.x; P.X[i].y = p.y; P.X[i].z = p.z; }
    P.hex.resize(P.num_hexes);
    P.hex_comp.resize(P.num_hexes);
    for (int h = 0; h < P.num_hexes; ++h) {
        for (int k = 0; k < 8; ++k) P.hex[h][k] = mesh.hex_nodes[h].v[k];
        P.hex_comp[h] = mesh.hex_component_ids[h];
    }
    P.hex_edge.resize(P.num_hexes);
    P.hex_sign.resize(P.num_hexes);
    for (int h = 0; h < P.num_hexes; ++h) for (int i = 0; i < 12; ++i) { P.hex_edge[h][i] = mesh.hex_edges[h * 12 + i]; P.hex_sign[h][i] = mesh.hex_edge_signs[h * 12 + i]; }
    P.edges.resize(E);
    for (int e = 0; e < E; ++e) { const int2 ed = mesh.edges[e]; P.edges[e] = {ed.x, ed.y}; }

    // ---- K, M_sigma, f ----
    P.f.assign(E, 0.0);
    const double g2 = 0.5773502691896257;
    const double gp2[2] = {-g2, g2};
    const double g3[3] = {-0.7745966692414834, 0.0, 0.7745966692414834}, w3[3] = {5.0 / 9, 8.0 / 9, 5.0 / 9};
    for (int h = 0; h < P.num_hexes; ++h) {
        double3 nd[8];
        hex_geometry(mesh, h, nd);
        const int comp = P.hex_comp[h];
        const double nu = 1.0 / (mu0 * mats.mu_r_of(comp));
        const double kap = mats.kappa_of(comp);
        double Ke[12][12] = {}, Me[12][12] = {}, fe[12] = {};
        for (int qa = 0; qa < 2; ++qa) for (int qb = 0; qb < 2; ++qb) for (int qc = 0; qc < 2; ++qc) {
            const double xi = gp2[qa], eta = gp2[qb], zeta = gp2[qc];
            double J[3][3], iJ[3][3], det;
            compute_jacobian(nd, xi, eta, zeta, J, iJ, det);
            double3 W[12], C[12];
            evaluate_nedelec_basis(xi, eta, zeta, W);
            evaluate_nedelec_curl(xi, eta, zeta, C);
            double3 rw[12], rc[12];
            for (int i = 0; i < 12; ++i) {
                // Produktionshelfer (NedelecDispatch.cuh): W = invJ * W_ref (zeilenweise), rot W = J^T rot W_ref / detJ
                rw[i] = piola_covariant(iJ, W[i]);
                rc[i] = piola_contravariant(J, det, C[i]);
            }
            for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
                Ke[i][j] += nu * (rc[i].x * rc[j].x + rc[i].y * rc[j].y + rc[i].z * rc[j].z) * det;
                Me[i][j] += kap * (rw[i].x * rw[j].x + rw[i].y * rw[j].y + rw[i].z * rw[j].z) * det;
            }
        }
        // Quelle: 3x3x3 Gauss
        for (int qa = 0; qa < 3; ++qa) for (int qb = 0; qb < 3; ++qb) for (int qc = 0; qc < 3; ++qc) {
            const double xi = g3[qa], eta = g3[qb], zeta = g3[qc];
            double J[3][3], iJ[3][3], det;
            compute_jacobian(nd, xi, eta, zeta, J, iJ, det);
            double3 W[12];
            evaluate_nedelec_basis(xi, eta, zeta, W);
            // Position
            double N[8];
            static const double sx[8] = {-1, 1, 1, -1, -1, 1, 1, -1}, sy[8] = {-1, -1, 1, 1, -1, -1, 1, 1}, sz[8] = {-1, -1, -1, -1, 1, 1, 1, 1};
            Vec3 x;
            for (int k = 0; k < 8; ++k) {
                N[k] = 0.125 * (1 + sx[k] * xi) * (1 + sy[k] * eta) * (1 + sz[k] * zeta);
                x.x += N[k] * nd[k].x; x.y += N[k] * nd[k].y; x.z += N[k] * nd[k].z;
            }
            const Vec3 js = Js(x, comp);
            const double wq = w3[qa] * w3[qb] * w3[qc] * det;
            for (int i = 0; i < 12; ++i) {
                const double3 rw = piola_covariant(iJ, W[i]);
                fe[i] += wq * (js.x * rw.x + js.y * rw.y + js.z * rw.z);
            }
        }
        for (int i = 0; i < 12; ++i) {
            const int gi = mesh.hex_edges[h * 12 + i], si = mesh.hex_edge_signs[h * 12 + i];
            P.f[gi] += si * fe[i];
            for (int j = 0; j < 12; ++j) {
                const int gj = mesh.hex_edges[h * 12 + j], sj = mesh.hex_edge_signs[h * 12 + j];
                P.K.push_back({gi, gj, si * sj * Ke[i][j]});
                if (Me[i][j] != 0.0) P.Msig.push_back({gi, gj, si * sj * Me[i][j]});
            }
        }
    }

    // ---- Eichung ----
    std::vector<MaterialProperties> mp(P.num_hexes);
    for (int h = 0; h < P.num_hexes; ++h) { mp[h].kappa = mats.kappa_of(P.hex_comp[h]); mp[h].mu_relative = mats.mu_r_of(P.hex_comp[h]); }
    const GaugeInfo gi = build_tree_cotree_gauge(mesh, mp.data(), 0.0);
    P.gauged = gi.is_gauged;
    P.num_gauged = gi.num_gauged;

    // ---- Randflaeche: nur genutzte Knoten, monoton ----
    std::vector<char> used(P.num_nodes, 0);
    std::vector<std::array<int, 4>> quads;
    for (size_t q = 0; q < mesh.num_boundary_faces; ++q) {
        const auto& f = mesh.boundary_faces[q];
        quads.push_back({f.v[0], f.v[1], f.v[2], f.v[3]});
        for (int k = 0; k < 4; ++k) used[f.v[k]] = 1;
    }
    std::vector<int> remap(P.num_nodes, -1);
    std::vector<Vec3> Xs;
    for (int v = 0; v < P.num_nodes; ++v) if (used[v]) { remap[v] = static_cast<int>(Xs.size()); Xs.push_back(P.X[v]); P.surf_node_to_fem.push_back(v); }
    for (auto& q : quads) for (int& v : q) v = remap[v];
    P.surface = build_trace_surface(Xs, quads);
    // Flaechenkante -> FEM-Kante
    P.surf_edge_to_fem.assign(P.surface.num_edges(), -1);
    for (size_t q = 0; q < mesh.num_boundary_faces; ++q)
        for (int i = 0; i < 4; ++i) {
            const int e = P.surface.quad_edges[q][i];
            const int fe = mesh.boundary_edges_to_fem_edges[4 * q + i];
            if (P.surf_edge_to_fem[e] >= 0 && P.surf_edge_to_fem[e] != fe) throw std::runtime_error("build_fem_problem: Flaechenkante auf zwei FEM-Kanten abgebildet");
            P.surf_edge_to_fem[e] = fe;
            if (P.surface.quad_sign[q][i] != mesh.boundary_edge_signs[4 * q + i]) throw std::runtime_error("build_fem_problem: Kantenvorzeichen weichen vom Bestand ab");
        }
    return P;
}

std::vector<char> make_tree_gauge(const FemProblem& F, unsigned seed)
{
    std::vector<int> order(F.num_edges);
    for (int e = 0; e < F.num_edges; ++e) order[e] = e;
    if (seed != 0) {
        unsigned st = seed;
        for (int i = F.num_edges - 1; i > 0; --i) { st = st * 1664525u + 1013904223u; std::swap(order[i], order[st % static_cast<unsigned>(i + 1)]); }
    }
    std::vector<int> parent(F.num_nodes);
    for (int v = 0; v < F.num_nodes; ++v) parent[v] = v;
    auto find = [&](int a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    std::vector<char> tree(F.num_edges, 0);
    for (int e : order) {
        const int a = find(F.edges[e][0]), b = find(F.edges[e][1]);
        if (a != b) { parent[a] = b; tree[e] = 1; }
    }
    return tree;
}

ProjectionResult project_source(const std::string& vtk_path, std::vector<double>& f, const std::vector<char>* blocked)
{
    Mesh mesh;
    load_mesh(vtk_path, mesh);
    const fem_direct::GradientProjectionReport r = fem_direct::project_out_gradient(mesh, f.data(), f.size(), blocked ? blocked->data() : nullptr, 1e-14, 20000);
    ProjectionResult res;
    res.rel_before = r.rel_before;
    res.rel_after = r.rel_after;
    return res;
}

CutLayers build_cut_layers(const FemProblem& F, int comp)
{
    CutLayers L;
    const double PI = 3.14159265358979323846;
    std::vector<char> in_node(F.num_nodes, 0);
    for (int h = 0; h < F.num_hexes; ++h) if (F.hex_comp[h] == comp) for (int k = 0; k < 8; ++k) in_node[F.hex[h][k]] = 1;
    std::vector<double> theta(F.num_nodes, 0.0);
    std::vector<double> levels;
    for (int v = 0; v < F.num_nodes; ++v) if (in_node[v]) { theta[v] = std::atan2(F.X[v].y, F.X[v].x); levels.push_back(theta[v]); }
    std::sort(levels.begin(), levels.end());
    std::vector<double> uniq;
    for (double t : levels) if (uniq.empty() || t - uniq.back() > 1e-7) uniq.push_back(t);
    L.K = static_cast<int>(uniq.size());
    auto slice = [&](int v) { // Index der Halbebene
        const double t = theta[v];
        int lo = 0, hi = L.K - 1;
        while (lo < hi) { const int mid = (lo + hi) / 2; if (uniq[mid] < t - 1e-7) lo = mid + 1; else hi = mid; }
        return lo;
    };
    L.layers.assign(L.K, {});
    L.in_comp_edge.assign(F.num_edges, 0);
    for (int e = 0; e < F.num_edges; ++e) {
        const int a = F.edges[e][0], b = F.edges[e][1];
        if (!in_node[a] || !in_node[b]) continue;
        L.in_comp_edge[e] = 1;
        const int sa = slice(a), sb = slice(b);
        if (sa == sb) continue;
        double d = theta[b] - theta[a];                       // Richtung vmin -> vmax
        while (d > PI) d -= 2 * PI;
        while (d <= -PI) d += 2 * PI;
        const int k = (d > 0) ? sa : sb;                      // Schicht zwischen Halbebene k und k+1 (zyklisch)
        if (((k + 1) % L.K) != ((d > 0) ? sb : sa)) throw std::runtime_error("build_cut_layers: Kante ueberspringt eine Halbebene");
        L.layers[k].push_back({e, d > 0 ? 1.0 : -1.0});
    }
    return L;
}

std::vector<double> prepare_source(const FemProblem& F, const std::string& vtk_path, const std::vector<CoilSource>& coils, bool project,
                                   bool renormalize, std::vector<SourceDiagnostics>* diag)
{
    std::vector<double> out(F.num_edges, 0.0);
    if (diag) diag->clear();
    for (const CoilSource& cs : coils) {
        const CutLayers L = build_cut_layers(F, cs.comp);
        std::vector<double> fc(F.num_edges, 0.0);
        for (int e = 0; e < F.num_edges; ++e) if (L.in_comp_edge[e]) fc[e] = F.f[e];
        auto gtn = [&](const std::vector<double>& f) {
            std::vector<double> g(F.num_nodes, 0.0);
            double fn = 0;
            for (int e = 0; e < F.num_edges; ++e) if (L.in_comp_edge[e]) { g[F.edges[e][1]] += f[e]; g[F.edges[e][0]] -= f[e]; fn += f[e] * f[e]; }
            double sn = 0;
            for (double v : g) sn += v * v;
            return fn > 0 ? std::sqrt(sn / fn) : 0.0;
        };
        SourceDiagnostics d;
        d.comp = cs.comp; d.K = L.K;
        d.rel_gt_before = gtn(fc);
        d.net_min_before = 1e300; d.net_max_before = -1e300;
        for (int k = 0; k < L.K; ++k) { const double n = L.net(fc, k); d.net_min_before = std::min(d.net_min_before, n); d.net_max_before = std::max(d.net_max_before, n); }
        if (project) {
            std::vector<char> blocked(F.num_edges, 1);
            for (int e = 0; e < F.num_edges; ++e) if (L.in_comp_edge[e]) blocked[e] = 0;
            project_source(vtk_path, fc, &blocked);
        }
        d.rel_gt_after = gtn(fc);
        auto stats = [&](double* mean, double* spread) {
            double mn = 1e300, mx = -1e300, sum = 0;
            for (int k = 0; k < L.K; ++k) { const double n = L.net(fc, k); mn = std::min(mn, n); mx = std::max(mx, n); sum += n; }
            *mean = sum / L.K;
            *spread = (mx - mn) / std::fabs(*mean);
        };
        stats(&d.net_proj, &d.net_proj_spread);
        if (renormalize) {
            d.scale = cs.target / d.net_proj;
            for (double& v : fc) v *= d.scale;
        }
        stats(&d.net_final, &d.net_final_spread);
        for (int e = 0; e < F.num_edges; ++e) out[e] += fc[e];
        if (diag) diag->push_back(d);
    }
    return out;
}

} // namespace trace
