// FEM-Seite der Maxwell-Spur-Kopplung, siehe include/TraceFem.h
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
    // lokale Kantenkoeffizienten: a[globale Kante] * Vorzeichen; Vorzeichen aus Knotenreihenfolge (v_min -> v_max)
    static const int le[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    // Kanten-Index suchen
    for (int i = 0; i < 12; ++i) {
        const int n0 = hex[h][le[i][0]], n1 = hex[h][le[i][1]];
        int ge = -1;
        for (int e = 0; e < num_edges; ++e) if (edges[e][0] == std::min(n0, n1) && edges[e][1] == std::max(n0, n1)) { ge = e; break; }
        const double s = (n0 < n1) ? 1.0 : -1.0;
        const double ai = a[ge] * s;
        // physikalische Rotation: J curl_ref / det
        const double3 c = piola_contravariant(J, det, C[i]);
        r.x += ai * c.x; r.y += ai * c.y; r.z += ai * c.z;
    }
    return r;
}

FemProblem build_fem_problem(const std::string& vtk_path, const FemMaterials& mats, double mu0,
                             const std::function<Vec3(const Vec3&, int)>& Js, double length_scale)
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
    P.edges.resize(E);
    for (int e = 0; e < E; ++e) { const int2 ed = mesh.edges[e]; P.edges[e] = {ed.x, ed.y}; }

    // ---- K, M_sigma, f ----
    P.K.assign(static_cast<size_t>(E) * E, 0.0);
    P.Msig.assign(static_cast<size_t>(E) * E, 0.0);
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
                P.K[static_cast<size_t>(gi) * E + gj] += si * sj * Ke[i][j];
                P.Msig[static_cast<size_t>(gi) * E + gj] += si * sj * Me[i][j];
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

} // namespace trace
