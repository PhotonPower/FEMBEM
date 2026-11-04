// =============================================================================
// Host-Test: T-Kopplung gegen eine dichte Host-Referenz (Korrektur der
// BEM-Kopplung, docs/design/bem-coupling-fixes.md)
//
// Auf einem echten Netz (data/team28/team28.vtk) mit perm != id: der
// T-Kernelrumpf, mit den Quads in BEM-Reihenfolge gespeist, muss die Matrix
// reproduzieren, die unabhaengig je Position pos aus der Geometrie des Quads
// perm[pos] aufgebaut ist. Gegenprobe: mit mesh.boundary_faces in Netzreihenfolge
// (die alte Indexierung) weicht das Ergebnis stark ab.
//
// Reiner Host-Code; braucht einen CUDA-Kontext nur fuer DeviceArray.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "BemCoupling.h"
#include "BemIntegration.cuh"
#include "ClusterTree.h"
#include "CouplingKernels.cuh"
#include "Mesh.cuh"
#include "TopologyExtractor.h"
#include "VtkReader.h"

static int g_fail = 0;
// greater = false: Soll v < lim; greater = true (Gegenprobe): Soll v > lim
static void check(bool ok, const char* what, double v, double lim, bool greater = false) {
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, v,
                greater ? (ok ? ">" : "<=") : (ok ? "<" : ">="), lim);
    if (!ok) ++g_fail;
}

static double rel(const std::vector<double>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (a[i] - b[i]); d += b[i] * b[i]; }
    return std::sqrt(n / (d > 0 ? d : 1.0));
}

// Referenz: je Position pos Geometrie des Quads perm[pos], explizit aus mesh.boundary_faces.
//   Zeile 1: yA[g_r] -= sum_c s_r s_c T[r][c] k[pos*4+c]
//   Zeile 2: yK[pos*4+r] -= 0.5 sum_c s_r s_c T[r][c] a[g_c]
static void reference(const Mesh& mesh, const BemCouplingMap& map, const std::vector<double>& nodes_xyz,
                      const std::vector<double>& a, const std::vector<double>& k,
                      std::vector<double>& yA, std::vector<double>& yK) {
    yA.assign(mesh.num_edges, 0.0);
    yK.assign(static_cast<size_t>(map.num_bem_dofs), 0.0);
    for (size_t pos = 0; pos < map.perm.size(); ++pos) {
        const Mesh::QuadFace& f = mesh.boundary_faces[static_cast<size_t>(map.perm[pos])];
        double3 nd[4];
        for (int i = 0; i < 4; ++i)
            nd[i] = make_double3(nodes_xyz[3 * f.v[i]], nodes_xyz[3 * f.v[i] + 1], nodes_xyz[3 * f.v[i] + 2]);
        double T[4][4];
        integrate_boundary_mass_matrix(nd, T);
        for (int r = 0; r < 4; ++r) {
            const int gr = map.bem_to_fem[pos * 4 + r];
            for (int c = 0; c < 4; ++c) {
                const int gc = map.bem_to_fem[pos * 4 + c];
                const double Trc = static_cast<double>(map.bem_sign[pos * 4 + r] * map.bem_sign[pos * 4 + c]) * T[r][c];
                yA[static_cast<size_t>(gr)] -= Trc * k[pos * 4 + c];
                yK[pos * 4 + r] -= 0.5 * Trc * a[static_cast<size_t>(gc)];
            }
        }
    }
}

static void run_kernel(const std::vector<Mesh::QuadFace>& faces, const std::vector<double3>& nodes,
                       const BemCouplingMap& map, const std::vector<double>& a, const std::vector<double>& k,
                       std::vector<double>& yA, std::vector<double>& yK) {
    yA.assign(a.size(), 0.0);
    yK.assign(static_cast<size_t>(map.num_bem_dofs), 0.0);
    for (size_t pos = 0; pos < map.perm.size(); ++pos)
        apply_T_matrix_kernel_body(static_cast<int>(pos), static_cast<int>(map.perm.size()), faces.data(),
                                   nodes.data(), map.bem_to_fem.data(), map.bem_sign.data(), a.data(),
                                   k.data(), yA.data(), yK.data());
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "team28.vtk";
    std::printf("=== test_bem_coupling_geometry (%s) ===\n", path);
    Mesh mesh;
    VtkReader::read_hexahedral_mesh(path, mesh);
    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);
    ClusterTree tree(30);
    tree.build_tree(mesh);
    const BemCouplingMap map = build_bem_coupling_map(mesh, tree);

    mesh.nodes_current.prefetch_to_host();
    mesh.boundary_faces.prefetch_to_host();
    std::vector<double3> nodes(mesh.num_nodes);
    std::vector<double> xyz(3 * mesh.num_nodes);
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        nodes[i] = mesh.nodes_current[i];
        xyz[3 * i] = nodes[i].x; xyz[3 * i + 1] = nodes[i].y; xyz[3 * i + 2] = nodes[i].z;
    }
    std::vector<Mesh::QuadFace> faces_mesh(mesh.num_boundary_faces);
    for (size_t i = 0; i < faces_mesh.size(); ++i) faces_mesh[i] = mesh.boundary_faces[i];
    const std::vector<Mesh::QuadFace> faces_bem = bem_ordered_boundary_faces(mesh, map);

    size_t moved = 0;
    for (size_t pos = 0; pos < map.perm.size(); ++pos) if (map.perm[pos] != static_cast<int>(pos)) ++moved;
    std::printf("  %zu Randquads, %zu Kanten, perm[pos] != pos fuer %zu\n", faces_mesh.size(),
                static_cast<size_t>(mesh.num_edges), moved);
    if (moved == 0) { std::printf("  [FEHLER] perm ist die Identitaet: das Netz taugt nicht fuer diesen Test\n"); return 1; }

    bool ordered_ok = true;
    for (size_t pos = 0; pos < faces_bem.size(); ++pos) {
        const Mesh::QuadFace& f = faces_mesh[static_cast<size_t>(map.perm[pos])];
        for (int i = 0; i < 4; ++i) if (faces_bem[pos].v[i] != f.v[i]) ordered_ok = false;
    }
    check(ordered_ok, "bem_ordered_boundary_faces[pos] = boundary_faces[perm[pos]]", ordered_ok ? 0.0 : 1.0, 0.5);

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> a(mesh.num_edges), k(static_cast<size_t>(map.num_bem_dofs));
    for (double& x : a) x = U(rng);
    for (double& x : k) x = U(rng);

    std::vector<double> rA, rK, yA, yK, oA, oK;
    reference(mesh, map, xyz, a, k, rA, rK);
    run_kernel(faces_bem, nodes, map, a, k, yA, yK);
    run_kernel(faces_mesh, nodes, map, a, k, oA, oK);
    check(rel(yA, rA) < 1e-13, "Zeile 1: Kernel mit Quads in BEM-Reihenfolge = Host-Referenz", rel(yA, rA), 1e-13);
    check(rel(yK, rK) < 1e-13, "Zeile 2: Kernel mit Quads in BEM-Reihenfolge = Host-Referenz", rel(yK, rK), 1e-13);
    check(rel(oA, rA) > 0.1, "Gegenprobe: alte Indexierung (Netzreihenfolge), Zeile 1, wird erkannt", rel(oA, rA), 0.1, true);
    check(rel(oK, rK) > 0.1, "Gegenprobe: alte Indexierung (Netzreihenfolge), Zeile 2, wird erkannt", rel(oK, rK), 0.1, true);

    std::printf("\n%s (%d Fehler)\n", g_fail ? "FEHLER" : "ALLE PRUEFUNGEN BESTANDEN", g_fail);
    return g_fail ? 1 : 0;
}
