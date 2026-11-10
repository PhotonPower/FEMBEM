// =============================================================================
// T9 am TEAM-28-Rand (Phase 1b, Schritt 1): Randflaeche von data/team28/team28_hex.vtk
// (Aussenspule, Innenspule, Platte). Erwartet: drei Zusammenhangskomponenten, Geschlechter g = 1, 1, 0.
// Host-Test; die Mesh-Klassen brauchen einen CUDA-Kontext (Unified Memory), die Rechnung nicht.
// Aufruf: test_trace_surface_team28 <team28_hex.vtk>
// =============================================================================
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>

#include "Mesh.cuh"
#include "TopologyExtractor.h"
#include "TraceSurface.h"
#include "VtkReader.h"

using namespace trace;

static int g_fail = 0;
static void check_eq(bool ok, const char* what, long a, long b)
{
    std::printf("  [%s] %s  (%ld %s %ld)\n", ok ? "OK " : "FEHLER", what, a, ok ? "==" : "!=", b);
    if (!ok) ++g_fail;
}

int main(int argc, char** argv)
{
    const std::string path = argc > 1 ? argv[1] : "team28_hex.vtk";
    Mesh mesh;
    VtkReader::read_hexahedral_mesh(path, mesh);
    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);
    std::printf("Netz %s: %zu Hexaeder, %zu Randquads\n", path.c_str(), mesh.num_hexes, mesh.num_boundary_faces);

    std::vector<Vec3> X(mesh.num_nodes);
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        const double3 p = mesh.nodes_initial[i];
        X[i].x = p.x; X[i].y = p.y; X[i].z = p.z;
    }
    std::vector<std::array<int, 4>> quads;
    for (size_t f = 0; f < mesh.num_boundary_faces; ++f) {
        const auto& q = mesh.boundary_faces[f];
        quads.push_back({q.v[0], q.v[1], q.v[2], q.v[3]});
    }
    // ungenutzte Knoten (Innenknoten der Hexaeder) entfernen; die Numerierung bleibt MONOTON, damit
    // die Richtung vmin -> vmax (und damit mesh.boundary_edge_signs) erhalten bleibt
    std::vector<int> remap(X.size(), -1);
    std::vector<Vec3> Xs;
    {
        std::vector<char> used(X.size(), 0);
        for (auto& q : quads) for (int v : q) used[v] = 1;
        for (size_t v = 0; v < X.size(); ++v) if (used[v]) { remap[v] = static_cast<int>(Xs.size()); Xs.push_back(X[v]); }
        for (auto& q : quads) for (int& v : q) v = remap[v];
    }

    TraceSurface S = build_trace_surface(Xs, quads);
    check_eq(S.manifold_oriented, "geschlossen und konsistent orientiert", S.manifold_oriented, 1);
    check_eq(S.num_components == 3, "3 Zusammenhangskomponenten (Aussenspule, Innenspule, Platte)", S.num_components, 3);
    int n0 = 0, n1 = 0;
    for (int g : S.comp_genus) { if (g == 0) ++n0; else if (g == 1) ++n1; }
    check_eq(n1 == 2 && n0 == 1, "Geschlechter: zweimal g = 1, einmal g = 0", n1 * 10 + n0, 21);
    for (int c = 0; c < S.num_components; ++c)
        std::printf("    Komponente %d: V=%d E=%d F=%d, Euler=%d, g=%d\n", c, S.comp_V[c], S.comp_E[c], S.comp_F[c], S.comp_V[c] - S.comp_E[c] + S.comp_F[c], S.comp_genus[c]);

    const int V = S.num_nodes(), E = S.num_edges(), F = S.num_quads();
    // R G = 0 exakt: ganzzahliges psi, Zirkulation von G psi um jedes Quad (Ganzzahlarithmetik, also exakt)
    long worst = 0;
    for (int q = 0; q < F; ++q) {
        long circ = 0;
        for (int i = 0; i < 4; ++i) {
            const int e = S.quad_edges[q][i];
            circ += S.quad_sign[q][i] * (static_cast<long>((S.edges[e][1] * 7919L) % 1013) - static_cast<long>((S.edges[e][0] * 7919L) % 1013));
        }
        worst = std::max(worst, std::labs(circ));
    }
    (void)V;
    // Vorzeichen und Kantenzuordnung gegen den Bestand (mesh.boundary_edge_signs, boundary_edges_to_fem_edges)
    {
        long sign_mismatch = 0, map_mismatch = 0;
        std::vector<int> fem_of_edge(E, -1);
        for (int q = 0; q < F; ++q)
            for (int i = 0; i < 4; ++i) {
                const int e = S.quad_edges[q][i];
                if (S.quad_sign[q][i] != mesh.boundary_edge_signs[4 * q + i]) ++sign_mismatch;
                const int fe = mesh.boundary_edges_to_fem_edges[4 * q + i];
                if (fem_of_edge[e] < 0) fem_of_edge[e] = fe;
                else if (fem_of_edge[e] != fe) ++map_mismatch;
                // die FEM-Kante verbindet dieselben Knoten (Knoten im Originalnetz): vmin/vmax mit remap abgleichen unten
            }
        check_eq(sign_mismatch == 0, "quad_sign = mesh.boundary_edge_signs", sign_mismatch, 0);
        check_eq(map_mismatch == 0, "jede Flaechenkante gehoert zu genau einer FEM-Kante", map_mismatch, 0);
    }
    check_eq(worst == 0, "R G psi = 0 exakt (ganzzahliges psi, alle Quads)", worst, 0);
    check_eq(E == 2 * F, "E = 2F", E, 2 * F);

    compute_harmonic_fields(S);
    check_eq(S.nH == 4, "dim H = 2 sum g = 4", S.nH, 4);
    std::printf("    max|R H| = %.2e, max|G^T H| = %.2e\n", S.harmonic_residual_R, S.harmonic_residual_G);
    check_eq(S.harmonic_residual_R <= 1e-9 && S.harmonic_residual_G <= 1e-9, "H in ker R und senkrecht range G", S.harmonic_residual_R <= 1e-9 && S.harmonic_residual_G <= 1e-9, 1);

    std::printf(g_fail == 0 ? "ALLE TESTS BESTANDEN\n" : "FEHLGESCHLAGEN: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
