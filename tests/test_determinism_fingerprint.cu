// =============================================================================
// Determinismus-Fingerprint: wo genau divergieren zwei Plattformen?
//
//   cmake --build build --config Release --target test_determinism_fingerprint
//   .\build\Release\test_determinism_fingerprint.exe      (aus dem Repo-Wurzelverz.)
//
// Hintergrund: derselbe Quellcode mit demselben team28.vtk erzeugte unter
// libstdc++/g++ und der MSVC-STL verschiedene Cluster-Bäume. Ein eindeutiger
// Tiebreak im Sortier-Komparator von ClusterTree::build_recursive hat das
// nicht vollständig behoben -- die Blattpartition unterscheidet sich weiterhin.
//
// Dieser Test bildet bitweise Prüfsummen (FNV-1a über die Rohbytes) über jede
// Stufe der Kette, damit sich die Divergenz lokalisieren lässt, statt sie zu
// erraten. Zwei Plattformen vergleichen ihre Ausgaben zeilenweise; die erste
// abweichende Zeile benennt die Stufe.
//
//   H1  Knotenkoordinaten, wie VtkReader sie liefert
//   H2  Randflächen-Konnektivität (ganzzahlig, sollte immer gleich sein)
//   H3  Quad-Schwerpunkte, hier nachgerechnet wie in ClusterTree
//   H4  Permutation nach dem Wurzel-Sortierschritt, Komparator nachgebildet
//   H5  Blattpartition des fertigen Baums
//   H6  boundary_edges_to_fem_edges (ganzzahlig)
//
// Lesart:
//   H1 weicht ab  -> Text-nach-double-Konvertierung im VtkReader
//   H2 oder H6    -> Topologie-Extraktion, waere ueberraschend (rein integer)
//   H3 weicht ab  -> Fliesskomma-Auswertung der Schwerpunkte
//                    (FMA-Kontraktion, Reassoziation, x87-Zwischengenauigkeit)
//   H4 bei gleichem H3 -> Sortierung selbst, also Komparator oder Algorithmus
//   H5 bei gleichem H4 -> Rekursion: Achsenwahl, Bounding Box, Median-Teilung
//
// Zusätzlich werden die ersten Splits im Klartext ausgegeben, damit man sieht,
// ab welcher Rekursionstiefe die Bäume auseinanderlaufen.
// =============================================================================

#include "Backend.cuh"
#include "VtkReader.h"
#include "TopologyExtractor.h"
#include "ClusterTree.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>

// FNV-1a über Rohbytes. Bitweise, also empfindlich für das letzte Bit eines
// double -- genau darum geht es hier.
static uint64_t fnv1a(const void* data, size_t n_bytes, uint64_t h = 1469598103934665603ULL)
{
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < n_bytes; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

int main(int argc, char** argv)
{
    // FEMBEM_BACKEND=cpu rechnet ohne GPU. Der CPU-Modus ist die Referenz fuer
    // diesen Test: seine Summationsreihenfolge ist fest (siehe BUILD.md).
    if (const char* env = std::getenv("FEMBEM_BACKEND")) {
        if (std::strcmp(env, "cpu") == 0 || std::strcmp(env, "CPU") == 0)
            bf::g_backend = bf::Backend::CPU;
    }

    const char* mesh_path = (argc > 1) ? argv[1] : "team28.vtk";

    Mesh mesh;
    VtkReader::read_hexahedral_mesh(mesh_path, mesh);
    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);

    printf("\n=============== Determinismus-Fingerprint ===============\n");
    printf("sizeof(double)=%zu  sizeof(double3)=%zu  sizeof(int)=%zu\n",
           sizeof(double), sizeof(double3), sizeof(int));
    printf("Knoten=%zu  Hexaeder=%zu  Kanten=%zu  Randflaechen=%zu\n\n",
           mesh.num_nodes, mesh.num_hexes, mesh.num_edges, mesh.num_boundary_faces);

    // ---- H1: Knotenkoordinaten ----
    uint64_t h1 = 1469598103934665603ULL;
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        double3 p = mesh.nodes_current[i];
        h1 = fnv1a(&p.x, sizeof(double), h1);
        h1 = fnv1a(&p.y, sizeof(double), h1);
        h1 = fnv1a(&p.z, sizeof(double), h1);
    }
    printf("H1 Knotenkoordinaten            = %016llx\n", (unsigned long long)h1);

    // ---- H2: Randflaechen-Konnektivitaet ----
    uint64_t h2 = 1469598103934665603ULL;
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i) {
        Mesh::QuadFace q = mesh.boundary_faces[i];
        for (int k = 0; k < 4; ++k) h2 = fnv1a(&q.v[k], sizeof(int), h2);
    }
    printf("H2 Randflaechen-Konnektivitaet  = %016llx\n", (unsigned long long)h2);

    // ---- H2b: globale Kantenliste ----
    // extract_edges vergibt die IDs in der Hexaeder-Schleife ueber
    // unique_edges.size() und iteriert edge_map nie, die Nummerierung ist also
    // deterministisch. Hier trotzdem gehasht, damit die Annahme geprueft wird.
    uint64_t h2b = 1469598103934665603ULL;
    for (size_t i = 0; i < mesh.num_edges; ++i) {
        int2 e = mesh.edges[i];
        h2b = fnv1a(&e.x, sizeof(int), h2b);
        h2b = fnv1a(&e.y, sizeof(int), h2b);
    }
    printf("H2b globale Kantenliste         = %016llx\n", (unsigned long long)h2b);

    // ---- H3: Schwerpunkte, exakt wie in ClusterTree::build_tree ----
    std::vector<double3> cen(mesh.num_boundary_faces);
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i) {
        Mesh::QuadFace quad = mesh.boundary_faces[i];
        double3 p0 = mesh.nodes_current[quad.v[0]];
        double3 p1 = mesh.nodes_current[quad.v[1]];
        double3 p2 = mesh.nodes_current[quad.v[2]];
        double3 p3 = mesh.nodes_current[quad.v[3]];
        cen[i].x = 0.25 * (p0.x + p1.x + p2.x + p3.x);
        cen[i].y = 0.25 * (p0.y + p1.y + p2.y + p3.y);
        cen[i].z = 0.25 * (p0.z + p1.z + p2.z + p3.z);
    }
    uint64_t h3 = 1469598103934665603ULL;
    for (size_t i = 0; i < cen.size(); ++i) {
        h3 = fnv1a(&cen[i].x, sizeof(double), h3);
        h3 = fnv1a(&cen[i].y, sizeof(double), h3);
        h3 = fnv1a(&cen[i].z, sizeof(double), h3);
    }
    printf("H3 Quad-Schwerpunkte            = %016llx\n", (unsigned long long)h3);

    // Rohwerte zum direkten Vergleich, falls H3 abweicht. %.17g ist fuer double
    // verlustfrei, ein Unterschied im letzten Bit ist damit sichtbar.
    const size_t probe[] = { 0, 1, 2, cen.size()/2, cen.size()-1 };
    for (size_t t = 0; t < sizeof(probe)/sizeof(probe[0]); ++t) {
        size_t i = probe[t];
        printf("   cen[%4zu] = %.17g  %.17g  %.17g\n", i, cen[i].x, cen[i].y, cen[i].z);
    }
    {   // Erster Knoten roh, um H1 eingrenzen zu koennen
        double3 p = mesh.nodes_current[0];
        double3 q = mesh.nodes_current[mesh.num_nodes-1];
        printf("   node[0]    = %.17g  %.17g  %.17g\n", p.x, p.y, p.z);
        printf("   node[last] = %.17g  %.17g  %.17g\n", q.x, q.y, q.z);
    }

    // Wie viele exakte Gleichstaende gibt es ueberhaupt pro Achse?
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<double> k(cen.size());
        for (size_t i = 0; i < cen.size(); ++i)
            k[i] = (axis == 0) ? cen[i].x : (axis == 1) ? cen[i].y : cen[i].z;
        std::sort(k.begin(), k.end());
        size_t distinct = static_cast<size_t>(std::unique(k.begin(), k.end()) - k.begin());
        printf("   Achse %c: %zu verschiedene Werte bei %zu Quads -> %zu Gleichstands-Paare\n",
               "xyz"[axis], distinct, cen.size(), cen.size() - distinct);
    }

    // ---- H4: Wurzel-Sortierschritt nachgebildet ----
    // Bounding Box ueber alle Quads, laengste Achse, Komparator wie im Original.
    double3 bb_min = mesh.nodes_current[mesh.boundary_faces[0].v[0]];
    double3 bb_max = bb_min;
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i) {
        Mesh::QuadFace q = mesh.boundary_faces[i];
        for (int k = 0; k < 4; ++k) {
            double3 p = mesh.nodes_current[q.v[k]];
            bb_min.x = std::min(bb_min.x, p.x); bb_max.x = std::max(bb_max.x, p.x);
            bb_min.y = std::min(bb_min.y, p.y); bb_max.y = std::max(bb_max.y, p.y);
            bb_min.z = std::min(bb_min.z, p.z); bb_max.z = std::max(bb_max.z, p.z);
        }
    }
    const double dx = bb_max.x - bb_min.x;
    const double dy = bb_max.y - bb_min.y;
    const double dz = bb_max.z - bb_min.z;
    int axis = 0;
    if (dy > dx && dy > dz) axis = 1;
    else if (dz > dx && dz > dy) axis = 2;
    printf("\n   Wurzel-BBox dx=%.17g dy=%.17g dz=%.17g -> Splitachse %c\n",
           dx, dy, dz, "xyz"[axis]);

    std::vector<int> perm(mesh.num_boundary_faces);
    for (size_t i = 0; i < perm.size(); ++i) perm[i] = static_cast<int>(i);
    std::sort(perm.begin(), perm.end(), [&cen, axis](int a, int b) {
        double ka, kb;
        if      (axis == 0) { ka = cen[a].x; kb = cen[b].x; }
        else if (axis == 1) { ka = cen[a].y; kb = cen[b].y; }
        else                { ka = cen[a].z; kb = cen[b].z; }
        if (ka != kb) return ka < kb;
        return a < b;
    });
    uint64_t h4 = fnv1a(perm.data(), perm.size() * sizeof(int));
    printf("H4 Permutation nach Wurzelsort  = %016llx\n", (unsigned long long)h4);

    // ---- H5: Blattpartition des fertigen Baums ----
    ClusterTree tree(30);
    tree.build_tree(mesh);
    uint64_t h5 = 1469598103934665603ULL;
    int leaves = 0;
    for (size_t i = 0; i < tree.get_num_nodes(); ++i) {
        const ClusterNode& nd = tree.get_node(i);
        if (!nd.is_leaf) continue;
        ++leaves;
        int sz = static_cast<int>(nd.quad_indices.size());
        h5 = fnv1a(&sz, sizeof(int), h5);
        h5 = fnv1a(nd.quad_indices.data(), nd.quad_indices.size() * sizeof(int), h5);
    }
    printf("H5 Blattpartition (%d Blaetter)  = %016llx\n", leaves, (unsigned long long)h5);

    // ---- H6: BEM->FEM Kantenabbildung ----
    uint64_t h6 = 1469598103934665603ULL;
    for (size_t i = 0; i < mesh.num_boundary_faces * 4; ++i) {
        int e = mesh.boundary_edges_to_fem_edges[i];
        h6 = fnv1a(&e, sizeof(int), h6);
    }
    printf("H6 boundary_edges_to_fem_edges  = %016llx\n", (unsigned long long)h6);

    // ---- Die ersten Blaetter im Klartext, um die Divergenztiefe zu sehen ----
    printf("\n   Erste 6 Blaetter (Groesse: erste 5 Indizes):\n");
    int shown = 0;
    for (size_t i = 0; i < tree.get_num_nodes() && shown < 6; ++i) {
        const ClusterNode& nd = tree.get_node(i);
        if (!nd.is_leaf) continue;
        printf("     Knoten %3zu  n=%2zu : ", i, nd.quad_indices.size());
        for (size_t k = 0; k < nd.quad_indices.size() && k < 5; ++k)
            printf("%d ", nd.quad_indices[k]);
        printf("\n");
        ++shown;
    }
    printf("\n");
    return 0;
}
