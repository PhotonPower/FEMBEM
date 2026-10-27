// =============================================================================
// Abnahmeprüfung: adressieren die BEM-Blöcke die richtigen FEM-Kanten?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_bem_index_mapping
//   .\build\Release\test_bem_index_mapping.exe          (aus dem Repo-Wurzelverz.)
//   ./build/test_bem_index_mapping                      (Linux)
// Optional ein anderes Netz: test_bem_index_mapping <pfad/zur/datei.vtk>
//
// Diese Datei ist reiner Host-Code, wird aber als .cu von nvcc übersetzt, damit
// sie direkt gegen die echten Header (Mesh.cuh, DeviceArray.cuh) baut statt
// gegen Kopien. Sie startet keine Kernel und braucht keine GPU-Rechenleistung,
// nur einen CUDA-Kontext für den Unified Memory von DeviceArray.
//
// HINTERGRUND
// -----------
// mainloop() in src/fem.cu hängt die komprimierten BEM-Blöcke so in den
// gekoppelten Operator ein (Zeilen 487/488 bzw. 529/530):
//
//     int row_off = tree.get_node(i).quad_indices[0] * 4;
//     int col_off = tree.get_node(j).quad_indices[0] * 4;
//     op.add_aca_block(m, n, aca.rank, row_off, col_off, d_U, d_V);
//
// Das setzt zwei Dinge voraus, die beide nicht gelten:
//
//   (a) Dass die Quad-Indizes eines Clusters zusammenhängend und aufsteigend
//       ab quad_indices[0] liegen. ClusterTree::build_recursive sortiert die
//       Quads per std::sort nach Schwerpunktkoordinate, quad_indices enthält
//       also verstreute Originalindizes.
//
//   (b) Dass "Randquad-Index * 4 + lokale Kante" dasselbe ist wie eine globale
//       FEM-Kanten-ID. Der Operator arbeitet auf total_dofs = mesh.num_edges.
//       Die Übersetzung leistet mesh.boundary_edges_to_fem_edges, das in
//       TopologyExtractor::extract_boundary_faces korrekt aufgebaut, im
//       Gen-B-Pfad aber nirgends verwendet wird. Ebenso ungenutzt bleibt
//       mesh.boundary_edge_signs für die Orientierung.
//
// Der Test bildet die Blockschleife exakt nach und vergleicht die tatsächlich
// adressierten Zeilen mit denen, die über boundary_edges_to_fem_edges korrekt
// wären.
//
// STAND
// -----
// Die Kopplung ist umgebaut: mainloop() benutzt build_bem_coupling_map
// (include/BemCoupling.h, src/BemCoupling.cpp), die Blockoffsets liegen im
// permutierten BEM-Elementraum, und CoupledMVOperator::apply() bindet ihn per
// Gather/Scatter an die FEM-Kanten. Dieser Test prüft die Invarianten des
// Ergebnisses und ist damit eine Regressionsprüfung, kein offenes To-do mehr.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
//
// Nach dem Umbau zusätzlich zu prüfen, was dieser Test NICHT abdeckt:
//   - boundary_edge_signs muss beim Gather/Scatter angewendet werden.
//   - evaluate_rooftop_basis in BemIntegration.cuh definiert seine Kanten als
//     0->1, 1->2, 3->2, 0->3. Die Kantentabelle in TopologyExtractor ist
//     zyklisch ({0,1},{1,2},{2,3},{3,0}), die lokalen Kanten 2 und 3 laufen
//     dort also gegenläufig.
// Vorlage für beides: k_gather_bem / k_scatter_add_bem.
// =============================================================================

#include "VtkReader.h"
#include "TopologyExtractor.h"
#include "ClusterTree.h"
#include "BemCoupling.h"

#include <cstdio>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>

int main(int argc, char** argv)
{
    const char* mesh_path = (argc > 1) ? argv[1] : "team28.vtk";

    Mesh mesh;
    VtkReader::read_hexahedral_mesh(mesh_path, mesh);
    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);

    const int total_dofs = static_cast<int>(mesh.num_edges);
    ClusterTree tree(30);
    tree.build_tree(mesh);

    const BemCouplingMap map = build_bem_coupling_map(mesh, tree);

    printf("\n=============== BEM -> FEM Indexabbildung ===============\n");
    printf("total_dofs (mesh.num_edges) = %d\n", total_dofs);
    printf("num_bem_dofs                = %d\n", map.num_bem_dofs);
    printf("Baumknoten                  = %zu\n\n", tree.get_num_nodes());

    int failures = 0;
    auto check = [&](bool ok, const char* name, const char* detail) {
        printf("  [%s] %s%s%s\n", ok ? "OK  " : "FEHL", name,
               ok ? "" : "  <- ", ok ? "" : detail);
        if (!ok) ++failures;
    };

    // ---- 1: perm ist eine Bijektion ueber alle Randquads ----
    {
        std::set<int> uniq(map.perm.begin(), map.perm.end());
        bool ok = (map.perm.size() == mesh.num_boundary_faces)
               && (uniq.size() == mesh.num_boundary_faces)
               && (*uniq.begin() == 0)
               && (*uniq.rbegin() == static_cast<int>(mesh.num_boundary_faces) - 1);
        check(ok, "perm ist Bijektion ueber alle Randquads", "Quad fehlt oder doppelt");
    }

    // ---- 2: Blaetter belegen zusammenhaengende, luecken- und ueberlappungsfreie Bereiche ----
    {
        std::vector<char> covered(map.num_bem_dofs, 0);
        bool ok = true;
        long long total = 0;
        for (size_t node = 0; node < tree.get_num_nodes(); ++node) {
            const ClusterNode& nd = tree.get_node(node);
            if (!nd.is_leaf) { if (map.leaf_offset[node] != -1) ok = false; continue; }
            const int off = map.leaf_offset[node] * 4;
            const int m   = static_cast<int>(nd.quad_indices.size()) * 4;
            if (off < 0 || off + m > map.num_bem_dofs) { ok = false; continue; }
            for (int t = 0; t < m; ++t) { if (covered[off + t]) ok = false; covered[off + t] = 1; }
            total += m;
        }
        if (total != map.num_bem_dofs) ok = false;
        for (int t = 0; t < map.num_bem_dofs; ++t) if (!covered[t]) ok = false;
        check(ok, "Blattbereiche sind zusammenhaengend und partitionieren den BEM-Raum",
              "Ueberlappung, Luecke oder Bereichsverletzung");
    }

    // ---- 3: jeder Blockeintrag adressiert die richtige FEM-Kante ----
    // Zeile row_off + lq*4 + e des Blocks von Blatt i muss auf dieselbe FEM-Kante
    // zeigen wie boundary_edges_to_fem_edges[quad_indices[lq]*4 + e].
    {
        long long checked = 0, wrong = 0;
        for (size_t node = 0; node < tree.get_num_nodes(); ++node) {
            const ClusterNode& nd = tree.get_node(node);
            if (!nd.is_leaf) continue;
            const int off = map.leaf_offset[node] * 4;
            for (int lq = 0; lq < static_cast<int>(nd.quad_indices.size()); ++lq) {
                for (int e = 0; e < 4; ++e) {
                    const int bem_dof = off + lq * 4 + e;
                    const int expect_edge = mesh.boundary_edges_to_fem_edges[nd.quad_indices[lq] * 4 + e];
                    const int expect_sign = mesh.boundary_edge_signs[nd.quad_indices[lq] * 4 + e];
                    ++checked;
                    if (map.bem_to_fem[bem_dof] != expect_edge) ++wrong;
                    else if (map.bem_sign[bem_dof] != expect_sign) ++wrong;
                }
            }
        }
        printf("      geprueft %lld Blockeintraege, falsch %lld\n", checked, wrong);
        check(wrong == 0 && checked == map.num_bem_dofs,
              "Blockzeilen zeigen auf die korrekte FEM-Kante samt Vorzeichen", "Abweichung");
    }

    // ---- 4: alle Indizes im gueltigen Bereich, Vorzeichen +/-1 ----
    {
        bool ok = true;
        for (int i = 0; i < map.num_bem_dofs; ++i) {
            if (map.bem_to_fem[i] < 0 || map.bem_to_fem[i] >= total_dofs) ok = false;
            if (map.bem_sign[i] != 1 && map.bem_sign[i] != -1) ok = false;
        }
        check(ok, "alle FEM-Indizes im Bereich, alle Vorzeichen +/-1", "Bereichs- oder Vorzeichenfehler");
    }

    // ---- 5: jede Randkante wird von genau zwei BEM-Freiheitsgraden getroffen ----
    // Auf einer geschlossenen Oberflaeche gehoert jede Kante zu genau zwei Quads.
    // Das ist die Voraussetzung dafuer, dass der Scatter addieren muss.
    {
        std::map<int,int> hits;
        for (int i = 0; i < map.num_bem_dofs; ++i) ++hits[map.bem_to_fem[i]];
        int not_two = 0, min_h = 1 << 30, max_h = 0;
        for (const auto& kv : hits) {
            if (kv.second != 2) ++not_two;
            min_h = std::min(min_h, kv.second);
            max_h = std::max(max_h, kv.second);
        }
        printf("      %zu verschiedene Randkanten, Treffer je Kante zwischen %d und %d\n",
               hits.size(), min_h, max_h);
        check(not_two == 0, "jede Randkante wird von genau 2 BEM-Freiheitsgraden getroffen",
              "Netz nicht mannigfaltig oder Abbildung falsch");
        check(static_cast<long long>(hits.size()) * 2 == map.num_bem_dofs,
              "Anzahl Randkanten mal 2 gleich num_bem_dofs", "Inkonsistenz");
    }

    // ---- 6: keine FEM-Kante geht leer aus, die einen Beitrag braucht ----
    {
        std::set<int> reached(map.bem_to_fem.begin(), map.bem_to_fem.end());
        std::set<int> needed;
        for (size_t i = 0; i < mesh.num_boundary_faces * 4; ++i)
            needed.insert(mesh.boundary_edges_to_fem_edges[i]);
        std::vector<int> missed;
        std::set_difference(needed.begin(), needed.end(), reached.begin(), reached.end(),
                            std::back_inserter(missed));
        printf("      benoetigt %zu, erreicht %zu, fehlend %zu\n",
               needed.size(), reached.size(), missed.size());
        check(missed.empty(), "keine benoetigte FEM-Kante ohne BEM-Beitrag", "Kanten fehlen");
    }

    printf("\n=============== %s ===============\n\n",
           failures == 0 ? "BESTANDEN: Indexabbildung korrekt"
                         : "FEHLGESCHLAGEN: Indexabbildung falsch");
    return failures == 0 ? 0 : 1;
}
