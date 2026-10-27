#include "CohomologyGauge.h"

#include <vector>
#include <queue>
#include <map>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace {

// -----------------------------------------------------------------------------
// Flächenrand-Tabelle zur Laufzeit ableiten (siehe Begründung in
// CohomologyGauge.h und im Projektstand-Dokument, Abschnitt "Konventionen"):
// eine von-Hand-Ableitung ist genau die Fehlerquelle, die diese Codebasis
// wiederholt getroffen hat.
// -----------------------------------------------------------------------------
struct FaceEdgeEntry { int local_edge; int sign; };

void build_face_edge_table(FaceEdgeEntry table[6][4])
{
    const int local_face_to_nodes[6][4] = {
        {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
        {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}
    };
    const int local_edge_to_nodes[12][2] = {
        {0, 1}, {1, 2}, {2, 3}, {3, 0}, {4, 5}, {5, 6},
        {6, 7}, {7, 4}, {0, 4}, {1, 5}, {2, 6}, {3, 7}
    };
    for (int f = 0; f < 6; ++f) {
        for (int k = 0; k < 4; ++k) {
            const int a = local_face_to_nodes[f][k];
            const int b = local_face_to_nodes[f][(k + 1) % 4];
            bool found = false;
            for (int le = 0; le < 12; ++le) {
                if (local_edge_to_nodes[le][0] == a && local_edge_to_nodes[le][1] == b) {
                    table[f][k] = {le, +1}; found = true; break;
                }
                if (local_edge_to_nodes[le][0] == b && local_edge_to_nodes[le][1] == a) {
                    table[f][k] = {le, -1}; found = true; break;
                }
            }
            if (!found) {
                throw std::runtime_error("build_face_edge_table: local_face_to_nodes und "
                    "local_edge_to_nodes sind inkonsistent.");
            }
        }
    }
}

// Ein 2-Zell (Fläche) als 4 (globale Kante, Vorzeichen)-Paare.
struct Face { int edge[4]; int sign[4]; };

// -----------------------------------------------------------------------------
// Einfacher CG-Löser für symmetrisch positiv definite (nach Regularisierung)
// Systeme, gegeben rein über ein Matrix-Vektor-Produkt (kein explizites
// Aufstellen der Matrix nötig). Dient nur dieser Datei.
// -----------------------------------------------------------------------------
template <class MatVec>
void solve_cg(int n, MatVec apply, const std::vector<double>& b,
             std::vector<double>& x, double tol, int max_iter)
{
    x.assign(n, 0.0);
    std::vector<double> r = b, p, Ap(n);
    double rr = 0.0;
    for (int i = 0; i < n; ++i) rr += r[i] * r[i];
    double b_norm = std::sqrt(rr);
    if (b_norm < 1e-300) return; // b ist Null -> Loesung ist Null
    p = r;
    for (int iter = 0; iter < max_iter; ++iter) {
        apply(p, Ap);
        double pAp = 0.0;
        for (int i = 0; i < n; ++i) pAp += p[i] * Ap[i];
        if (std::fabs(pAp) < 1e-300) break;
        const double alpha = rr / pAp;
        double rr_new = 0.0;
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
            rr_new += r[i] * r[i];
        }
        if (std::sqrt(rr_new) / b_norm < tol) { rr = rr_new; break; }
        const double beta = rr_new / rr;
        for (int i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];
        rr = rr_new;
    }
}

} // namespace

CohomologyGaugeResult find_cohomology_generators(
    const Mesh& mesh,
    const GaugeInfo& base_gauge,
    const MaterialProperties* materials,
    double kappa_threshold,
    bool allow_simply_connected)
{
    if (materials == nullptr) {
        throw std::runtime_error("find_cohomology_generators: materials ist nullptr.");
    }
    if (static_cast<int>(base_gauge.is_gauged.size()) != static_cast<int>(mesh.num_edges)) {
        throw std::runtime_error("find_cohomology_generators: base_gauge passt nicht zu "
            "diesem Netz.");
    }

    const int num_nodes = static_cast<int>(mesh.num_nodes);
    const int num_hexes = static_cast<int>(mesh.num_hexes);
    const int num_edges = static_cast<int>(mesh.num_edges);

    // ---------------------------------------------------------------------
    // 1. Leitende Kanten, Baum-Adjazenz, Komponenten (wie im verworfenen
    //    ersten Versuch -- diese Schritte waren nicht das Problem).
    // ---------------------------------------------------------------------
    std::vector<char> conductor_edge(num_edges, 0);
    for (int h = 0; h < num_hexes; ++h) {
        if (!(materials[h].kappa > kappa_threshold)) continue;
        const int ne = element_num_edges(mesh.elem_type(h));
        for (int e = 0; e < ne; ++e) {
            const int ge = mesh.hex_edges[h * 12 + e];
            if (ge < 0) continue;   // ungenutzter Platz eines Tetraeders/Prismas
            conductor_edge[ge] = 1;
        }
    }

    std::vector<int> component_id(num_nodes, -1);
    {
        std::vector<std::vector<int>> adj(num_nodes);
        for (int e = 0; e < num_edges; ++e) {
            if (!base_gauge.is_gauged[e] || conductor_edge[e]) continue;
            const int2 ed = mesh.edges[e];
            adj[ed.x].push_back(ed.y);
            adj[ed.y].push_back(ed.x);
        }
        int num_components = 0;
        for (int start = 0; start < num_nodes; ++start) {
            if (component_id[start] != -1 || adj[start].empty()) continue;
            const int comp = num_components++;
            std::queue<int> q; q.push(start); component_id[start] = comp;
            while (!q.empty()) {
                const int u = q.front(); q.pop();
                for (int v : adj[u]) {
                    if (component_id[v] != -1) continue;
                    component_id[v] = comp; q.push(v);
                }
            }
        }
    }
    int num_components = 0;
    for (int c : component_id) num_components = std::max(num_components, c + 1);

    // ---------------------------------------------------------------------
    // 2. Pro Komponente: Kantenmenge, Flächenliste (C_2, dedupliziert)
    // ---------------------------------------------------------------------
    FaceEdgeEntry face_table[6][4];
    build_face_edge_table(face_table);

    CohomologyGaugeResult result;

    for (int comp = 0; comp < num_components; ++comp) {
        // Kanten dieser Komponente
        std::vector<int> comp_edges;
        for (int e = 0; e < num_edges; ++e) {
            if (conductor_edge[e]) continue;
            const int2 ed = mesh.edges[e];
            if (component_id[ed.x] == comp) comp_edges.push_back(e);
        }

        // Flächen dieser Komponente, dedupliziert über die sortierten
        // globalen Knoten-IDs. Erste Begegnung legt die Orientierung fest;
        // das ist beliebig waehlbar, da nur der von den Flaechenraendern
        // aufgespannte Unterraum (Im(delta_1^T)) gebraucht wird.
        std::map<std::array<int,4>, Face> face_map;
        for (int h = 0; h < num_hexes; ++h) {
            if (materials[h].kappa > kappa_threshold) continue;
            // Ein Hexaeder gehoert vollstaendig zu einer Komponente (Knoten
            // eines Hexaeders liegen nie in zwei verschiedenen Komponenten).
            const int n0 = mesh.hex_nodes[h].v[0];
            if (component_id[n0] != comp) continue;

            for (int f = 0; f < 6; ++f) {
                Face face;
                std::array<int,4> gnodes;
                for (int k = 0; k < 4; ++k) {
                    const int local_edge = face_table[f][k].local_edge;
                    const int fsign      = face_table[f][k].sign;
                    const int ge = mesh.hex_edges[h * 12 + local_edge];
                    const int hs = mesh.hex_edge_signs[h * 12 + local_edge];
                    if (ge < 0) continue;
                    face.edge[k] = ge;
                    face.sign[k] = fsign * hs;
                }
                // Knoten dieser Flaeche für den Dedup-Schlüssel (sortiert,
                // damit beide Hexaeder, die sich eine Flaeche teilen, auf
                // denselben Schlüssel kommen unabhängig von Durchlaufrichtung).
                const int local_face_to_nodes[6][4] = {
                    {0,3,2,1},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7}
                };
                for (int k = 0; k < 4; ++k)
                    gnodes[k] = mesh.hex_nodes[h].v[local_face_to_nodes[f][k]];
                std::sort(gnodes.begin(), gnodes.end());

                if (face_map.find(gnodes) == face_map.end()) {
                    face_map[gnodes] = face;
                }
            }
        }
        std::vector<Face> faces;
        faces.reserve(face_map.size());
        for (auto& kv : face_map) faces.push_back(kv.second);

        // ---------------------------------------------------------------------
        // 3. Matvecs: delta_1 (Kanten->Flaechen), delta_1^T, delta_0 (Knoten->
        //    Kanten), delta_0^T -- alle auf global indizierten, ausserhalb der
        //    Komponente konsequent auf Null gehaltenen Vektoren.
        // ---------------------------------------------------------------------
        auto apply_delta1 = [&](const std::vector<double>& a, std::vector<double>& out_faces) {
            out_faces.assign(faces.size(), 0.0);
            for (size_t f = 0; f < faces.size(); ++f) {
                double s = 0.0;
                for (int k = 0; k < 4; ++k) s += faces[f].sign[k] * a[faces[f].edge[k]];
                out_faces[f] = s;
            }
        };
        auto apply_delta1T = [&](const std::vector<double>& y_faces, std::vector<double>& out_edges) {
            out_edges.assign(num_edges, 0.0);
            for (size_t f = 0; f < faces.size(); ++f) {
                for (int k = 0; k < 4; ++k)
                    out_edges[faces[f].edge[k]] += faces[f].sign[k] * y_faces[f];
            }
        };
        auto apply_delta0 = [&](const std::vector<double>& phi, std::vector<double>& out_edges) {
            out_edges.assign(num_edges, 0.0);
            for (int e : comp_edges) {
                const int2 ed = mesh.edges[e];
                out_edges[e] = phi[ed.y] - phi[ed.x];
            }
        };
        auto apply_delta0T = [&](const std::vector<double>& a, std::vector<double>& out_nodes) {
            out_nodes.assign(num_nodes, 0.0);
            for (int e : comp_edges) {
                const int2 ed = mesh.edges[e];
                out_nodes[ed.y] += a[e];
                out_nodes[ed.x] -= a[e];
            }
        };

        // ---------------------------------------------------------------------
        // 4. Schritt 1: zufaelligen Startvektor auf den Kanten der Komponente
        // ---------------------------------------------------------------------
        std::vector<double> a0(num_edges, 0.0);
        unsigned seed = 0x9E3779B9u + static_cast<unsigned>(comp) * 2654435761u;
        for (int e : comp_edges) {
            seed = seed * 1103515245u + 12345u;
            a0[e] = static_cast<double>((seed >> 16) & 0x7fffu) / 32768.0 - 0.5;
        }

        // ---------------------------------------------------------------------
        // 5. Schritt 2: auf Ker(delta_1) projizieren
        //    (delta_1 delta_1^T + eps I) y = delta_1 a0,  a_proj = a0 - delta_1^T y
        // ---------------------------------------------------------------------
        std::vector<double> rhs_faces;
        apply_delta1(a0, rhs_faces);
        const double eps1 = 1e-9;
        std::vector<double> y;
        solve_cg(static_cast<int>(faces.size()),
            [&](const std::vector<double>& v, std::vector<double>& out) {
                std::vector<double> tmp_edges;
                apply_delta1T(v, tmp_edges);
                std::vector<double> tmp_faces;
                apply_delta1(tmp_edges, tmp_faces);
                out.resize(v.size());
                for (size_t i = 0; i < v.size(); ++i) out[i] = tmp_faces[i] + eps1 * v[i];
            },
            rhs_faces, y, 1e-10, 20000);

        std::vector<double> delta1T_y;
        apply_delta1T(y, delta1T_y);
        std::vector<double> a_proj(num_edges);
        for (int e : comp_edges) a_proj[e] = a0[e] - delta1T_y[e];

        // ---------------------------------------------------------------------
        // 6. Schritt 3: Gradientenanteil herausprojizieren
        //    (delta_0^T delta_0 + eps I) phi = delta_0^T a_proj,
        //    a_harm = a_proj - delta_0(phi)
        // ---------------------------------------------------------------------
        std::vector<double> rhs_nodes;
        apply_delta0T(a_proj, rhs_nodes);
        const double eps0 = 1e-9;
        std::vector<double> phi;
        solve_cg(num_nodes,
            [&](const std::vector<double>& v, std::vector<double>& out) {
                std::vector<double> tmp_edges;
                apply_delta0(v, tmp_edges);
                std::vector<double> tmp_nodes;
                apply_delta0T(tmp_edges, tmp_nodes);
                out.resize(v.size());
                for (size_t i = 0; i < v.size(); ++i) out[i] = tmp_nodes[i] + eps0 * v[i];
            },
            rhs_nodes, phi, 1e-10, 20000);

        std::vector<double> delta0_phi;
        apply_delta0(phi, delta0_phi);

        CohomologyGenerator gen;
        gen.component_id = comp;
        gen.vector.assign(num_edges, 0.0);
        for (int e : comp_edges) gen.vector[e] = a_proj[e] - delta0_phi[e];

        // ---------------------------------------------------------------------
        // 7. Verifikation: ist a_harm (numerisch) ein Kozykel und nicht trivial?
        // ---------------------------------------------------------------------
        std::vector<double> check_faces;
        apply_delta1(gen.vector, check_faces);
        double res_sq = 0.0, norm_sq = 0.0;
        for (double v : check_faces) res_sq += v * v;
        for (int e : comp_edges) norm_sq += gen.vector[e] * gen.vector[e];
        gen.norm = std::sqrt(norm_sq);
        gen.residual_cocycle = (gen.norm > 1e-300) ? std::sqrt(res_sq) / gen.norm : 1.0;

        if (gen.norm < 1e-6 && allow_simply_connected) {
            // b1 = 0 fuer diese Komponente: sie ist einfach zusammenhaengend,
            // es gibt nichts zu projizieren. Kein Generator wird aufgenommen.
            ++result.num_simply_connected;
            continue;
        }
        if (gen.norm < 1e-6) {
            throw std::runtime_error("find_cohomology_generators: Komponente " + std::to_string(comp)
                + " lieferte einen (nahezu) verschwindenden Generator -- entweder ist diese "
                  "Komponente tatsaechlich einfach zusammenhaengend (b1=0) und wurde faelschlich "
                  "als Loch behandelt, oder die Regularisierung eps ist zu gross.");
        }
        if (gen.residual_cocycle > 1e-5) {
            throw std::runtime_error("find_cohomology_generators: Komponente " + std::to_string(comp)
                + " lieferte keinen gueltigen Kozykel (relatives Residuum "
                + std::to_string(gen.residual_cocycle) + "). CG nicht konvergiert oder eps "
                  "falsch gewaehlt.");
        }

        result.generators.push_back(std::move(gen));
    }

    return result;
}
