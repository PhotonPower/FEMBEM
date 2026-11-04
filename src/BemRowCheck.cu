// =============================================================================
// Gegenprobe 2 im produktiven harmonischen BEM-FEM-Pfad (--bem-row-check)
// =============================================================================
//
// Setzt das exakte Biot-Savart-Paar (a_ex, K) in beide Blockzeilen ein:
//
//   Zeile 1 (FEM auf Randkanten):  K a_ex - T k - b           (K = W - omega*C)
//   Zeile 2 (BEM):                 (-0.5 T - D_BEM) a_ex + mu0 G_BEM k
//
// mit k aus K1 (mu0 K1 = (dA/dn)_tan) und K2 (mu0 K2 = -n x B), je mit beiden
// Vorzeichen. Operatoren: die bestehenden, ueber CoupledMVOperator::
// row_check_parts(). Nichts wird veraendert, kein Vorzeichen stillschweigend
// gewaehlt. Bericht: docs/design/bem-row-check.md.
//
// Zusaetzlich (Messungen, keine Reparatur):
//   - T-Anwendung des Operators gegen eine Host-Nachrechnung mit derselben
//     Indexierung (Kontrolle der Nachrechnung) und gegen zwei Varianten
//     (Geometrie in BEM-Reihenfolge; konsistente Vorzeichen),
//   - Rekonstruktion der Randspur aus a_ex ueber die Rooftop-Funktionen.
// =============================================================================

#include "BemRowCheckRun.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "BemRowCheck.h"
#include "Backend.cuh"

using namespace bem_row_check;

namespace {

constexpr int kNumBodies = 3;                       // 1 Platte, 2 Innen-, 3 Aussenspule
const char* kBodyName[kNumBodies] = {"Platte", "Innenspule", "Aussenspule"};
constexpr int kNumLabels = kNumBodies * kNumClasses;

inline int label_of(int body, int cls) { return (body - 1) * kNumClasses + cls; }

struct Prepared {
    int E = 0;                       // FEM-Kanten
    int nq = 0, nb = 0;              // Quads, BEM-Freiheitsgrade
    std::vector<int> perm;           // Position -> Quad
    std::vector<int> bem_to_fem, bem_sign;
    std::vector<QuadGeom> geom;      // je Position (BEM-Reihenfolge)
    std::vector<int> quad_body;      // je Position
    std::vector<int> quad_cls;       // je Position
    std::vector<int> pos_label;      // je Position: Koerper/Klasse-Label
    std::vector<int> edge_label;     // je FEM-Kante: Label oder -1 (nicht auf dem Rand)
    std::vector<int> edge_body;      // je FEM-Kante: Koerper (1..3) oder 0
    std::vector<double3> nodes;
    std::vector<Mesh::QuadFace> faces;   // Mesh-Reihenfolge
};

double rel_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (a[i] - b[i]); d += b[i] * b[i]; }
    return d > 0 ? std::sqrt(n / d) : std::sqrt(n);
}

// Host-Nachrechnung der T-Anwendung: yA = -T k (Laenge E), yK = -0.5 T a (Laenge nb).
//   perm_geom : Geometrie des Quads perm[pos] statt faces[pos]
//   consistent: Vorzeichen so, dass x_K roher Rooftop-Koeffizient ist und die Testfunktion
//               in Zeile 2 roh (Zeile 1: sign[r], Zeile 2: nur Gather-Vorzeichen sign[c])
void host_T_apply(const Prepared& P, bool perm_geom, bool consistent,
                  const std::vector<double>& a, const std::vector<double>& k,
                  std::vector<double>& yA, std::vector<double>& yK) {
    yA.assign(static_cast<size_t>(P.E), 0.0);
    yK.assign(static_cast<size_t>(P.nb), 0.0);
    for (int pos = 0; pos < P.nq; ++pos) {
        const int qi = perm_geom ? P.perm[static_cast<size_t>(pos)] : pos;
        const Mesh::QuadFace& f = P.faces[static_cast<size_t>(qi)];
        double3 nd[4];
        for (int i = 0; i < 4; ++i) nd[i] = P.nodes[static_cast<size_t>(f.v[i])];
        double T[4][4];
        integrate_boundary_mass_matrix(nd, T);
        int g[4], s[4];
        for (int i = 0; i < 4; ++i) {
            g[i] = P.bem_to_fem[static_cast<size_t>(pos * 4 + i)];
            s[i] = P.bem_sign[static_cast<size_t>(pos * 4 + i)];
        }
        for (int r = 0; r < 4; ++r) {
            double va = 0.0, vk = 0.0;
            for (int c = 0; c < 4; ++c) {
                if (!consistent) {
                    const double Trc = static_cast<double>(s[r] * s[c]) * T[r][c];
                    va -= Trc * k[static_cast<size_t>(pos * 4 + c)];
                    vk -= 0.5 * Trc * a[static_cast<size_t>(g[c])];
                } else {
                    va -= static_cast<double>(s[r]) * T[r][c] * k[static_cast<size_t>(pos * 4 + c)];
                    vk -= 0.5 * T[r][c] * static_cast<double>(s[c]) * a[static_cast<size_t>(g[c])];
                }
            }
            yA[static_cast<size_t>(g[r])] += va;
            yK[static_cast<size_t>(pos * 4 + r)] += vk;
        }
    }
}

// Residuen je Koerper/Klasse: gibt (Label 0..8 | 9..11 je Koerper | 12 alle) zurueck.
std::vector<Residual> by_body_and_class(const std::vector<double>& r, const std::vector<double>& ref,
                                        const std::vector<int>& label) {
    std::vector<Residual> raw = accumulate_by_label(r, ref, label, kNumLabels);   // 0..8, [9] = alle
    std::vector<Residual> out(static_cast<size_t>(kNumLabels) + kNumBodies + 1);
    for (int i = 0; i < kNumLabels; ++i) out[static_cast<size_t>(i)] = raw[static_cast<size_t>(i)];
    for (int b = 0; b < kNumBodies; ++b) {
        Residual s;
        for (int c = 0; c < kNumClasses; ++c) {
            const Residual& x = raw[static_cast<size_t>(b * kNumClasses + c)];
            s.num2 += x.num2; s.ref2 += x.ref2; s.n += x.n;
        }
        out[static_cast<size_t>(kNumLabels + b)] = s;
    }
    out[static_cast<size_t>(kNumLabels + kNumBodies)] = raw[static_cast<size_t>(kNumLabels)];
    return out;
}

void print_row_header(const char* title) {
    std::printf("\n%s\n\n", title);
    std::printf("| Koerper | Klasse | n | +K1 | -K1 | +K2 | -K2 |\n");
    std::printf("|---|---|---:|---:|---:|---:|---:|\n");
}

// vier Residuensaetze (+K1, -K1, +K2, -K2) je Zeile in Prozent
void print_table(const char* title, const std::vector<Residual> res[4]) {
    print_row_header(title);
    for (int b = 0; b < kNumBodies; ++b) {
        for (int c = 0; c <= kNumClasses; ++c) {
            const int idx = (c < kNumClasses) ? b * kNumClasses + c : kNumLabels + b;
            if (res[0][static_cast<size_t>(idx)].n == 0) continue;
            std::printf("| %s | %s | %ld |", kBodyName[b], c < kNumClasses ? class_name(c) : "alle",
                        res[0][static_cast<size_t>(idx)].n);
            for (int v = 0; v < 4; ++v) std::printf(" %.2f %% |", 100.0 * res[v][static_cast<size_t>(idx)].rel());
            std::printf("\n");
        }
    }
    const size_t all = static_cast<size_t>(kNumLabels + kNumBodies);
    std::printf("| alle | alle | %ld |", res[0][all].n);
    for (int v = 0; v < 4; ++v) std::printf(" %.2f %% |", 100.0 * res[v][all].rel());
    std::printf("\n");
    std::cout.flush();
}

}  // namespace

int run_bem_row_check(CoupledMVOperator& op, Mesh& mesh, const BemCouplingMap& map,
                      const std::vector<int>& body_of_hex, CSRMatrix& csr_W, CSRMatrix& csr_C,
                      double omega, const std::vector<double>& b_phys, double spacing_m) {
    using clk = std::chrono::steady_clock;
    const auto t_start = clk::now();
    std::cout << "\n[RowCheck] ===== Gegenprobe 2: exaktes Biot-Savart-Paar in beide Blockzeilen =====\n"
              << std::flush;

    // -- Host-Daten --------------------------------------------------------------
    mesh.edges.prefetch_to_host();
    mesh.nodes_current.prefetch_to_host();
    mesh.boundary_faces.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();
    mesh.boundary_edges_to_fem_edges.prefetch_to_host();
    mesh.boundary_edge_signs.prefetch_to_host();
    csr_W.row_offsets.prefetch_to_host(); csr_W.col_indices.prefetch_to_host(); csr_W.values.prefetch_to_host();
    csr_C.row_offsets.prefetch_to_host(); csr_C.col_indices.prefetch_to_host(); csr_C.values.prefetch_to_host();
    bf::sync();

    Prepared P;
    P.E = static_cast<int>(mesh.num_edges);
    P.nq = static_cast<int>(mesh.num_boundary_faces);
    P.nb = map.num_bem_dofs;
    if (P.nb != 4 * P.nq) {
        std::cout << "[RowCheck] ABBRUCH: num_bem_dofs " << P.nb << " != 4 * Randquads " << P.nq << "\n";
        return 1;
    }
    P.perm = map.perm;
    P.bem_to_fem = map.bem_to_fem;
    P.bem_sign = map.bem_sign;
    P.nodes.resize(mesh.num_nodes);
    for (size_t i = 0; i < mesh.num_nodes; ++i) P.nodes[i] = mesh.nodes_current[i];
    P.faces.resize(static_cast<size_t>(P.nq));
    for (int i = 0; i < P.nq; ++i) P.faces[static_cast<size_t>(i)] = mesh.boundary_faces[static_cast<size_t>(i)];

    // -- Quadgeometrie, Orientierung, Klassen (je BEM-Position) ----------------------
    P.geom.resize(static_cast<size_t>(P.nq));
    P.quad_body.assign(static_cast<size_t>(P.nq), 0);
    P.quad_cls.assign(static_cast<size_t>(P.nq), 0);
    P.pos_label.assign(static_cast<size_t>(P.nq), -1);
    long n_inward = 0, n_perm_moved = 0;
    for (int pos = 0; pos < P.nq; ++pos) {
        const int qi = P.perm[static_cast<size_t>(pos)];
        if (qi != pos) ++n_perm_moved;
        const Mesh::QuadFace& f = P.faces[static_cast<size_t>(qi)];
        QuadGeom g;
        for (int i = 0; i < 4; ++i) g.n[i] = P.nodes[static_cast<size_t>(f.v[i])];
        // aeussere Normale: von der Mitte des Elternhexaeders weg
        const Hexahedron& h = mesh.hex_nodes[static_cast<size_t>(f.parent_hex_id)];
        V3 c{0, 0, 0};
        for (int i = 0; i < 8; ++i) c = c + to_v3(P.nodes[static_cast<size_t>(h.v[i])]);
        c = 0.125 * c;
        const V3 n0 = to_v3(evaluate_quad_normal(0.0, 0.0, g.n));
        const V3 p0 = quad_point(g.n, 0.0, 0.0);
        g.outward = dot(n0, p0 - c) >= 0.0 ? 1.0 : -1.0;
        if (g.outward < 0.0) ++n_inward;
        P.geom[static_cast<size_t>(pos)] = g;
        P.quad_body[static_cast<size_t>(pos)] = body_of_hex[static_cast<size_t>(f.parent_hex_id)];
    }

    // Kante -> angrenzende Quads (Positionen), fuer die Kantenerkennung
    std::vector<std::vector<int>> edge_quads(static_cast<size_t>(P.E));
    for (int pos = 0; pos < P.nq; ++pos)
        for (int e = 0; e < 4; ++e)
            edge_quads[static_cast<size_t>(P.bem_to_fem[static_cast<size_t>(pos * 4 + e)])].push_back(pos);
    for (int pos = 0; pos < P.nq; ++pos) {
        const V3 n = outward_normal(P.geom[static_cast<size_t>(pos)], 0.0, 0.0);
        bool sharp = false;
        for (int e = 0; e < 4 && !sharp; ++e)
            for (int q2 : edge_quads[static_cast<size_t>(P.bem_to_fem[static_cast<size_t>(pos * 4 + e)])]) {
                if (q2 == pos) continue;
                const V3 m = outward_normal(P.geom[static_cast<size_t>(q2)], 0.0, 0.0);
                if (std::fabs(dot(n, m)) < 0.9) { sharp = true; break; }
            }
        const int cls = classify_quad(n, sharp);
        P.quad_cls[static_cast<size_t>(pos)] = cls;
        const int body = P.quad_body[static_cast<size_t>(pos)];
        if (body >= 1 && body <= kNumBodies) P.pos_label[static_cast<size_t>(pos)] = label_of(body, cls);
    }
    // Kantenlabel: beide Quads gleich -> diese Klasse; ebene Nachbarn -> eben;
    // sonst Kante.
    P.edge_label.assign(static_cast<size_t>(P.E), -1);
    P.edge_body.assign(static_cast<size_t>(P.E), 0);
    for (int e = 0; e < P.E; ++e) {
        const auto& qs = edge_quads[static_cast<size_t>(e)];
        if (qs.empty()) continue;
        const int body = P.quad_body[static_cast<size_t>(qs[0])];
        if (body < 1 || body > kNumBodies) continue;
        int cls = P.quad_cls[static_cast<size_t>(qs[0])];
        for (int q : qs) if (P.quad_cls[static_cast<size_t>(q)] != cls) cls = kEdgeQuad;
        P.edge_label[static_cast<size_t>(e)] = label_of(body, cls);
        P.edge_body[static_cast<size_t>(e)] = body;
    }
    // Koerper aller Kanten (innere inbegriffen) ueber die Knoten
    std::vector<int> node_body(mesh.num_nodes, 0);
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const int b = body_of_hex[h];
        if (b <= 0) continue;
        const Hexahedron& hx = mesh.hex_nodes[h];
        for (int i = 0; i < 8; ++i) node_body[static_cast<size_t>(hx.v[i])] = b;
    }

    {
        long cls_n[kNumBodies + 1][kNumClasses] = {};
        for (int pos = 0; pos < P.nq; ++pos)
            if (P.quad_body[static_cast<size_t>(pos)] >= 1 && P.quad_body[static_cast<size_t>(pos)] <= kNumBodies)
                ++cls_n[P.quad_body[static_cast<size_t>(pos)]][P.quad_cls[static_cast<size_t>(pos)]];
        long n_neg = 0;
        for (int i = 0; i < P.nb; ++i) if (P.bem_sign[static_cast<size_t>(i)] < 0) ++n_neg;
        std::printf("[RowCheck] %d Randquads, %d FEM-Kanten, %d BEM-Freiheitsgrade (%ld mit Vorzeichen -1)\n",
                    P.nq, P.E, P.nb, n_neg);
        std::printf("[RowCheck] Normalenorientierung: %ld von %d Quads haben eine nach INNEN zeigende "
                    "evaluate_quad_normal()-Normale\n", n_inward, P.nq);
        std::printf("[RowCheck] BEM-Reihenfolge: %ld von %d Positionen mit perm[pos] != pos\n",
                    n_perm_moved, P.nq);
        for (int b = 1; b <= kNumBodies; ++b)
            std::printf("[RowCheck]   %-11s Quads: eben %ld, Seite %ld, Kante %ld\n", kBodyName[b - 1],
                        cls_n[b][kPlane], cls_n[b][kSide], cls_n[b][kEdgeQuad]);
        std::cout.flush();
    }

    // -- Exakte Felder ---------------------------------------------------------------
    const std::vector<Filament> fil = make_filaments(team28_coils(), 20.0, spacing_m);
    std::printf("[RowCheck] Quelle: %zu Faeden (Abstand %.3g mm), Scheitelstrom 20 A\n", fil.size(),
                spacing_m * 1e3);
    std::cout.flush();

    auto t0 = clk::now();
    std::vector<double> a_ex(static_cast<size_t>(P.E), 0.0);
    parallel_for(static_cast<size_t>(P.E), [&](size_t e) {
        const int2 ed = mesh.edges[e];
        const V3 p0 = to_v3(P.nodes[static_cast<size_t>(ed.x)]);
        const V3 p1 = to_v3(P.nodes[static_cast<size_t>(ed.y)]);
        a_ex[e] = edge_integral(p0, p1, [&](V3 p) { return exact_field(fil, p).A; });
    });
    std::printf("[RowCheck] a_ex: Kantenintegrale auf %d Kanten in %.1f s\n", P.E,
                std::chrono::duration<double>(clk::now() - t0).count());
    std::cout.flush();

    // K1, K2 je Position (Galerkin-Projektion in den Rooftop-Raum)
    t0 = clk::now();
    std::vector<double> k1(static_cast<size_t>(P.nb), 0.0), k2(static_cast<size_t>(P.nb), 0.0);
    std::vector<char> proj_ok(static_cast<size_t>(P.nq), 1);
    parallel_for(static_cast<size_t>(P.nq), [&](size_t pos) {
        const QuadGeom& g = P.geom[pos];
        double c1[4], c2[4];
        auto F1 = [&](V3 p, V3 n) {
            const ExactField ef = exact_field(fil, p);
            V3 dn = normal_derivative_A(ef, n);
            dn = dn - dot(dn, n) * n;                      // tangentialer Anteil
            return (1.0 / kMu0) * dn;
        };
        auto F2 = [&](V3 p, V3 n) {
            const ExactField ef = exact_field(fil, p);
            return (-1.0 / kMu0) * cross(n, ef.B);
        };
        const bool ok1 = project_to_rooftop(g, F1, c1);
        const bool ok2 = project_to_rooftop(g, F2, c2);
        if (!ok1 || !ok2) { proj_ok[pos] = 0; return; }
        for (int i = 0; i < 4; ++i) { k1[pos * 4 + i] = c1[i]; k2[pos * 4 + i] = c2[i]; }
    });
    long n_bad = 0;
    for (char c : proj_ok) if (!c) ++n_bad;
    std::printf("[RowCheck] k1, k2: Projektion auf %d Quads in %.1f s (%ld Quads mit singulaerer Massenmatrix)\n",
                P.nq, std::chrono::duration<double>(clk::now() - t0).count(), n_bad);
    std::cout.flush();

    // -- Operatoren anwenden ----------------------------------------------------------
    DeviceArray<double> d_a(static_cast<size_t>(P.E)), d_k(static_cast<size_t>(P.nb));
    DeviceArray<double> d_TK(static_cast<size_t>(P.E)), d_TA(static_cast<size_t>(P.nb)),
                        d_DA(static_cast<size_t>(P.nb)), d_GK(static_cast<size_t>(P.nb));
    d_a.prefetch_to_host(); d_k.prefetch_to_host(); bf::sync();
    for (int e = 0; e < P.E; ++e) d_a[static_cast<size_t>(e)] = a_ex[static_cast<size_t>(e)];

    std::vector<double> TK[2], Gk[2], TAK, DaK;   // TK[v] = -T k_v (Zeile 1), Gk[v] = mu0 G k_v
    for (int v = 0; v < 2; ++v) {
        const std::vector<double>& kv = (v == 0) ? k1 : k2;
        for (int i = 0; i < P.nb; ++i) d_k[static_cast<size_t>(i)] = kv[static_cast<size_t>(i)];
        d_a.prefetch_to_device(); d_k.prefetch_to_device(); bf::sync();
        op.row_check_parts(d_a.data(), d_k.data(), d_TK.data(), d_TA.data(), d_DA.data(), d_GK.data());
        d_TK.prefetch_to_host(); d_TA.prefetch_to_host(); d_DA.prefetch_to_host(); d_GK.prefetch_to_host();
        bf::sync();
        TK[v].assign(d_TK.data(), d_TK.data() + P.E);
        Gk[v].assign(d_GK.data(), d_GK.data() + P.nb);
        if (v == 0) {
            TAK.assign(d_TA.data(), d_TA.data() + P.nb);
            DaK.assign(d_DA.data(), d_DA.data() + P.nb);
        }
        d_a.prefetch_to_host(); d_k.prefetch_to_host(); bf::sync();
    }

    // -- Zeile 1: K a_ex - b und W a_ex - b auf den Hostmatrizen ----------------------
    std::vector<double> Ka_b(static_cast<size_t>(P.E)), Wa_b(static_cast<size_t>(P.E));
    for (int r = 0; r < P.E; ++r) {
        double w = 0.0, c = 0.0;
        for (int p = csr_W.row_offsets[static_cast<size_t>(r)]; p < csr_W.row_offsets[static_cast<size_t>(r) + 1]; ++p)
            w += csr_W.values[static_cast<size_t>(p)] * a_ex[static_cast<size_t>(csr_W.col_indices[static_cast<size_t>(p)])];
        for (int p = csr_C.row_offsets[static_cast<size_t>(r)]; p < csr_C.row_offsets[static_cast<size_t>(r) + 1]; ++p)
            c += csr_C.values[static_cast<size_t>(p)] * a_ex[static_cast<size_t>(csr_C.col_indices[static_cast<size_t>(p)])];
        Wa_b[static_cast<size_t>(r)] = w - b_phys[static_cast<size_t>(r)];
        Ka_b[static_cast<size_t>(r)] = (w - omega * c) - b_phys[static_cast<size_t>(r)];
    }

    // Kontrolle: innere Zeilen (nicht auf dem Rand) je Koerper, K-Form
    {
        double num[kNumBodies + 1] = {}, den[kNumBodies + 1] = {}, ka2[kNumBodies + 1] = {}, kab[kNumBodies + 1] = {};
        for (int r = 0; r < P.E; ++r) {
            if (P.edge_label[static_cast<size_t>(r)] >= 0) continue;
            const int b = node_body[static_cast<size_t>(mesh.edges[static_cast<size_t>(r)].x)];
            if (b < 1) continue;
            const double bb = b_phys[static_cast<size_t>(r)];
            const double ka = Ka_b[static_cast<size_t>(r)] + bb;     // K a_ex
            num[b] += Ka_b[static_cast<size_t>(r)] * Ka_b[static_cast<size_t>(r)];
            den[b] += bb * bb;
            ka2[b] += ka * ka;
            kab[b] += ka * bb;
        }
        for (int b = 1; b <= kNumBodies; ++b)
            if (den[b] > 0)
                std::printf("[RowCheck] innere Zeilen %-11s ||K a_ex - b|| / ||b|| = %.2f %% "
                            "(Bezug: nur Zeilen ohne BEM; b = 0 in der Platte); ||K a_ex|| / ||b|| = %.4f, "
                            "bester Faktor <K a_ex, b>/<b, b> = %.4f, Kosinus = %.4f\n", kBodyName[b - 1],
                            100.0 * std::sqrt(num[b] / den[b]), std::sqrt(ka2[b] / den[b]),
                            kab[b] / den[b], kab[b] / std::sqrt(ka2[b] * den[b]));
            else
                std::printf("[RowCheck] innere Zeilen %-11s ||K a_ex - b|| = %.3e (||b|| = 0)\n",
                            kBodyName[b - 1], std::sqrt(num[b]));
    }

    // -- T-Kontrolle: Operator gegen Host-Nachrechnung ---------------------------------
    std::vector<double> yA0, yK0, yA1, yK1, yA2, yK2;
    host_T_apply(P, false, false, a_ex, k1, yA0, yK0);   // V0: wie der Operator indexiert
    host_T_apply(P, true, false, a_ex, k1, yA1, yK1);    // V1: Geometrie des Quads perm[pos]
    host_T_apply(P, true, true, a_ex, k1, yA2, yK2);     // V2: V1 plus konsistente Vorzeichen
    std::printf("[RowCheck] T-Kontrolle (Operator gegen Host, k = k1): "
                "Zeile 1 %.2e, Zeile 2 %.2e (Soll ~1e-12: die Nachrechnung gibt den Operator wieder)\n",
                rel_diff(TK[0], yA0), rel_diff(TAK, yK0));
    std::printf("[RowCheck] T-Varianten gegen Operator: V1 (Geometrie perm[pos]) Zeile 1 %.2e, Zeile 2 %.2e; "
                "V2 (zusaetzlich konsistente Vorzeichen) Zeile 1 %.2e, Zeile 2 %.2e\n",
                rel_diff(yA1, TK[0]), rel_diff(yK1, TAK), rel_diff(yA2, TK[0]), rel_diff(yK2, TAK));
    std::cout.flush();

    // -- Residuentabellen ---------------------------------------------------------------
    // Variante: 0 Operator, 1 Geometrie perm[pos], 2 plus konsistente Vorzeichen.
    const char* vname[3] = {"V0: Operator unveraendert", "V1: T-Geometrie in BEM-Reihenfolge (Messung)",
                            "V2: V1 plus konsistente Vorzeichen (Messung)"};
    std::vector<double> TKv[3][2], TAKv[3];
    TKv[0][0] = TK[0]; TKv[0][1] = TK[1]; TAKv[0] = TAK;
    for (int var = 1; var < 3; ++var) {
        for (int v = 0; v < 2; ++v) {
            std::vector<double> yA, yK;
            host_T_apply(P, true, var == 2, a_ex, v == 0 ? k1 : k2, yA, yK);
            TKv[var][v] = yA;
            if (v == 0) TAKv[var] = yK;
        }
    }

    for (int var = 0; var < 3; ++var) {
        std::printf("\n[RowCheck] ---------- %s ----------\n", vname[var]);
        // Zeile 1: r = (K a - b) + s*(-T k_v); Bezug ||K a - b||
        std::vector<Residual> r1[4], r2[4];
        std::vector<double> ref1(static_cast<size_t>(P.E));
        for (int e = 0; e < P.E; ++e) ref1[static_cast<size_t>(e)] = Ka_b[static_cast<size_t>(e)];
        std::vector<double> t2(static_cast<size_t>(P.nb));
        for (int i = 0; i < P.nb; ++i) t2[static_cast<size_t>(i)] = TAKv[var][static_cast<size_t>(i)] + DaK[static_cast<size_t>(i)];
        std::vector<int> lab2(static_cast<size_t>(P.nb));
        for (int i = 0; i < P.nb; ++i) lab2[static_cast<size_t>(i)] = P.pos_label[static_cast<size_t>(i / 4)];
        int col = 0;
        for (int v = 0; v < 2; ++v) {
            for (int sgn = +1; sgn >= -1; sgn -= 2, ++col) {
                r1[col] = by_body_and_class(axpy(Ka_b, static_cast<double>(sgn), TKv[var][v]), ref1, P.edge_label);
                r2[col] = by_body_and_class(axpy(t2, static_cast<double>(sgn), Gk[v]), t2, lab2);
            }
        }
        char title[160];
        std::snprintf(title, sizeof title, "Zeile 1 (FEM auf Randkanten), %s: ||K a - T k - b|| / ||K a - b||", vname[var]);
        print_table(title, r1);
        std::snprintf(title, sizeof title, "Zeile 2 (BEM), %s: ||(-0.5 T - D) a + mu0 G k|| / ||(-0.5 T - D) a||", vname[var]);
        print_table(title, r2);

        if (var == 0) {
            // Bezugsnormen
            std::vector<double> mt(static_cast<size_t>(P.E));
            double nKab = 0, nTk1 = 0, nTk2 = 0, nW = 0;
            for (int e = 0; e < P.E; ++e) {
                if (P.edge_label[static_cast<size_t>(e)] < 0) continue;
                nKab += Ka_b[static_cast<size_t>(e)] * Ka_b[static_cast<size_t>(e)];
                nW += Wa_b[static_cast<size_t>(e)] * Wa_b[static_cast<size_t>(e)];
                nTk1 += TK[0][static_cast<size_t>(e)] * TK[0][static_cast<size_t>(e)];
                nTk2 += TK[1][static_cast<size_t>(e)] * TK[1][static_cast<size_t>(e)];
            }
            std::printf("\n[RowCheck] Normen (V0, nur Randkanten): ||K a - b|| = %.4e, ||T k1|| = %.4e, "
                        "||T k2|| = %.4e, ||W a - b|| = %.4e\n",
                        std::sqrt(nKab), std::sqrt(nTk1), std::sqrt(nTk2), std::sqrt(nW));
            std::printf("[RowCheck]                Zeile 2: ||(-0.5 T - D) a|| = %.4e, ||mu0 G k1|| = %.4e, "
                        "||mu0 G k2|| = %.4e, ||D a|| = %.4e, ||0.5 T a|| = %.4e\n",
                        vnorm(t2), vnorm(Gk[0]), vnorm(Gk[1]), vnorm(DaK), vnorm(TAK));
            // W-Form (omega*C a_ex mitgerechnet) je Koerper
            std::vector<Residual> w1[4];
            int col2 = 0;
            for (int v = 0; v < 2; ++v)
                for (int sgn = +1; sgn >= -1; sgn -= 2, ++col2)
                    w1[col2] = by_body_and_class(axpy(Wa_b, static_cast<double>(sgn), TK[v]), Wa_b, P.edge_label);
            print_table("Zeile 1 mit W statt K (omega*C a_ex mitgerechnet; nur fuer die Platte relevant), V0", w1);
        }
    }

    // -- Referenz-Randterm: Zeile 1 ohne Rooftop-Darstellung und ohne T ---------------------
    // g_v[r] = int W3D_r . K_v dS ueber die Randquads der Kante r, W3D = tangentiale Spur der
    // Nedelec-Basisfunktion des Hexaeders (kovariant: W_0 = 0.25 (1-v) g^u, W_1 = 0.25 (1+u) g^v,
    // W_2 = -0.25 (1+v) g^u, W_3 = -0.25 (1-u) g^v mit der reziproken Basis g^u, g^v der Flaeche),
    // mal sign (lokal -> global). Exakte Felder, 16-Punkt-Gauss je Quad. Stimmt das exakte Paar
    // mit der FEM-Zeile ueberein, ist K a - b = g_v (K1 bzw. K2).
    std::vector<double> gref[2];
    {
        std::vector<double> loc[2];
        loc[0].assign(static_cast<size_t>(P.nb), 0.0);
        loc[1].assign(static_cast<size_t>(P.nb), 0.0);
        parallel_for(static_cast<size_t>(P.nq), [&](size_t pos) {
            const QuadGeom& g = P.geom[pos];
            double x[4], w[4];
            gauss4(x, w);
            for (int ia = 0; ia < 4; ++ia) for (int ib = 0; ib < 4; ++ib) {
                const double u = x[ia], v = x[ib];
                // Tangenten und Normale
                const double h = 1e-7;
                (void)h;
                double detJ; double3 Wdummy[4];
                evaluate_rooftop_basis(u, v, g.n, Wdummy, detJ);
                // dr_du, dr_dv (wie evaluate_rooftop_basis)
                V3 dru, drv;
                {
                    const double3* nd = g.n;
                    dru = {0.25 * (-(1 - v) * nd[0].x + (1 - v) * nd[1].x + (1 + v) * nd[2].x - (1 + v) * nd[3].x),
                           0.25 * (-(1 - v) * nd[0].y + (1 - v) * nd[1].y + (1 + v) * nd[2].y - (1 + v) * nd[3].y),
                           0.25 * (-(1 - v) * nd[0].z + (1 - v) * nd[1].z + (1 + v) * nd[2].z - (1 + v) * nd[3].z)};
                    drv = {0.25 * (-(1 - u) * nd[0].x - (1 + u) * nd[1].x + (1 + u) * nd[2].x + (1 - u) * nd[3].x),
                           0.25 * (-(1 - u) * nd[0].y - (1 + u) * nd[1].y + (1 + u) * nd[2].y + (1 - u) * nd[3].y),
                           0.25 * (-(1 - u) * nd[0].z - (1 + u) * nd[1].z + (1 + u) * nd[2].z + (1 - u) * nd[3].z)};
                }
                const V3 N = cross(dru, drv);
                const double N2 = dot(N, N);
                const V3 gu = (1.0 / N2) * cross(drv, N);
                const V3 gv = (1.0 / N2) * cross(N, dru);
                const V3 p = quad_point(g.n, u, v);
                const V3 n = outward_normal(g, u, v);
                const ExactField ef = exact_field(fil, p);
                V3 dn = normal_derivative_A(ef, n);
                dn = dn - dot(dn, n) * n;
                const V3 K1 = (1.0 / kMu0) * dn;
                const V3 K2 = (-1.0 / kMu0) * cross(n, ef.B);
                const V3 Wc[4] = {0.25 * (1 - v) * gu, 0.25 * (1 + u) * gv, -0.25 * (1 + v) * gu, -0.25 * (1 - u) * gv};
                const double wt = w[ia] * w[ib] * detJ;
                for (int i = 0; i < 4; ++i) {
                    loc[0][pos * 4 + static_cast<size_t>(i)] += wt * dot(Wc[i], K1);
                    loc[1][pos * 4 + static_cast<size_t>(i)] += wt * dot(Wc[i], K2);
                }
            }
        });
        for (int v = 0; v < 2; ++v) {
            gref[v].assign(static_cast<size_t>(P.E), 0.0);
            for (int i = 0; i < P.nb; ++i)
                gref[v][static_cast<size_t>(P.bem_to_fem[static_cast<size_t>(i)])] +=
                    static_cast<double>(P.bem_sign[static_cast<size_t>(i)]) * loc[v][static_cast<size_t>(i)];
        }
        // r = (K a - b) - s * g_v, gleiche Spaltenfolge (+K1, -K1, +K2, -K2)
        std::vector<double> neg(static_cast<size_t>(P.E));
        std::vector<Residual> r3[4];
        int col = 0;
        for (int v = 0; v < 2; ++v) {
            for (int e = 0; e < P.E; ++e) neg[static_cast<size_t>(e)] = -gref[v][static_cast<size_t>(e)];
            for (int sgn = +1; sgn >= -1; sgn -= 2, ++col)
                r3[col] = by_body_and_class(axpy(Ka_b, static_cast<double>(sgn), neg), Ka_b, P.edge_label);
        }
        print_table("Zeile 1 mit exakten Randintegralen (Nedelec-Spur, ohne T und Rooftop): "
                    "||K a - g - b|| / ||K a - b||, g = int W3D . K dS", r3);
        // T-Test: wie gut gibt T_v k_v das exakte Randintegral wieder?
        std::printf("\nT-Test: ||T k - g|| / ||g|| (g = exaktes Randintegral zu K1 bzw. K2), je Variante\n\n");
        std::printf("| Variante | Koerper | Klasse | n | K1 | K2 |\n|---|---|---|---:|---:|---:|\n");
        for (int var = 0; var < 3; ++var) {
            std::vector<Residual> t[2];
            for (int v = 0; v < 2; ++v) {
                std::vector<double> d(static_cast<size_t>(P.E));
                for (int e = 0; e < P.E; ++e) d[static_cast<size_t>(e)] = -TKv[var][v][static_cast<size_t>(e)] - gref[v][static_cast<size_t>(e)];
                t[v] = by_body_and_class(d, gref[v], P.edge_label);
            }
            for (int b = 0; b < kNumBodies; ++b)
                for (int c = 0; c <= kNumClasses; ++c) {
                    const size_t idx = static_cast<size_t>((c < kNumClasses) ? b * kNumClasses + c : kNumLabels + b);
                    if (t[0][idx].n == 0) continue;
                    std::printf("| V%d | %s | %s | %ld | %.2f %% | %.2f %% |\n", var, kBodyName[b],
                                c < kNumClasses ? class_name(c) : "alle", t[0][idx].n,
                                100.0 * t[0][idx].rel(), 100.0 * t[1][idx].rel());
                }
            const size_t all = static_cast<size_t>(kNumLabels + kNumBodies);
            std::printf("| V%d | alle | alle | %ld | %.2f %% | %.2f %% |\n", var, t[0][all].n,
                        100.0 * t[0][all].rel(), 100.0 * t[1][all].rel());
        }
        std::cout.flush();
    }

    // -- Zeile 2 mit konsistenter Rooftop-Darstellung (Messung, kein Operatoreingriff) -----
    // Statt a_ex ueber den Gather des Operators (Rooftop-Koeffizient = Kantenintegral) wird die
    // tangentiale Randspur P_t A_ex ebenso wie K1/K2 per Galerkin in den Rooftop-Raum projiziert
    // (Koeffizienten cA). D_BEM und G_BEM sind die bestehenden Bloecke, T kommt vom Host mit
    // Geometrie perm[pos] und rohen Koeffizienten (V2-Vorzeichen: nur die Testfunktion roh).
    // Trennt Formulierung/Bloecke (hier) von der Darstellung der FEM-Werte im BEM-Raum (oben).
    {
        std::vector<double> cA(static_cast<size_t>(P.nb), 0.0);
        parallel_for(static_cast<size_t>(P.nq), [&](size_t pos) {
            double c[4];
            auto FA = [&](V3 p, V3 n) {
                const V3 A = exact_field(fil, p).A;
                return A - dot(A, n) * n;
            };
            if (project_to_rooftop(P.geom[pos], FA, c))
                for (int i = 0; i < 4; ++i) cA[pos * 4 + static_cast<size_t>(i)] = c[i];
        });
        // T-Term (Zeile 2): -0.5 * T_loc * cA, Geometrie perm[pos], Testfunktion roh
        std::vector<double> TAc(static_cast<size_t>(P.nb), 0.0);
        for (int pos = 0; pos < P.nq; ++pos) {
            const Mesh::QuadFace& f = P.faces[static_cast<size_t>(P.perm[static_cast<size_t>(pos)])];
            double3 nd[4];
            for (int i = 0; i < 4; ++i) nd[i] = P.nodes[static_cast<size_t>(f.v[i])];
            double T[4][4];
            integrate_boundary_mass_matrix(nd, T);
            for (int r = 0; r < 4; ++r) {
                double v = 0.0;
                for (int c = 0; c < 4; ++c) v += T[r][c] * cA[static_cast<size_t>(pos * 4 + c)];
                TAc[static_cast<size_t>(pos * 4 + r)] = -0.5 * v;
            }
        }
        // D_BEM cA und mu0 G k_v ueber den Operator (xbem-Eingabe)
        DeviceArray<double> d_cA(static_cast<size_t>(P.nb));
        d_cA.prefetch_to_host(); bf::sync();
        for (int i = 0; i < P.nb; ++i) d_cA[static_cast<size_t>(i)] = cA[static_cast<size_t>(i)];
        std::vector<double> DcA;
        std::vector<double> Gk2[2];
        for (int v = 0; v < 2; ++v) {
            const std::vector<double>& kv = (v == 0) ? k1 : k2;
            for (int i = 0; i < P.nb; ++i) d_k[static_cast<size_t>(i)] = kv[static_cast<size_t>(i)];
            d_a.prefetch_to_device(); d_k.prefetch_to_device(); d_cA.prefetch_to_device(); bf::sync();
            op.row_check_parts(d_a.data(), d_k.data(), d_TK.data(), d_TA.data(), d_DA.data(), d_GK.data(),
                               d_cA.data());
            d_DA.prefetch_to_host(); d_GK.prefetch_to_host(); bf::sync();
            if (v == 0) DcA.assign(d_DA.data(), d_DA.data() + P.nb);
            Gk2[v].assign(d_GK.data(), d_GK.data() + P.nb);
            d_a.prefetch_to_host(); d_k.prefetch_to_host(); d_cA.prefetch_to_host(); bf::sync();
        }
        std::vector<double> t2(static_cast<size_t>(P.nb));
        for (int i = 0; i < P.nb; ++i) t2[static_cast<size_t>(i)] = TAc[static_cast<size_t>(i)] + DcA[static_cast<size_t>(i)];
        std::vector<int> lab2(static_cast<size_t>(P.nb));
        for (int i = 0; i < P.nb; ++i) lab2[static_cast<size_t>(i)] = P.pos_label[static_cast<size_t>(i / 4)];
        std::vector<Residual> r4[4];
        int col = 0;
        for (int v = 0; v < 2; ++v)
            for (int sgn = +1; sgn >= -1; sgn -= 2, ++col)
                r4[col] = by_body_and_class(axpy(t2, static_cast<double>(sgn), Gk2[v]), t2, lab2);
        print_table("Zeile 2 mit konsistenter Darstellung (Galerkin-Projektion von P_t A_ex und K, T mit "
                    "Geometrie perm[pos]): ||(-0.5 T - D) cA + mu0 G k|| / ||(-0.5 T - D) cA||", r4);
        std::printf("\n[RowCheck] Normen (konsistent): ||(-0.5 T - D) cA|| = %.4e, ||mu0 G k1|| = %.4e, ||mu0 G k2|| = %.4e\n",
                    vnorm(t2), vnorm(Gk2[0]), vnorm(Gk2[1]));
        // Vorzeichen des D-Terms umgekehrt: (-0.5 T + D) cA + mu0 G k. (Die Gleichungen mit
        // umgekehrtem Vorzeichen von 0.5 T sind bis auf das Vorzeichen von k gleich diesen beiden;
        // beide Vorzeichen von k stehen in den Tabellen.)
        std::vector<double> t2f(static_cast<size_t>(P.nb));
        for (int i = 0; i < P.nb; ++i) t2f[static_cast<size_t>(i)] = TAc[static_cast<size_t>(i)] - DcA[static_cast<size_t>(i)];
        std::vector<Residual> r5[4];
        col = 0;
        for (int v = 0; v < 2; ++v)
            for (int sgn = +1; sgn >= -1; sgn -= 2, ++col)
                r5[col] = by_body_and_class(axpy(t2f, static_cast<double>(sgn), Gk2[v]), t2f, lab2);
        print_table("Zeile 2, konsistente Darstellung, Vorzeichen des D-Terms UMGEKEHRT (Messung): "
                    "||(-0.5 T + D) cA + mu0 G k|| / ||(-0.5 T + D) cA||", r5);
        std::printf("\n[RowCheck] Normen (D umgekehrt): ||(-0.5 T + D) cA|| = %.4e\n", vnorm(t2f));
        std::cout.flush();
    }

    // -- k1 gegen k2 ----------------------------------------------------------------------
    {
        std::vector<double> d(static_cast<size_t>(P.nb)), ref(static_cast<size_t>(P.nb));
        for (int i = 0; i < P.nb; ++i) { d[static_cast<size_t>(i)] = k1[static_cast<size_t>(i)] - k2[static_cast<size_t>(i)]; ref[static_cast<size_t>(i)] = k1[static_cast<size_t>(i)]; }
        std::vector<int> lab(static_cast<size_t>(P.nb));
        for (int i = 0; i < P.nb; ++i) lab[static_cast<size_t>(i)] = P.pos_label[static_cast<size_t>(i / 4)];
        const std::vector<Residual> r = by_body_and_class(d, ref, lab);
        std::printf("\n||k1 - k2|| / ||k1|| je Koerper und Quadklasse (Rooftop-Koeffizienten)\n\n");
        std::printf("| Koerper | Klasse | n | ||k1-k2|| / ||k1|| |\n|---|---|---:|---:|\n");
        for (int b = 0; b < kNumBodies; ++b)
            for (int c = 0; c <= kNumClasses; ++c) {
                const size_t idx = static_cast<size_t>((c < kNumClasses) ? b * kNumClasses + c : kNumLabels + b);
                if (r[idx].n == 0) continue;
                std::printf("| %s | %s | %ld | %.4f |\n", kBodyName[b], c < kNumClasses ? class_name(c) : "alle", r[idx].n, r[idx].rel());
            }
    }

    // -- Randspur: Rekonstruktion aus a_ex ueber die Rooftop-Funktionen --------------------
    {
        struct Acc { double e2 = 0, r2 = 0, cross = 0, a2 = 0; long n = 0; };
        std::vector<Acc> acc(static_cast<size_t>(kNumLabels) + 1);
        std::vector<double> qe2(static_cast<size_t>(P.nq)), qr2(static_cast<size_t>(P.nq)), qc(static_cast<size_t>(P.nq)), qa2(static_cast<size_t>(P.nq));
        parallel_for(static_cast<size_t>(P.nq), [&](size_t pos) {
            const QuadGeom& g = P.geom[pos];
            double c[4];
            for (int i = 0; i < 4; ++i)
                c[i] = static_cast<double>(P.bem_sign[pos * 4 + static_cast<size_t>(i)]) *
                       a_ex[static_cast<size_t>(P.bem_to_fem[pos * 4 + static_cast<size_t>(i)])];
            double x[4], w[4];
            gauss4(x, w);
            double e2 = 0, r2 = 0, cr = 0, a2 = 0;
            for (int ia = 0; ia < 4; ++ia) for (int ib = 0; ib < 4; ++ib) {
                double detJ; double3 W[4];
                evaluate_rooftop_basis(x[ia], x[ib], g.n, W, detJ);
                const V3 p = quad_point(g.n, x[ia], x[ib]);
                const V3 n = outward_normal(g, x[ia], x[ib]);
                V3 At = exact_field(fil, p).A;
                At = At - dot(At, n) * n;
                const V3 rec = rooftop_value(g, c, x[ia], x[ib]);
                const double wt = w[ia] * w[ib] * detJ;
                e2 += wt * dot(rec - At, rec - At);
                r2 += wt * dot(rec, rec);
                cr += wt * dot(rec, At);
                a2 += wt * dot(At, At);
            }
            qe2[pos] = e2; qr2[pos] = r2; qc[pos] = cr; qa2[pos] = a2;
        });
        for (int pos = 0; pos < P.nq; ++pos) {
            const int lab = P.pos_label[static_cast<size_t>(pos)];
            if (lab < 0) continue;
            for (size_t k : {static_cast<size_t>(lab), static_cast<size_t>(kNumLabels)}) {
                acc[k].e2 += qe2[static_cast<size_t>(pos)]; acc[k].r2 += qr2[static_cast<size_t>(pos)];
                acc[k].cross += qc[static_cast<size_t>(pos)]; acc[k].a2 += qa2[static_cast<size_t>(pos)];
                ++acc[k].n;
            }
        }
        std::printf("\nRandspur: Rooftop-Rekonstruktion sum_i c_i W_i mit c_i = sign_i * a_ex (Gather des Operators) gegen den "
                    "tangentialen Anteil von A_ex\n\n");
        std::printf("| Koerper | Klasse | Quads | relativer L2-Fehler | Projektion <rec, A_t>/<A_t, A_t> | Betragsverhaeltnis |\n|---|---|---:|---:|---:|---:|\n");
        for (int b = 0; b < kNumBodies; ++b)
            for (int c = 0; c < kNumClasses; ++c) {
                const Acc& a = acc[static_cast<size_t>(b * kNumClasses + c)];
                if (a.n == 0) continue;
                std::printf("| %s | %s | %ld | %.4f | %.4f | %.4f |\n", kBodyName[b], class_name(c), a.n,
                            std::sqrt(a.e2 / a.a2), a.cross / a.a2, std::sqrt(a.r2 / a.a2));
            }
        const Acc& all = acc[static_cast<size_t>(kNumLabels)];
        std::printf("| alle | alle | %ld | %.4f | %.4f | %.4f |\n", all.n, std::sqrt(all.e2 / all.a2),
                    all.cross / all.a2, std::sqrt(all.r2 / all.a2));
    }

    std::printf("\n[RowCheck] fertig nach %.1f s. Operatoren, Matrizen und Loesung unveraendert; "
                "das Loesen entfaellt.\n",
                std::chrono::duration<double>(clk::now() - t_start).count());
    std::cout.flush();
    return 0;
}
