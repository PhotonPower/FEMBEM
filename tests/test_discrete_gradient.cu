// =============================================================================
// test_discrete_gradient.cu -- Grundbausteine des Hiptmair-Vorkonditionierers
// =============================================================================
//
// Geprueft wird an einem kleinen strukturierten Hexaedernetz:
//
//   1. G hat je Zeile genau -1 und +1, Spalten aufsteigend
//   2. G^T G ist der Knoten-Graph-Laplace: Diagonale = Kantengrad,
//      Zeilensumme = 0. Das prueft Aufbau und beide Anwendungsrichtungen.
//   3. Die DEFINIERENDE Eigenschaft: curl(grad) = 0. Wird die Kanten-
//      Steifigkeitsmatrix S (rein curl-curl, ohne Massenterm) mit einem
//      Gradientenfeld multipliziert, muss exakt null herauskommen:
//          S * G * phi = 0  fuer jedes phi
//      Das ist die Grundlage des ganzen Verfahrens -- und zugleich ein
//      scharfer Test der curl-curl-Assemblierung: stimmen Vorzeichen oder
//      Kantenzuordnung nicht, faellt es hier auf.
//   4. A_n = G^T A G stimmt mit der direkten Auswertung phi^T A_n psi
//      = (G phi)^T A (G psi) ueberein.
//   5. Fuer eine reine curl-curl-Matrix ist A_n exakt null -- genau die
//      Schwierigkeit im magnetostatischen Fall.
//
// Braucht keine GPU.
// =============================================================================

#include "DiscreteGradient.cuh"
#include "NedelecDispatch.cuh"   // piola_contravariant; bindet NedelecHexahedron.cuh ein

#include <cmath>
#include <cstdio>
#include <array>
#include <map>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
void ck(bool ok, const char* w) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", w);
    if (!ok) ++g_fail;
}
void ck_lt(double v, double b, const char* w) {
    const bool ok = (v < b);
    std::printf("  [%s] %s  (%.3e < %.3e)\n", ok ? "OK " : "FEHL", w, v, b);
    if (!ok) ++g_fail;
}

// Lokale Kanten wie in NedelecHexahedron.cuh: W[le] ist von a nach b orientiert.
const int LOC[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
                         {0,4},{1,5},{2,6},{3,7} };

struct SmallMesh {
    Mesh mesh;
    std::vector<int> hex_edge;   // 12 je Hexaeder
    std::vector<int> hex_sign;
};

/// n x n x n Wuerfelblock, absichtlich leicht verzerrt.
/// Mesh ist weder kopier- noch verschiebbar (DeviceArray-Member), deshalb wird
/// in ein uebergebenes Objekt gebaut statt eines zurueckzugeben.
void build(int n, SmallMesh& sm) {
    std::map<std::array<int,3>, int> lut;
    std::vector<double3> pts;
    auto nid = [&](int i, int j, int k) {
        std::array<int,3> a{i,j,k};
        auto it = lut.find(a);
        if (it != lut.end()) return it->second;
        const int id = static_cast<int>(pts.size());
        pts.push_back(make_double3(i + 0.11 * std::sin(1.3 * j + k),
                                   j + 0.07 * k,
                                   k + 0.05 * i * j));
        lut[a] = id;
        return id;
    };
    std::vector<Hexahedron> hx;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        Hexahedron h{};
        h.v[0]=nid(i,j,k);     h.v[1]=nid(i+1,j,k);   h.v[2]=nid(i+1,j+1,k);   h.v[3]=nid(i,j+1,k);
        h.v[4]=nid(i,j,k+1);   h.v[5]=nid(i+1,j,k+1); h.v[6]=nid(i+1,j+1,k+1); h.v[7]=nid(i,j+1,k+1);
        hx.push_back(h);
    }

    // Kanten global sammeln, (v_min, v_max)
    std::map<std::pair<int,int>, int> emap;
    for (const auto& h : hx) {
        for (int le = 0; le < 12; ++le) {
            const int a = h.v[LOC[le][0]], b = h.v[LOC[le][1]];
            emap.emplace(std::make_pair(std::min(a,b), std::max(a,b)),
                         static_cast<int>(emap.size()));
        }
    }
    // Neu durchnummerieren, damit die Reihenfolge deterministisch sortiert ist
    int idx = 0;
    for (auto& kv : emap) kv.second = idx++;

    sm.mesh.num_nodes = pts.size();
    sm.mesh.num_hexes = hx.size();
    sm.mesh.num_edges = emap.size();
    sm.mesh.nodes_current.allocate(pts.size());
    sm.mesh.hex_nodes.allocate(hx.size());
    sm.mesh.edges.allocate(emap.size());
    for (size_t q = 0; q < pts.size(); ++q) sm.mesh.nodes_current[q] = pts[q];
    for (size_t q = 0; q < hx.size(); ++q)  sm.mesh.hex_nodes[q] = hx[q];
    for (const auto& kv : emap) sm.mesh.edges[kv.second] = make_int2(kv.first.first, kv.first.second);

    sm.hex_edge.resize(hx.size() * 12);
    sm.hex_sign.resize(hx.size() * 12);
    for (size_t h = 0; h < hx.size(); ++h) {
        for (int le = 0; le < 12; ++le) {
            const int a = hx[h].v[LOC[le][0]], b = hx[h].v[LOC[le][1]];
            sm.hex_edge[h*12+le] = emap.at({std::min(a,b), std::max(a,b)});
            sm.hex_sign[h*12+le] = (a < b) ? 1 : -1;
        }
    }
}

/// Reine curl-curl-Steifigkeit, ohne Massenterm.
HostCsr assemble_curl_curl(const SmallMesh& sm) {
    const int ne = static_cast<int>(sm.mesh.num_edges);
    std::vector<std::map<int,double>> rows(ne);

    const double gp = 1.0 / std::sqrt(3.0);
    for (size_t h = 0; h < sm.mesh.num_hexes; ++h) {
        double3 nd[8];
        for (int i = 0; i < 8; ++i) nd[i] = sm.mesh.nodes_current[sm.mesh.hex_nodes[h].v[i]];

        double Ke[12][12] = {};
        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
            const double xi = (a ? gp : -gp), et = (b ? gp : -gp), ze = (c ? gp : -gp);
            double J[3][3], invJ[3][3], detJ;
            compute_jacobian(nd, xi, et, ze, J, invJ, detJ);
            double3 cw[12];
            evaluate_nedelec_curl(xi, et, ze, cw);
            // kontravariant: rot W_phys = J_std * rot W_ref / detJ. Weil
            // compute_jacobian() J transponiert ablegt (J = J_std^T), ist das
            // die SPALTENweise Form ueber J -- piola_contravariant, wie in der
            // produktiven Assemblierung. Hier stand bis 31.10.2026 die
            // zeilenweise Form; fuer die Nullraumpruefung ist das gleichgueltig
            // (rot grad = 0 gilt schon im Referenzelement), fuer jede
            // Aussage ueber den Operator selbst nicht.
            double3 rc[12];
            for (int i = 0; i < 12; ++i) rc[i] = piola_contravariant(J, detJ, cw[i]);
            for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
                Ke[i][j] += detJ * (rc[i].x*rc[j].x + rc[i].y*rc[j].y + rc[i].z*rc[j].z);
            }
        }
        for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
            const int gi = sm.hex_edge[h*12+i], gj = sm.hex_edge[h*12+j];
            const double s = sm.hex_sign[h*12+i] * sm.hex_sign[h*12+j];
            rows[gi][gj] += s * Ke[i][j];
        }
    }

    HostCsr A;
    A.num_rows = ne; A.num_cols = ne;
    A.row_offsets.resize(ne + 1);
    for (int i = 0; i < ne; ++i) {
        A.row_offsets[i] = static_cast<int>(A.col_indices.size());
        for (const auto& kv : rows[i]) {
            A.col_indices.push_back(kv.first);
            A.values.push_back(kv.second);
        }
    }
    A.row_offsets[ne] = static_cast<int>(A.col_indices.size());
    return A;
}

} // namespace

int main() {
    bf::g_backend = bf::Backend::CPU;
    std::puts("========================================================");
    std::puts(" Diskreter Gradient und Hilfsraum (Hiptmair)             ");
    std::puts("========================================================");

    SmallMesh sm;
    build(3, sm);
    const int nn = static_cast<int>(sm.mesh.num_nodes);
    const int ne = static_cast<int>(sm.mesh.num_edges);
    std::printf("\nTestnetz: %d Knoten, %zu Hexaeder, %d Kanten\n",
                nn, sm.mesh.num_hexes, ne);

    HostCsr G = build_discrete_gradient(sm.mesh);

    // ---- 1. Struktur
    std::puts("\n[1] Struktur des diskreten Gradienten");
    {
        bool ok = (G.num_rows == ne && G.num_cols == nn && G.nnz() == 2 * ne);
        bool signs_ok = true, sorted_ok = true;
        for (int e = 0; e < ne && ok; ++e) {
            const int k0 = G.row_offsets[e];
            if (G.row_offsets[e+1] - k0 != 2) { ok = false; break; }
            double s = G.values[k0] + G.values[k0+1];
            if (std::fabs(s) > 1e-15) signs_ok = false;
            if (G.col_indices[k0] >= G.col_indices[k0+1]) sorted_ok = false;
        }
        ck(ok, "Dimensionen und genau 2 Eintraege je Zeile");
        ck(signs_ok, "je Zeile -1 und +1 (Summe null)");
        ck(sorted_ok, "Spaltenindizes aufsteigend");
    }

    // ---- 2. G^T G ist der Graph-Laplace
    std::puts("\n[2] G^T G ist der Knoten-Graph-Laplace");
    {
        std::vector<double> ones(nn, 1.0), ge(ne), gtg(nn);
        apply_gradient(G, ones.data(), ge.data());
        double mx = 0.0;
        for (double v : ge) mx = std::fmax(mx, std::fabs(v));
        ck_lt(mx, 1e-15, "G * (konstant) = 0, also grad(const) = 0");

        std::mt19937 rng(7);
        std::uniform_real_distribution<double> uni(-1, 1);
        std::vector<double> phi(nn);
        for (auto& v : phi) v = uni(rng);
        apply_gradient(G, phi.data(), ge.data());
        apply_gradient_transpose(G, ge.data(), gtg.data(), nn);
        double s = 0.0;
        for (double v : gtg) s += v;
        ck_lt(std::fabs(s), 1e-10, "Zeilensumme von G^T G ist null");
    }

    // ---- 3. Die definierende Eigenschaft: curl(grad) = 0
    std::puts("\n[3] S * G * phi = 0   (curl von grad verschwindet)");
    std::puts("    Grundlage des Verfahrens UND scharfer Test der curl-curl-Assemblierung.");
    HostCsr S = assemble_curl_curl(sm);
    {
        std::mt19937 rng(1234);
        std::uniform_real_distribution<double> uni(-1, 1);
        std::vector<double> phi(nn), gphi(ne), sgphi(ne);
        for (auto& v : phi) v = uni(rng);
        apply_gradient(G, phi.data(), gphi.data());
        apply_host_csr(S, gphi.data(), sgphi.data());

        double n_sg = 0.0, n_g = 0.0, n_s = 0.0;
        for (double v : sgphi) n_sg += v * v;
        for (double v : gphi)  n_g  += v * v;
        for (double v : S.values) n_s = std::fmax(n_s, std::fabs(v));
        n_sg = std::sqrt(n_sg); n_g = std::sqrt(n_g);
        std::printf("      ||S G phi|| = %.3e,  ||G phi|| = %.3e,  max|S| = %.3e\n",
                    n_sg, n_g, n_s);
        ck_lt(n_sg / (n_g * n_s), 1e-12, "relativ verschwindend");
    }

    // ---- 4. A_n = G^T A G, gegen die direkte Auswertung
    std::puts("\n[4] A_n = G^T A G gegen phi^T A_n psi = (G phi)^T A (G psi)");
    {
        double smax = 0.0;
        for (double v : S.values) smax = std::fmax(smax, std::fabs(v));
        int tiny = 0; double anmax = 0.0;
        // Bezugsgroesse ist max|S|, NICHT das eigene Maximum von A_n: bei
        // reinem curl-curl ist A_n vollstaendig null, sein Maximum also nur
        // Rundungsrauschen, und eine Normierung darauf waere bedeutungslos.
        HostCsr An = build_auxiliary_nodal_matrix(S, sm.mesh, &tiny, smax, 1e-12, &anmax);
        std::mt19937 rng(99);
        std::uniform_real_distribution<double> uni(-1, 1);
        std::vector<double> phi(nn), psi(nn), t1(ne), t2(ne), t3(ne), t4(nn);
        for (auto& v : phi) v = uni(rng);
        for (auto& v : psi) v = uni(rng);

        apply_host_csr(An, psi.data(), t4.data());
        double lhs = 0.0;
        for (int i = 0; i < nn; ++i) lhs += phi[i] * t4[i];

        apply_gradient(G, phi.data(), t1.data());
        apply_gradient(G, psi.data(), t2.data());
        apply_host_csr(S, t2.data(), t3.data());
        double rhs = 0.0;
        for (int e = 0; e < ne; ++e) rhs += t1[e] * t3[e];

        std::printf("      phi^T A_n psi = %.6e,  (G phi)^T S (G psi) = %.6e\n", lhs, rhs);
        ck_lt(std::fabs(lhs - rhs) / std::fmax(1.0, std::fabs(rhs)), 1e-12,
              "Dreifachprodukt stimmt");

        // ---- 5. Bei reinem curl-curl ist A_n exakt null
        std::puts("\n[5] Reine curl-curl-Matrix: A_n muss EXAKT null sein");
        std::puts("    Das ist die Schwierigkeit im magnetostatischen Fall -- der");
        std::puts("    Hilfsraum traegt dort nichts und braucht eigene Behandlung.");
        std::printf("      max|A_n| = %.3e,  max|S| = %.3e,  Verhaeltnis %.3e\n",
                    anmax, smax, anmax / smax);
        std::printf("      Zeilen mit verschwindender Diagonale: %d von %d\n", tiny, nn);
        ck_lt(anmax / smax, 1e-12, "A_n ist identisch null (relativ zu max|S|)");
        ck(tiny == nn, "alle Diagonaleintraege als verschwindend erkannt");
    }


    // ---- 6. Vektorieller Hilfsraum Pi
    std::puts("\n[6] Interpolationsoperator Pi (voller AMS)");
    HostCsr Pi = build_nodal_vector_interpolation(sm.mesh);
    {
        bool ok = (Pi.num_rows == ne && Pi.num_cols == 3 * nn && Pi.nnz() == 6 * ne);
        bool sorted_ok = true;
        for (int e = 0; e < ne; ++e) {
            for (int k = Pi.row_offsets[e]; k + 1 < Pi.row_offsets[e+1]; ++k) {
                if (Pi.col_indices[k] >= Pi.col_indices[k+1]) sorted_ok = false;
            }
        }
        ck(ok, "Dimensionen und 6 Eintraege je Zeile");
        ck(sorted_ok, "Spaltenindizes aufsteigend");
    }

    // ---- 7. Pi auf ein KONSTANTES Feld: curl(const) = 0, also S Pi v = 0
    std::puts("\n[7] S * Pi * (konstantes Vektorfeld) = 0");
    std::puts("    Gegenstueck zu S*G*phi = 0 und ebenso scharf: die Trapezregel in Pi");
    std::puts("    liefert fuer ein konstantes Feld exakt die Nedelec-Freiheitsgrade,");
    std::puts("    und ein konstantes Feld hat verschwindende Rotation.");
    {
        std::vector<double> v(3 * nn), pv(ne), spv(ne);
        const double c0[3] = { 0.83, -1.27, 0.41 };
        for (int n = 0; n < nn; ++n) for (int c = 0; c < 3; ++c) v[3*n+c] = c0[c];
        apply_host_csr(Pi, v.data(), pv.data());

        // Gegenprobe: (Pi v)_e muss v . (x_n2 - x_n1) sein
        double emax = 0.0;
        for (int e = 0; e < ne; ++e) {
            const int2 ed = sm.mesh.edges[e];
            const double3 p1 = sm.mesh.nodes_current[ed.x], p2 = sm.mesh.nodes_current[ed.y];
            const double exact = c0[0]*(p2.x-p1.x) + c0[1]*(p2.y-p1.y) + c0[2]*(p2.z-p1.z);
            emax = std::fmax(emax, std::fabs(pv[e] - exact));
        }
        ck_lt(emax, 1e-14, "Pi v = v . (x_n2 - x_n1) fuer konstantes v");

        apply_host_csr(S, pv.data(), spv.data());
        double n_sp = 0.0, n_p = 0.0, smax = 0.0;
        for (double x : spv) n_sp += x * x;
        for (double x : pv)  n_p  += x * x;
        for (double x : S.values) smax = std::fmax(smax, std::fabs(x));
        n_sp = std::sqrt(n_sp); n_p = std::sqrt(n_p);
        std::printf("      ||S Pi v|| = %.3e,  ||Pi v|| = %.3e\n", n_sp, n_p);
        ck_lt(n_sp / (n_p * smax), 1e-12, "relativ verschwindend");
    }

    // ---- 8. A_Pi = Pi^T A Pi, und der entscheidende Unterschied zu A_n
    std::puts("\n[8] A_Pi = Pi^T A Pi -- und warum der vektorielle Hilfsraum noetig ist");
    {
        auto n2e = build_node_to_edges(sm.mesh);
        double apmax = 0.0;
        HostCsr Ap = build_auxiliary_vector_matrix(S, Pi, sm.mesh, n2e, &apmax);

        std::mt19937 rng(4242);
        std::uniform_real_distribution<double> uni(-1, 1);
        std::vector<double> u(3*nn), w(3*nn), t1(ne), t2(ne), t3(ne), t4(3*nn);
        for (auto& x : u) x = uni(rng);
        for (auto& x : w) x = uni(rng);

        apply_host_csr(Ap, w.data(), t4.data());
        double lhs = 0.0;
        for (int i = 0; i < 3*nn; ++i) lhs += u[i] * t4[i];

        apply_host_csr(Pi, u.data(), t1.data());
        apply_host_csr(Pi, w.data(), t2.data());
        apply_host_csr(S, t2.data(), t3.data());
        double rhs = 0.0;
        for (int e = 0; e < ne; ++e) rhs += t1[e] * t3[e];

        std::printf("      u^T A_Pi w = %.6e,  (Pi u)^T S (Pi w) = %.6e\n", lhs, rhs);
        ck_lt(std::fabs(lhs - rhs) / std::fmax(1.0, std::fabs(rhs)), 1e-12,
              "Dreifachprodukt stimmt");

        double smax = 0.0;
        for (double x : S.values) smax = std::fmax(smax, std::fabs(x));
        std::printf("      max|A_Pi| = %.3e,  max|S| = %.3e,  Verhaeltnis %.3e\n",
                    apmax, smax, apmax / smax);
        std::puts("      Zum Vergleich: max|A_n|/max|S| war 6.5e-16, also identisch null.");
        ck(apmax / smax > 1e-3,
           "A_Pi ist bei reinem curl-curl NICHT null -- genau das leistet der "
           "vektorielle Hilfsraum");
    }

    std::puts("\n========================================================");
    if (g_fail == 0) { std::puts(" ALLE PRUEFUNGEN BESTANDEN"); std::puts("========================================================"); return 0; }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
