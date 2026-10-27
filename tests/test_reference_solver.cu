// =============================================================================
// test_reference_solver.cu -- Abnahmepruefung der Bausteine des Referenzloesers
// =============================================================================
//
// Geprueft werden die Teile, die neu geschrieben sind und in denen ein Fehler
// tatsaechlich moeglich ist:
//
//   1. Q1-Formfunktionen: Zerlegung der Eins, Gradientensumme null,
//      Reproduktion linearer Felder, detJ > 0
//   2. Duffy-Transformation fuer das singulaere Einfachschichtintegral gegen
//      den analytischen Wert Int_[0,1]^2 dA/r = 2*ln(1+sqrt(2))
//   3. Randmassenmatrix: Zeilensumme = Flaeche
//   4. Randintegralgleichungen auf einer GESCHLOSSENEN Flaeche
//        innen:  (1/2 I + K) u = V q
//        aussen: V q = (K - 1/2 I) u
//      auf der KUGEL (glatt) mit Konvergenzordnung, und auf dem WUERFEL, um
//      die Kantenbeschraenkung zu dokumentieren statt sie zu verschweigen.
//   5. Dichte LU: A x = b gegen eine bekannte Loesung
//
// Braucht keine GPU (setzt bf::g_backend selbst auf CPU).
// =============================================================================

#include "Backend.cuh"
#include "DenseLinalg.cuh"
#include "NodalBem.cuh"
#include "NodalHexahedron.cuh"
#include "NodalMaxwell.cuh"

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", what);
    if (!ok) ++g_fail;
}
void check_lt(double value, double bound, const char* what) {
    const bool ok = (value < bound);
    std::printf("  [%s] %s  (%.3e < %.3e)\n", ok ? "OK " : "FEHL", what, value, bound);
    if (!ok) ++g_fail;
}

// -----------------------------------------------------------------------------
// 1. Q1-Formfunktionen
// -----------------------------------------------------------------------------
void test_shape_functions() {
    std::puts("\n[1] Q1-Formfunktionen (NodalHexahedron.cuh)");

    // Verzerrter Hexaeder, damit die Pruefungen nicht nur den achsparallelen
    // Sonderfall treffen.
    double3 nd[8] = {
        {0.0, 0.0, 0.0}, {1.1, 0.0, 0.0}, {1.0, 0.9, 0.1}, {0.1, 1.0, 0.0},
        {0.0, 0.1, 1.0}, {1.0, 0.0, 1.1}, {1.2, 1.1, 1.0}, {0.0, 1.0, 1.0}
    };

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> uni(-0.95, 0.95);

    double max_pu = 0.0, max_gsum = 0.0, max_lin = 0.0, min_det = 1e30;
    for (int t = 0; t < 200; ++t) {
        const double xi = uni(rng), eta = uni(rng), zeta = uni(rng);

        double N[8];
        evaluate_nodal_shape(xi, eta, zeta, N);
        double s = 0.0;
        for (int i = 0; i < 8; ++i) s += N[i];
        max_pu = std::fmax(max_pu, std::fabs(s - 1.0));

        double3 g[8]; double detJ;
        evaluate_nodal_gradients(nd, xi, eta, zeta, g, detJ);
        min_det = std::fmin(min_det, detJ);

        // Summe der Gradienten muss verschwinden (Folge der Zerlegung der Eins)
        double3 gs = make_double3(0, 0, 0);
        for (int i = 0; i < 8; ++i) { gs.x += g[i].x; gs.y += g[i].y; gs.z += g[i].z; }
        max_gsum = std::fmax(max_gsum, norm3(gs));

        // Ein LINEARES Feld f = a + b.x muss exakt reproduziert werden, und
        // sein Gradient exakt b. Das ist der eigentliche Patch-Test: er prueft
        // Formfunktionen und Jacobi-Inverse gemeinsam.
        const double3 b = make_double3(0.37, -1.2, 0.8);
        double3 grad = make_double3(0, 0, 0);
        for (int i = 0; i < 8; ++i) {
            const double f = b.x * nd[i].x + b.y * nd[i].y + b.z * nd[i].z;
            grad.x += f * g[i].x; grad.y += f * g[i].y; grad.z += f * g[i].z;
        }
        max_lin = std::fmax(max_lin, norm3(sub3(grad, b)));
    }

    check_lt(max_pu,   1e-14, "Zerlegung der Eins: sum N_i = 1");
    check_lt(max_gsum, 1e-12, "sum grad N_i = 0");
    check_lt(max_lin,  1e-12, "linearer Patch-Test: grad(b.x) = b exakt");
    check(min_det > 0.0, "detJ > 0 im gesamten Element");
    std::printf("      kleinstes detJ = %.6f\n", min_det);
}

// -----------------------------------------------------------------------------
// 2. Singulaeres Einfachschichtintegral
// -----------------------------------------------------------------------------
void test_duffy_singular() {
    std::puts("\n[2] Duffy-Transformation, singulaeres Int G dA");

    double3 nd[4] = { {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0} };
    double V[4], K[4];
    nodal_bem_panel_singular(nd, 0, 1.0, V, K);

    const double sum   = V[0] + V[1] + V[2] + V[3];
    const double exact = 2.0 * std::log(1.0 + std::sqrt(2.0)) / (4.0 * M_PI);
    check_lt(std::fabs(sum - exact) / exact, 1e-5,
             "Int_[0,1]^2 G dA gegen 2*ln(1+sqrt(2))/(4pi)");
    std::printf("      erhalten %.12f, analytisch %.12f\n", sum, exact);

    // Auf einem EBENEN Panel ist (x-y).n identisch null -> K-Beitrag exakt 0.
    double maxK = 0.0;
    for (int j = 0; j < 4; ++j) maxK = std::fmax(maxK, std::fabs(K[j]));
    check_lt(maxK, 1e-14, "K-Beitrag des ebenen singulaeren Panels ist null");

    // Symmetrie: bei Singularitaet in Ecke 0 muessen die beiden Nachbarecken
    // denselben Wert liefern.
    check_lt(std::fabs(V[1] - V[3]) / V[1], 1e-12, "Symmetrie V_1 == V_3");
}

// -----------------------------------------------------------------------------
// Geschlossene Testflaeche: Wuerfel [-1,1]^3 oder darauf projizierte Kugel
// -----------------------------------------------------------------------------
struct Surface {
    std::vector<double3> nodes;
    std::vector<std::array<int, 4>> panels;
    bool is_sphere = false;
};

Surface make_surface(int n, bool sphere) {
    Surface s; s.is_sphere = sphere;
    std::map<std::array<long long, 3>, int> lut;
    auto add = [&](double3 p) {
        auto q = [](double v) { return static_cast<long long>(std::llround(v * 1e7)); };
        std::array<long long, 3> k{ q(p.x), q(p.y), q(p.z) };
        auto it = lut.find(k);
        if (it != lut.end()) return it->second;
        const int id = static_cast<int>(s.nodes.size());
        s.nodes.push_back(p); lut[k] = id; return id;
    };

    for (int axis = 0; axis < 3; ++axis) {
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const double a0 = -1.0 + 2.0 * i / n, a1 = -1.0 + 2.0 * (i + 1) / n;
                    const double b0 = -1.0 + 2.0 * j / n, b1 = -1.0 + 2.0 * (j + 1) / n;
                    const double aa[4] = { a0, a1, a1, a0 };
                    const double bb[4] = { b0, b0, b1, b1 };
                    std::array<int, 4> P{};
                    for (int k = 0; k < 4; ++k) {
                        // Umlaufsinn bei sgn < 0 spiegeln, damit cross(x_u,x_v)
                        // nach aussen zeigt.
                        const int kk = (sgn > 0) ? k : 3 - k;
                        double3 p{};
                        if (axis == 0) p = make_double3(sgn * 1.0, aa[kk], bb[kk]);
                        if (axis == 1) p = make_double3(bb[kk], sgn * 1.0, aa[kk]);
                        if (axis == 2) p = make_double3(aa[kk], bb[kk], sgn * 1.0);
                        if (sphere) {
                            const double L = norm3(p);
                            p = make_double3(p.x / L, p.y / L, p.z / L);
                        }
                        P[k] = add(p);
                    }
                    s.panels.push_back(P);
                }
            }
        }
    }
    return s;
}

/// Gemittelte Knotennormale; auf der Kugel die exakte.
double3 node_normal(const Surface& s, int i) {
    if (s.is_sphere) {
        const double3 p = s.nodes[i];
        return scale3(1.0 / norm3(p), p);
    }
    double3 acc = make_double3(0, 0, 0);
    for (const auto& P : s.panels) {
        bool has = false;
        for (int k = 0; k < 4; ++k) if (P[k] == i) has = true;
        if (!has) continue;
        double3 nd[4];
        for (int k = 0; k < 4; ++k) nd[k] = s.nodes[P[k]];
        double3 y, nr; double jc;
        bem_quad_geometry(nd, 0.0, 0.0, y, nr, jc);
        acc = add3(acc, scale3(1.0 / jc, nr));
    }
    return scale3(1.0 / norm3(acc), acc);
}

void assemble_VK(const Surface& s, std::vector<double>& V, std::vector<double>& K) {
    const int N = static_cast<int>(s.nodes.size());
    const int NP = static_cast<int>(s.panels.size());
    V.assign(static_cast<size_t>(N) * N, 0.0);
    K.assign(static_cast<size_t>(N) * N, 0.0);

    for (int i = 0; i < N; ++i) {
        const double3 x = s.nodes[i];
        for (int p = 0; p < NP; ++p) {
            double3 nd[4];
            for (int k = 0; k < 4; ++k) nd[k] = s.nodes[s.panels[p][k]];
            int corner = -1;
            for (int k = 0; k < 4; ++k) if (s.panels[p][k] == i) corner = k;

            double lV[4], lK[4];
            if (corner >= 0) {
                nodal_bem_panel_singular(nd, corner, 1.0, lV, lK);
            } else {
                const double3 c = nodal_bem_panel_center(nd);
                const double  d = norm3(sub3(x, c));
                const double  h = nodal_bem_panel_size(nd);
                nodal_bem_panel_regular(x, nd, 1.0, nodal_bem_subdivisions(d, h), lV, lK);
            }
            for (int k = 0; k < 4; ++k) {
                V[static_cast<size_t>(i) * N + s.panels[p][k]] += lV[k];
                K[static_cast<size_t>(i) * N + s.panels[p][k]] += lK[k];
            }
        }
        // Diagonale aus der Zeilensumme (1/2 I + K)*1 = 0
        double off = 0.0;
        for (int j = 0; j < N; ++j) if (j != i) off += K[static_cast<size_t>(i) * N + j];
        K[static_cast<size_t>(i) * N + i] = -0.5 - off;
    }
}

double rel_l2(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = a[i] - b[i];
        num += d * d; den += b[i] * b[i];
    }
    return (den > 0.0) ? std::sqrt(num / den) : 0.0;
}

/// Residuum der Innen- und Aussengleichung.
void bie_residuals(const Surface& s, double& err_in, double& err_out) {
    const int N = static_cast<int>(s.nodes.size());
    std::vector<double> V, K;
    assemble_VK(s, V, K);

    // ---- innen: u = x (harmonisch), q = n_x
    {
        std::vector<double> u(N), q(N), lhs(N, 0.0), rhs(N, 0.0);
        for (int i = 0; i < N; ++i) {
            u[i] = s.nodes[i].x;
            q[i] = node_normal(s, i).x;
        }
        for (int i = 0; i < N; ++i) {
            double a = 0.5 * u[i], b = 0.0;
            for (int j = 0; j < N; ++j) {
                a += K[static_cast<size_t>(i) * N + j] * u[j];
                b += V[static_cast<size_t>(i) * N + j] * q[j];
            }
            lhs[i] = a; rhs[i] = b;
        }
        err_in = rel_l2(lhs, rhs);
    }

    // ---- aussen: u = 1/(4 pi r), Punktquelle im Ursprung (im Inneren)
    {
        std::vector<double> u(N), q(N), lhs(N, 0.0), rhs(N, 0.0);
        for (int i = 0; i < N; ++i) {
            const double3 p = s.nodes[i];
            const double  r = norm3(p);
            u[i] = 1.0 / (4.0 * M_PI * r);
            const double3 nh = node_normal(s, i);
            q[i] = -dot3(p, nh) / (4.0 * M_PI * r * r * r);
        }
        for (int i = 0; i < N; ++i) {
            double a = 0.0, b = -0.5 * u[i];
            for (int j = 0; j < N; ++j) {
                a += V[static_cast<size_t>(i) * N + j] * q[j];
                b += K[static_cast<size_t>(i) * N + j] * u[j];
            }
            lhs[i] = a; rhs[i] = b;
        }
        err_out = rel_l2(lhs, rhs);
    }
}

// -----------------------------------------------------------------------------
// 3. Randmassenmatrix
// -----------------------------------------------------------------------------
void test_boundary_mass() {
    std::puts("\n[3] Randmassenmatrix: Gesamtsumme = Flaeche");

    const Surface s = make_surface(4, false);   // Wuerfel [-1,1]^3, Flaeche 24
    const int N = static_cast<int>(s.nodes.size());
    std::vector<double> T(static_cast<size_t>(N) * N, 0.0);

    for (const auto& P : s.panels) {
        double3 nd[4];
        for (int k = 0; k < 4; ++k) nd[k] = s.nodes[P[k]];
        const int n = bem_gl_n();
        for (int gu = 0; gu < n; ++gu) {
            for (int gv = 0; gv < n; ++gv) {
                const double u = bem_gl_x(gu), v = bem_gl_x(gv);
                const double w = bem_gl_w(gu) * bem_gl_w(gv);
                double3 y, nr; double jac;
                bem_quad_geometry(nd, u, v, y, nr, jac);
                double Nn[4];
                bem_quad_shape(u, v, Nn);
                for (int a = 0; a < 4; ++a)
                    for (int b = 0; b < 4; ++b)
                        T[static_cast<size_t>(P[a]) * N + P[b]] += w * jac * Nn[a] * Nn[b];
            }
        }
    }

    double total = 0.0;
    for (double v : T) total += v;
    // sum_ij Int N_i N_j = Int (sum N_i)(sum N_j) = Int 1 = Flaeche
    check_lt(std::fabs(total - 24.0) / 24.0, 1e-12,
             "sum_ij T_ij = Wuerfelflaeche 24");
    std::printf("      erhalten %.12f\n", total);
}

// -----------------------------------------------------------------------------
// 4. Randintegralgleichungen
// -----------------------------------------------------------------------------
void test_bie_sphere() {
    std::puts("\n[4a] Randintegralgleichungen auf der KUGEL (glatt)");
    std::puts("     Erwartung: O(h^2) in beiden Gleichungen.");

    const int    ns[3] = { 4, 8, 12 };
    double e_in[3], e_out[3];
    for (int t = 0; t < 3; ++t) {
        const Surface s = make_surface(ns[t], true);
        bie_residuals(s, e_in[t], e_out[t]);
        std::printf("      %2dx%2d Panels/Seite (%4d Knoten):  innen %.3e   aussen %.3e\n",
                    ns[t], ns[t], static_cast<int>(s.nodes.size()), e_in[t], e_out[t]);
    }

    check_lt(e_in[2],  5e-3, "innen  (1/2 I + K)u = Vq, feinstes Netz");
    check_lt(e_out[2], 5e-3, "aussen Vq = (K - 1/2 I)u, feinstes Netz");

    // Konvergenzordnung zwischen 8x8 und 12x12 (h-Verhaeltnis 1.5)
    const double r_in  = std::log(e_in[1]  / e_in[2])  / std::log(1.5);
    const double r_out = std::log(e_out[1] / e_out[2]) / std::log(1.5);
    std::printf("      gemessene Ordnung: innen %.2f, aussen %.2f\n", r_in, r_out);
    check(r_in > 1.5 && r_out > 1.5, "Konvergenzordnung > 1.5 (erwartet ~2)");
}

void test_bie_cube_documents_limit() {
    std::puts("\n[4b] Dieselben Gleichungen am WUERFEL -- dokumentiert die Grenze");
    std::puts("     An geometrischen Kanten ist die Neumann-Spur q UNSTETIG. Eine");
    std::puts("     stetige Knotenbasis kann das nicht darstellen; der Fehler bleibt");
    std::puts("     deshalb bei ~10 %% und die Ordnung faellt auf etwa O(h^0.75).");
    std::puts("     Das ist keine Fehlfunktion, sondern die Grenze der Formulierung.");
    std::puts("     Platte und Spulen bei TEAM 28 haben scharfe Kanten -- dieser");
    std::puts("     Fehler ist im Referenzlauf vorhanden und wirkt auf die Kraft.");
    std::puts("     Abhilfe waere unstetiges q (Doppelknoten), siehe ReferenceSolver.cuh.");

    const int ns[2] = { 4, 6 };
    double e_in[2], e_out[2];
    for (int t = 0; t < 2; ++t) {
        const Surface s = make_surface(ns[t], false);
        bie_residuals(s, e_in[t], e_out[t]);
        std::printf("      %2dx%2d Panels/Seite (%4d Knoten):  innen %.3e   aussen %.3e\n",
                    ns[t], ns[t], static_cast<int>(s.nodes.size()), e_in[t], e_out[t]);
    }

    // Die Pruefung ist bewusst schwach: sie soll nur festhalten, dass der
    // Fehler UEBERHAUPT faellt (die Operatoren also stimmen) und nicht
    // davonlaeuft. Eine scharfe Schranke waere hier eine Luege.
    check(e_in[1]  < e_in[0],  "innen:  Fehler faellt bei Verfeinerung");
    check(e_out[1] < e_out[0], "aussen: Fehler faellt bei Verfeinerung");
    check_lt(e_in[1],  0.2, "innen:  Fehler bleibt unter 20 %");
    check_lt(e_out[1], 0.2, "aussen: Fehler bleibt unter 20 %");
}

// -----------------------------------------------------------------------------
// 5. Dichte LU
// -----------------------------------------------------------------------------
void test_dense_lu() {
    std::puts("\n[5] Dichte LU (bf::DenseLU, CPU-Pfad)");

    const int n = 60;
    std::vector<double> A(static_cast<size_t>(n) * n), x_ref(n), b(n, 0.0);
    std::mt19937 rng(31337);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    // Column-major, diagonaldominant damit die Pivotisierung nicht das Thema ist
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[static_cast<size_t>(j) * n + i] = uni(rng) + ((i == j) ? 2.0 * n : 0.0);
    for (int i = 0; i < n; ++i) x_ref[i] = uni(rng);

    for (int i = 0; i < n; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += A[static_cast<size_t>(j) * n + i] * x_ref[j];
        b[i] = s;
    }

    std::vector<double> Afac = A, xs = b;
    bf::DenseLU lu;
    lu.factorize(Afac.data(), n);
    lu.solve(xs.data(), 1);

    double e = 0.0, nrm = 0.0;
    for (int i = 0; i < n; ++i) { e += (xs[i]-x_ref[i])*(xs[i]-x_ref[i]); nrm += x_ref[i]*x_ref[i]; }
    check_lt(std::sqrt(e / nrm), 1e-12, "A x = b gegen bekannte Loesung");

    // Mehrere rechte Seiten gleichzeitig -- so wird V^-1 C gebildet
    std::vector<double> B(static_cast<size_t>(n) * 3);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < n; ++i)
            B[static_cast<size_t>(c) * n + i] = b[i] * (c + 1);
    lu.solve(B.data(), 3);
    double e3 = 0.0;
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < n; ++i)
            e3 = std::fmax(e3, std::fabs(B[static_cast<size_t>(c)*n+i] - (c+1)*x_ref[i]));
    check_lt(e3, 1e-10, "mehrere rechte Seiten (wie bei D = V^-1 C)");
}


// -----------------------------------------------------------------------------
// 6. Maxwell-Spannungstensor fuer Knotenfelder
// -----------------------------------------------------------------------------
//
// Geprueft wird gegen drei analytisch bekannte Faelle. Alle drei benutzen ein
// UNIFORMES B, das durch ein in x lineares A erzeugt wird -- Q1 interpoliert
// lineare Felder exakt, curl A ist also exakt uniform und der Vergleich hat
// keinen Diskretisierungsfehler.
//
//   (a) B rein NORMAL zur Flaeche:      T.n = + |B|^2/(2 mu_0) * n   (Zug nach aussen)
//   (b) B rein TANGENTIAL zur Flaeche:  T.n = - |B|^2/(2 mu_0) * n   (Zug nach innen)
//   (c) geschlossene Flaeche, uniformes B:  Gesamtkraft = 0
//
// (a) und (b) haben ENTGEGENGESETZTES Vorzeichen -- damit faellt ein
// Vorzeichenfehler in der Normale auf, was (c) allein nicht leisten wuerde
// (dort ist 0 == -0).
void test_nodal_maxwell() {
    std::puts("\n[6] Maxwell-Spannungstensor fuer Knotenfelder (NodalMaxwell.cuh)");

    // Einheitswuerfel [0,1]^3 als einzelner Hexaeder, Knotenreihenfolge wie
    // in Hexahedron::v (siehe NodalHexahedron.cuh).
    double3 nodes[8] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };
    Hexahedron hex;
    for (int i = 0; i < 8; ++i) hex.v[i] = i;

    const double B0 = 0.75;   // T
    const double pressure = B0 * B0 / (2.0 * NODAL_MU_0);

    // A-Felder, die ein uniformes B erzeugen:
    //   B = B0*z_hat  <-  A = B0/2 * (-y, x, 0)
    //   B = B0*x_hat  <-  A = B0/2 * (0, -z, y)
    auto set_A_for_Bz = [&](double* A) {
        for (int i = 0; i < 8; ++i) {
            A[i*3+0] = -0.5 * B0 * nodes[i].y;
            A[i*3+1] =  0.5 * B0 * nodes[i].x;
            A[i*3+2] =  0.0;
        }
    };
    auto set_A_for_Bx = [&](double* A) {
        for (int i = 0; i < 8; ++i) {
            A[i*3+0] =  0.0;
            A[i*3+1] = -0.5 * B0 * nodes[i].z;
            A[i*3+2] =  0.5 * B0 * nodes[i].y;
        }
    };

    auto integrate = [&](const double* A, int local_face,
                         double3& F, double3& M) {
        Mesh::QuadFace q;
        const int lf2n[6][4] = {
            {0,3,2,1}, {4,5,6,7}, {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7}
        };
        for (int k = 0; k < 4; ++k) q.v[k] = lf2n[local_face][k];
        q.parent_hex_id = 0;
        q.local_face_id = local_face;

        double fx=0, fy=0, fz=0, mx=0, my=0, mz=0;
        reference_maxwell_body(0, 1, &q, &hex, nodes, A,
                               make_double3(0.5, 0.5, 0.5),
                               &fx, &fy, &fz, &mx, &my, &mz);
        F = make_double3(fx, fy, fz);
        M = make_double3(mx, my, mz);
    };

    double A[24];
    double3 F, M;

    // ---- (a) Deckflaeche (local_face 1, Normale +z), B = B0 z_hat -> NORMAL
    set_A_for_Bz(A);
    integrate(A, 1, F, M);
    std::printf("      (a) B normal   : F = (%.4f, %.4f, %.4f) N, erwartet (0,0,%+.4f)\n",
                F.x, F.y, F.z, pressure);
    check_lt(std::fabs(F.z - pressure) / pressure, 1e-12,
             "B normal: Zug nach AUSSEN mit |B|^2/(2 mu_0)");
    check_lt(std::sqrt(F.x*F.x + F.y*F.y) / pressure, 1e-12, "keine Querkraft");

    // ---- (b) dieselbe Flaeche, B = B0 x_hat -> TANGENTIAL
    set_A_for_Bx(A);
    integrate(A, 1, F, M);
    std::printf("      (b) B tangential: F = (%.4f, %.4f, %.4f) N, erwartet (0,0,%+.4f)\n",
                F.x, F.y, F.z, -pressure);
    check_lt(std::fabs(F.z + pressure) / pressure, 1e-12,
             "B tangential: Zug nach INNEN, entgegengesetztes Vorzeichen");

    // ---- (c) geschlossene Wuerfeloberflaeche, uniformes B -> Gesamtkraft 0
    set_A_for_Bz(A);
    double3 Fsum = make_double3(0, 0, 0), Msum = make_double3(0, 0, 0);
    for (int lf = 0; lf < 6; ++lf) {
        integrate(A, lf, F, M);
        Fsum = add3(Fsum, F); Msum = add3(Msum, M);
    }
    std::printf("      (c) geschlossen : |F| = %.3e N, |M| = %.3e Nm (Skala %.1f N)\n",
                norm3(Fsum), norm3(Msum), pressure);
    check_lt(norm3(Fsum) / pressure, 1e-12, "geschlossene Flaeche: Gesamtkraft null");
    check_lt(norm3(Msum) / pressure, 1e-12, "geschlossene Flaeche: Gesamtmoment null");

    // ---- Gegenprobe: B = curl A wird ueberhaupt richtig ausgewertet
    {
        double3 g[8]; double detJ;
        evaluate_nodal_gradients(nodes, 0.1, -0.3, 0.2, g, detJ);
        set_A_for_Bz(A);
        double3 B = make_double3(0, 0, 0);
        for (int i = 0; i < 8; ++i) {
            const double3 Ai = make_double3(A[i*3+0], A[i*3+1], A[i*3+2]);
            const double3 c = cross3(g[i], Ai);
            B.x += c.x; B.y += c.y; B.z += c.z;
        }
        std::printf("      B = curl A     : (%.6f, %.6f, %.6f), erwartet (0,0,%.6f)\n",
                    B.x, B.y, B.z, B0);
        check_lt(norm3(sub3(B, make_double3(0, 0, B0))) / B0, 1e-13,
                 "B = sum grad N_i x A_i reproduziert uniformes Feld exakt");
    }
}

} // namespace

int main() {
    bf::g_backend = bf::Backend::CPU;

    std::puts("========================================================");
    std::puts(" Abnahmepruefung Referenzloeser (--reference)            ");
    std::puts("========================================================");

    test_shape_functions();
    test_duffy_singular();
    test_boundary_mass();
    test_bie_sphere();
    test_bie_cube_documents_limit();
    test_dense_lu();
    test_nodal_maxwell();

    std::puts("\n========================================================");
    if (g_fail == 0) {
        std::puts(" ALLE PRUEFUNGEN BESTANDEN");
        std::puts("========================================================");
        return 0;
    }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
