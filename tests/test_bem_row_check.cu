// =============================================================================
// Host-Test fuer include/BemRowCheck.h (Gegenprobe 2, --bem-row-check)
//
//   [1] elliptische Integrale und Schleifenfeld gegen unabhaengige Referenzen
//   [2] rot A_ex = B_ex (zentrale Differenz), B_z auf der Achse gegen
//       BiotSavart.cuh, (n.grad)A analytisch gegen zentrale Differenz
//   [3] Rooftop-Projektion (Rundlauf) mit Gegenprobe; Beziehung zwischen
//       Rooftop-Koeffizient und Kantenintegral (gemessen = analytisch)
//   [4] Residuenbuchhaltung auf einem kleinen synthetischen Fall, mit
//       Gegenproben (falsches Vorzeichen, vertauschte Zeilen)
//
// Reiner Host-Code, keine GPU, kein Netz.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include "BemRowCheck.h"
#include "BiotSavart.cuh"

using namespace bem_row_check;

static int g_fail = 0;
static void check(bool ok, const char* what, double value, double limit) {
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value,
                ok ? "<" : ">=", limit);
    if (!ok) ++g_fail;
}
static void check_true(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHLER", what);
    if (!ok) ++g_fail;
}

// ---------------------------------------------------------------------------
// Unabhaengige Referenz: A_phi und B einer Schleife durch numerische
// Integration des Biot-Savart-Integrals ueber den Umfang (kein elliptisches
// Integral).
// ---------------------------------------------------------------------------
static void loop_numeric(double a, double I, double r, double z, double& Aphi, double& Br,
                         double& Bz) {
    const int N = 20000;
    double Ax = 0, Ay = 0, Bxs = 0, Bzs = 0;
    const V3 p{r, 0.0, z};
    for (int i = 0; i < N; ++i) {
        const double ph = 2.0 * kPi * (i + 0.5) / N;
        const V3 q{a * std::cos(ph), a * std::sin(ph), 0.0};
        const V3 dl{-a * std::sin(ph) * 2.0 * kPi / N, a * std::cos(ph) * 2.0 * kPi / N, 0.0};
        const V3 R = p - q;
        const double d = norm(R);
        Ax += dl.x / d;
        Ay += dl.y / d;
        const V3 dB = cross(dl, R);
        Bxs += dB.x / (d * d * d);
        Bzs += dB.z / (d * d * d);
    }
    const double c = kMu0 * I / (4.0 * kPi);
    Aphi = c * Ay;   // am Punkt (r, 0, z) ist e_phi = e_y
    Br = c * Bxs;
    Bz = c * Bzs;
    (void)Ax;
}

static void test_fields() {
    std::printf("[1] Elliptische Integrale und Schleifenfeld\n");
    double K, E;
    ellipke(0.5, K, E);
    check(std::fabs(K - 1.8540746773013719) < 1e-14, "K(0.5)", std::fabs(K - 1.8540746773013719), 1e-14);
    check(std::fabs(E - 1.3506438810476755) < 1e-14, "E(0.5)", std::fabs(E - 1.3506438810476755), 1e-14);

    double worst = 0;
    const double pts[4][2] = {{0.03, 0.01}, {0.08, -0.02}, {0.1, 0.05}, {0.02, -0.03}};
    for (auto& p : pts) {
        const LoopField f = loop_field(0.055, 20.0, p[0], p[1]);
        double Ap, Br, Bz;
        loop_numeric(0.055, 20.0, p[0], p[1], Ap, Br, Bz);
        worst = std::max({worst, std::fabs(f.Aphi - Ap) / std::fabs(Ap), std::fabs(f.Br - Br) / std::fabs(Br),
                          std::fabs(f.Bz - Bz) / std::fabs(Bz)});
    }
    check(worst < 1e-8, "Schleife: geschlossene Form gegen Umfangsintegral (A_phi, B_r, B_z)", worst, 1e-8);

    // Gegenprobe: ein falscher Radius muss auffallen
    const LoopField g = loop_field(0.056, 20.0, 0.03, 0.01);
    double Ap, Br, Bz;
    loop_numeric(0.055, 20.0, 0.03, 0.01, Ap, Br, Bz);
    check(std::fabs(g.Bz - Bz) / std::fabs(Bz) > 1e-3, "Gegenprobe: falscher Radius wird erkannt",
          std::fabs(g.Bz - Bz) / std::fabs(Bz), 1e-3);
}

static void test_derivatives() {
    std::printf("[2] rot A = B, Achse, Normalableitung\n");
    // grobe Faeden (2 mm) reichen: geprueft wird die Identitaet, nicht die Spule
    const std::vector<Filament> fil = make_filaments(team28_coils(), 20.0, 2e-3);
    auto A_at = [&](V3 p) { return exact_field(fil, p).A; };

    // Punkte abseits der Faeden (Faeden bei 2 mm Raster, Mitte der Zellen +1 mm)
    const V3 pts[3] = {{0.062, 0.021, -0.0101}, {0.0123, 0.0456, 0.012}, {0.0713, -0.0377, -0.0334}};
    double worst = 0, worst_n = 0;
    const double h = 1e-6;
    for (const V3& p : pts) {
        auto d = [&](int c, double s) {
            V3 q = p;
            (c == 0 ? q.x : (c == 1 ? q.y : q.z)) += s;
            return A_at(q);
        };
        const V3 dx = (1.0 / (2 * h)) * (d(0, h) - d(0, -h));
        const V3 dy = (1.0 / (2 * h)) * (d(1, h) - d(1, -h));
        const V3 dz = (1.0 / (2 * h)) * (d(2, h) - d(2, -h));
        const V3 curl{dy.z - dz.y, dz.x - dx.z, dx.y - dy.x};
        const V3 B = exact_field(fil, p).B;
        worst = std::max(worst, norm(curl - B) / norm(B));

        // (n.grad)A gegen zentrale Differenz laengs n
        const V3 n = (1.0 / std::sqrt(0.36 + 0.64 + 0.25)) * V3{0.6, -0.8, 0.5};
        const V3 an = (1.0 / (2 * h)) * (A_at(p + h * n) - A_at(p - h * n));
        const V3 ex = normal_derivative_A(exact_field(fil, p), n);
        worst_n = std::max(worst_n, norm(an - ex) / norm(ex));
    }
    check(worst < 1e-6, "rot A_ex = B_ex (zentrale Differenz, h = 1e-6)", worst, 1e-6);
    check(worst_n < 1e-6, "(n.grad)A analytisch gegen zentrale Differenz", worst_n, 1e-6);

    // Gegenprobe: ohne den Term -(A_phi/r)(n.e_phi)e_r waere die Formel falsch
    const V3 p = pts[0];
    const V3 n = (1.0 / std::sqrt(1.25)) * V3{0.6, -0.8, 0.5};
    const ExactField e = exact_field(fil, p);
    const double dAr = e.Bz - e.Aphi / e.r;
    const V3 falsch = (dot(n, e.e_r) * dAr + n.z * (-e.Br)) * e.e_phi;
    const V3 an = (1.0 / (2 * h)) * (A_at(p + h * n) - A_at(p - h * n));
    check(norm(an - falsch) / norm(an) > 1e-3, "Gegenprobe: ohne Kruemmungsterm weicht (n.grad)A ab",
          norm(an - falsch) / norm(an), 1e-3);

    // B_z auf der Achse gegen BiotSavart.cuh (homogene Stromdichte); die Faeden
    // approximieren sie mit Mittelpunktsregel
    const std::vector<Filament> f05 = make_filaments(team28_coils(), 20.0, 0.5e-3);
    const std::vector<Filament> f10 = make_filaments(team28_coils(), 20.0, 1.0e-3);
    double w05 = 0, w10 = 0;
    for (double z : {0.0113, 0.0143, 0.02, 0.05}) {
        const double ref = biot_savart::team28_bz_on_axis(z, 20.0);
        w05 = std::max(w05, std::fabs(sum_field(f05, 0.0, z).Bz - ref) / std::fabs(ref));
        w10 = std::max(w10, std::fabs(sum_field(f10, 0.0, z).Bz - ref) / std::fabs(ref));
    }
    std::printf("      B_z Achse gegen BiotSavart.cuh: Faden 1 mm %.2e, 0.5 mm %.2e\n", w10, w05);
    check(w05 < 5e-4, "B_z auf der Achse, Faden 0.5 mm (gemessener Fadenfehler)", w05, 5e-4);
    check(w05 < w10, "feineres Fadenraster liegt naeher an der homogenen Stromdichte", w05, w10);
}

// ---------------------------------------------------------------------------
// Rooftop-Projektion
// ---------------------------------------------------------------------------
static QuadGeom make_quad(double a, double b, double skew) {
    QuadGeom q;
    // Rechteck mit Halbseiten a (u) und b (v) in der Ebene z = 0.2, leicht geschert
    q.n[0] = make_double3(-a, -b, 0.2);
    q.n[1] = make_double3(a, -b + skew, 0.2);
    q.n[2] = make_double3(a, b, 0.2);
    q.n[3] = make_double3(-a, b - skew, 0.2);
    q.outward = 1.0;
    return q;
}

static void test_projection() {
    std::printf("[3] Rooftop-Projektion\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> U(-1, 1);
    double worst = 0, worst_bad = 1e300;
    for (int t = 0; t < 20; ++t) {
        QuadGeom q = make_quad(0.01 + 0.01 * std::fabs(U(rng)), 0.01 + 0.01 * std::fabs(U(rng)),
                               0.002 * U(rng));
        double c0[4] = {U(rng), U(rng), U(rng), U(rng)};
        // Rundlauf ueber die Gauss-Punkte: project_to_rooftop ruft F(p, n) an
        // den 16 Gauss-Punkten auf; das Feld sum c_i W_i wird dafuer in einer
        // Tabelle nach Punkt abgelegt.
        struct Tab { V3 p; V3 val; };
        std::vector<Tab> tab;
        double x[4], w[4];
        gauss4(x, w);
        for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b)
            tab.push_back({quad_point(q.n, x[a], x[b]), rooftop_value(q, c0, x[a], x[b])});
        auto Ftab = [&](V3 p, V3) {
            for (const Tab& e : tab) if (norm(e.p - p) < 1e-14) return e.val;
            return V3{1e9, 0, 0};
        };
        double c1[4];
        if (!project_to_rooftop(q, Ftab, c1)) { check_true(false, "Massenmatrix regulaer"); continue; }
        double err = 0;
        for (int i = 0; i < 4; ++i) err = std::max(err, std::fabs(c1[i] - c0[i]));
        worst = std::max(worst, err);

        // Gegenprobe: ohne T^-1 (f direkt als Koeffizient) ist der Rundlauf falsch
        double f[4] = {0, 0, 0, 0};
        for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
            double detJ; double3 W[4];
            evaluate_rooftop_basis(x[a], x[b], q.n, W, detJ);
            const V3 val = Ftab(quad_point(q.n, x[a], x[b]), V3{});
            for (int i = 0; i < 4; ++i) f[i] += w[a] * w[b] * detJ * dot(to_v3(W[i]), val);
        }
        double bad = 0;
        for (int i = 0; i < 4; ++i) bad = std::max(bad, std::fabs(f[i] - c0[i]));
        worst_bad = std::min(worst_bad, bad);
    }
    check(worst < 1e-10, "Rundlauf: Koeffizienten -> Feld -> Projektion", worst, 1e-10);
    check(worst_bad > 1e-3, "Gegenprobe: Projektion ohne Massenmatrix wird erkannt", worst_bad, 1e-3);

    // Kantenintegral eines konstanten Feldes gegen den Rooftop-Koeffizienten.
    // Konstantes A = (1,0,0): Kantenintegral der Kante {0,1} (in +x) = 2a. Das
    // Rooftop-Feld sum a_i W_i hat im Zentrum die Tangentialkomponente
    // (a_0 W_0 + a_2 W_2).x. Das Verhaeltnis zu A_x ist gemessen und wird mit der
    // Vorhersage 2 a / b (u-Kanten) verglichen.
    double worst_ratio = 0;
    for (double ab : {1.0, 2.0, 0.5}) {
        QuadGeom q = make_quad(0.01 * ab, 0.01, 0.0);
        const V3 A{1.0, 0.0, 0.0};
        double c[4];
        for (int i = 0; i < 4; ++i) {
            const V3 p0 = to_v3(q.n[i]), p1 = to_v3(q.n[(i + 1) % 4]);
            c[i] = dot(A, p1 - p0);   // Kantenintegral in lokaler Richtung
        }
        const V3 rec = rooftop_value(q, c, 0.0, 0.0);
        const double measured = rec.x;               // Soll bei richtiger Skalierung: 1
        const double predicted = 2.0 * ab;           // 2 * (a/b)
        std::printf("      a/b = %.1f: Rooftop-Rekonstruktion von A_x = 1 ergibt %.6f "
                    "(Vorhersage 2a/b = %.6f)\n", ab, measured, predicted);
        worst_ratio = std::max(worst_ratio, std::fabs(measured - predicted));
    }
    check(worst_ratio < 1e-10, "Rekonstruktion aus Kantenintegralen = 2a/b (Messung = Vorhersage)",
          worst_ratio, 1e-10);
}

// ---------------------------------------------------------------------------
// Residuenbuchhaltung, synthetischer Fall
// ---------------------------------------------------------------------------
struct Mat {
    int n, m;
    std::vector<double> v;
    Mat(int n_, int m_) : n(n_), m(m_), v(static_cast<size_t>(n_) * m_, 0.0) {}
    double& operator()(int i, int j) { return v[static_cast<size_t>(i) * m + j]; }
    double operator()(int i, int j) const { return v[static_cast<size_t>(i) * m + j]; }
    std::vector<double> apply(const std::vector<double>& x) const {
        std::vector<double> y(static_cast<size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) y[i] += (*this)(i, j) * x[j];
        return y;
    }
};

static void test_residuals() {
    std::printf("[4] Residuenbuchhaltung, synthetischer Fall\n");
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> U(-1, 1);
    const int n = 12, m = 8;
    Mat A(n, n), T(n, m), D(m, n), G(m, m);
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) A(i, j) = U(rng) + (i == j ? 4.0 : 0.0);
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) T(i, j) = U(rng);
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) D(i, j) = U(rng);
    for (int i = 0; i < m; ++i) for (int j = 0; j < m; ++j) G(i, j) = U(rng) + (i == j ? 3.0 : 0.0);
    std::vector<double> a(n), k(m);
    for (double& x : a) x = U(rng);
    // k so waehlen, dass Zeile 2 exakt null ist: G k = D a  (Vorzeichen der Gegenprobe
    // folgt aus r2 = -D a + G k)
    {
        // Gauss-Elimination G k = D a
        std::vector<double> rhs = D.apply(a);
        Mat M = G;
        for (int c = 0; c < m; ++c) {
            int p = c;
            for (int i = c + 1; i < m; ++i) if (std::fabs(M(i, c)) > std::fabs(M(p, c))) p = i;
            for (int j = 0; j < m; ++j) std::swap(M(p, j), M(c, j));
            std::swap(rhs[p], rhs[c]);
            for (int i = c + 1; i < m; ++i) {
                const double f = M(i, c) / M(c, c);
                for (int j = c; j < m; ++j) M(i, j) -= f * M(c, j);
                rhs[i] -= f * rhs[c];
            }
        }
        for (int i = m - 1; i >= 0; --i) {
            double s = rhs[i];
            for (int j = i + 1; j < m; ++j) s -= M(i, j) * k[j];
            k[i] = s / M(i, i);
        }
    }
    // b so, dass Zeile 1 exakt null ist: A a - T k - b = 0
    std::vector<double> Aa = A.apply(a), Tk = T.apply(k), Da = D.apply(a), Gk = G.apply(k);
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = Aa[i] - Tk[i];

    std::vector<int> lab1(n), lab2(m);
    for (int i = 0; i < n; ++i) lab1[i] = i % 3;
    for (int i = 0; i < m; ++i) lab2[i] = i % 3;

    // Zeile 1: r1 = (A a - b) + s*(-T k); Zeile 2: r2 = -D a + s*G k
    auto row1 = [&](double s) {
        std::vector<double> t1(n);
        for (int i = 0; i < n; ++i) t1[i] = Aa[i] - b[i];
        std::vector<double> mt(n);
        for (int i = 0; i < n; ++i) mt[i] = -Tk[i];
        return accumulate_by_label(axpy(t1, s, mt), t1, lab1, 3);
    };
    auto row2 = [&](double s) {
        std::vector<double> t2(m), g(m);
        for (int i = 0; i < m; ++i) { t2[i] = -Da[i]; g[i] = Gk[i]; }
        return accumulate_by_label(axpy(t2, s, g), t2, lab2, 3);
    };
    const auto r1p = row1(+1.0), r2p = row2(+1.0);
    check(r1p[3].rel() < 1e-13, "exaktes Paar: Zeile 1 verschwindet", r1p[3].rel(), 1e-13);
    check(r2p[3].rel() < 1e-13, "exaktes Paar: Zeile 2 verschwindet", r2p[3].rel(), 1e-13);
    for (int c = 0; c < 3; ++c)
        check(r1p[c].rel() < 1e-13 && r2p[c].rel() < 1e-13, "  je Klasse null", r1p[c].rel(), 1e-13);

    const auto r1m = row1(-1.0), r2m = row2(-1.0);
    check(r1m[3].rel() > 0.1, "Gegenprobe: falsches Vorzeichen von k, Zeile 1", r1m[3].rel(), 0.1);
    check(r2m[3].rel() > 0.1, "Gegenprobe: falsches Vorzeichen von k, Zeile 2", r2m[3].rel(), 0.1);

    // vertauschte Zeilen: die Terme von Zeile 2 in Zeile 1 einsetzen (gleiche
    // Laenge noetig -> auf min(n, m) kuerzen)
    // r1 mit den Termen von Zeile 2 gerechnet: (A a - b)_i + G k_i  statt  - T k_i
    std::vector<double> sw(m);
    for (int i = 0; i < m; ++i) sw[i] = (Aa[i] - b[i]) + Gk[i];
    double nrm = 0, ref = 0;
    for (int i = 0; i < m; ++i) { nrm += sw[i] * sw[i]; ref += (Aa[i] - b[i]) * (Aa[i] - b[i]); }
    check(std::sqrt(nrm / ref) > 0.1, "Gegenprobe: vertauschte Zeilen werden erkannt",
          std::sqrt(nrm / ref), 0.1);
}

int main() {
    std::printf("=== test_bem_row_check ===\n");
    test_fields();
    test_derivatives();
    test_projection();
    test_residuals();
    std::printf("\n%s (%d Fehler)\n", g_fail ? "FEHLER" : "ALLE PRUEFUNGEN BESTANDEN", g_fail);
    return g_fail ? 1 : 0;
}
