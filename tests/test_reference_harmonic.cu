// =============================================================================
// test_reference_harmonic.cu -- Bausteine des harmonischen Referenzpfads
// =============================================================================
//
// Geprueft wird, was --reference --harmonic neu einfuehrt:
//
//   1. nodal_lorentz_harmonic an einem analytischen Fall (uniformes B,
//      konstantes A_i): F = (sigma*w/2) Int A_i x B_r dV exakt.
//   2. nodal_lorentz_harmonic gegen das Zyklusmittel des transienten
//      nodal_lorentz auf einem verzerrten Hexaeder mit Zufallsfeldern, Kraft
//      UND Moment, Quadraturordnung 2 und 3.
//      Gegenprobe: vertauschte Rollen von A_r und A_i werden erkannt.
//   3. Phaseninvarianz, keine Kraft ohne Phasenverzug, Maske und kappa = 0.
//      Gegenprobe: eine Variante mit falschem Vorzeichen ist NICHT
//      phaseninvariant -- die Pruefung kann also scheitern.
//   4. ReferenceHarmonicOperator gegen das komplexe Produkt (K+G+jwC) x.
//      Gegenprobe: das Vorzeichen von w vertauscht wird erkannt.
//   5. FGMRES mit diag(M, M) gegen die komplexe Direktloesung, mit exaktem
//      und mit wechselndem, ungenauem inneren Loeser; sigma = 0 liefert
//      A_i = 0 exakt in einer Iteration.
//      Gegenprobe: ein Operator mit falschem Kopplungsvorzeichen liefert eine
//      andere Loesung, und der Vergleich erkennt sie.
//   6. reference_apply_dirichlet (--reference-dirichlet): n = 40, Maske mit
//      Randknoten, Zeilen UND Spalten eliminiert; FGMRES auf dem
//      eliminierten 2n-Blocksystem gegen die dichte komplexe Direktloesung
//      des reduzierten Systems (1e-10), Dirichlet-Eintraege exakt 0, Muster
//      und C = (W - K)/w in den Dirichlet-Zeilen exakt null.
//      Gegenprobe: nur Zeilen zu eliminieren verletzt die Symmetrie, und die
//      Pruefung meldet das.
//
// Braucht keine GPU (setzt bf::g_backend selbst auf CPU).
// =============================================================================

#include "Backend.cuh"
#include "DeviceArray.cuh"
#include "HostLinalg.cuh"
#include "NodalMaxwell.cuh"
#include "ReferenceHarmonic.cuh"

#include <cmath>
#include <complex>
#include <algorithm>
#include <cstdio>
#include <random>
#include <stdexcept>
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
void check_gt(double value, double bound, const char* what) {
    const bool ok = (value > bound);
    std::printf("  [%s] %s  (%.3e > %.3e)\n", ok ? "OK " : "FEHL", what, value, bound);
    if (!ok) ++g_fail;
}

constexpr double PI = 3.14159265358979323846;

struct Loads { double3 F, M; };

double rel3(double3 a, double3 b) {
    return norm3(sub3(a, b)) / std::fmax(norm3(b), 1e-300);
}

// Ein Hexaeder mit Knotenfeldern
struct OneHex {
    double3 nodes[8];
    Hexahedron hex;
    MaterialProperties mat;
    OneHex() { for (int i = 0; i < 8; ++i) hex.v[i] = i; }
};

Loads harmonic_loads(const OneHex& e, const unsigned char* mask,
                     const double* Ar, const double* Ai, double omega, int order,
                     double3 cm) {
    double f[6] = {0, 0, 0, 0, 0, 0};
    nodal_lorentz_harmonic_body(0, 1, &e.hex, e.nodes, &e.mat, mask, Ar, Ai, omega,
                                order, cm, &f[0], &f[1], &f[2], &f[3], &f[4], &f[5]);
    return { make_double3(f[0], f[1], f[2]), make_double3(f[3], f[4], f[5]) };
}

/// Zyklusmittel ueber den transienten Kernel: A(t) = A_r cos wt - A_i sin wt,
/// J aus dem Rueckwaertsdifferenzenquotienten mit sehr kleinem Schritt.
Loads cycle_average_transient(const OneHex& e, const double* Ar, const double* Ai,
                              double omega, int order, double3 cm) {
    const int K = 16;                   // exakt fuer Grad-2-Trigonometrie
    const double delta = 1e-7 / omega;  // w*delta = 1e-7
    double A[24], Ap[24];
    Loads acc = { make_double3(0, 0, 0), make_double3(0, 0, 0) };
    for (int k = 0; k < K; ++k) {
        const double t = (2.0 * PI / omega) * k / K;
        for (int i = 0; i < 24; ++i) {
            A[i]  = Ar[i] * std::cos(omega * t) - Ai[i] * std::sin(omega * t);
            Ap[i] = Ar[i] * std::cos(omega * (t - delta)) - Ai[i] * std::sin(omega * (t - delta));
        }
        double f[6] = {0, 0, 0, 0, 0, 0};
        nodal_lorentz_body(0, 1, &e.hex, e.nodes, &e.mat, A, Ap, delta, order, cm,
                           &f[0], &f[1], &f[2], &f[3], &f[4], &f[5]);
        acc.F = add3(acc.F, make_double3(f[0] / K, f[1] / K, f[2] / K));
        acc.M = add3(acc.M, make_double3(f[3] / K, f[4] / K, f[5] / K));
    }
    return acc;
}

OneHex distorted_hex(double kappa) {
    OneHex e;
    const double3 nd[8] = {
        {0.0, 0.0, 0.0}, {1.1, 0.0, 0.0}, {1.0, 0.9, 0.1}, {0.1, 1.0, 0.0},
        {0.0, 0.1, 1.0}, {1.0, 0.0, 1.1}, {1.2, 1.1, 1.0}, {0.0, 1.0, 1.0}
    };
    for (int i = 0; i < 8; ++i) e.nodes[i] = nd[i];
    e.mat.kappa = kappa;
    return e;
}

// -----------------------------------------------------------------------------
// 1. Analytischer Fall
// -----------------------------------------------------------------------------
void test_lorentz_analytic() {
    std::puts("\n[1] nodal_lorentz_harmonic: uniformes B_r, konstantes A_i");

    OneHex e;
    const double3 cube[8] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };
    for (int i = 0; i < 8; ++i) e.nodes[i] = cube[i];
    e.mat.kappa = 3.0;
    const double omega = 2.0 * PI * 50.0, B0 = 0.4, a = 0.25;

    // A_r = B0/2 (-y, x, 0) -> B_r = B0 z;  A_i = (a, 0, 0) -> B_i = 0.
    // <f> = (sigma w/2) A_i x B_r = (sigma w/2) (0, -a B0, 0), Volumen 1.
    double Ar[24], Ai[24];
    for (int i = 0; i < 8; ++i) {
        Ar[i*3+0] = -0.5 * B0 * cube[i].y; Ar[i*3+1] = 0.5 * B0 * cube[i].x; Ar[i*3+2] = 0.0;
        Ai[i*3+0] = a; Ai[i*3+1] = 0.0; Ai[i*3+2] = 0.0;
    }
    const double3 expect = make_double3(0.0, -0.5 * e.mat.kappa * omega * a * B0, 0.0);
    for (int order = 2; order <= 3; ++order) {
        const Loads L = harmonic_loads(e, nullptr, Ar, Ai, omega, order,
                                       make_double3(0.5, 0.5, 0.5));
        std::printf("      Ordnung %d: F = (%.6f, %.6f, %.6f), erwartet (0, %.6f, 0)\n",
                    order, L.F.x, L.F.y, L.F.z, expect.y);
        check_lt(rel3(L.F, expect), 1e-13, "Kraft analytisch");
        check_lt(norm3(L.M) / std::fabs(expect.y), 1e-13,
                 "uniforme Kraftdichte: kein Moment um den Mittelpunkt");
    }
}

// -----------------------------------------------------------------------------
// 2. Gegen das Zyklusmittel von nodal_lorentz
// -----------------------------------------------------------------------------
void test_lorentz_against_transient() {
    std::puts("\n[2] nodal_lorentz_harmonic gegen Zyklusmittel von nodal_lorentz");

    OneHex e = distorted_hex(2.5);
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    double Ar[24], Ai[24];
    for (int i = 0; i < 24; ++i) { Ar[i] = uni(rng); Ai[i] = uni(rng); }
    const double omega = 2.0 * PI * 50.0;
    const double3 cm = make_double3(0.3, 0.6, 0.4);   // bewusst nicht die Mitte

    for (int order = 2; order <= 3; ++order) {
        const Loads H = harmonic_loads(e, nullptr, Ar, Ai, omega, order, cm);
        const Loads T = cycle_average_transient(e, Ar, Ai, omega, order, cm);
        std::printf("      Ordnung %d: F_harm = (%+.6e, %+.6e, %+.6e)\n"
                    "                 F_tran = (%+.6e, %+.6e, %+.6e)\n",
                    order, H.F.x, H.F.y, H.F.z, T.F.x, T.F.y, T.F.z);
        check_lt(rel3(H.F, T.F), 1e-5, "Kraft gleich dem Zyklusmittel");
        check_lt(rel3(H.M, T.M), 1e-5, "Moment gleich dem Zyklusmittel");
    }

    // Gegenprobe: A_r und A_i vertauscht. Das ist die naheliegende Verwechslung
    // der Konvention A = A_r cos - A_i sin; die Kraft kehrt ihr Vorzeichen um.
    const Loads Hs = harmonic_loads(e, nullptr, Ai, Ar, omega, 3, cm);
    const Loads T  = cycle_average_transient(e, Ar, Ai, omega, 3, cm);
    check_gt(rel3(Hs.F, T.F), 1.0, "Gegenprobe: vertauschte A_r/A_i weicht ab");
}

// -----------------------------------------------------------------------------
// 3. Phaseninvarianz, Phasenverzug, Maske
// -----------------------------------------------------------------------------
// Test-eigene Auswertung der Kraftdichte in einem Punkt, nur fuer die
// Gegenprobe mit falschem Vorzeichen.
double3 density(double sign, double3 ar, double3 ai, double3 br, double3 bi) {
    const double3 t1 = cross3(ai, br), t2 = cross3(ar, bi);
    return make_double3(t1.x + sign * t2.x, t1.y + sign * t2.y, t1.z + sign * t2.z);
}

void test_lorentz_invariance() {
    std::puts("\n[3] Phaseninvarianz, Phasenverzug, Maske");

    OneHex e = distorted_hex(1.7);
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    double Ar[24], Ai[24], Ar2[24], Ai2[24];
    for (int i = 0; i < 24; ++i) { Ar[i] = uni(rng); Ai[i] = uni(rng); }
    const double omega = 2.0 * PI * 50.0, phi = 0.7;
    const double3 cm = make_double3(0.5, 0.5, 0.5);

    // A^' = e^{j phi} A^:  A_r' = A_r cos - A_i sin,  A_i' = A_r sin + A_i cos
    for (int i = 0; i < 24; ++i) {
        Ar2[i] = Ar[i] * std::cos(phi) - Ai[i] * std::sin(phi);
        Ai2[i] = Ar[i] * std::sin(phi) + Ai[i] * std::cos(phi);
    }
    const Loads L1 = harmonic_loads(e, nullptr, Ar, Ai, omega, 3, cm);
    const Loads L2 = harmonic_loads(e, nullptr, Ar2, Ai2, omega, 3, cm);
    check_lt(rel3(L2.F, L1.F), 1e-12, "Kraft invariant gegen globale Phase");
    check_lt(rel3(L2.M, L1.M), 1e-12, "Moment invariant gegen globale Phase");

    // Gegenprobe: dieselbe Drehung mit der falschen Variante A_i x B_r + A_r x B_i
    // in einem Punkt -- sie ist nicht invariant.
    {
        const double3 ar  = make_double3(0.3, -0.2, 0.9), ai  = make_double3(-0.5, 0.4, 0.1);
        const double3 br  = make_double3(0.7, 0.1, -0.3), bi  = make_double3(0.2, -0.8, 0.5);
        auto rot = [&](double3 r, double3 i, bool real) {
            return real ? make_double3(r.x*std::cos(phi) - i.x*std::sin(phi),
                                       r.y*std::cos(phi) - i.y*std::sin(phi),
                                       r.z*std::cos(phi) - i.z*std::sin(phi))
                        : make_double3(r.x*std::sin(phi) + i.x*std::cos(phi),
                                       r.y*std::sin(phi) + i.y*std::cos(phi),
                                       r.z*std::sin(phi) + i.z*std::cos(phi));
        };
        const double3 ar2 = rot(ar, ai, true), ai2 = rot(ar, ai, false);
        const double3 br2 = rot(br, bi, true), bi2 = rot(br, bi, false);
        const double good = rel3(density(-1.0, ar2, ai2, br2, bi2), density(-1.0, ar, ai, br, bi));
        const double bad  = rel3(density(+1.0, ar2, ai2, br2, bi2), density(+1.0, ar, ai, br, bi));
        check_lt(good, 1e-14, "Punktdichte (richtige Variante) invariant");
        check_gt(bad, 1e-2, "Gegenprobe: falsches Vorzeichen ist nicht invariant");
    }

    // Kein Phasenverzug -> keine Kraft
    for (int i = 0; i < 24; ++i) Ai2[i] = 0.7 * Ar[i];
    const Loads L0 = harmonic_loads(e, nullptr, Ar, Ai2, omega, 3, cm);
    check_lt(norm3(L0.F) / norm3(L1.F), 1e-14, "A_i parallel A_r: keine Kraft");

    // Maske und Nichtleiter
    const unsigned char off = 0, on = 1;
    const Loads Lm = harmonic_loads(e, &off, Ar, Ai, omega, 3, cm);
    const Loads Lo = harmonic_loads(e, &on, Ar, Ai, omega, 3, cm);
    check(norm3(Lm.F) == 0.0 && norm3(Lm.M) == 0.0, "Maske 0: Element traegt nicht bei");
    check(rel3(Lo.F, L1.F) == 0.0, "Maske 1: gleich ohne Maske");
    e.mat.kappa = 0.0;
    const Loads Lk = harmonic_loads(e, &on, Ar, Ai, omega, 3, cm);
    check(norm3(Lk.F) == 0.0, "kappa = 0: Element traegt nicht bei");
}

// -----------------------------------------------------------------------------
// Kleines Modellsystem fuer Operator und FGMRES
// -----------------------------------------------------------------------------
// K: Tridiagonalmatrix (d, -1), G: kleine dichte Stoerung (unsymmetrisch wie
// der DtN-Block), C: diagonal, nur in der ersten Haelfte der Zeilen von null
// verschieden (Leiter), auf dem CSR-Muster von K abgelegt.
struct Model {
    int n = 0;
    double omega = 0.0;
    DeviceArray<int> ro, ci;
    DeviceArray<double> kv, cv;
    std::vector<double> G;          // n x n zeilenweise

    void build(int n_, double omega_, double c_scale, unsigned seed) {
        n = n_; omega = omega_;
        std::vector<int> r, c; std::vector<double> k, cc;
        for (int i = 0; i < n; ++i) {
            r.push_back(static_cast<int>(c.size()));
            for (int j = i - 1; j <= i + 1; ++j) {
                if (j < 0 || j >= n) continue;
                c.push_back(j);
                k.push_back(j == i ? 2.6 : -1.0);
                cc.push_back(j == i && i < n / 2 ? c_scale * (1.0 + 0.1 * i) : 0.0);
            }
        }
        r.push_back(static_cast<int>(c.size()));
        ro.allocate(r.size()); ci.allocate(c.size());
        kv.allocate(k.size()); cv.allocate(cc.size());
        for (size_t i = 0; i < r.size(); ++i) ro[i] = r[i];
        for (size_t i = 0; i < c.size(); ++i) { ci[i] = c[i]; kv[i] = k[i]; cv[i] = cc[i]; }
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);
        G.assign(static_cast<size_t>(n) * n, 0.0);
        for (auto& g : G) g = 0.05 * uni(rng);
    }
    void apply_kg(const double* x, double* y) const {
        bf::csr_spmv(n, ro.data(), ci.data(), kv.data(), x, y);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) y[i] += G[static_cast<size_t>(i) * n + j] * x[j];
    }
    /// Dichte Matrizen fuer die Vergleichsrechnungen
    std::vector<double> dense(const DeviceArray<double>& v) const {
        std::vector<double> D(static_cast<size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int k = ro[i]; k < ro[i + 1]; ++k) D[static_cast<size_t>(i) * n + ci[k]] = v[k];
        return D;
    }
};

/// Gauss mit Spaltenpivot, zeilenweise Matrix, ueberschreibt A und b.
template <class T>
void gauss_solve(std::vector<T>& A, std::vector<T>& b, int n) {
    for (int k = 0; k < n; ++k) {
        int p = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(A[static_cast<size_t>(i) * n + k]) > std::abs(A[static_cast<size_t>(p) * n + k])) p = i;
        if (p != k) {
            for (int j = 0; j < n; ++j) std::swap(A[static_cast<size_t>(k) * n + j], A[static_cast<size_t>(p) * n + j]);
            std::swap(b[k], b[p]);
        }
        for (int i = k + 1; i < n; ++i) {
            const T f = A[static_cast<size_t>(i) * n + k] / A[static_cast<size_t>(k) * n + k];
            for (int j = k; j < n; ++j) A[static_cast<size_t>(i) * n + j] -= f * A[static_cast<size_t>(k) * n + j];
            b[i] -= f * b[k];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        T s = b[i];
        for (int j = i + 1; j < n; ++j) s -= A[static_cast<size_t>(i) * n + j] * b[j];
        b[i] = s / A[static_cast<size_t>(i) * n + i];
    }
}

void setup_operator(const Model& m, ReferenceHarmonicOperator& op, double omega) {
    op.n = m.n;
    op.apply_kg = [&m](const double* x, double* y) { m.apply_kg(x, y); };
    op.c_row_offsets = m.ro.data();
    op.c_col_indices = m.ci.data();
    op.c_values = m.cv.data();
    op.omega = omega;
    op.cublas = nullptr;
    op.allocate_buffers();
}

/// Komplexe Loesung von (K + G + jwC) z = b
std::vector<std::complex<double>> complex_solution(const Model& m, const std::vector<double>& b) {
    const int n = m.n;
    const std::vector<double> K = m.dense(m.kv), C = m.dense(m.cv);
    std::vector<std::complex<double>> A(static_cast<size_t>(n) * n), z(n);
    for (size_t i = 0; i < A.size(); ++i)
        A[i] = std::complex<double>(K[i] + m.G[i], m.omega * C[i]);
    for (int i = 0; i < n; ++i) z[i] = b[i];
    gauss_solve(A, z, n);
    return z;
}

// -----------------------------------------------------------------------------
// 4. Operator gegen das komplexe Produkt
// -----------------------------------------------------------------------------
void test_operator() {
    std::puts("\n[4] ReferenceHarmonicOperator gegen (K+G+jwC) x");

    Model m;
    m.build(14, 2.0 * PI * 50.0, 1e-3, 3);
    const int n = m.n;
    std::mt19937 rng(9);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    DeviceArray<double> x(2 * n), y(2 * n);
    for (int i = 0; i < 2 * n; ++i) x[i] = uni(rng);

    const std::vector<double> K = m.dense(m.kv), C = m.dense(m.cv);
    auto compare = [&](double omega_used) {
        ReferenceHarmonicOperator op;
        setup_operator(m, op, omega_used);
        op.apply(x.data(), y.data());
        double err = 0.0, ref = 0.0;
        for (int i = 0; i < n; ++i) {
            std::complex<double> s = 0.0;
            for (int j = 0; j < n; ++j) {
                const size_t ij = static_cast<size_t>(i) * n + j;
                s += std::complex<double>(K[ij] + m.G[ij], m.omega * C[ij])
                   * std::complex<double>(x[j], x[n + j]);
            }
            err = std::fmax(err, std::abs(s - std::complex<double>(y[i], y[n + i])));
            ref = std::fmax(ref, std::abs(s));
        }
        return err / ref;
    };
    check_lt(compare(m.omega), 1e-14, "Blockprodukt gleich komplexem Produkt");
    check_gt(compare(-m.omega), 1e-3, "Gegenprobe: Vorzeichen von w vertauscht wird erkannt");
}

// -----------------------------------------------------------------------------
// 5. FGMRES gegen die komplexe Direktloesung
// -----------------------------------------------------------------------------
void test_fgmres() {
    std::puts("\n[5] FGMRES mit diag(M, M) gegen komplexe Direktloesung");

    const int n = 40;
    // Kopplung so stark, dass A_i nicht klein gegen A_r ist: w*C bis 1.45
    // gegen den kleinsten Eigenwert 0.6 von K, also mu_max ~ 2.4.
    Model m;
    m.build(n, 2.0 * PI * 50.0, 0.5 / (2.0 * PI * 50.0), 17);
    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = std::sin(0.3 * i) + 0.2;
    const std::vector<std::complex<double>> z = complex_solution(m, b);

    // M = K + G + wC dicht, fuer den exakten inneren Loeser
    const std::vector<double> K = m.dense(m.kv), C = m.dense(m.cv);
    std::vector<double> M(static_cast<size_t>(n) * n);
    for (size_t i = 0; i < M.size(); ++i) M[i] = K[i] + m.G[i] + m.omega * C[i];

    auto exact_m = [&](const double* rhs, double* x) {
        std::vector<double> A = M, r(rhs, rhs + n);
        gauss_solve(A, r, n);
        for (int i = 0; i < n; ++i) x[i] = r[i];
        return 0;
    };
    // Ungenau UND wechselnd: 1 bis 3 Jacobi-Schritte je Aufruf. Genau dafuer
    // ist das flexible GMRES da.
    int calls = 0;
    auto jacobi_m = [&](const double* rhs, double* x) {
        const int sweeps = 1 + (calls++ % 3);
        std::vector<double> xn(n);
        for (int s = 0; s < sweeps; ++s) {
            for (int i = 0; i < n; ++i) {
                double r = rhs[i];
                for (int j = 0; j < n; ++j) r -= M[static_cast<size_t>(i) * n + j] * x[j];
                xn[i] = x[i] + r / M[static_cast<size_t>(i) * n + i];
            }
            for (int i = 0; i < n; ++i) x[i] = xn[i];
        }
        return sweeps;
    };

    DeviceArray<double> b2(2 * n), x2(2 * n);
    for (int i = 0; i < n; ++i) { b2[i] = b[i]; b2[n + i] = 0.0; }

    auto solve = [&](double omega_used, std::function<int(const double*, double*)> inner,
                     ReferenceFgmresResult& res) {
        ReferenceHarmonicOperator op;
        setup_operator(m, op, omega_used);
        ReferenceFgmresOptions opt;
        opt.tol = 1e-11; opt.restart = 30; opt.max_iter = 400;
        x2.fill_zero();
        res = reference_harmonic_fgmres(
            [&op](const double* x, double* y) { op.apply(x, y); },
            reference_harmonic_block_precond(n, inner), 2 * n, b2.data(), x2.data(),
            opt, nullptr, nullptr);
        double err = 0.0, ref = 0.0;
        for (int i = 0; i < n; ++i) {
            err = std::fmax(err, std::abs(std::complex<double>(x2[i], x2[n + i]) - z[i]));
            ref = std::fmax(ref, std::abs(z[i]));
        }
        return err / ref;
    };

    double ai_max = 0.0, ar_max = 0.0;
    for (int i = 0; i < n; ++i) {
        ai_max = std::fmax(ai_max, std::fabs(z[i].imag()));
        ar_max = std::fmax(ar_max, std::fabs(z[i].real()));
    }
    std::printf("      Modell: n = %d, max|A_i|/max|A_r| = %.3f\n", n, ai_max / ar_max);

    ReferenceFgmresResult r1, r2, r3;
    const double e1 = solve(m.omega, exact_m, r1);
    std::printf("      exakter innerer Loeser: %d Iterationen, Residuum %.2e\n",
                r1.iterations, r1.rel_residual);
    check(r1.converged, "exakt innen: konvergiert");
    check_lt(e1, 1e-9, "exakt innen: Loesung gleich komplexer Direktloesung");
    // Wirkung des Vorkonditionierers: gegen die Identitaet (M = I).
    {
        auto identity_m = [&](const double* rhs, double* x) {
            for (int i = 0; i < n; ++i) x[i] = rhs[i];
            return 0;
        };
        ReferenceFgmresResult r0;
        solve(m.omega, identity_m, r0);
        std::printf("      ohne Vorkonditionierer: %d Iterationen, Residuum %.2e\n",
                    r0.iterations, r0.rel_residual);
        check(2 * r1.iterations < r0.iterations,
              "diag(M, M) halbiert mindestens die Iterationszahl gegen M = I");
    }

    const double e2 = solve(m.omega, jacobi_m, r2);
    std::printf("      Jacobi innen (1..3 Schritte): %d Iterationen, Residuum %.2e\n",
                r2.iterations, r2.rel_residual);
    check(r2.converged, "wechselnder ungenauer Vorkonditionierer: konvergiert");
    check_lt(e2, 1e-8, "wechselnd innen: Loesung gleich komplexer Direktloesung");

    // Protokoll: je Iteration ein Eintrag, Schaetzung innerhalb eines Zyklus
    // monoton (Eigenschaft von GMRES), innere Iterationen mitgezaehlt.
    bool mono = true;
    for (size_t i = 1; i < r2.history.size(); ++i) {
        if ((i % 30) != 0 && r2.history[i] > r2.history[i - 1] * (1.0 + 1e-12)) mono = false;
    }
    check(static_cast<int>(r2.history.size()) == r2.iterations &&
          static_cast<int>(r2.inner_iterations.size()) == r2.iterations,
          "Protokoll: ein Eintrag je Iteration");
    check(mono, "Residuumsschaetzung im Zyklus monoton fallend");
    check(r2.inner_iterations[0] >= 2 && r2.inner_iterations[0] <= 6,
          "innere Iterationen beider Bloecke werden gezaehlt");

    // Gegenprobe: falsches Kopplungsvorzeichen -> konvergiert, aber gegen eine
    // ANDERE Loesung; der Vergleich muss das melden.
    const double e3 = solve(-m.omega, exact_m, r3);
    check_gt(e3, 1e-2, "Gegenprobe: falsches Kopplungsvorzeichen wird erkannt");

    // sigma = 0: C = 0 -> diag(M, M) ist exakt, eine Iteration, A_i = 0 exakt.
    Model m0;
    m0.build(n, 2.0 * PI * 50.0, 0.0, 17);
    {
        const std::vector<double> K0 = m0.dense(m0.kv);
        std::vector<double> M0(static_cast<size_t>(n) * n);
        for (size_t i = 0; i < M0.size(); ++i) M0[i] = K0[i] + m0.G[i];
        auto exact0 = [&](const double* rhs, double* x) {
            std::vector<double> A = M0, r(rhs, rhs + n);
            gauss_solve(A, r, n);
            for (int i = 0; i < n; ++i) x[i] = r[i];
            return 0;
        };
        ReferenceHarmonicOperator op;
        setup_operator(m0, op, m0.omega);
        ReferenceFgmresOptions opt;
        opt.tol = 1e-11;
        x2.fill_zero();
        const ReferenceFgmresResult r0 = reference_harmonic_fgmres(
            [&op](const double* x, double* y) { op.apply(x, y); },
            reference_harmonic_block_precond(n, exact0), 2 * n, b2.data(), x2.data(),
            opt, nullptr, nullptr);
        double ai = 0.0;
        for (int i = 0; i < n; ++i) ai = std::fmax(ai, std::fabs(x2[n + i]));
        const std::vector<std::complex<double>> z0 = complex_solution(m0, b);
        double err = 0.0, ref = 0.0;
        for (int i = 0; i < n; ++i) {
            err = std::fmax(err, std::fabs(x2[i] - z0[i].real()));
            ref = std::fmax(ref, std::fabs(z0[i].real()));
        }
        std::printf("      sigma = 0: %d Iteration(en), max|A_i| = %.1e\n", r0.iterations, ai);
        check(r0.converged && r0.iterations == 1, "sigma = 0: eine Iteration");
        check(ai == 0.0, "sigma = 0: A_i exakt null");
        check_lt(err / ref, 1e-12, "sigma = 0: A_r loest (K+G) A_r = b");
    }
}

// -----------------------------------------------------------------------------
// 6. Dirichlet-Elimination
// -----------------------------------------------------------------------------
double max_asymmetry(int n, const int* ro, const int* ci, const double* v) {
    std::vector<double> D(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = ro[i]; k < ro[i + 1]; ++k) D[static_cast<size_t>(i) * n + ci[k]] = v[k];
    double a = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j)
            a = std::fmax(a, std::fabs(D[static_cast<size_t>(i) * n + j] - D[static_cast<size_t>(j) * n + i]));
    return a;
}

void test_dirichlet() {
    std::puts("\n[6] reference_apply_dirichlet: Elimination von Zeilen UND Spalten");

    const int n = 40;
    Model m;
    m.build(n, 2.0 * PI * 50.0, 0.4 / (2.0 * PI * 50.0), 5);
    std::fill(m.G.begin(), m.G.end(), 0.0);     // reines CSR-Problem, SPD

    std::vector<unsigned char> dir(n, 0);
    for (int i : {0, 7, 8, 21, 22, 23, 39}) dir[i] = 1;     // darunter Nachbarn
    int n_free = 0;
    for (int i = 0; i < n; ++i) if (!dir[i]) ++n_free;

    const size_t nnz = static_cast<size_t>(m.ro[n]);
    std::vector<double> kv(nnz), wv(nnz);
    for (size_t k = 0; k < nnz; ++k) { kv[k] = m.kv[k]; wv[k] = m.kv[k] + m.omega * m.cv[k]; }
    double dsum = 0.0; int dcount = 0;
    for (int i = 0; i < n; ++i) {
        if (dir[i]) continue;
        for (int k = m.ro[i]; k < m.ro[i + 1]; ++k) if (m.ci[k] == i) { dsum += kv[k]; ++dcount; }
    }
    const double diag = dsum / dcount;
    const std::vector<double> kv0 = kv, wv0 = wv;      // Kopien ohne Elimination

    // Fehlerhafte Variante fuer die Gegenprobe: nur Zeilen eliminieren
    auto rows_only = [&](std::vector<double> v) {
        for (int i = 0; i < n; ++i) {
            if (!dir[i]) continue;
            for (int k = m.ro[i]; k < m.ro[i + 1]; ++k) v[k] = (m.ci[k] == i) ? diag : 0.0;
        }
        return v;
    };
    const double asym_before = max_asymmetry(n, m.ro.data(), m.ci.data(), kv.data());
    const std::vector<double> kv_rows = rows_only(kv);
    const double asym_rows = max_asymmetry(n, m.ro.data(), m.ci.data(), kv_rows.data());

    const std::vector<int> ro_before(m.ro.data(), m.ro.data() + n + 1);
    const std::vector<int> ci_before(m.ci.data(), m.ci.data() + nnz);
    reference_apply_dirichlet(n, m.ro.data(), m.ci.data(), kv.data(), dir.data(), diag);
    reference_apply_dirichlet(n, m.ro.data(), m.ci.data(), wv.data(), dir.data(), diag);

    check(std::equal(ro_before.begin(), ro_before.end(), m.ro.data()) &&
          std::equal(ci_before.begin(), ci_before.end(), m.ci.data()),
          "Muster von K und W bleibt unveraendert (gleiches CSR-Muster)");
    check(max_asymmetry(n, m.ro.data(), m.ci.data(), kv.data()) <= asym_before + 1e-300 &&
          max_asymmetry(n, m.ro.data(), m.ci.data(), kv.data()) == 0.0,
          "K nach der Elimination symmetrisch");
    check_gt(asym_rows, 0.5, "Gegenprobe: nur Zeilen eliminiert -> Asymmetrie wird gemeldet");

    // C = (W - K)/w: in Dirichlet-Zeilen und -Spalten exakt null
    DeviceArray<double> cv(static_cast<size_t>(nnz));
    bool c_zero = true;
    for (int i = 0; i < n; ++i)
        for (int k = m.ro[i]; k < m.ro[i + 1]; ++k) {
            cv[k] = (wv[k] - kv[k]) / m.omega;
            if ((dir[i] || dir[m.ci[k]]) && cv[k] != 0.0) c_zero = false;
        }
    check(c_zero, "C = (W - K)/w in den Dirichlet-Zeilen und -Spalten exakt null");

    // Operator mit eliminierten Matrizen; rechte Seite in Dirichlet-Zeilen null
    DeviceArray<double> kdev(static_cast<size_t>(nnz));
    for (size_t k = 0; k < nnz; ++k) kdev[k] = kv[k];
    ReferenceHarmonicOperator op;
    op.n = n;
    op.apply_kg = [&](const double* x, double* y) {
        bf::csr_spmv(n, m.ro.data(), m.ci.data(), kdev.data(), x, y);
    };
    op.c_row_offsets = m.ro.data(); op.c_col_indices = m.ci.data(); op.c_values = cv.data();
    op.omega = m.omega; op.cublas = nullptr; op.allocate_buffers();

    std::vector<double> b(n);
    for (int i = 0; i < n; ++i) b[i] = dir[i] ? 0.0 : std::sin(0.4 * i) + 0.3;
    DeviceArray<double> b2(2 * n), x2(2 * n);
    for (int i = 0; i < n; ++i) { b2[i] = b[i]; b2[n + i] = 0.0; }
    x2.fill_zero();

    // exakter innerer Loeser: M = W (eliminiert), dicht
    std::vector<double> M(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = m.ro[i]; k < m.ro[i + 1]; ++k) M[static_cast<size_t>(i) * n + m.ci[k]] = wv[k];
    auto exact_m = [&](const double* rhs, double* x) {
        std::vector<double> A = M, r(rhs, rhs + n);
        gauss_solve(A, r, n);
        for (int i = 0; i < n; ++i) x[i] = r[i];
        return 0;
    };
    ReferenceFgmresOptions opt;
    opt.tol = 1e-12; opt.restart = 30; opt.max_iter = 200;
    const ReferenceFgmresResult res = reference_harmonic_fgmres(
        [&op](const double* x, double* y) { op.apply(x, y); },
        reference_harmonic_block_precond(n, exact_m), 2 * n, b2.data(), x2.data(),
        opt, nullptr, nullptr);
    check(res.converged, "FGMRES auf dem eliminierten Blocksystem konvergiert");

    // dichte komplexe Direktloesung des REDUZIERTEN Systems (nur freie Zeilen/Spalten)
    std::vector<int> fi;
    for (int i = 0; i < n; ++i) if (!dir[i]) fi.push_back(i);
    std::vector<std::complex<double>> Ar(static_cast<size_t>(n_free) * n_free), zr(n_free);
    {
        std::vector<double> Kd(static_cast<size_t>(n) * n, 0.0), Cd(static_cast<size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int k = m.ro[i]; k < m.ro[i + 1]; ++k) {
                Kd[static_cast<size_t>(i) * n + m.ci[k]] = kv[k];
                Cd[static_cast<size_t>(i) * n + m.ci[k]] = cv[k];
            }
        for (int a = 0; a < n_free; ++a) {
            zr[a] = b[fi[a]];
            for (int c = 0; c < n_free; ++c)
                Ar[static_cast<size_t>(a) * n_free + c] = std::complex<double>(
                    Kd[static_cast<size_t>(fi[a]) * n + fi[c]],
                    m.omega * Cd[static_cast<size_t>(fi[a]) * n + fi[c]]);
        }
        gauss_solve(Ar, zr, n_free);
    }
    double err = 0.0, ref = 0.0;
    for (int a = 0; a < n_free; ++a) {
        err = std::fmax(err, std::abs(std::complex<double>(x2[fi[a]], x2[n + fi[a]]) - zr[a]));
        ref = std::fmax(ref, std::abs(zr[a]));
    }
    double ai_max = 0.0, ar_max = 0.0;
    for (int a = 0; a < n_free; ++a) {
        ai_max = std::fmax(ai_max, std::fabs(zr[a].imag()));
        ar_max = std::fmax(ar_max, std::fabs(zr[a].real()));
    }
    std::printf("      n = %d, davon %d Dirichlet, %d FGMRES-Iterationen, "
                "max|A_i|/max|A_r| = %.2f\n", n, n - n_free, res.iterations, ai_max / ar_max);
    check_lt(err / ref, 1e-10, "Loesung gleich dichter Direktloesung des reduzierten Systems");
    double dmax = 0.0;
    for (int i = 0; i < n; ++i)
        if (dir[i]) dmax = std::fmax(dmax, std::fmax(std::fabs(x2[i]), std::fabs(x2[n + i])));
    check(dmax == 0.0, "Dirichlet-Eintraege exakt 0");

    // Gegenprobe der LOESUNG: dieselbe rechte Seite, aber ohne Elimination
    // (Original-K und -W) -- die Loesung weicht vom reduzierten System ab, und
    // der Vergleich muss das erkennen.
    {
        DeviceArray<double> k_orig(static_cast<size_t>(nnz)), c_orig(static_cast<size_t>(nnz));
        for (size_t k = 0; k < nnz; ++k) {
            k_orig[k] = kv0[k];
            c_orig[k] = (wv0[k] - kv0[k]) / m.omega;
        }
        ReferenceHarmonicOperator op2;
        op2.n = n;
        op2.apply_kg = [&](const double* x, double* y) {
            bf::csr_spmv(n, m.ro.data(), m.ci.data(), k_orig.data(), x, y);
        };
        op2.c_row_offsets = m.ro.data(); op2.c_col_indices = m.ci.data();
        op2.c_values = c_orig.data(); op2.omega = m.omega; op2.cublas = nullptr;
        op2.allocate_buffers();
        std::vector<double> M2(static_cast<size_t>(n) * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int k = m.ro[i]; k < m.ro[i + 1]; ++k) M2[static_cast<size_t>(i) * n + m.ci[k]] = wv0[k];
        auto exact2 = [&](const double* rhs, double* x) {
            std::vector<double> A = M2, r(rhs, rhs + n);
            gauss_solve(A, r, n);
            for (int i = 0; i < n; ++i) x[i] = r[i];
            return 0;
        };
        x2.fill_zero();
        reference_harmonic_fgmres(
            [&op2](const double* x, double* y) { op2.apply(x, y); },
            reference_harmonic_block_precond(n, exact2), 2 * n, b2.data(), x2.data(),
            opt, nullptr, nullptr);
        double e2 = 0.0;
        for (int a = 0; a < n_free; ++a)
            e2 = std::fmax(e2, std::abs(std::complex<double>(x2[fi[a]], x2[n + fi[a]]) - zr[a]));
        check_gt(e2 / ref, 1e-3, "Gegenprobe: ohne Elimination weicht die Loesung ab");
    }

    // fehlender Diagonaleintrag in einer Dirichlet-Zeile wird abgewiesen
    {
        const int ro2[3] = {0, 1, 2};
        const int ci2[2] = {1, 0};            // keine Diagonale
        double v2[2] = {1.0, 1.0};
        const unsigned char d2[2] = {1, 0};
        bool thrown = false;
        try { reference_apply_dirichlet(2, ro2, ci2, v2, d2, 1.0); }
        catch (const std::runtime_error&) { thrown = true; }
        check(thrown, "Dirichlet-Zeile ohne Diagonaleintrag wird abgewiesen");
    }
}

} // namespace

int main() {
    bf::g_backend = bf::Backend::CPU;

    std::puts("========================================================");
    std::puts(" Harmonischer Referenzpfad (--reference --harmonic)       ");
    std::puts("========================================================");

    test_lorentz_analytic();
    test_lorentz_against_transient();
    test_lorentz_invariance();
    test_operator();
    test_fgmres();
    test_dirichlet();

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
