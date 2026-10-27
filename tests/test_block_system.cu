// =============================================================================
// Abnahmepruefung: reelle 2n-Blockform des harmonischen Systems (AP5)
//
//   cmake --build build --config Release --target test_block_system
//   ./build/test_block_system
//
// Reiner Host-Code, keine GPU.
//
// WORUM ES GEHT
// -------------
// cuDSS rechnet reell. Das harmonische System ist komplex:
//
//     (K + j*omega*C) (x_r + j x_i) = b_r + j b_i
//
// Es wird deshalb auf ein reelles System doppelter Groesse gebracht. Dabei
// gibt es genau zwei Fehlerquellen, und beide sind lautlos:
//
//   1. Ein Vorzeichen an einem der Nebenblocke oder auf der rechten Seite.
//      Das Ergebnis ist dann eine Loesung -- nur nicht die des gestellten
//      Problems. Der Imaginaerteil bekommt das falsche Vorzeichen, und damit
//      die Kraft.
//   2. Unsortierte Spaltenindizes. CSR verlangt aufsteigende Reihenfolge je
//      Zeile; cuDSS prueft das nicht zwingend, und ein Loeser auf
//      fehlsortierten Indizes liefert Zahlen, die nach nichts aussehen.
//
// DIE ENTSCHEIDENDE PRUEFUNG
// --------------------------
// Nummer 3 loest dasselbe Problem zweimal auf voellig verschiedenen Wegen:
// einmal komplex mit std::complex und Gauss, einmal ueber die reelle
// 2n-Blockform. Beide muessen uebereinstimmen. Das prueft alle Vorzeichen
// gleichzeitig und ist gegen den naheliegenden Fehler immun, dieselbe
// Herleitung zweimal falsch zu machen -- der komplexe Weg benutzt die
// Blockform gar nicht.
//
// Und zwar fuer BEIDE Formen: General (unsymmetrisch, LU) und Symmetric
// (symmetrisch indefinit, LDLt).
// =============================================================================

#include "HarmonicBlockSystem.h"

#include <complex>
#include <cstdio>
#include <cmath>
#include <vector>

static int g_failures = 0;

static void check(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  %-54s %10.3e  (Grenze %8.1e)  %s\n",
                name, err, tol, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static void check_int(const char* name, long got, long want) {
    const bool ok = (got == want);
    std::printf("  %-54s %10ld  (erwartet %8ld)  %s\n",
                name, got, want, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static void check_true(const char* name, bool ok) {
    std::printf("  %-54s %10s                      %s\n",
                name, ok ? "ja" : "nein", ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

// -----------------------------------------------------------------------------
// Dichte Gauss-Loeser, reell und komplex. Nur fuer diese winzigen Systeme.
// -----------------------------------------------------------------------------
template <typename T>
static bool gauss(std::vector<T> a, std::vector<T> b, int n, std::vector<T>& x) {
    auto at = [&](int i, int j) -> T& { return a[static_cast<size_t>(i) * n + j]; };
    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::abs(at(i, k)) > std::abs(at(piv, k))) piv = i;
        if (std::abs(at(piv, k)) < 1e-300) return false;
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(at(k, j), at(piv, j));
            std::swap(b[k], b[piv]);
        }
        for (int i = k + 1; i < n; ++i) {
            const T f = at(i, k) / at(k, k);
            if (std::abs(f) == 0.0) continue;
            for (int j = k; j < n; ++j) at(i, j) -= f * at(k, j);
            b[i] -= f * b[k];
        }
    }
    x.assign(n, T(0));
    for (int i = n - 1; i >= 0; --i) {
        T s = b[i];
        for (int j = i + 1; j < n; ++j) s -= at(i, j) * x[j];
        x[i] = s / at(i, i);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Kleines Testsystem: n = 5, K und C symmetrisch ueber EINEM CSR-Muster
// -----------------------------------------------------------------------------
// Das Muster ist absichtlich nicht bandartig, damit die Spaltensortierung der
// Blockform wirklich geprueft wird.
struct Problem {
    static const int n = 5;
    std::vector<int> row_offsets, col_indices;
    std::vector<double> K, C;
    std::vector<double> b_r, b_i;
    double omega = 2.0 * 3.14159265358979323846 * 50.0;

    Problem() {
        std::vector<std::vector<int>> pat = {
            {0, 1, 4},
            {0, 1, 2},
            {1, 2, 3, 4},
            {2, 3},
            {0, 2, 4}
        };
        row_offsets.push_back(0);
        for (int r = 0; r < n; ++r) {
            for (int c : pat[r]) col_indices.push_back(c);
            row_offsets.push_back(static_cast<int>(col_indices.size()));
        }

        // K symmetrisch positiv definit, C symmetrisch positiv semidefinit --
        // wie die Curl-Curl- und die Sigma-Massenmatrix.
        auto sym = [&](int i, int j, double base) {
            return base * (1.0 + 0.5 * std::sin(3.0 * (i + j)) );
        };
        K.resize(col_indices.size());
        C.resize(col_indices.size());
        for (int r = 0; r < n; ++r) {
            for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
                const int c = col_indices[k];
                if (r == c) {
                    K[k] = 10.0 + r;
                    C[k] = 1e-3 * (2.0 + r);
                } else {
                    K[k] = -sym(r, c, 1.0);
                    C[k] = -1e-4 * sym(r, c, 1.0);
                }
            }
        }

        b_r = {1.0, -2.0, 0.5, 3.0, -1.5};
        b_i = {0.25, 0.0, -1.0, 0.75, 2.0};
    }

    // Dichte Fassung der komplexen Matrix
    std::vector<std::complex<double>> dense_complex() const {
        std::vector<std::complex<double>> a(static_cast<size_t>(n) * n, {0.0, 0.0});
        for (int r = 0; r < n; ++r)
            for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
                a[static_cast<size_t>(r) * n + col_indices[k]] =
                    std::complex<double>(K[k], omega * C[k]);
        return a;
    }
};

static std::vector<double> dense_from_csr(int nr, const std::vector<int>& ro,
                                          const std::vector<int>& ci,
                                          const std::vector<double>& v) {
    std::vector<double> a(static_cast<size_t>(nr) * nr, 0.0);
    for (int r = 0; r < nr; ++r)
        for (int k = ro[r]; k < ro[r + 1]; ++k)
            a[static_cast<size_t>(r) * nr + ci[k]] = v[k];
    return a;
}

// -----------------------------------------------------------------------------
// 1. Struktur
// -----------------------------------------------------------------------------
static void test_structure(const Problem& p, std::vector<int>& ro2,
                           std::vector<int>& ci2) {
    fem_direct::build_block_structure(p.n, p.row_offsets.data(),
                                      p.col_indices.data(), ro2, ci2);

    check_int("Zeilen des Blocksystems", static_cast<long>(ro2.size()) - 1, 2 * Problem::n);
    check_int("nnz = 4 * nnz der Eingabe", static_cast<long>(ci2.size()),
              4 * static_cast<long>(p.col_indices.size()));

    // Spaltenindizes je Zeile aufsteigend UND im Bereich
    bool sorted = true, in_range = true;
    for (size_t r = 0; r + 1 < ro2.size(); ++r) {
        for (int k = ro2[r]; k < ro2[r + 1]; ++k) {
            if (ci2[k] < 0 || ci2[k] >= 2 * Problem::n) in_range = false;
            if (k > ro2[r] && ci2[k] <= ci2[k - 1]) sorted = false;
        }
    }
    check_true("Spaltenindizes je Zeile streng aufsteigend", sorted);
    check_true("Spaltenindizes im Bereich [0, 2n)", in_range);
}

// -----------------------------------------------------------------------------
// 2. Blockstruktur der Werte
// -----------------------------------------------------------------------------
static void test_block_layout(const Problem& p, const std::vector<int>& ro2,
                              const std::vector<int>& ci2) {
    std::vector<double> v(ci2.size());
    fem_direct::fill_block_values(p.n, p.row_offsets.data(), p.col_indices.data(),
                                  p.K.data(), p.C.data(), p.omega,
                                  fem_direct::BlockForm::General, v.data());
    const auto H = dense_from_csr(2 * Problem::n, ro2, ci2, v);
    const auto Kd = dense_from_csr(Problem::n, p.row_offsets, p.col_indices, p.K);
    const auto Cd = dense_from_csr(Problem::n, p.row_offsets, p.col_indices, p.C);

    const int n = Problem::n, N = 2 * n;
    double e11 = 0, e12 = 0, e21 = 0, e22 = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            e11 = std::fmax(e11, std::fabs(H[i*N + j]         - Kd[i*n + j]));
            e12 = std::fmax(e12, std::fabs(H[i*N + n + j]     + p.omega * Cd[i*n + j]));
            e21 = std::fmax(e21, std::fabs(H[(n+i)*N + j]     - p.omega * Cd[i*n + j]));
            e22 = std::fmax(e22, std::fabs(H[(n+i)*N + n + j] - Kd[i*n + j]));
        }
    check("General: Block (1,1) = K", e11, 0.0);
    check("General: Block (1,2) = -wC", e12, 0.0);
    check("General: Block (2,1) = +wC", e21, 0.0);
    check("General: Block (2,2) = K", e22, 0.0);

    // Symmetric
    fem_direct::fill_block_values(p.n, p.row_offsets.data(), p.col_indices.data(),
                                  p.K.data(), p.C.data(), p.omega,
                                  fem_direct::BlockForm::Symmetric, v.data());
    const auto S = dense_from_csr(N, ro2, ci2, v);
    double asym = 0.0, scale = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            asym = std::fmax(asym, std::fabs(S[i*N + j] - S[j*N + i]));
            scale = std::fmax(scale, std::fabs(S[i*N + j]));
        }
    check("Symmetric: Matrix ist symmetrisch", asym / scale, 1e-15);

    // General muss dagegen UNsymmetrisch sein -- sonst waeren die Vorzeichen
    // der Nebenblocke gleich, und das waere der Fehler, den wir suchen.
    fem_direct::fill_block_values(p.n, p.row_offsets.data(), p.col_indices.data(),
                                  p.K.data(), p.C.data(), p.omega,
                                  fem_direct::BlockForm::General, v.data());
    const auto G = dense_from_csr(N, ro2, ci2, v);
    double gasym = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            gasym = std::fmax(gasym, std::fabs(G[i*N + j] - G[j*N + i]));
    check_true("General ist erwartungsgemaess NICHT symmetrisch", gasym > 1e-6 * scale);
}

// -----------------------------------------------------------------------------
// 3. Der Aequivalenznachweis gegen die komplexe Loesung
// -----------------------------------------------------------------------------
static void test_against_complex(const Problem& p, const std::vector<int>& ro2,
                                 const std::vector<int>& ci2) {
    const int n = Problem::n, N = 2 * n;

    // Weg A: komplex loesen, ohne jede Blockform
    std::vector<std::complex<double>> bc(n);
    for (int i = 0; i < n; ++i) bc[i] = {p.b_r[i], p.b_i[i]};
    std::vector<std::complex<double>> xc;
    check_true("komplexes System loesbar", gauss(p.dense_complex(), bc, n, xc));

    // Weg B: beide reellen Blockformen
    const fem_direct::BlockForm forms[2] = {fem_direct::BlockForm::General,
                                            fem_direct::BlockForm::Symmetric};
    const char* names[2] = {"General", "Symmetric"};

    for (int f = 0; f < 2; ++f) {
        std::vector<double> v(ci2.size());
        fem_direct::fill_block_values(p.n, p.row_offsets.data(), p.col_indices.data(),
                                      p.K.data(), p.C.data(), p.omega, forms[f], v.data());
        std::vector<double> rhs(N);
        fem_direct::fill_block_rhs(p.n, p.b_r.data(), p.b_i.data(), forms[f], rhs.data());

        std::vector<double> x2n;
        char buf[128];
        std::snprintf(buf, sizeof buf, "%s: 2n-System loesbar", names[f]);
        check_true(buf, gauss(dense_from_csr(N, ro2, ci2, v), rhs, N, x2n));

        std::vector<double> xr(n), xi(n);
        fem_direct::split_block_solution(p.n, x2n.data(), xr.data(), xi.data());

        double err = 0.0, norm = 0.0;
        for (int i = 0; i < n; ++i) {
            err = std::fmax(err, std::fabs(xr[i] - xc[i].real()));
            err = std::fmax(err, std::fabs(xi[i] - xc[i].imag()));
            norm = std::fmax(norm, std::abs(xc[i]));
        }
        std::snprintf(buf, sizeof buf, "%s: Loesung gleich der komplexen", names[f]);
        check(buf, err / norm, 1e-12);
    }

    // b_i == nullptr muss wie b_i == 0 wirken
    std::vector<double> v(ci2.size());
    fem_direct::fill_block_values(p.n, p.row_offsets.data(), p.col_indices.data(),
                                  p.K.data(), p.C.data(), p.omega,
                                  fem_direct::BlockForm::General, v.data());
    std::vector<double> rhs_null(N), rhs_zero(N);
    const std::vector<double> zeros(n, 0.0);
    fem_direct::fill_block_rhs(p.n, p.b_r.data(), nullptr,
                               fem_direct::BlockForm::General, rhs_null.data());
    fem_direct::fill_block_rhs(p.n, p.b_r.data(), zeros.data(),
                               fem_direct::BlockForm::General, rhs_zero.data());
    double d = 0.0;
    for (int i = 0; i < N; ++i) d = std::fmax(d, std::fabs(rhs_null[i] - rhs_zero[i]));
    check("b_i == nullptr wirkt wie b_i == 0", d, 0.0);
}

// -----------------------------------------------------------------------------
// 4. Oberes Dreieck
// -----------------------------------------------------------------------------
static void test_triangle(const Problem& p, const std::vector<int>& ro2,
                          const std::vector<int>& ci2) {
    const int N = 2 * Problem::n;

    std::vector<double> v(ci2.size());
    fem_direct::fill_block_values(p.n, p.row_offsets.data(), p.col_indices.data(),
                                  p.K.data(), p.C.data(), p.omega,
                                  fem_direct::BlockForm::Symmetric, v.data());

    std::vector<int> tro, tci;
    std::vector<double> tv;
    const size_t nnz_tri = fem_direct::extract_upper_triangle(
        N, ro2.data(), ci2.data(), v.data(), tro, tci, tv);

    check_true("Dreieck kleiner als die volle Matrix", nnz_tri < v.size());

    // Aus dem Dreieck die volle Matrix rekonstruieren und vergleichen. Das ist
    // genau, was cuDSS bei MTYPE_SYMMETRIC intern tut.
    std::vector<double> rec(static_cast<size_t>(N) * N, 0.0);
    for (int r = 0; r < N; ++r)
        for (int k = tro[r]; k < tro[r + 1]; ++k) {
            const int c = tci[k];
            rec[static_cast<size_t>(r) * N + c] = tv[k];
            rec[static_cast<size_t>(c) * N + r] = tv[k];
        }
    const auto full = dense_from_csr(N, ro2, ci2, v);
    double err = 0.0;
    for (size_t i = 0; i < full.size(); ++i)
        err = std::fmax(err, std::fabs(full[i] - rec[i]));
    check("aus dem Dreieck rekonstruiert = volle Matrix", err, 0.0);

    // Aus General darf KEIN Dreieck gezogen werden
    fem_direct::fill_block_values(p.n, p.row_offsets.data(), p.col_indices.data(),
                                  p.K.data(), p.C.data(), p.omega,
                                  fem_direct::BlockForm::General, v.data());
    bool threw = false;
    try {
        fem_direct::extract_upper_triangle(N, ro2.data(), ci2.data(), v.data(),
                                           tro, tci, tv);
    } catch (const std::exception&) { threw = true; }
    check_true("Dreieck aus General bricht ab", threw);
}

int main() {
    Problem p;
    std::vector<int> ro2, ci2;

    std::printf("\n== Muster des 2n-Systems ==\n");
    test_structure(p, ro2, ci2);

    std::printf("\n== Lage der Bloecke ==\n");
    test_block_layout(p, ro2, ci2);

    std::printf("\n== Gegen die komplexe Loesung ==\n");
    test_against_complex(p, ro2, ci2);

    std::printf("\n== Oberes Dreieck fuer CUDSS_MTYPE_SYMMETRIC ==\n");
    test_triangle(p, ro2, ci2);

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
