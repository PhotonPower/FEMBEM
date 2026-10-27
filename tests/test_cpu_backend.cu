// =============================================================================
// test_cpu_backend.cu -- Abnahmepruefung der CPU-Linearalgebra
// =============================================================================
//
// Der CPU-Modus (--cpu) ersetzt cuBLAS und cuSPARSE durch eigene Routinen in
// src/HostLinalg.cu. Die Kernel selbst teilen ihren __host__ __device__ Rumpf
// mit dem GPU-Pfad und koennen daher nicht auseinanderlaufen -- die
// Linearalgebra dagegen ist neu geschriebener Code und damit die Stelle, an
// der ein Fehler tatsaechlich moeglich ist. Genau die wird hier geprueft:
//
//   1. bf::csr_spmv   gegen eine dichte Referenzrechnung
//   2. bf::dgemv      (N und T) gegen eine dichte Referenzrechnung
//   3. bf::csr_ilu0   gegen die definierende ILU(0)-Eigenschaft:
//                     (L*U)_ij == A_ij fuer alle (i,j) im Besetzungsmuster
//   4. bf::csr_ilu0 + bf::csr_lu_solve als EXAKTER Loeser fuer eine
//                     Tridiagonalmatrix (dort entsteht bei der LU-Zerlegung
//                     kein Fill-in, ILU(0) ist also die exakte LU)
//   5. bf::dnrm2      Ueberlaufsicherheit bei extremen Komponenten
//   6. bf::ddot / daxpy / dscal / dcopy gegen Handrechnung
//
// Bauen und ausfuehren:
//   cmake --build build --config Release --target test_cpu_backend
//   ./build/test_cpu_backend
//
// Der Test braucht KEINE GPU -- er stellt bf::g_backend selbst auf CPU.
// =============================================================================

#include "Backend.cuh"
#include "HostLinalg.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", what);
    if (!ok) ++g_failures;
}

void check_close(double a, double b, double tol, const char* what) {
    const double scale = std::fmax(1.0, std::fmax(std::fabs(a), std::fabs(b)));
    const bool ok = std::fabs(a - b) <= tol * scale;
    std::printf("  [%s] %s (erhalten %.17g, erwartet %.17g)\n",
                ok ? "OK " : "FEHL", what, a, b);
    if (!ok) ++g_failures;
}

// -----------------------------------------------------------------------------
// Kleine dichte Hilfsmatrix, Zeilen-Major, nur fuer die Referenzrechnungen.
// -----------------------------------------------------------------------------
struct Dense {
    int n;
    std::vector<double> a;
    explicit Dense(int n_) : n(n_), a(static_cast<size_t>(n_) * n_, 0.0) {}
    double&       at(int i, int j)       { return a[static_cast<size_t>(i) * n + j]; }
    const double& at(int i, int j) const { return a[static_cast<size_t>(i) * n + j]; }
};

/// Baut eine CSR-Darstellung aus einer dichten Matrix. Es werden genau die
/// Eintraege aufgenommen, die dort nicht exakt 0 sind -- die Spaltenindizes
/// entstehen dabei automatisch aufsteigend sortiert, so wie
/// build_csr_structure() sie ueber std::set ebenfalls liefert (Voraussetzung
/// von bf::csr_ilu0).
void to_csr(const Dense& d, std::vector<int>& rp, std::vector<int>& ci,
            std::vector<double>& v)
{
    rp.assign(static_cast<size_t>(d.n) + 1, 0);
    ci.clear();
    v.clear();
    for (int i = 0; i < d.n; ++i) {
        rp[i] = static_cast<int>(ci.size());
        for (int j = 0; j < d.n; ++j) {
            if (d.at(i, j) != 0.0) { ci.push_back(j); v.push_back(d.at(i, j)); }
        }
    }
    rp[d.n] = static_cast<int>(ci.size());
}

// =============================================================================
// 1. CSR-Matrix-Vektor-Produkt
// =============================================================================
void test_csr_spmv() {
    std::puts("\n[1] bf::csr_spmv gegen dichte Referenz");

    const int n = 40;
    Dense d(n);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    for (int i = 0; i < n; ++i) {
        d.at(i, i) = 4.0 + uni(rng);                 // Diagonale immer besetzt
        if (i > 0)     d.at(i, i - 1) = uni(rng);
        if (i + 1 < n) d.at(i, i + 1) = uni(rng);
        if (i + 7 < n) d.at(i, i + 7) = uni(rng);    // etwas Fernkopplung
    }

    std::vector<int> rp, ci;
    std::vector<double> vals;
    to_csr(d, rp, ci, vals);

    std::vector<double> x(n), y(n, 0.0), y_ref(n, 0.0);
    for (int i = 0; i < n; ++i) x[i] = uni(rng);

    for (int i = 0; i < n; ++i) {
        double acc = 0.0;
        for (int j = 0; j < n; ++j) acc += d.at(i, j) * x[j];
        y_ref[i] = acc;
    }

    bf::csr_spmv(n, rp.data(), ci.data(), vals.data(), x.data(), y.data());

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) max_err = std::fmax(max_err, std::fabs(y[i] - y_ref[i]));
    check(max_err < 1e-13, "y = A*x stimmt mit der dichten Rechnung ueberein");
    std::printf("      max|y - y_ref| = %.3e\n", max_err);
}

// =============================================================================
// 2. dgemv, beide Transpositionen
// =============================================================================
void test_dgemv() {
    std::puts("\n[2] bf::dgemv (CUBLAS_OP_N und CUBLAS_OP_T) gegen dichte Referenz");

    // A ist m x n in COLUMN-MAJOR -- genau das Format, in dem die BEM-Bloecke
    // (dense_blocks, ACA-Faktoren U/V) im Speicher liegen.
    const int m = 7, n = 5, lda = m;
    std::vector<double> A(static_cast<size_t>(lda) * n);
    std::mt19937 rng(777);
    std::uniform_real_distribution<double> uni(-2.0, 2.0);
    for (auto& e : A) e = uni(rng);

    auto Aij = [&](int i, int j) { return A[static_cast<size_t>(j) * lda + i]; };

    const double alpha = 1.7, beta = -0.4;

    // --- OP_N: y (m) = alpha*A*x (n) + beta*y
    {
        std::vector<double> x(n), y(m), y_ref(m);
        for (int j = 0; j < n; ++j) x[j] = uni(rng);
        for (int i = 0; i < m; ++i) { y[i] = uni(rng); y_ref[i] = y[i]; }

        for (int i = 0; i < m; ++i) {
            double acc = 0.0;
            for (int j = 0; j < n; ++j) acc += Aij(i, j) * x[j];
            y_ref[i] = alpha * acc + beta * y_ref[i];
        }
        bf::dgemv(nullptr, /*transpose=*/false, m, n, alpha, A.data(), lda,
                  x.data(), beta, y.data());

        double e = 0.0;
        for (int i = 0; i < m; ++i) e = std::fmax(e, std::fabs(y[i] - y_ref[i]));
        check(e < 1e-13, "OP_N: y = alpha*A*x + beta*y");
    }

    // --- OP_T: y (n) = alpha*A^T*x (m) + beta*y
    {
        std::vector<double> x(m), y(n), y_ref(n);
        for (int i = 0; i < m; ++i) x[i] = uni(rng);
        for (int j = 0; j < n; ++j) { y[j] = uni(rng); y_ref[j] = y[j]; }

        for (int j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int i = 0; i < m; ++i) acc += Aij(i, j) * x[i];
            y_ref[j] = alpha * acc + beta * y_ref[j];
        }
        bf::dgemv(nullptr, /*transpose=*/true, m, n, alpha, A.data(), lda,
                  x.data(), beta, y.data());

        double e = 0.0;
        for (int j = 0; j < n; ++j) e = std::fmax(e, std::fabs(y[j] - y_ref[j]));
        check(e < 1e-13, "OP_T: y = alpha*A^T*x + beta*y");
    }
}

// =============================================================================
// 3. ILU(0): definierende Eigenschaft auf dem Besetzungsmuster
// =============================================================================
void test_ilu0_pattern_property() {
    std::puts("\n[3] bf::csr_ilu0: (L*U)_ij == A_ij auf dem Besetzungsmuster");

    const int n = 30;
    Dense d(n);
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    // Diagonaldominant, damit ILU(0) ohne Pivotierung nicht entartet -- das
    // ist auch die Situation, die k_ilu_gauge_aware_setup() mit dem
    // Gershgorin-Shift im Loeser gezielt herstellt.
    for (int i = 0; i < n; ++i) {
        double off = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            if ((i + j) % 5 == 0 || j == i - 1 || j == i + 1) {
                const double v = uni(rng);
                d.at(i, j) = v;
                off += std::fabs(v);
            }
        }
        d.at(i, i) = off + 1.5;
    }

    std::vector<int> rp, ci;
    std::vector<double> vals;
    to_csr(d, rp, ci, vals);

    std::vector<double> lu = vals;   // in-place-Faktorisierung auf der Kopie
    std::vector<int> diag;
    const int zp = bf::csr_ilu0(n, rp.data(), ci.data(), lu.data(), diag);
    check(zp < 0, "Faktorisierung ohne Nullpivot");
    if (zp >= 0) return;

    // Packed L\U in dichte L und U aufloesen und ausmultiplizieren.
    Dense L(n), U(n);
    for (int i = 0; i < n; ++i) {
        L.at(i, i) = 1.0;                                  // Einheitsdiagonale
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            const int j = ci[k];
            if (j < i) L.at(i, j) = lu[k];
            else       U.at(i, j) = lu[k];
        }
    }

    double max_err_pattern = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int k = rp[i]; k < rp[i + 1]; ++k) {
            const int j = ci[k];
            double acc = 0.0;
            for (int t = 0; t < n; ++t) acc += L.at(i, t) * U.at(t, j);
            max_err_pattern = std::fmax(max_err_pattern, std::fabs(acc - d.at(i, j)));
        }
    }
    check(max_err_pattern < 1e-11,
          "L*U reproduziert A exakt auf allen besetzten Positionen");
    std::printf("      max|(LU)_ij - A_ij| auf dem Muster = %.3e\n", max_err_pattern);
}

// =============================================================================
// 4. ILU(0) + Substitution als exakter Loeser fuer eine Tridiagonalmatrix
// =============================================================================
void test_ilu0_tridiagonal_is_exact() {
    std::puts("\n[4] Tridiagonalfall: ILU(0) ist die exakte LU, also exakter Loeser");

    // Bei einer Tridiagonalmatrix entsteht in der LU-Zerlegung kein Fill-in.
    // ILU(0) (das Fill-in gerade verwirft) ist damit die EXAKTE Zerlegung --
    // die Substitutionen muessen A*x = b bis auf Rundung genau loesen. Das
    // prueft Faktorisierung und beide Substitutionen in einem Zug und ist
    // deutlich scharfer als eine reine Toleranzabfrage auf einem
    // Vorkonditionierer.
    const int n = 64;
    Dense d(n);
    for (int i = 0; i < n; ++i) {
        d.at(i, i) = 2.0;
        if (i > 0)     d.at(i, i - 1) = -1.0;
        if (i + 1 < n) d.at(i, i + 1) = -1.0;
    }

    std::vector<int> rp, ci;
    std::vector<double> vals;
    to_csr(d, rp, ci, vals);

    std::vector<double> lu = vals;
    std::vector<int> diag;
    const int zp = bf::csr_ilu0(n, rp.data(), ci.data(), lu.data(), diag);
    check(zp < 0, "Faktorisierung ohne Nullpivot");
    if (zp >= 0) return;

    std::vector<double> b(n), x(n, 0.0), tmp(n, 0.0);
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int i = 0; i < n; ++i) b[i] = uni(rng);

    bf::csr_lu_solve(n, rp.data(), ci.data(), lu.data(), diag,
                     b.data(), x.data(), tmp.data());

    // Residuum ||A*x - b|| / ||b||
    std::vector<double> Ax(n, 0.0);
    bf::csr_spmv(n, rp.data(), ci.data(), vals.data(), x.data(), Ax.data());

    double num = 0.0, den = 0.0;
    bf::dnrm2(nullptr, n, b.data(), &den);
    for (int i = 0; i < n; ++i) { const double r = Ax[i] - b[i]; num += r * r; }
    const double rel = std::sqrt(num) / den;

    check(rel < 1e-12, "relatives Residuum ||A*x - b||/||b|| auf Rundungsniveau");
    std::printf("      rel. Residuum = %.3e\n", rel);
}

// =============================================================================
// 5. dnrm2: Ueberlaufsicherheit
// =============================================================================
void test_dnrm2_extremes() {
    std::puts("\n[5] bf::dnrm2 bei extremen Komponenten");

    // Naive Summation von x_i^2 wuerde hier auf +inf laufen (1e200^2 = 1e400).
    // Die skalierte Summation in HostLinalg muss sqrt(2)*1e200 liefern.
    // Relevant, weil die Diagnoseausgaben des Loesers Komponenten von ~1e-8
    // bis ~1e9 im selben Vektor zeigen und der SDE-Schritt daran noch dreht.
    {
        std::vector<double> x = {1e200, 1e200};
        double r = 0.0;
        bf::dnrm2(nullptr, 2, x.data(), &r);
        check(std::isfinite(r), "kein Ueberlauf bei sehr grossen Komponenten");
        check_close(r, std::sqrt(2.0) * 1e200, 1e-14, "Wert korrekt (gross)");
    }
    {
        std::vector<double> x = {1e-200, 1e-200};
        double r = 0.0;
        bf::dnrm2(nullptr, 2, x.data(), &r);
        check(r > 0.0, "kein Unterlauf auf 0 bei sehr kleinen Komponenten");
        check_close(r, std::sqrt(2.0) * 1e-200, 1e-14, "Wert korrekt (klein)");
    }
    {
        std::vector<double> x = {3.0, 4.0};
        double r = 0.0;
        bf::dnrm2(nullptr, 2, x.data(), &r);
        check_close(r, 5.0, 1e-15, "Normalfall ||(3,4)|| = 5");
    }
    {
        double r = 123.0;
        bf::dnrm2(nullptr, 0, nullptr, &r);
        check_close(r, 0.0, 1e-15, "n == 0 liefert 0");
    }
}

// =============================================================================
// 6. Uebrige BLAS-1-Operationen
// =============================================================================
void test_blas1() {
    std::puts("\n[6] bf::ddot / daxpy / dscal / dcopy");

    std::vector<double> x = {1.0, 2.0, 3.0, 4.0};
    std::vector<double> y = {10.0, 20.0, 30.0, 40.0};

    double dot = 0.0;
    bf::ddot(nullptr, 4, x.data(), y.data(), &dot);
    check_close(dot, 10.0 + 40.0 + 90.0 + 160.0, 1e-15, "ddot");

    std::vector<double> y1 = y;
    bf::daxpy(nullptr, 4, 2.0, x.data(), y1.data());   // y += 2*x
    check_close(y1[0], 12.0, 1e-15, "daxpy Eintrag 0");
    check_close(y1[3], 48.0, 1e-15, "daxpy Eintrag 3");

    std::vector<double> x1 = x;
    bf::dscal(nullptr, 4, -3.0, x1.data());
    check_close(x1[2], -9.0, 1e-15, "dscal");

    std::vector<double> dst(4, 0.0);
    bf::dcopy(nullptr, 4, x.data(), dst.data());
    check_close(dst[1], 2.0, 1e-15, "dcopy");

    // dcopy mit identischem Quell- und Zielzeiger muss ein No-Op sein
    // (std::memcpy auf sich selbst waere undefiniertes Verhalten).
    bf::dcopy(nullptr, 4, dst.data(), dst.data());
    check_close(dst[1], 2.0, 1e-15, "dcopy mit x == y ist unschaedlich");
}

// =============================================================================
// 7. Speicher-Wrapper im CPU-Modus
// =============================================================================
void test_memory_wrappers() {
    std::puts("\n[7] bf::malloc_d / memset_d / memcpy_d / free_d im CPU-Modus");

    double* p = nullptr;
    bf::malloc_d(&p, 8 * sizeof(double));
    check(p != nullptr, "malloc_d liefert einen Zeiger");

    bf::memset_d(p, 0, 8 * sizeof(double));
    bool all_zero = true;
    for (int i = 0; i < 8; ++i) all_zero = all_zero && (p[i] == 0.0);
    check(all_zero, "memset_d nullt den Block");

    const double src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    // Richtung ist im CPU-Modus bedeutungslos, wird aber wie im GPU-Pfad
    // uebergeben -- der Aufrufcode ist in beiden Modi derselbe.
    bf::memcpy_d(p, src, 8 * sizeof(double), cudaMemcpyHostToDevice);
    check_close(p[5], 6.0, 1e-15, "memcpy_d kopiert");

    bf::free_d(p);
    bf::free_d(nullptr);   // muss folgenlos bleiben
    check(true, "free_d(nullptr) ist folgenlos");

    // sync() darf im CPU-Modus nichts tun und insbesondere nicht werfen.
    bf::sync();
    check(true, "sync() im CPU-Modus ist ein No-Op");
}

} // namespace

int main() {
    // Der Test prueft ausschliesslich den CPU-Pfad und braucht deshalb weder
    // Geraet noch Treiber.
    bf::g_backend = bf::Backend::CPU;

    std::puts("========================================================");
    std::puts(" Abnahmepruefung CPU-Backend (bf::, src/HostLinalg.cu)  ");
    std::puts("========================================================");
    std::printf("Aktives Backend: %s\n", bf::backend_name());

    test_csr_spmv();
    test_dgemv();
    test_ilu0_pattern_property();
    test_ilu0_tridiagonal_is_exact();
    test_dnrm2_extremes();
    test_blas1();
    test_memory_wrappers();

    std::puts("\n========================================================");
    if (g_failures == 0) {
        std::puts(" ALLE PRUEFUNGEN BESTANDEN");
        std::puts("========================================================");
        return EXIT_SUCCESS;
    }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_failures);
    std::puts("========================================================");
    return EXIT_FAILURE;
}
