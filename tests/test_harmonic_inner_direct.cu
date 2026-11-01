// =============================================================================
// Abnahmepruefung: --harmonic-inner-direct (C-to-R mit dichter Zerlegung)
//
//   cmake --build build --config Release --target test_harmonic_inner_direct
//   ./build/Release/test_harmonic_inner_direct
//
// Reiner Host-Code, keine GPU, kein Netz.
//
// WORUM ES GEHT
// -------------
// Auf dem BEM-FEM-Hexaedernetz erreicht das innere GMRES von C-to-R seine
// Toleranz nicht (docs/design/gmsh-phase9-report.md). --harmonic-inner-direct
// tastet (W+T) = K + omega*C spaltenweise ab, zerlegt die dichte Matrix und
// loest jede innere Gleichung exakt. Geprueft wird:
//
//   1. die Abtastung: A e_j landet in Spalte j (column-major, wie cuSOLVER es
//      erwartet). K ist dafuer bewusst UNSYMMETRISCH -- bei symmetrischem K
//      waere eine vertauschte Ablage unsichtbar.
//   2. solve_harmonic() mit Zerlegung gegen die dichte Loesung des komplexen
//      Systems (K + j omega C) x = b, aufgeschrieben als reelle 2n-Blockform.
//   3. dass der exakte Zweig wirklich laeuft: je innerer Loesung genau ein
//      "Schritt", und einige Dutzend aeussere Iterationen bis 1e-12.
//
// GEGENPROBEN
// -----------
//   * die abgetastete Matrix ZEILENWEISE gelesen muss deutlich abweichen
//     (sonst prueft Punkt 1 nichts);
//   * ohne Zerlegung, mit innerem GMRES auf einen Schritt begrenzt (und ohne
//     ILU, die auf diesem Tridiagonalsystem exakt waere): der aeussere Loeser
//     verfehlt die Toleranz innerhalb desselben Deckels.
// =============================================================================

#include "Backend.cuh"
#include "CoupledSolver.cuh"
#include "DenseLinalg.cuh"
#include "HarmonicSolver.cuh"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace bf { extern Backend g_backend; }

static int g_failures = 0;

static void check(bool ok, const char* what) {
    std::printf("  %-14s %s\n", ok ? "ok" : "FEHLGESCHLAGEN", what);
    if (!ok) ++g_failures;
}

struct Csr {
    std::vector<int> row, col;
    std::vector<double> val;
};

// K: 1D-Laplace plus unsymmetrischer Konvektionsterm, omega*C: Masse nur auf
// einem "Leiter" in der Mitte -- wie bei TEAM 28 traegt C nicht ueberall.
static const int    N     = 40;
static const double OMEGA = 3.0;
static bool in_conductor(int i) { return i >= 12 && i < 28; }

static double k_entry(int i, int j) {
    if (i == j) return 2.0;
    if (j == i - 1) return -1.0 - 0.3;   // unsymmetrisch: K_i,i-1 != K_i-1,i
    if (j == i + 1) return -1.0 + 0.3;
    return 0.0;
}
static double c_entry(int i) { return in_conductor(i) ? 0.8 : 0.0; }

static Csr build_W() {
    Csr a;
    a.row.push_back(0);
    for (int i = 0; i < N; ++i) {
        for (int j = std::max(0, i - 1); j <= std::min(N - 1, i + 1); ++j) {
            double v = k_entry(i, j);
            if (i == j) v += OMEGA * c_entry(i);
            a.col.push_back(j);
            a.val.push_back(v);
        }
        a.row.push_back(static_cast<int>(a.col.size()));
    }
    return a;
}

static void fill_C(CSRMatrix& C) {
    C.num_rows = N;
    C.nnz = N;
    C.values.allocate(N);
    C.col_indices.allocate(N);
    C.row_offsets.allocate(N + 1);
    for (int i = 0; i < N; ++i) {
        C.values.data()[i] = c_entry(i);
        C.col_indices.data()[i] = i;
        C.row_offsets.data()[i] = i;
    }
    C.row_offsets.data()[N] = N;
}

// Bezug: [[K, -wC], [wC, K]] [x_r; x_i] = [b_r; b_i], dicht geloest.
static void reference(const std::vector<double>& br, std::vector<double>& xr,
                      std::vector<double>& xi) {
    const int n2 = 2 * N;
    std::vector<double> H(static_cast<size_t>(n2) * n2, 0.0);
    auto at = [&](int i, int j) -> double& { return H[static_cast<size_t>(j) * n2 + i]; };
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            at(i, j) = k_entry(i, j);
            at(N + i, N + j) = k_entry(i, j);
        }
        at(i, N + i) = -OMEGA * c_entry(i);
        at(N + i, i) = +OMEGA * c_entry(i);
    }
    std::vector<double> rhs(n2, 0.0);
    for (int i = 0; i < N; ++i) rhs[i] = br[i];
    bf::DenseLU lu;
    lu.factorize(H.data(), n2);
    lu.solve(rhs.data(), 1);
    xr.assign(rhs.begin(), rhs.begin() + N);
    xi.assign(rhs.begin() + N, rhs.end());
}

static double rel_err(const std::vector<double>& ar, const std::vector<double>& ai,
                      const double* br, const double* bi) {
    double d = 0.0, s = 0.0;
    for (int i = 0; i < N; ++i) {
        d += (ar[i] - br[i]) * (ar[i] - br[i]) + (ai[i] - bi[i]) * (ai[i] - bi[i]);
        s += ar[i] * ar[i] + ai[i] * ai[i];
    }
    return std::sqrt(d / s);
}

struct Run { bf::HarmonicSolveResult res; double err; };

static Run run(CoupledMVOperator& op, const CSRMatrix& C, const std::vector<double>& br,
               const std::vector<double>& ref_r, const std::vector<double>& ref_i,
               const bf::DenseLU* direct, int inner_max_iter, int outer_cap) {
    DeviceArray<double> d_br(N), d_bi(N), d_xr(N), d_xi(N);
    for (int i = 0; i < N; ++i) {
        d_br.data()[i] = br[i];
        d_bi.data()[i] = 0.0;
        d_xr.data()[i] = 0.0;
        d_xi.data()[i] = 0.0;
    }
    Run r;
    r.res = bf::solve_harmonic(op, C, N, N, OMEGA, d_br.data(), d_bi.data(),
                               d_xr.data(), d_xi.data(), 1e-12, outer_cap, 64,
                               1e-4, inner_max_iter, direct);
    r.err = rel_err(ref_r, ref_i, d_xr.data(), d_xi.data());
    return r;
}

int main() {
    bf::g_backend = bf::Backend::CPU;

    Csr W = build_W();
    CoupledMVOperator op(N);
    op.setup_fem_matrix(N, static_cast<int>(W.val.size()), W.row.data(), W.col.data(),
                        W.val.data());
    op.compute_ilu();
    CSRMatrix C;
    fill_C(C);

    std::mt19937 rng(7u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<double> br(N);
    for (double& v : br) v = uni(rng);
    std::vector<double> ref_r, ref_i;
    reference(br, ref_r, ref_i);

    std::printf("Abtastung von (W+T)\n");
    std::vector<double> A(static_cast<size_t>(N) * N, 0.0);
    bf::sample_operator_dense(op, N, A.data());
    double worst = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double expect = k_entry(i, j) + (i == j ? OMEGA * c_entry(i) : 0.0);
            worst = std::max(worst, std::fabs(A[static_cast<size_t>(j) * N + i] - expect));
        }
    char buf[160];
    std::snprintf(buf, sizeof buf, "A(i,j) = (W+T)_ij column-major, max. Abweichung %.1e", worst);
    check(worst < 1e-14, buf);

    std::printf("C-to-R mit dichter Zerlegung\n");
    bf::DenseLU lu;
    lu.factorize(A.data(), N);
    Run d = run(op, C, br, ref_r, ref_i, &lu, 100, 200);
    // Die Schranke [1/2, 1] fuer das Spektrum von P^-1 H gilt nur bei
    // symmetrischem K; hier ist K absichtlich unsymmetrisch (Abtastung), und
    // gemessen sind 32 Iterationen bis 1e-12. Die Grenze 40 haelt nur fest,
    // dass es bei wenigen Dutzend bleibt; die Gegenprobe unten zeigt den
    // Unterschied zum begrenzten inneren GMRES.
    std::snprintf(buf, sizeof buf, "konvergiert in %d aeusseren Iterationen (<= 40)",
                  d.res.iterations);
    check(d.res.converged && d.res.iterations <= 40, buf);
    std::snprintf(buf, sizeof buf, "Fehler gegen die dichte 2n-Loesung %.1e < 1e-9", d.err);
    check(d.err < 1e-9, buf);
    std::snprintf(buf, sizeof buf, "exakter Zweig: %ld innere Loesungen, %ld Schritte",
                  d.res.inner_solves, d.res.inner_iters);
    check(d.res.inner_solves > 0 && d.res.inner_iters == d.res.inner_solves, buf);

    std::printf("Gegenproben\n");
    std::vector<double> At(A.size(), 0.0);
    bf::sample_operator_dense(op, N, At.data());
    double worst_t = 0.0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            double expect = k_entry(i, j) + (i == j ? OMEGA * c_entry(i) : 0.0);
            // Ablage zeilenweise gelesen: so saehe eine vertauschte Abtastung aus
            worst_t = std::max(worst_t, std::fabs(At[static_cast<size_t>(i) * N + j] - expect));
        }
    std::snprintf(buf, sizeof buf,
                  "zeilenweise gelesen weicht die Ablage um %.2f ab (K unsymmetrisch)", worst_t);
    check(worst_t > 0.5, buf);

    // Ohne ILU: ILU(0) ist auf einer Tridiagonalmatrix exakt, dann waere auch
    // ein einziger innerer GMRES-Schritt schon eine exakte Loesung.
    op.set_ilu_bypass(true);
    Run g = run(op, C, br, ref_r, ref_i, nullptr, 1, d.res.iterations);
    std::snprintf(buf, sizeof buf,
                  "inneres GMRES mit 1 Schritt: nach %d Iterationen nicht konvergiert "
                  "(Fehler %.1e)", d.res.iterations, g.err);
    check(!g.res.converged, buf);

    if (g_failures) {
        std::printf("\n%d Pruefung(en) fehlgeschlagen\n", g_failures);
        return 1;
    }
    std::printf("\nalle Pruefungen bestanden\n");
    return 0;
}
