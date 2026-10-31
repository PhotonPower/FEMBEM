// =============================================================================
// Abnahmeprüfung: brechen GMRES und BiCGStab bei NaN/Inf sofort ab?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_solver_nonfinite_abort
//   ./build/test_solver_nonfinite_abort
//
// Reiner Host-Code, keine GPU, kein Netz von der Platte. Läuft in Sekunden.
//
// HINTERGRUND
// -----------
// Die Abbruchkriterien der Krylov-Löser haben die Form "res/b_norm <= tol".
// Für NaN ist das falsch; der Löser meldet also keine Konvergenz. Das ist die
// sichere Richtung, aber er lief bis max_iter weiter -- mit Vorkonditionierer
// bis zu 1000 Iterationen -- und gab am Ende ein NaN-x zurück
// (docs/design/nan-and-thresholds.md, „Bewusst stehen gelassen“).
//
// Schlimmer waren zwei Aufrufer in fem.cu, die "last_relative_residual() >
// tol" als Kriterium für NICHT konvergiert nahmen. Für NaN ist auch das
// falsch: ein NaN-Ergebnis ging dort als konvergiert durch.
//
// Jetzt prüfen die Löser ||b||, das Residuum am Zyklusbeginn und (GMRES) das
// mitgeführte Hessenberg-Residuum auf Endlichkeit, brechen sofort ab, lassen
// x beim letzten endlichen Stand und melden ein NaN-Residuum.
//
// GEPRÜFT
// -------
//   1. Kontrollfall: endliches System, GMRES konvergiert (der Aufbau taugt).
//   2. NaN in b                    -> GMRES bricht bei Iteration 0 ab.
//   3. NaN in der Matrix           -> dito (A*0 = NaN*0 = NaN im Startresiduum).
//   4. Überlauf erst IM Zyklus     -> GMRES bricht bei Iteration 1 ab, x bleibt
//      endlich. A*0 ist hier endlich (1.5e308 * 0 = 0); erst A*v mit v != 0
//      läuft über. Das deckt die Prüfung des Hessenberg-Residuums ab, die
//      Fälle 2 und 3 gar nicht erreichen.
//   5. BiCGStab mit NaN in b       -> x bleibt endlich (vorher: NaN-x).
//   6. BiCGStab mit Überlauf im Lauf -> x bleibt endlich.
//
// Ohne die Änderung fallen 2, 3 und 4 durch (Iterationen = max_iter) und 5, 6
// ebenfalls (x enthält NaN).
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "Backend.cuh"
#include "CoupledSolver.cuh"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace bf { extern Backend g_backend; }

static int g_failures = 0;

static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " OK " : "FEHL", what);
    if (!ok) ++g_failures;
}

// Tridiagonale SPD-Matrix (2, -1) der Größe n als CSR. Optional wird ein
// Eintrag der ersten Zeile ersetzt bzw. die erste Zeile ganz überschrieben.
struct Csr {
    std::vector<int> row, col;
    std::vector<double> val;
};

static Csr laplace_1d(int n) {
    Csr a;
    a.row.push_back(0);
    for (int i = 0; i < n; ++i) {
        if (i > 0)     { a.col.push_back(i - 1); a.val.push_back(-1.0); }
        a.col.push_back(i); a.val.push_back(2.0);
        if (i < n - 1) { a.col.push_back(i + 1); a.val.push_back(-1.0); }
        a.row.push_back(static_cast<int>(a.col.size()));
    }
    return a;
}

// Erste Zeile dicht mit 1.5e308 auf den ersten drei Spalten: A*0 = 0, aber
// A*v mit v = b/||b|| (b = Einsen, n = 4, v_i = 0.5) ergibt 2.25e308 > DBL_MAX,
// also Inf.
static Csr overflow_in_cycle(int n) {
    Csr a;
    a.row.push_back(0);
    for (int i = 0; i < n; ++i) {
        if (i == 0) {
            for (int j = 0; j < 3; ++j) { a.col.push_back(j); a.val.push_back(1.5e308); }
        } else {
            if (i > 0)     { a.col.push_back(i - 1); a.val.push_back(-1.0); }
            a.col.push_back(i); a.val.push_back(2.0);
            if (i < n - 1) { a.col.push_back(i + 1); a.val.push_back(-1.0); }
        }
        a.row.push_back(static_cast<int>(a.col.size()));
    }
    return a;
}

static void setup(CoupledMVOperator& op, Csr& a, int n) {
    op.setup_fem_matrix(n, static_cast<int>(a.val.size()),
                        a.row.data(), a.col.data(), a.val.data());
    // Ohne Vorkonditionierer: die Prüfung gilt dem Krylov-Löser selbst, und
    // ILU(0) auf einer Matrix mit NaN oder 1e308 wäre ein eigenes Thema.
    op.set_ilu_bypass(true);
}

static bool all_finite(const std::vector<double>& v) {
    for (double x : v) if (!std::isfinite(x)) return false;
    return true;
}

struct GmresResult { int iters; double res; bool x_finite; };

static GmresResult run_gmres(Csr a, std::vector<double> b, int max_iter) {
    const int n = static_cast<int>(b.size());
    CoupledMVOperator op(n);
    setup(op, a, n);
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    GMRESSolver solver(n, 10);
    solver.solve(op, b.data(), x.data(), 1e-10, max_iter);
    return { solver.last_iterations(), solver.last_relative_residual(), all_finite(x) };
}

static bool run_bicgstab(Csr a, std::vector<double> b, int max_iter) {
    const int n = static_cast<int>(b.size());
    CoupledMVOperator op(n);
    setup(op, a, n);
    std::vector<double> x(static_cast<size_t>(n), 0.0);
    BiCGStabSolver solver(n);
    solver.solve(op, b.data(), x.data(), 1e-10, max_iter);
    return all_finite(x);
}

int main() {
    bf::g_backend = bf::Backend::CPU;

    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const int n = 16;
    const int max_iter = 200;
    const double tol = 1e-10;

    std::printf("\n=== 1. Kontrollfall: endliches System ===\n");
    {
        const auto r = run_gmres(laplace_1d(n), std::vector<double>(n, 1.0), max_iter);
        std::printf("  Iterationen %d, rel. Residuum %g\n", r.iters, r.res);
        check(r.res <= tol, "GMRES konvergiert");
        check(r.iters < max_iter, "  und braucht weniger als max_iter Iterationen");
    }

    std::printf("\n=== 2. NaN in der rechten Seite ===\n");
    {
        std::vector<double> b(n, 1.0);
        b[3] = qnan;
        const auto r = run_gmres(laplace_1d(n), b, max_iter);
        std::printf("  Iterationen %d, rel. Residuum %g\n", r.iters, r.res);
        check(r.iters == 0, "GMRES bricht bei Iteration 0 ab");
        check(!(r.res <= tol), "Residuum meldet keinen Erfolg");
        check(std::isnan(r.res), "Residuum ist NaN");
        check(r.x_finite, "x bleibt endlich (Startvektor)");
    }

    std::printf("\n=== 3. NaN in der Matrix ===\n");
    {
        Csr a = laplace_1d(n);
        a.val[5] = qnan;
        const auto r = run_gmres(a, std::vector<double>(n, 1.0), max_iter);
        std::printf("  Iterationen %d, rel. Residuum %g\n", r.iters, r.res);
        check(r.iters == 0, "GMRES bricht bei Iteration 0 ab");
        check(!(r.res <= tol), "Residuum meldet keinen Erfolg");
        check(r.x_finite, "x bleibt endlich (Startvektor)");
    }

    std::printf("\n=== 4. Überlauf erst im Arnoldi-Schritt ===\n");
    {
        const int m = 4;
        const auto r = run_gmres(overflow_in_cycle(m), std::vector<double>(m, 1.0), max_iter);
        std::printf("  Iterationen %d, rel. Residuum %g\n", r.iters, r.res);
        check(r.iters == 1, "GMRES bricht bei Iteration 1 ab");
        check(std::isnan(r.res), "Residuum ist NaN");
        check(r.x_finite, "x bleibt endlich (x wird vor dem Abbruch nicht angefasst)");
    }

    std::printf("\n=== 5. BiCGStab, NaN in der rechten Seite ===\n");
    {
        std::vector<double> b(n, 1.0);
        b[3] = qnan;
        check(run_bicgstab(laplace_1d(n), b, max_iter), "x bleibt endlich");
    }

    std::printf("\n=== 6. BiCGStab, Überlauf im Lauf ===\n");
    {
        const int m = 4;
        check(run_bicgstab(overflow_in_cycle(m), std::vector<double>(m, 1.0), max_iter),
              "x bleibt endlich");
    }

    if (g_failures == 0) {
        std::printf("\nAlle Pruefungen bestanden.\n");
        return 0;
    }
    std::printf("\n%d Pruefung(en) fehlgeschlagen.\n", g_failures);
    return 1;
}
