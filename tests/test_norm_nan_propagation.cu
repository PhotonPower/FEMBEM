// =============================================================================
// Abnahmeprüfung: propagiert bf::dnrm2 NaN und Inf?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_norm_nan_propagation
//   ./build/test_norm_nan_propagation
//
// HINTERGRUND
// -----------
// Die CPU-Fassung von bf::dnrm2 benutzt die skalierte Summation des
// Referenz-BLAS. Darin steckte ein Fehler mit erheblicher Wirkung: jeder
// Vergleich mit NaN ist falsch, also war "scale < a" für ein NaN nie erfüllt
// und scale blieb 0.0. Der Rückgabewert ist aber "(scale == 0.0) ? 0.0 : ...".
//
// Ein Vektor aus LAUTER NaN lieferte damit die Norm 0.0.
//
// Aufgefallen ist es nie, weil ein Vektor mit GEMISCHTEN Einträgen korrekt NaN
// liefert -- die endlichen Werte setzen scale. Erst wenn ein Vektor vollständig
// entartet, kippt das Ergebnis nach 0.
//
// Genau das ist der gefährliche Fall. GMRES prüft die Konvergenz über
// r_norm/b_norm. Wird die Lösung irgendwo NaN, wird das Residuum durchgehend
// NaN -- und meldete sich als 0.0 zurück. Der Löser gab
//   [GMRES] Converged at iteration 50 | Rel. Res: 0
// aus und lieferte NaN-Kräfte. Ebenso alle Diagnosenormen: eine Ausgabe
// "||FEM-Anteil von y|| = 0" bedeutete nicht "der Vektor ist null", sondern
// "der Vektor ist komplett NaN". Ein Fehler, der Fehler unsichtbar macht.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "Backend.cuh"
#include "HostLinalg.cuh"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace bf { extern Backend g_backend; }

static int g_failures = 0;

static void check(bool ok, const char* what, double got) {
    std::printf("  [%s] %-46s -> %g\n", ok ? " OK " : "FEHL", what, got);
    if (!ok) ++g_failures;
}

int main() {
    bf::g_backend = bf::Backend::CPU;
    // Im CPU-Modus entscheiden die bf::-Wrapper nach bf::g_backend, das Handle
    // darf nullptr sein (siehe require_handle in src/HostLinalg.cu).
    cublasHandle_t h = nullptr;

    const double qnan = std::nan("");
    const double pinf = std::numeric_limits<double>::infinity();
    double r = 0.0;

    std::printf("\n=== bf::dnrm2: Propagation von NaN und Inf ===\n\n");

    { std::vector<double> v = {qnan, qnan, qnan};
      bf::dnrm2(h, 3, v.data(), &r);
      check(std::isnan(r), "Vektor aus lauter NaN", r); }

    { std::vector<double> v = {3.0, 4.0, qnan};
      bf::dnrm2(h, 3, v.data(), &r);
      check(std::isnan(r), "gemischt, ein NaN", r); }

    { std::vector<double> v = {0.0, 0.0, qnan};
      bf::dnrm2(h, 3, v.data(), &r);
      check(std::isnan(r), "Nullen und ein NaN", r); }

    { std::vector<double> v = {pinf, pinf};
      bf::dnrm2(h, 2, v.data(), &r);
      check(std::isinf(r), "Vektor aus lauter Inf", r); }

    { std::vector<double> v = {3.0, 4.0, pinf};
      bf::dnrm2(h, 3, v.data(), &r);
      check(std::isinf(r), "gemischt, ein Inf", r); }

    { std::vector<double> v = {qnan, pinf};
      bf::dnrm2(h, 2, v.data(), &r);
      check(std::isnan(r), "NaN und Inf -> NaN gewinnt", r); }

    // Die gesunden Faelle duerfen sich nicht geaendert haben.
    { std::vector<double> v = {3.0, 4.0};
      bf::dnrm2(h, 2, v.data(), &r);
      check(std::fabs(r - 5.0) < 1e-14, "{3, 4} = 5 (Regression)", r); }

    { std::vector<double> v = {0.0, 0.0, 0.0};
      bf::dnrm2(h, 3, v.data(), &r);
      check(r == 0.0, "Nullvektor = 0 (Regression)", r); }

    { std::vector<double> v = {1e-200, 1e200};
      bf::dnrm2(h, 2, v.data(), &r);
      check(std::isfinite(r) && r > 1e199, "extreme Skalen ohne Ueberlauf", r); }

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
