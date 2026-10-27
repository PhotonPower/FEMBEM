// =============================================================================
// Abnahmeprüfung: liest bf::dgemv bei beta == 0 den Ausgabevektor?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_dgemv_beta_zero
//   ./build/test_dgemv_beta_zero
//
// HINTERGRUND
// -----------
// Die BLAS-Spezifikation verlangt ausdrücklich: bei beta == 0 wird y NICHT
// gelesen. Der Aufrufer darf y also uninitialisiert übergeben. Die CPU-Fassung
// von bf::dgemv rechnete stattdessen unbedingt
//
//     y[i] = alpha * acc + beta * y[i];
//
// und las y damit auch bei beta == 0. Enthält der Puffer ein NaN-Bitmuster,
// ist 0 * NaN = NaN -- das Ergebnis ist vollständig NaN, obwohl A und x
// endlich sind.
//
// So sind die NaN-Kräfte des TEAM-28-Laufs entstanden. apply_physical()
// benutzt d_temp_z als Zwischenpuffer für z = V^T*x der ACA-Blöcke, mit
// beta = 0. DeviceArray::allocate() initialisiert nicht (fill_zero() ist eine
// eigene Methode), d_temp_z enthielt also rohen malloc-Speicher. Ein NaN darin
// machte z zu NaN, und der anschließende U*z-Schritt trug es in ALLE Zeilen
// des Blocks. Weil G- und H-Blöcke denselben Puffer benutzen, waren beide
// betroffen: 16544 von 16544 Einträgen NaN, bei sauberen Blockdaten und
// sauberer Eingabe.
//
// Die Prüfung füllt y bewusst mit NaN vor und verlangt ein endliches Ergebnis.
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
static void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " OK " : "FEHL", what);
    if (!ok) ++g_failures;
}

int main() {
    bf::g_backend = bf::Backend::CPU;
    cublasHandle_t h = nullptr;               // im CPU-Modus zulaessig
    const double qnan = std::nan("");

    std::printf("\n=== bf::dgemv: beta == 0 darf y nicht lesen ===\n\n");

    // A (2x2, column-major) = [[1, 2], [3, 4]],  x = [1, 1]
    std::vector<double> A = {1.0, 3.0, 2.0, 4.0};
    std::vector<double> x = {1.0, 1.0};

    {   // Nicht transponiert: A*x = [3, 7]
        std::vector<double> y = {qnan, qnan};
        bf::dgemv(h, false, 2, 2, 1.0, A.data(), 2, x.data(), 0.0, y.data());
        std::printf("        y = [%g, %g]  (erwartet [3, 7])\n", y[0], y[1]);
        check(std::isfinite(y[0]) && std::isfinite(y[1]),
              "y mit NaN vorbelegt, beta = 0 -> Ergebnis endlich");
        check(std::fabs(y[0] - 3.0) < 1e-14 && std::fabs(y[1] - 7.0) < 1e-14,
              "Ergebnis numerisch korrekt");
    }

    {   // Transponiert: A^T*x = [4, 6]
        std::vector<double> y = {qnan, qnan};
        bf::dgemv(h, true, 2, 2, 1.0, A.data(), 2, x.data(), 0.0, y.data());
        std::printf("        y = [%g, %g]  (erwartet [4, 6])\n", y[0], y[1]);
        check(std::isfinite(y[0]) && std::isfinite(y[1]),
              "dasselbe transponiert (der Fall der ACA-Bloecke: z = V^T*x)");
        check(std::fabs(y[0] - 4.0) < 1e-14 && std::fabs(y[1] - 6.0) < 1e-14,
              "transponiertes Ergebnis numerisch korrekt");
    }

    {   // Regression: beta != 0 muss y weiterhin lesen und akkumulieren.
        std::vector<double> y = {10.0, 20.0};
        bf::dgemv(h, false, 2, 2, 1.0, A.data(), 2, x.data(), 1.0, y.data());
        std::printf("        y = [%g, %g]  (erwartet [13, 27])\n", y[0], y[1]);
        check(std::fabs(y[0] - 13.0) < 1e-14 && std::fabs(y[1] - 27.0) < 1e-14,
              "beta = 1 akkumuliert weiterhin korrekt (Regression)");
    }

    {   // Regression: beta = 2 skaliert das vorhandene y.
        std::vector<double> y = {1.0, 1.0};
        bf::dgemv(h, false, 2, 2, 1.0, A.data(), 2, x.data(), 2.0, y.data());
        check(std::fabs(y[0] - 5.0) < 1e-14 && std::fabs(y[1] - 9.0) < 1e-14,
              "beta = 2 skaliert korrekt (Regression)");
    }

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
