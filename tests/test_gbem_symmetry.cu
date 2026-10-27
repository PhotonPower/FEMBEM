// =============================================================================
// Abnahmeprüfung: ist G_BEM als Operator exakt symmetrisch?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_gbem_symmetry
//   ./build/test_gbem_symmetry
//
// Läuft im CPU-Modus, keine GPU, kein Netz.
//
// HINTERGRUND
// -----------
// Der Single-Layer-Operator ist analytisch symmetrisch. Bisher wurden die
// Blöcke (I,J) und (J,I) UNABHÄNGIG voneinander integriert bzw.
// ACA-komprimiert -- das Ergebnis war nur bis auf Quadratur- und
// Approximationsfehler symmetrisch.
//
// Für das innere CG des blocktriangulären Vorkonditionierers ist das fatal:
// CG setzt SPD voraus. Am TEAM-28-Problem gemessen erreichte es relative
// Residuen von 1.7 bis 2.1 bei einem Ziel von 0.01 -- es VERGRÖSSERTE das
// Residuum, statt es zu senken, und der äußere Löser bekam eine Korrektur,
// die schlechter ist als gar keine.
//
// Seit fem.cu den gespiegelten Block als Transponierte des bereits berechneten
// aufsetzt (add_dense_block_transposed / add_aca_block_mirrored), ist die
// Symmetrie eine Eigenschaft der KONSTRUKTION und nicht mehr der
// Approximationsgüte. Nebeneffekt: der Speicher der Fernfeldblöcke halbiert
// sich, weil sich Spiegelpaare einen Puffer teilen.
//
// Dieser Test prüft die Symmetrie so, wie sie zählt: am angewandten Operator,
// über <x, G y> == <G x, y> für Zufallsvektoren -- nicht an den Blockdaten.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "CoupledSolver.cuh"
#include "Mesh.cuh"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace bf { extern Backend g_backend; }

namespace {

int g_failures = 0;

void check(bool ok, const char* what, double value) {
    std::printf("  [%s] %-46s %.3e\n", ok ? " OK " : "FEHL", what, value);
    if (!ok) ++g_failures;
}

} // namespace

int main() {
    bf::g_backend = bf::Backend::CPU;
    std::mt19937 rng(4242);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    std::printf("\n=== G_BEM: Symmetrie des angewandten Operators ===\n\n");

    // Zwei Cluster mit je 3 bzw. 4 BEM-Freiheitsgraden, ein unsymmetrisch
    // gewaehlter Nichtdiagonalblock -- genau die Lage der Spiegelpaare.
    const int NA = 12, NB = 16, N = NA + NB;

    // Ein bewusst UNSYMMETRISCHER Block (I,J). Wird er samt Transponierter
    // eingetragen, muss der Gesamtoperator trotzdem symmetrisch sein.
    std::vector<double> Bij(static_cast<size_t>(NA) * NB);
    for (auto& v : Bij) v = uni(rng);

    // Symmetrische Diagonalbloecke.
    auto make_sym = [&](int n) {
        std::vector<double> M(static_cast<size_t>(n) * n);
        for (int c = 0; c < n; ++c)
            for (int r = 0; r <= c; ++r) {
                const double v = (r == c) ? (5.0 + uni(rng)) : 0.3 * uni(rng);
                M[static_cast<size_t>(c) * n + r] = v;
                M[static_cast<size_t>(r) * n + c] = v;
            }
        return M;
    };
    std::vector<double> Baa = make_sym(NA), Bbb = make_sym(NB);

    // Dummy-Abbildung: der T-Kopplungsblock wird hier nicht benutzt, die
    // Zeiger duerfen aber nicht null sein.
    std::vector<int> dummy_map(N, 0), dummy_sign(N, 1);

    CoupledMVOperator op(1);
    op.setup_bem_coupling(N, dummy_map.data(), dummy_sign.data());

    op.add_dense_block(NA, NA, 0,  0,  Baa.data());
    op.add_dense_block(NB, NB, NA, NA, Bbb.data());
    op.add_dense_block(NA, NB, 0,  NA, Bij.data());
    // Der Spiegelpartner (J,I): DERSELBE Speicher, als Transponierte angewandt.
    op.add_dense_block_transposed(NB, NA, NA, 0, Bij.data());

    // <x, G y> gegen <G x, y> fuer mehrere Zufallspaare.
    double worst = 0.0, scale = 0.0;
    for (int t = 0; t < 20; ++t) {
        std::vector<double> x(N), y(N), Gx(N, 0.0), Gy(N, 0.0);
        for (int i = 0; i < N; ++i) { x[i] = uni(rng); y[i] = uni(rng); }

        op.apply_schur_tilde(x.data(), Gx.data());
        op.apply_schur_tilde(y.data(), Gy.data());

        double xGy = 0.0, Gxy = 0.0;
        for (int i = 0; i < N; ++i) { xGy += x[i] * Gy[i]; Gxy += Gx[i] * y[i]; }
        worst = std::fmax(worst, std::fabs(xGy - Gxy));
        scale = std::fmax(scale, std::fabs(xGy));
    }
    check(worst <= 1e-13 * std::fmax(scale, 1.0),
          "<x, G y> == <G x, y> fuer 20 Zufallspaare", worst / std::fmax(scale, 1.0));

    // Gegenprobe: OHNE Spiegelung, also mit zwei unabhaengigen Bloecken, muss
    // die Symmetrie verletzt sein -- sonst prueft der Test nichts.
    {
        std::vector<double> Bji(static_cast<size_t>(NB) * NA);
        for (auto& v : Bji) v = uni(rng);          // unabhaengig, nicht transponiert

        CoupledMVOperator op2(1);
        op2.setup_bem_coupling(N, dummy_map.data(), dummy_sign.data());
        op2.add_dense_block(NA, NA, 0,  0,  Baa.data());
        op2.add_dense_block(NB, NB, NA, NA, Bbb.data());
        op2.add_dense_block(NA, NB, 0,  NA, Bij.data());
        op2.add_dense_block(NB, NA, NA, 0,  Bji.data());

        std::vector<double> x(N), y(N), Gx(N, 0.0), Gy(N, 0.0);
        for (int i = 0; i < N; ++i) { x[i] = uni(rng); y[i] = uni(rng); }
        op2.apply_schur_tilde(x.data(), Gx.data());
        op2.apply_schur_tilde(y.data(), Gy.data());
        double xGy = 0.0, Gxy = 0.0;
        for (int i = 0; i < N; ++i) { xGy += x[i] * Gy[i]; Gxy += Gx[i] * y[i]; }
        const double rel = std::fabs(xGy - Gxy) / std::fmax(std::fabs(xGy), 1.0);
        check(rel > 1e-6,
              "Gegenprobe: unabhaengige Bloecke sind unsymmetrisch", rel);
    }

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
