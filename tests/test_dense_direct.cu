// =============================================================================
// Abnahmeprüfung: taugt der Direktweg von --direct selbst?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_dense_direct
//   ./build/test_dense_direct
//
// WARUM ES DIESEN TEST GIBT
// -------------------------
// --direct baut das gekoppelte System spaltenweise auf, zerlegt es dicht und
// meldet das Residuum der exakten Lösung. Die Deutung lautet:
//
//   ~1e-15  -> System lösbar, GMRES scheitert aus einem anderen Grund
//   ~1e-2   -> System INKONSISTENT, kein Vorkonditionierer hilft
//
// Diese Deutung trägt nur, wenn das Verfahren selbst funktioniert. Käme auf
// einem nachweislich lösbaren System ebenfalls 1e-2 heraus, wäre das Ergebnis
// wertlos und würde in die Irre führen -- man würde einen Modellierungsfehler
// suchen, wo ein Werkzeugfehler vorliegt.
//
// Dieser Test schließt das aus: reguläre, diagonaldominante Matrix mit
// bekannter Lösbarkeit, gemessen 1.1e-15.
//
// Er prüft ausdrücklich AUCH, dass factorize() in place arbeitet und die
// Eingabematrix zerstört. In --direct wird deshalb vorher eine Kopie gesichert;
// ohne sie würde das Residuum gegen die LU-Faktoren statt gegen A gerechnet und
// wäre bedeutungslos -- ohne dass irgendetwas auffiele.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "DenseLinalg.cuh"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace bf { extern Backend g_backend; }

int main() {
    bf::g_backend = bf::Backend::CPU;
    int failures = 0;
    std::printf("\n=== DenseLU: Selbstpruefung des --direct-Verfahrens ===\n\n");

    const int n = 300;
    std::mt19937 rng(5);
    std::uniform_real_distribution<double> u(-1.0, 1.0);

    std::vector<double> A(static_cast<size_t>(n) * n), b(n), x(n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            A[static_cast<size_t>(j) * n + i] = (i == j) ? (n + u(rng)) : u(rng);
    for (int i = 0; i < n; ++i) b[i] = u(rng);

    const std::vector<double> A_copy(A);
    x = b;

    bf::DenseLU lu;
    lu.factorize(A.data(), n);
    lu.solve(x.data(), 1);

    double rn = 0.0, bn = 0.0;
    for (int i = 0; i < n; ++i) {
        double ax = 0.0;
        for (int j = 0; j < n; ++j)
            ax += A_copy[static_cast<size_t>(j) * n + i] * x[j];
        rn += (b[i] - ax) * (b[i] - ax);
        bn += b[i] * b[i];
    }
    const double rel = std::sqrt(rn / bn);
    const bool ok = rel < 1e-12;
    std::printf("  [%s] regulaeres System: Rel. Res. = %.3e (erwartet < 1e-12)\n",
                ok ? " OK " : "FEHL", rel);
    if (!ok) ++failures;

    // factorize() arbeitet in place. Das ist dokumentiert, aber leicht zu
    // uebersehen -- und in --direct waere die Folge ein Residuum, das gegen die
    // LU-Faktoren statt gegen A gerechnet wird.
    bool destroyed = false;
    for (size_t t = 0; t < A.size(); ++t)
        if (A[t] != A_copy[t]) { destroyed = true; break; }
    std::printf("  [%s] factorize() zerstoert die Eingabematrix "
                "(--direct muss vorher kopieren)\n", destroyed ? " OK " : "FEHL");
    if (!destroyed) ++failures;

    // -----------------------------------------------------------------------
    // Misst die Konditionszahlschaetzung von --direct richtig?
    // -----------------------------------------------------------------------
    //
    // --direct meldet kappa_2 ueber Potenzmethoden auf A^T A und (A^T A)^-1.
    // Diese Zahl entscheidet, ob die Stagnation des Loesers erklaert ist --
    // sie muss also selbst stimmen. Geprueft an Diagonalmatrizen mit
    // VORGEGEBENER Konditionszahl, wo die Antwort bekannt ist.
    for (double kappa_soll : {1e3, 1e6, 1e9}) {
        const int m = 200;
        std::vector<double> D(static_cast<size_t>(m) * m, 0.0), Dt(D);
        for (int i = 0; i < m; ++i) {
            // Singulaerwerte logarithmisch von 1 bis kappa_soll verteilt
            const double sv = std::pow(kappa_soll, i / double(m - 1));
            D[static_cast<size_t>(i) * m + i] = sv;
        }
        std::vector<double> Dc(D);
        for (int j = 0; j < m; ++j)
            for (int i = 0; i < m; ++i)
                Dt[static_cast<size_t>(i) * m + j] = Dc[static_cast<size_t>(j) * m + i];

        bf::DenseLU l1, l2;
        l1.factorize(D.data(), m);
        l2.factorize(Dt.data(), m);

        std::mt19937 r2(7);
        std::uniform_real_distribution<double> uu(-1.0, 1.0);
        std::vector<double> v(m), w(m);
        auto nrm = [&](std::vector<double>& u) {
            double s2 = 0.0; for (double t : u) s2 += t*t; s2 = std::sqrt(s2);
            if (s2 > 0.0) for (double& t : u) t /= s2;
            return s2;
        };
        auto mul = [&](const std::vector<double>& in, std::vector<double>& out) {
            std::fill(out.begin(), out.end(), 0.0);
            for (int j = 0; j < m; ++j)
                for (int i = 0; i < m; ++i)
                    out[i] += Dc[static_cast<size_t>(j)*m + i] * in[j];
        };
        auto mulT = [&](const std::vector<double>& in, std::vector<double>& out) {
            for (int j = 0; j < m; ++j) {
                double s2 = 0.0;
                for (int i = 0; i < m; ++i) s2 += Dc[static_cast<size_t>(j)*m + i] * in[i];
                out[j] = s2;
            }
        };

        for (int i = 0; i < m; ++i) v[i] = uu(r2);
        nrm(v);
        double smax = 0.0;
        for (int it = 0; it < 80; ++it) { mul(v, w); mulT(w, v); smax = std::sqrt(nrm(v)); }
        for (int i = 0; i < m; ++i) v[i] = uu(r2);
        nrm(v);
        double smin = 0.0;
        for (int it = 0; it < 80; ++it) {
            w = v; l2.solve(w.data(), 1); l1.solve(w.data(), 1); v = w;
            const double g = nrm(v);
            smin = (g > 0.0) ? 1.0 / std::sqrt(g) : 0.0;
        }
        const double kappa = (smin > 0.0) ? smax / smin : 0.0;
        const double err = std::fabs(kappa - kappa_soll) / kappa_soll;
        const bool ok2 = err < 0.05;
        std::printf("  [%s] kappa-Schaetzung: Soll %.0e, gemessen %.3e (Abw. %.1f %%)\n",
                    ok2 ? " OK " : "FEHL", kappa_soll, kappa, 100.0 * err);
        if (!ok2) ++failures;
    }

    std::printf("\n");
    if (failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", failures);
    return 1;
}
