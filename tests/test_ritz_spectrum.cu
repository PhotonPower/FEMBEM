// =============================================================================
// test_ritz_spectrum.cu -- Eigenwerte einer Hessenberg-Matrix
// =============================================================================
//
// Die Konditionsschaetzung des Loesers steht und faellt damit, dass die
// Eigenwertberechnung stimmt. Geprueft wird gegen Matrizen mit bekanntem
// Spektrum:
//
//   1. Diagonalmatrix -- Eigenwerte sind die Diagonaleintraege
//   2. Bidiagonal (obere Dreiecksform) -- ebenso
//   3. 2x2-Block mit komplexem Paar -- analytisch bekannt
//   4. Companion-Matrix eines Polynoms mit vorgegebenen Wurzeln
//   5. Eine Matrix mit stark gespreiztem Spektrum, wie sie hier auftritt
//
// Braucht keine GPU.
// =============================================================================

#include "RitzSpectrum.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace {
int g_fail = 0;
void ck(bool ok, const char* w) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", w);
    if (!ok) ++g_fail;
}
/// Vergleicht Betragsspektren nach Sortierung.
bool spectra_match(std::vector<std::complex<double>> got,
                   std::vector<double> want, double tol) {
    if (got.size() != want.size()) return false;
    std::vector<double> g;
    for (auto& z : got) g.push_back(std::abs(z));
    std::sort(g.begin(), g.end());
    std::sort(want.begin(), want.end());
    for (size_t i = 0; i < g.size(); ++i) {
        const double s = std::fmax(1.0, std::fabs(want[i]));
        if (std::fabs(g[i] - want[i]) > tol * s) return false;
    }
    return true;
}
} // namespace

int main() {
    std::puts("========================================================");
    std::puts(" Eigenwerte der Hessenberg-Matrix (Ritz-Spektrum)        ");
    std::puts("========================================================\n");
    std::vector<std::complex<double>> ev;

    // ---- 1. Diagonal
    {
        const int n = 5, ld = 5;
        std::vector<double> H(n*ld, 0.0);
        const double d[5] = {3.0, -1.0, 7.5, 0.25, 12.0};
        for (int i = 0; i < n; ++i) H[i*ld+i] = d[i];
        hessenberg_eigenvalues(H, n, ld, ev);
        ck(spectra_match(ev, {3.0, 1.0, 7.5, 0.25, 12.0}, 1e-10), "Diagonalmatrix");
    }

    // ---- 2. Obere Dreiecksmatrix
    {
        const int n = 4, ld = 4;
        std::vector<double> H(n*ld, 0.0);
        const double d[4] = {2.0, 5.0, -3.0, 0.5};
        for (int i = 0; i < n; ++i) { H[i*ld+i] = d[i]; for (int j = i+1; j < n; ++j) H[i*ld+j] = 1.7; }
        hessenberg_eigenvalues(H, n, ld, ev);
        ck(spectra_match(ev, {2.0, 5.0, 3.0, 0.5}, 1e-10), "obere Dreiecksmatrix");
    }

    // ---- 3. Komplexes Paar: [[0,-1],[1,0]] hat Eigenwerte +-i
    {
        const int n = 2, ld = 2;
        std::vector<double> H = {0.0, -1.0, 1.0, 0.0};
        hessenberg_eigenvalues(H, n, ld, ev);
        bool ok = (ev.size() == 2);
        for (auto& z : ev) ok = ok && std::fabs(std::abs(z) - 1.0) < 1e-12
                                   && std::fabs(z.real()) < 1e-12;
        std::printf("      Eigenwerte: (%.3f, %.3fi) und (%.3f, %.3fi)\n",
                    ev[0].real(), ev[0].imag(), ev[1].real(), ev[1].imag());
        ck(ok, "komplex konjugiertes Paar +-i");
    }

    // ---- 4. Companion-Matrix zu (x-1)(x-2)(x-3)(x-4) = x^4-10x^3+35x^2-50x+24
    {
        const int n = 4, ld = 4;
        std::vector<double> H(n*ld, 0.0);
        // obere Hessenberg-Companion: Subdiagonale 1, letzte Spalte die Koeffizienten
        for (int i = 1; i < n; ++i) H[i*ld + (i-1)] = 1.0;
        H[0*ld+3] = -24.0; H[1*ld+3] = 50.0; H[2*ld+3] = -35.0; H[3*ld+3] = 10.0;
        hessenberg_eigenvalues(H, n, ld, ev);
        std::printf("      Betraege:");
        std::vector<double> m; for (auto& z : ev) m.push_back(std::abs(z));
        std::sort(m.begin(), m.end());
        for (double v : m) std::printf(" %.6f", v);
        std::printf("\n");
        ck(spectra_match(ev, {1.0, 2.0, 3.0, 4.0}, 1e-8),
           "Companion-Matrix, Wurzeln 1 2 3 4");
    }

    // ---- 5. Stark gespreiztes Spektrum, wie im Loeser erwartet
    {
        const int n = 20, ld = 20;
        std::vector<double> H(n*ld, 0.0);
        std::vector<double> want;
        for (int i = 0; i < n; ++i) {
            const double d = std::pow(10.0, -4.0 + 0.5 * i);   // 1e-4 bis 1e5.5
            H[i*ld+i] = d; want.push_back(d);
            if (i + 1 < n) H[i*ld + i + 1] = 0.01 * d;         // schwach gekoppelt
        }
        hessenberg_eigenvalues(H, n, ld, ev);
        ck(spectra_match(ev, want, 1e-6), "Spektrum ueber neun Dekaden");
        report_ritz_spectrum("Testmatrix mit kappa = 1e9.5", ev);
    }

    std::puts("========================================================");
    if (g_fail == 0) { std::puts(" ALLE PRUEFUNGEN BESTANDEN");
                       std::puts("========================================================"); return 0; }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
