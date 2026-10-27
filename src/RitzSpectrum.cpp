#include "RitzSpectrum.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace {

/// Eigenwerte eines 2x2-Blocks [[a,b],[c,d]] -- analytisch, auch komplex.
void eig2x2(double a, double b, double c, double d,
            std::complex<double>& l1, std::complex<double>& l2)
{
    const double tr  = a + d;
    const double det = a * d - b * c;
    const double disc = 0.25 * tr * tr - det;
    if (disc >= 0.0) {
        const double s = std::sqrt(disc);
        l1 = std::complex<double>(0.5 * tr + s, 0.0);
        l2 = std::complex<double>(0.5 * tr - s, 0.0);
    } else {
        const double s = std::sqrt(-disc);
        l1 = std::complex<double>(0.5 * tr,  s);
        l2 = std::complex<double>(0.5 * tr, -s);
    }
}

} // namespace

bool hessenberg_eigenvalues(std::vector<double>& H, int n, int ld,
                            std::vector<std::complex<double>>& out)
{
    out.clear();
    if (n <= 0) return true;
    auto at = [&](int i, int j) -> double& { return H[static_cast<size_t>(i) * ld + j]; };

    // Skala fuer die Abbruchschranke
    double scale = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = std::max(0, i - 1); j < n; ++j)
            scale = std::fmax(scale, std::fabs(at(i, j)));
    if (scale == 0.0) { out.assign(static_cast<size_t>(n), std::complex<double>(0.0, 0.0)); return true; }

    int hi = n - 1;
    int budget = 60 * n;          // Sicherheitsnetz gegen Nichtkonvergenz
    bool ok = true;

    while (hi >= 0) {
        if (budget-- <= 0) { ok = false; break; }

        // Kleines Subdiagonalelement suchen -> Blockanfang lo
        int lo = hi;
        while (lo > 0) {
            const double s = std::fabs(at(lo - 1, lo - 1)) + std::fabs(at(lo, lo));
            if (std::fabs(at(lo, lo - 1)) <= 1e-14 * (s > 0.0 ? s : scale)) break;
            --lo;
        }

        if (lo == hi) {                       // 1x1 -> reeller Eigenwert
            out.emplace_back(at(hi, hi), 0.0);
            --hi;
            continue;
        }
        if (lo == hi - 1) {                   // 2x2 -> analytisch
            std::complex<double> l1, l2;
            eig2x2(at(hi-1, hi-1), at(hi-1, hi), at(hi, hi-1), at(hi, hi), l1, l2);
            out.push_back(l1);
            out.push_back(l2);
            hi -= 2;
            continue;
        }

        // Wilkinson-Shift aus dem abschliessenden 2x2-Block
        std::complex<double> s1, s2;
        eig2x2(at(hi-1, hi-1), at(hi-1, hi), at(hi, hi-1), at(hi, hi), s1, s2);
        const double hh = at(hi, hi);
        double mu = (std::fabs(s1.real() - hh) < std::fabs(s2.real() - hh)) ? s1.real() : s2.real();
        if (std::fabs(s1.imag()) > 0.0) mu = hh;   // komplexes Paar: einfacher Shift

        // QR-Schritt auf dem Block [lo..hi] per Givens, verschoben um mu
        for (int i = lo; i <= hi; ++i) at(i, i) -= mu;

        std::vector<double> cs(static_cast<size_t>(hi - lo + 1), 1.0);
        std::vector<double> sn(static_cast<size_t>(hi - lo + 1), 0.0);
        for (int k = lo; k < hi; ++k) {
            const double x = at(k, k), y = at(k + 1, k);
            const double r = std::hypot(x, y);
            const double c = (r > 0.0) ? x / r : 1.0;
            const double s = (r > 0.0) ? y / r : 0.0;
            cs[k - lo] = c; sn[k - lo] = s;
            for (int j = k; j <= hi; ++j) {
                const double t1 = at(k, j), t2 = at(k + 1, j);
                at(k, j)     =  c * t1 + s * t2;
                at(k + 1, j) = -s * t1 + c * t2;
            }
        }
        for (int k = lo; k < hi; ++k) {     // RQ bilden
            const double c = cs[k - lo], s = sn[k - lo];
            for (int i = lo; i <= std::min(hi, k + 2); ++i) {
                const double t1 = at(i, k), t2 = at(i, k + 1);
                at(i, k)     =  c * t1 + s * t2;
                at(i, k + 1) = -s * t1 + c * t2;
            }
        }
        for (int i = lo; i <= hi; ++i) at(i, i) += mu;
    }

    // Nicht konvergierte Reste als Diagonalwerte uebernehmen
    for (int i = hi; i >= 0; --i) out.emplace_back(at(i, i), 0.0);
    return ok;
}

void report_ritz_spectrum(const char* label,
                          const std::vector<std::complex<double>>& ritz)
{
    if (ritz.empty()) {
        std::printf("  [Ritz] %s: keine Werte -- Auswertung nicht moeglich.\n", label);
        return;
    }

    std::vector<double> mag;
    mag.reserve(ritz.size());
    for (const auto& z : ritz) mag.push_back(std::abs(z));
    std::sort(mag.begin(), mag.end());

    const double mn = mag.front(), mx = mag.back();
    const double kappa = (mn > 0.0) ? mx / mn : 0.0;

    int n_small = 0;
    for (double v : mag) if (v < 1e-6 * mx) ++n_small;

    int n_complex = 0;
    for (const auto& z : ritz) if (std::fabs(z.imag()) > 1e-12 * std::abs(z)) ++n_complex;

    std::printf("\n  [Ritz] Spektrum aus dem Arnoldi-Prozess (%s), %zu Werte\n",
                label, ritz.size());
    std::printf("  [Ritz]   |lambda| von %.4e bis %.4e\n", mn, mx);
    if (mn > 0.0)
        std::printf("  [Ritz]   Schaetzung kappa = max/min = %.3e  (UNTERE Schranke)\n", kappa);
    else
        std::printf("  [Ritz]   kleinster Betrag ist exakt null -- der Krylov-Raum "
                    "enthaelt Nullraumrichtungen\n");
    std::printf("  [Ritz]   Quartile |lambda|: %.3e / %.3e / %.3e\n",
                mag[mag.size()/4], mag[mag.size()/2], mag[3*mag.size()/4]);
    std::printf("  [Ritz]   unter 1e-6 des Maximums: %d von %zu\n", n_small, ritz.size());
    std::printf("  [Ritz]   komplexe Werte: %d von %zu\n", n_complex, ritz.size());

    // Einordnung. Fuer GMRES gilt als Faustformel, dass die Iterationszahl fuer
    // eine Reduktion um den Faktor 10 etwa mit sqrt(kappa) waechst.
    if (mn > 0.0) {
        const double it_per_decade = std::sqrt(kappa);
        std::printf("  [Ritz]   Faustformel: rund %.0f Iterationen je Dekade "
                    "Residuumsreduktion\n", it_per_decade);
        if (it_per_decade > 500.0)
            std::printf("  [Ritz]   -> Bei diesem kappa koennen 1000 Iterationen das Ziel\n"
                        "  [Ritz]      nicht erreichen. Ein besserer Vorkonditionierer ist\n"
                        "  [Ritz]      noetig, kein weiterer Einzelfix.\n");
        else
            // ACHTUNG, die Schlussfolgerung traegt nur in EINE Richtung.
            //
            // kappa ist hier eine UNTERE Schranke: die Ritz-Werte eines
            // Arnoldi-Zyklus naehern zuerst die aeusseren Teile des Spektrums
            // an. Bei 400 Werten von 35234 Eigenwerten koennen ganze Cluster
            // kleiner Eigenwerte unentdeckt bleiben. Ein grosses kappa beweist
            // also, dass es schlecht konditioniert IST -- ein kleines beweist
            // NICHT, dass es gut konditioniert ist.
            //
            // Genau dieser Fehlschluss ist am TEAM-28-Problem aufgetreten: der
            // Bericht meldete kappa >= 1.9e3 und "44 Iterationen je Dekade",
            // gemessen waren es rund 2500 -- ein Faktor 57. Entweder ist das
            // Spektrum deutlich schlechter als die 400 Werte zeigen, oder die
            // Konvergenz wird von etwas anderem als der Konditionierung
            // begrenzt. Die Ausgabe darf diese Unterscheidung nicht
            // vorwegnehmen.
            std::printf("  [Ritz]   -> Aus DIESEN Werten allein spricht nichts gegen 1000\n"
                        "  [Ritz]      Iterationen. Achtung: kappa ist eine UNTERE Schranke\n"
                        "  [Ritz]      (Arnoldi findet zuerst die aeusseren Eigenwerte, bei\n"
                        "  [Ritz]      %zu von vielen Werten koennen Cluster kleiner\n"
                        "  [Ritz]      Eigenwerte unentdeckt bleiben). Ein kleines kappa\n"
                        "  [Ritz]      BEWEIST also keine gute Konditionierung.\n"
                        "  [Ritz]      Vergleichen Sie die Faustformel mit der tatsaechlich\n"
                        "  [Ritz]      gemessenen Rate: weichen sie um mehr als etwa eine\n"
                        "  [Ritz]      Groessenordnung ab, ist die Konditionierung NICHT die\n"
                        "  [Ritz]      Erklaerung.\n", ritz.size());
    }
    if (n_small > 0)
        std::printf("  [Ritz]   -> %d Werte nahe null bedeuten einen Nahnullraum, den der\n"
                    "  [Ritz]      Vorkonditionierer nicht erfasst.\n", n_small);
    std::printf("\n");
}
