// =============================================================================
// test_adjacent_bem.cu -- Quadratur fuer benachbarte BEM-Panelpaare
// =============================================================================
//
// Bis zuletzt liefen ALLE Paare ausser dem exakten Self-Patch durch die glatte
// 4-Punkt-Gauss-Regel, mit der Begruendung, das sei "hinreichend genau". Fuer
// die Einfachschicht stimmt das; fuer die Doppelschicht an GEOMETRISCHEN
// Kanten nicht -- dort ist der Integrand (r-r').n'/|r-r'|^3 stark singulaer.
//
// Geprueft wird gegen eine im Test selbst gerechnete Referenz (starke
// Unterteilung mit niedriger Ordnung, also dasselbe Verfahren mit weit
// hoeherem Aufwand). Das ist bewusst keine unabhaengige Formel: es geht um die
// Konvergenz der Quadratur, nicht um die Richtigkeit des Kerns.
//
// Braucht keine GPU.
// =============================================================================

#include "BemIntegration.cuh"
#include "SingularQuadPair.cuh"

#include <cmath>
#include <initializer_list>
#include <cstdio>

namespace {

int g_fail = 0;
void ck(bool ok, const char* w) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", w);
    if (!ok) ++g_fail;
}

double3 P3(double x, double y, double z) { return make_double3(x, y, z); }

/// Summe ueber den 4x4-Block -- eine skalare Kennzahl je Paar.
double block_sum(const double b[4][4]) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) s += b[i][j];
    return s;
}

} // namespace

int main() {
    std::puts("========================================================");
    std::puts(" Quadratur benachbarter BEM-Panelpaare                   ");
    std::puts("========================================================");

    // Zwei Einheitsquadrate an einer 90-Grad-Kante:
    // P in der Ebene z=0, Q in der Ebene x=0, gemeinsame Kante x=0,z=0.
    const double3 P[4] = { P3(-1,0,0), P3(0,0,0), P3(0,1,0), P3(-1,1,0) };
    const double3 Q[4] = { P3( 0,0,0), P3(0,1,0), P3(0,1,1), P3( 0,0,1) };

    // Zwei koplanare Nachbarn in derselben Ebene.
    const double3 C[4] = { P3(0,0,0), P3(1,0,0), P3(1,1,0), P3(0,1,0) };

    double blk[4][4];

    std::puts("\n[1] Doppelschicht an einer 90-Grad-Kante");
    {
        integrate_adjacent_quad_pair_dlp(P, Q, blk, 24);   // Referenz
        const double ref = block_sum(blk);
        integrate_regular_quad_pair_dlp(P, Q, blk);
        const double alt = block_sum(blk);
        integrate_adjacent_quad_pair_dlp(P, Q, blk, 8);
        const double neu = block_sum(blk);

        const double e_alt = std::fabs(alt / ref - 1.0);
        const double e_neu = std::fabs(neu / ref - 1.0);
        std::printf("      Referenz (24x24 Teilquads) = %+.8e\n", ref);
        std::printf("      4-Punkt-Gauss (bisher)     = %+.8e   Fehler %6.2f %%\n", alt, e_alt*100);
        std::printf("      8x8 Teilquads (jetzt)      = %+.8e   Fehler %6.2f %%\n", neu, e_neu*100);
        std::puts("\n      Konvergenz ueber die Unterteilungsstufe:");
        for (int n : {2, 4, 8, 12, 16}) {
            integrate_adjacent_quad_pair_dlp(P, Q, blk, n);
            std::printf("        %2dx%2d Teilquads: %+.8e   Fehler %6.2f %%   "
                        "%9d Kernauswertungen\n",
                        n, n, block_sum(blk), std::fabs(block_sum(blk)/ref-1)*100,
                        (n*n*4)*(n*n*4));
        }
        ck(e_alt > 0.05, "die alte Regel liegt hier deutlich daneben");
        ck(e_neu < 0.05, "die unterteilte Quadratur trifft auf unter 5 %");
        ck(e_neu < e_alt / 5.0, "mindestens Faktor 5 besser");
    }

    // -------------------------------------------------------------------------
    // [1b] Sauter-Schwab -- das Verfahren, das der Loeser jetzt benutzt
    // -------------------------------------------------------------------------
    // Verglichen wird gegen die unterteilte Quadratur mit 24x24 Teilquads. Die
    // hat selbst noch Fehler (16x16 lag 0.77 % neben 24x24), eine Abweichung
    // von rund einem Prozent sagt also nicht, welches der beiden naeher an der
    // Wahrheit liegt. Sauter-Schwab hebt die Singularitaet auf und konvergiert
    // exponentiell; die Unterteilung ueberdeckt sie nur. Im Zweifel ist
    // Sauter-Schwab der genauere Wert.
    std::puts("\n[1b] Sauter-Schwab (das jetzt verwendete Verfahren)");
    {
        // Globale Knotennummern: P und Q teilen die Kante, also die Knoten
        // (0,0,0) und (0,1,0). In P sind das die lokalen Ecken 1 und 2, in Q
        // die Ecken 0 und 1.
        const int gp[4] = {10, 20, 21, 11};
        const int gq[4] = {20, 21, 31, 30};

        integrate_adjacent_quad_pair_dlp(P, Q, blk, 24);
        const double ref = block_sum(blk);
        integrate_singular_quad_pair(P, gp, Q, gq, 1, blk);
        const double ss = block_sum(blk);
        std::printf("      Referenz (24x24 Unterteilung) = %+.8e\n", ref);
        std::printf("      Sauter-Schwab                 = %+.8e   Abweichung %6.2f %%\n",
                    ss, std::fabs(ss/ref-1)*100);
        ck(std::fabs(ss/ref-1.0) < 0.03,
           "Doppelschicht: Sauter-Schwab trifft die Referenz auf unter 3 %");

        integrate_adjacent_quad_pair(P, Q, blk, 24);
        const double refs = block_sum(blk);
        integrate_singular_quad_pair(P, gp, Q, gq, 0, blk);
        const double sss = block_sum(blk);
        std::printf("      Einfachschicht: Referenz = %+.8e, Sauter-Schwab = %+.8e"
                    "   Abweichung %6.2f %%\n", refs, sss, std::fabs(sss/refs-1)*100);
        ck(std::fabs(sss/refs-1.0) < 0.01,
           "Einfachschicht: Sauter-Schwab trifft die Referenz auf unter 1 %");
    }

    std::puts("\n[2] Doppelschicht bei KOPLANAREN Nachbarn");
    std::puts("    (r-r').n' verschwindet identisch, das Integral muss null sein");
    {
        integrate_adjacent_quad_pair_dlp(P, C, blk, 8);
        double mx = 0.0;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) mx = std::fmax(mx, std::fabs(blk[i][j]));
        std::printf("      groesster Betrag im Block = %.3e\n", mx);
        ck(mx < 1e-12, "verschwindet");
    }

    std::puts("\n[3] Wahl der Unterteilungsstufe");
    {
        const int lc = adjacent_subdivision_level(P, C);   // koplanar
        const int lk = adjacent_subdivision_level(P, Q);   // 90 Grad
        std::printf("      koplanar    -> Stufe %d\n", lc);
        std::printf("      90-Grad-Knick -> Stufe %d\n", lk);
        ck(lc == 2, "koplanare Nachbarn bekommen die billige Stufe");
        ck(lk == 8, "geknickte Nachbarn bekommen die volle Stufe");
    }

    std::puts("\n[4] Einfachschicht, gemeinsame Kante: war schon vorher genau");
    {
        integrate_adjacent_quad_pair(P, Q, blk, 24);
        const double ref = block_sum(blk);
        integrate_regular_quad_pair(P, Q, blk);
        const double alt = block_sum(blk);
        integrate_adjacent_quad_pair(P, Q, blk, 8);
        const double neu = block_sum(blk);
        std::printf("      Referenz = %+.8e\n", ref);
        std::printf("      4-Punkt  = %+.8e   Fehler %6.2f %%\n", alt, std::fabs(alt/ref-1)*100);
        std::printf("      8x8      = %+.8e   Fehler %6.2f %%\n", neu, std::fabs(neu/ref-1)*100);
        // Anmerkung zur eigenen Fehleinschaetzung: eine skalare Vorabrechnung
        // OHNE die Rooftop-Basisfunktionen ergab fuer diesen Fall nur 0.23 %
        // und liess die Einfachschicht harmlos erscheinen. Mit den
        // Basisfunktionen sind es 5.77 %. Die Gewichtung mit W_i . W_j
        // verschiebt das Gewicht zur gemeinsamen Kante hin, also genau zur
        // Singularitaet. Beide Kerne profitieren.
        ck(std::fabs(alt/ref-1.0) > 0.02,
           "auch die Einfachschicht war merklich ungenau (nicht 0.23 % wie skalar geschaetzt)");
        ck(std::fabs(neu/ref-1.0) < 0.01, "unterteilt unter 1 %");
        ck(std::fabs(neu/ref-1.0) < std::fabs(alt/ref-1.0) / 5.0, "mindestens Faktor 5 besser");
    }

    std::puts("\n========================================================");
    if (g_fail == 0) { std::puts(" ALLE PRUEFUNGEN BESTANDEN");
                       std::puts("========================================================"); return 0; }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
