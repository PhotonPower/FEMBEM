// =============================================================================
// test_sauter_schwab_coverage.cu -- decken die Transformationen ihr Gebiet ab?
// =============================================================================
//
// Die Sauter-Schwab-Transformationen bilden [0,1]^4 auf das Produkt zweier
// Referenzdreiecke ab. Integriert man darueber die konstante Funktion 1, MUSS
// die Flaeche dieses Produkts herauskommen:
//
//     (1/2) * (1/2) = 1/4
//
// Das ist ein Test ohne jeden Kern und ohne jede Singularitaet. Er prueft
// ausschliesslich, ob Teilgebiete und Jacobi-Determinanten zusammenpassen --
// und genau dort sass ein Fehler: der Common-Edge-Fall deckte nur 5/6 des
// Gebiets ab, weil sein erstes Teilgebiet einen Faktor eta2 zu viel trug.
//
// Der Test ist billig und faengt eine ganze Fehlerklasse ab, die sich sonst
// als "das Ergebnis ist etwas daneben" tarnt. Er haette von Anfang an
// dabeisein sollen.
//
// Braucht keine GPU.
// =============================================================================

#include "SauterSchwab.cuh"

#include <cmath>
#include <cstdio>

namespace {

int g_fail = 0;
int g_outside = 0;

/// Integriert f ueber das Produkt zweier Referenzdreiecke, transformiert.
/// f == 0: konstante 1  -> Sollwert 1/4  (prueft Teilgebiete und Jacobi)
/// f == 1: x1 + 2 x2 + 3 y1 + 4 y2 -> Sollwert 7/6  (prueft die KOORDINATEN)
///
/// Der zweite Fall ist der schaerfere: eine konstante Funktion merkt nicht, ob
/// die Teilgebiete das Gebiet an der richtigen STELLE ueberdecken, solange die
/// Gesamtflaeche stimmt. Ein linearer Integrand merkt es.
double integrate_over_pair(int n_cases, int kind, int f) {
    double tot = 0.0;
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
    for (int k = 0; k < 4; ++k)
    for (int l = 0; l < 4; ++l) {
        const double xi = GL_PT(i), e1 = GL_PT(j), e2 = GL_PT(k), e3 = GL_PT(l);
        const double gw = GL_WG(i) * GL_WG(j) * GL_WG(k) * GL_WG(l);
        for (int c = 0; c < n_cases; ++c) {
            double x1, x2, y1, y2, jac;
            if (kind == 0) {
                get_coincident_coords(c, xi, e1, e2, e3, x1, x2, y1, y2);
                jac = (xi*xi*xi) * (e1*e1) * e2;      // wie in integrate_triangle_coincident
            } else if (kind == 1) {
                get_common_edge_coords(c, xi, e1, e2, e3, x1, x2, y1, y2, jac);
            } else {
                get_common_vertex_coords(c, xi, e1, e2, e3, x1, x2, y1, y2, jac);
            }
            // Simplex-Bedingung mitpruefen: die Bilder muessen im
            // Referenzdreieck 0 <= x2 <= x1 <= 1 liegen. Verlassen sie es,
            // wuerden die Rooftop-Basisfunktionen ausserhalb ihres
            // Definitionsbereichs ausgewertet -- ein Fehler, den die reine
            // Flaechenpruefung nicht sieht.
            if (x2 > x1 + 1e-12 || y2 > y1 + 1e-12 ||
                x1 > 1.0 + 1e-12 || y1 > 1.0 + 1e-12) ++g_outside;
            const double val = (f == 0) ? 1.0 : (x1 + 2.0*x2 + 3.0*y1 + 4.0*y2);
            tot += gw * jac * val;
        }
    }
    return tot;
}

void check(const char* name, int n_cases, int kind, int f, double soll) {
    g_outside = 0;
    const double v = integrate_over_pair(n_cases, kind, f);
    const double rel = std::fabs(v / soll - 1.0);
    const bool ok = rel < 1e-12;
    std::printf("  [%s] %-30s = %.12f   Soll %.12f   Verhaeltnis %.6f%s\n",
                (ok && g_outside == 0) ? "OK " : "FEHL", name, v, soll, v / soll,
                (g_outside > 0) ? "   PUNKTE AUSSERHALB!" : "");
    if (!ok || g_outside > 0) ++g_fail;
}

} // namespace

int main() {
    std::puts("========================================================");
    std::puts(" Sauter-Schwab: Deckung des Integrationsgebiets          ");
    std::puts("========================================================");
    std::puts("\nSollwert fuer alle drei Faelle: 1/4 = 0.25\n");

    std::puts("[1] Konstante Funktion 1 -- prueft Teilgebiete und Jacobi-Determinanten");
    check("Coincident   (6 Teilgebiete)", 6, 0, 0, 0.25);
    check("Common Edge  (5 Teilgebiete)", 5, 1, 0, 0.25);
    check("Common Vertex(2 Teilgebiete)", 2, 2, 0, 0.25);

    std::puts("\n[2] f = x1 + 2 x2 + 3 y1 + 4 y2 -- prueft die KOORDINATEN");
    std::puts("    Exakt ueber das Produkt zweier Referenzdreiecke: 7/6 = 1.166666666667");
    std::puts("    Eine konstante Funktion merkt nicht, ob die Teilgebiete das Gebiet an");
    std::puts("    der richtigen STELLE ueberdecken -- ein linearer Integrand merkt es.");
    const double soll2 = 7.0 / 6.0;
    check("Coincident   (6 Teilgebiete)", 6, 0, 1, soll2);
    check("Common Edge  (5 Teilgebiete)", 5, 1, 1, soll2);
    check("Common Vertex(2 Teilgebiete)", 2, 2, 1, soll2);

    std::puts("\n  Zur Einordnung: vor der Korrektur lieferte Common Edge");
    std::puts("  0.208333 = 5/24, also eine Abdeckung von 0.8333. Sein erstes");
    std::puts("  Teilgebiet trug einen Faktor eta2 zu viel und damit nur die");
    std::puts("  Haelfte seines Beitrags.");

    std::puts("\n========================================================");
    if (g_fail == 0) { std::puts(" ALLE PRUEFUNGEN BESTANDEN");
                       std::puts("========================================================"); return 0; }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
