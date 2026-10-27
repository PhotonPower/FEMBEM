// =============================================================================
// Abnahmeprüfung: trifft die 4-Punkt-Sauter-Schwab-Regel die konvergierten Werte?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_singular_quadrature
//   ./build/test_singular_quadrature
//
// Reiner Host-Code, keine GPU, kein Netz, in Sekundenbruchteilen durch.
//
// HINTERGRUND
// -----------
// Es gab den Einwand, die Behandlung des Doppelschichtkerns bei GEMEINSAMER
// KANTE (shared == 2) sei unzureichend: der DLP-Kern ist r.n'/(4 pi R^3), der
// Zähler verschwindet dort nur linear, der Nenner kubisch, effektiv also
// O(1/R^2) -- und Sauter-Schwab transformiere nur das schwach singuläre 1/R,
// weshalb man dort numerisches Rauschen einfange.
//
// Das ist nachgemessen worden und trifft nicht zu. Die Common-Edge-
// Transformation zerlegt in FÜNF Teilgebiete (n_cases == 5 in
// sqp_accumulate) mit einer Jacobi-Determinante, die wie rho^3 verschwindet,
// während R proportional zu rho ist. Für einen Kern ~1/R^3 ergibt das
// rho^3/rho^3 = O(1): der transformierte Integrand ist beschränkt und
// analytisch. Gemessen wurde entsprechend EXPONENTIELLE Konvergenz unter
// Ordnungserhöhung, bis auf Maschinengenauigkeit:
//
//   Fall (DLP)                    N=4       N=8       N=12      N=20
//   Kante, koplanar               0 exakt   0         0         0
//   Kante, 60 Grad geknickt       3.8e-04   9.6e-08   5.1e-11   6.6e-14
//   Kante, geknickt + verzerrt    1.1e-03   3.6e-06   1.8e-08   1.2e-13
//   gemeinsame Ecke               6.2e-05   1.1e-11   7.7e-14   5.3e-14
//
// Ein Integrand mit unbehandelter Singularität würde bei Ordnungserhöhung
// nicht konvergieren, sondern wandern oder bestenfalls algebraisch schleichen.
// Sauter-Schwab ist für Kerne bis 1/R^3 konstruiert; die Maue-Identität
// braucht man erst für den hypersingulären Operator (1/R^4).
//
// WAS DIESER TEST FESTHÄLT
// ------------------------
// Die Referenzblöcke unten sind mit derselben Routine bei Ordnung N=24
// gerechnet (24^4 Punkte je Teilgebiet, konvergiert auf ~1e-13). Geprüft wird,
// dass die Produktionsordnung N=4 sie auf 2e-3 relativ trifft.
//
// Diese Schranke ist bewusst nicht enger: 1e-3 ist die tatsächliche
// Quadraturgenauigkeit der 4-Punkt-Regel auf stark verzerrten, geknickten
// Paaren. Solange der Löser bei Residuen um 1e-1 steht, ist das ohne Belang.
// Falls die Konvergenz später unter 1e-4 gehen soll, wäre N=6 zu erwägen
// (Abweichung dann 3.8e-05, Kosten Faktor 5 je singulärem Paar).
//
// Der koplanare Fall wird gesondert gegen EXAKT null geprüft -- dort ist
// r.n' identisch null, und das muss die Quadratur strukturell treffen, nicht
// nur näherungsweise.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "SingularQuadPair.cuh"

#include <cmath>
#include <cstdio>

namespace {

int g_failures = 0;

static const double REF_EDGE_FLAT_DLP[4][4] = {
    {0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00},
    {0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00},
    {0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00},
    {0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00}
};
static const double REF_EDGE_BENT_DLP[4][4] = {
    {3.961355115527707e-02, 1.962412393552174e-19, -2.913928714913471e-02, 5.892313948216469e-20},
    {0.000000000000000e+00, 3.286856597981123e-02, 0.000000000000000e+00, -1.424103538146164e-01},
    {-2.913928714913567e-02, -2.226270841793100e-19, 3.961355115528474e-02, 6.569641052532826e-20},
    {0.000000000000000e+00, -2.973131984528939e-02, 0.000000000000000e+00, 7.000111357795602e-02}
};
static const double REF_EDGE_BENT_SLP[4][4] = {
    {5.031725305937081e-02, 9.129634996109261e-19, -4.587550336131020e-02, -1.123945101312783e-19},
    {0.000000000000000e+00, 8.968063494205836e-02, 0.000000000000000e+00, -1.368584011364766e-01},
    {-4.587550336130777e-02, -9.411105357809760e-19, 5.031725305937925e-02, 2.588043501182639e-19},
    {0.000000000000000e+00, -6.855135466209113e-02, 0.000000000000000e+00, 8.968063494208499e-02}
};
static const double REF_EDGE_SKEW_DLP[4][4] = {
    {3.917866711053992e-02, 3.002525910931006e-03, -2.716570070413125e-02, -2.137631647524119e-02},
    {4.158433728584828e-03, 2.763563414818491e-02, -5.076138877059938e-03, -1.124095281340458e-01},
    {-3.260283083512131e-02, -5.216585292513248e-03, 4.150128927577744e-02, 3.135684505640447e-02},
    {1.386896754345113e-03, -2.134153957167447e-02, -8.110613973950009e-04, 4.710309900912744e-02}
};
static const double REF_VERTEX_DLP[4][4] = {
    {4.417212280540292e-03, -2.693607831819101e-20, -2.567045738420852e-03, 2.257342046402793e-20},
    {0.000000000000000e+00, 2.928517214925792e-03, 0.000000000000000e+00, -6.571847871900284e-03},
    {-8.431589727170093e-03, 4.707599638273697e-20, 4.417212280540991e-03, -4.043228404309460e-20},
    {0.000000000000000e+00, -3.579326753477792e-03, 0.000000000000000e+00, 6.753368186368370e-03}
};

double block_norm(const double b[4][4]) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) s += b[i][j] * b[i][j];
    return std::sqrt(s);
}

double block_diff(const double a[4][4], const double b[4][4]) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
        s += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]);
    return std::sqrt(s);
}

void check_against_reference(const char* name,
                             const double3 P[4], const int gp[4],
                             const double3 Q[4], const int gq[4],
                             int mode, const double ref[4][4])
{
    double blk[4][4];
    integrate_singular_quad_pair(P, gp, Q, gq, mode, blk);

    const double rn  = block_norm(ref);
    const double d   = block_diff(blk, ref);
    const double rel = (rn > 0.0) ? d / rn : d;
    const bool   ok  = rel <= 2e-3;

    std::printf("  [%s] %-38s rel. Abw. = %.3e\n", ok ? " OK " : "FEHL", name, rel);
    if (!ok) ++g_failures;
}

} // namespace

int main() {
    std::printf("\n=== Sauter-Schwab: 4-Punkt-Regel gegen konvergierte Referenz (N=24) ===\n\n");

    const double a = M_PI / 3.0;

    // --- gemeinsame Kante, koplanar -----------------------------------------
    // r.n' verschwindet identisch -> das Integral ist exakt null.
    {
        const double3 P[4] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
        const int    gp[4] = {10,11,12,13};
        const double3 Q[4] = {{1,0,0},{2,0,0},{2,1,0},{1,1,0}};
        const int    gq[4] = {11,14,15,12};

        double blk[4][4];
        integrate_singular_quad_pair(P, gp, Q, gq, 1, blk);
        const double n = block_norm(blk);
        const bool ok = (n == 0.0);
        std::printf("  [%s] %-38s ||Block|| = %.3e (exakt 0 erwartet)\n",
                    ok ? " OK " : "FEHL", "Kante koplanar, DLP", n);
        if (!ok) ++g_failures;
    }

    // --- gemeinsame Kante, 60 Grad geknickt ---------------------------------
    // Der kritische Fall: r.n' verschwindet NICHT, der Kern ist dort effektiv
    // O(1/R^2). Genau der Fall, um den der Einwand ging.
    {
        const double3 P[4] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
        const int    gp[4] = {10,11,12,13};
        const double3 Q[4] = {{1,0,0},{1+std::cos(a),0,std::sin(a)},
                              {1+std::cos(a),1,std::sin(a)},{1,1,0}};
        const int    gq[4] = {11,14,15,12};
        check_against_reference("Kante 60 Grad geknickt, DLP", P, gp, Q, gq, 1,
                                REF_EDGE_BENT_DLP);
        check_against_reference("Kante 60 Grad geknickt, SLP", P, gp, Q, gq, 0,
                                REF_EDGE_BENT_SLP);
    }

    // --- gemeinsame Kante, geknickt UND stark verzerrt ------------------------
    {
        const double3 P[4] = {{0,0,0},{1,0,0},{1.3,1.1,0.2},{-0.1,0.9,0}};
        const int    gp[4] = {10,11,12,13};
        const double3 Q[4] = {{1,0,0},{1.9,-0.2,0.7},{1.7,1.0,0.9},{1.3,1.1,0.2}};
        const int    gq[4] = {11,14,15,12};
        check_against_reference("Kante geknickt + verzerrt, DLP", P, gp, Q, gq, 1,
                                REF_EDGE_SKEW_DLP);
    }

    // --- gemeinsame Ecke (shared == 1) ---------------------------------------
    {
        const double3 P[4] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
        const int    gp[4] = {10,11,12,13};
        const double3 Q[4] = {{1,1,0},{2,1,0.3},{2,2,0.3},{1,2,0}};
        const int    gq[4] = {12,20,21,22};
        check_against_reference("gemeinsame Ecke, DLP", P, gp, Q, gq, 1,
                                REF_VERTEX_DLP);
    }

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
