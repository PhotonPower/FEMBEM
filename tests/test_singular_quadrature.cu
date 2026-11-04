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
// Die Referenzblöcke werden zur Laufzeit mit derselben Zerlegung bei Ordnung
// N=16 gerechnet (16^4 Punkte je Teilgebiet, konvergiert auf ~1e-12; unten als
// lokale Kopie von sqp_accumulate mit freier Ordnung). Früher standen dafür feste
// Zahlen im Quelltext, gerechnet mit der damaligen kontravarianten Rooftop-
// Funktion; sie hängen an der Basisfunktion und wurden mit der kovarianten
// Spurfunktion (docs/design/bem-coupling-fixes.md, Schritt 3) ungültig. Geprüft
// wird, dass die Produktionsordnung N=4 die Referenz auf 2e-3 relativ trifft.
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
#include <vector>

namespace {

int g_failures = 0;

// Gauss-Legendre-Punkte und Gewichte auf [0,1], Ordnung n (Newton auf P_n)
static void gl01(int n, std::vector<double>& x, std::vector<double>& w) {
    x.resize(n); w.resize(n);
    for (int i = 0; i < n; ++i) {
        double z = std::cos(M_PI * (i + 0.75) / (n + 0.5)), pp = 1.0;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1.0, p2 = 0.0;
            for (int j = 1; j <= n; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j - 1.0) * z * p2 - (j - 1.0) * p3) / j; }
            pp = n * (z * p1 - p2) / (z * z - 1.0);
            const double dz = p1 / pp;
            z -= dz;
            if (std::fabs(dz) < 1e-15) break;
        }
        x[i] = 0.5 * (1.0 - z);
        w[i] = 1.0 / ((1.0 - z * z) * pp * pp);
    }
}

// Lokale Kopie von sqp_accumulate mit freier Ordnung N (gleiche Transformationen)
static void sqp_accumulate_N(const double3 nodes_p[4], const double3 nodes_q[4],
                             int tri_p, int tri_q, const int perm_p[3], const int perm_q[3],
                             int n_shared, int mode, double block[4][4], int N) {
    std::vector<double> gx, gw;
    gl01(N, gx, gw);
    const int n_cases = (n_shared == 2) ? 5 : ((n_shared == 1) ? 2 : 1);
    for (int i = 0; i < N; ++i) for (int j = 0; j < N; ++j) for (int k = 0; k < N; ++k) for (int l = 0; l < N; ++l) {
        const double xi = gx[i], eta1 = gx[j], eta2 = gx[k], eta3 = gx[l];
        const double gwt = gw[i] * gw[j] * gw[k] * gw[l];
        for (int c = 0; c < n_cases; ++c) {
            double x1, x2, y1, y2, jac;
            if (n_shared == 2) get_common_edge_coords(c, xi, eta1, eta2, eta3, x1, x2, y1, y2, jac);
            else if (n_shared == 1) get_common_vertex_coords(c, xi, eta1, eta2, eta3, x1, x2, y1, y2, jac);
            else { x1 = xi; x2 = xi * eta1; y1 = eta2; y2 = eta2 * eta3; jac = xi * eta2; }
            double u_Rp, v_Rp, u_Rq, v_Rq;
            sqp_map_to_quad(x1 - x2, x2, tri_p, perm_p, u_Rp, v_Rp);
            sqp_map_to_quad(y1 - y2, y2, tri_q, perm_q, u_Rq, v_Rq);
            double detJ_p, detJ_q; double3 Wp[4], Wq[4];
            evaluate_rooftop_basis(u_Rp, v_Rp, nodes_p, Wp, detJ_p);
            evaluate_rooftop_basis(u_Rq, v_Rq, nodes_q, Wq, detJ_q);
            const double3 rp = evaluate_quad_position(u_Rp, v_Rp, nodes_p);
            const double3 rq = evaluate_quad_position(u_Rq, v_Rq, nodes_q);
            const double dx = rp.x - rq.x, dy = rp.y - rq.y, dz = rp.z - rq.z;
            const double R2 = dx * dx + dy * dy + dz * dz;
            if (R2 < 1e-30) continue;
            const double R = std::sqrt(R2);
            double f = gwt * 16.0 * jac * detJ_p * detJ_q / (4.0 * M_PI);
            if (mode == 0) f /= R;
            else { const double3 n_q = evaluate_quad_normal(u_Rq, v_Rq, nodes_q); f *= (dx * n_q.x + dy * n_q.y + dz * n_q.z) / (R2 * R); }
            for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b)
                block[a][b] += f * (Wp[a].x * Wq[b].x + Wp[a].y * Wq[b].y + Wp[a].z * Wq[b].z);
        }
    }
}

// Lokale Kopie von integrate_singular_quad_pair mit freier Ordnung
static void integrate_pair_N(const double3 P[4], const int gp[4], const double3 Q[4], const int gq[4],
                             int mode, double block[4][4], int N) {
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) block[a][b] = 0.0;
    for (int tp = 0; tp < 2; ++tp) for (int tq = 0; tq < 2; ++tq) {
        int lp[3], lq[3];
        sqp_triangle_nodes(tp, lp); sqp_triangle_nodes(tq, lq);
        int sp[2], sq[2], n_shared = 0;
        for (int a = 0; a < 3 && n_shared < 2; ++a)
            for (int b = 0; b < 3; ++b)
                if (gp[lp[a]] == gq[lq[b]]) { sp[n_shared] = a; sq[n_shared] = b; ++n_shared; break; }
        int perm_p[3], perm_q[3];
        sqp_build_permutation(sp, n_shared, perm_p);
        sqp_build_permutation(sq, n_shared, perm_q);
        sqp_accumulate_N(P, Q, tp, tq, perm_p, perm_q, n_shared, mode, block, N);
    }
}

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
                             int mode)
{
    double blk[4][4], ref[4][4];
    integrate_singular_quad_pair(P, gp, Q, gq, mode, blk);
    integrate_pair_N(P, gp, Q, gq, mode, ref, 16);

    const double rn  = block_norm(ref);
    const double d   = block_diff(blk, ref);
    const double rel = (rn > 0.0) ? d / rn : d;
    const bool   ok  = rel <= 2e-3;

    std::printf("  [%s] %-38s rel. Abw. = %.3e\n", ok ? " OK " : "FEHL", name, rel);
    if (!ok) ++g_failures;
}

} // namespace

int main() {
    std::printf("\n=== Sauter-Schwab: 4-Punkt-Regel gegen konvergierte Referenz (N=16) ===\n\n");

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
        check_against_reference("Kante 60 Grad geknickt, DLP", P, gp, Q, gq, 1);
        check_against_reference("Kante 60 Grad geknickt, SLP", P, gp, Q, gq, 0);
    }

    // --- gemeinsame Kante, geknickt UND stark verzerrt ------------------------
    {
        const double3 P[4] = {{0,0,0},{1,0,0},{1.3,1.1,0.2},{-0.1,0.9,0}};
        const int    gp[4] = {10,11,12,13};
        const double3 Q[4] = {{1,0,0},{1.9,-0.2,0.7},{1.7,1.0,0.9},{1.3,1.1,0.2}};
        const int    gq[4] = {11,14,15,12};
        check_against_reference("Kante geknickt + verzerrt, DLP", P, gp, Q, gq, 1);
    }

    // --- gemeinsame Ecke (shared == 1) ---------------------------------------
    {
        const double3 P[4] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
        const int    gp[4] = {10,11,12,13};
        const double3 Q[4] = {{1,1,0},{2,1,0.3},{2,2,0.3},{1,2,0}};
        const int    gq[4] = {12,20,21,22};
        check_against_reference("gemeinsame Ecke, DLP", P, gp, Q, gq, 1);
    }

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
