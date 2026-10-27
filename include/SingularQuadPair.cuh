#ifndef SINGULAR_QUAD_PAIR_CUH
#define SINGULAR_QUAD_PAIR_CUH

// =============================================================================
// SingularQuadPair.cuh -- Sauter-Schwab fuer benachbarte Panelpaare
// =============================================================================
//
// Vorgeschichte, weil sie fuer das Vertrauen in diese Datei zaehlt: eine erste
// Fassung lag um 66 % (Doppelschicht) bzw. 47 % (Einfachschicht) neben der
// Referenz. Der Fehler lag nicht hier, sondern in get_common_edge_coords --
// dort waren sowohl der Jacobi-Faktor des ersten Teilgebiets als auch alle
// fuenf Koordinatenformeln verstellt. Gefunden wurden beide mit zwei exakten
// Tests ohne jede Singularitaet (tests/test_sauter_schwab_coverage): die
// konstante Funktion 1 muss ueber das Produkt zweier Referenzdreiecke 1/4
// ergeben, ein linearer Integrand 7/6. Nach der Korrektur trifft diese Datei
// die Referenz auf 0.02 % (Einfachschicht) und 1.53 % (Doppelschicht).
//
// =============================================================================
//
// Ersetzt die unterteilte Quadratur, die zuvor an dieser Stelle stand. Diese
// war ein Notbehelf: sie brachte den Fehler an einer 90-Grad-Kante von 20.13 %
// auf 2.71 %, aber die Konvergenz blieb langsam (12x12 Teilquads 1.48 %,
// 16x16 noch 0.77 % bei 1.05e6 Kernauswertungen je Paar), weil sie die
// Singularitaet nur ueberdeckt statt sie aufzuheben.
//
// -----------------------------------------------------------------------------
// Verfahren
// -----------------------------------------------------------------------------
// Jedes Quad wird in zwei Dreiecke zerlegt -- A = (0,1,2) und B = (0,2,3),
// dieselbe Zerlegung, die map_tri_to_quad in SauterSchwab.cuh voraussetzt. Fuer
// jedes der vier Dreieckspaare wird bestimmt, wie viele Ecken es teilt, und die
// passende Sauter-Schwab-Transformation angewandt:
//
//   2 gemeinsame Ecken (Common Edge)   5 Teilgebiete, Jacobi xi^3 eta1^2 eta2
//   1 gemeinsame Ecke  (Common Vertex) 2 Teilgebiete, Jacobi xi^3 eta2
//   0 gemeinsame Ecken (getrennt)      Produktregel ueber beide Dreiecke
//
// Die Jacobi-Determinante der Duffy-artigen Transformation hebt die
// Singularitaet des Kerns auf: am Beruehrpunkt geht |r-r'| wie O(xi), der
// 1/r-Kern also wie O(1/xi) und der Doppelschichtkern wie O(1/xi^2), waehrend
// xi^3 dagegensteht. Der transformierte Integrand ist damit beschraenkt, und
// Gauss-Legendre konvergiert wieder exponentiell statt mit n^-0.5.
//
// -----------------------------------------------------------------------------
// Die Permutation, an der es haengt
// -----------------------------------------------------------------------------
// Sauter-Schwab setzt voraus, dass die geteilten Ecken in BEIDEN Dreiecken auf
// denselben lokalen Positionen liegen und in derselben Reihenfolge: bei
// gemeinsamer Kante auf 0 und 1, bei gemeinsamer Ecke auf 0. Genau diese
// Vorsortierung ist der Grund, aus dem der urspruengliche Code den Fall
// uebersprang ("benoetigt eine CPU-Vorsortierung der Knotenindizes"). Sie laesst
// sich aber lokal aus den globalen Knotennummern des Paares bestimmen, ohne
// Vorverarbeitung auf dem Host -- das tut build_permutation() unten.
//
// Nach der Transformation wird ueber die Permutation zurueckgerechnet und vom
// Dreieck auf das Quad-Referenzgebiet [-1,1]^2 abgebildet, weil die
// Rooftop-Basisfunktionen dort definiert sind. Eine Zerlegung in echte
// Teilpanels waere eine andere Basis und damit falsch.
// =============================================================================

#include "BemIntegration.cuh"
#include "SauterSchwab.cuh"

// -----------------------------------------------------------------------------
// Zerlegung des Quads in zwei Dreiecke, passend zu map_tri_to_quad
// -----------------------------------------------------------------------------
__host__ __device__ inline void sqp_triangle_nodes(int tri_id, int out[3]) {
    if (tri_id == 0) { out[0] = 0; out[1] = 1; out[2] = 2; }   // Dreieck A
    else             { out[0] = 0; out[1] = 2; out[2] = 3; }   // Dreieck B
}

/// Baryzentrische Dreieckskoordinaten -> Quad-Referenzgebiet, mit
/// Ent-Permutierung. perm[k] ist die ORIGINALE lokale Dreiecksecke, die in der
/// Sauter-Schwab-Numerierung an Position k steht.
__host__ __device__ inline void sqp_map_to_quad(
    double u_ss, double v_ss, int tri_id, const int perm[3],
    double& u_R, double& v_R)
{
    const double l_ss[3] = { 1.0 - u_ss - v_ss, u_ss, v_ss };
    double l_orig[3];
    l_orig[perm[0]] = l_ss[0];
    l_orig[perm[1]] = l_ss[1];
    l_orig[perm[2]] = l_ss[2];

    map_tri_to_quad(l_orig[1], l_orig[2], tri_id, u_R, v_R);
}

/// Permutation, die die gemeinsamen Ecken nach vorne holt.
/// 'shared_local' enthaelt die 'n_shared' lokalen Indizes der geteilten Ecken
/// in der Reihenfolge, die fuer BEIDE Dreiecke gleich sein muss.
__host__ __device__ inline void sqp_build_permutation(
    const int shared_local[2], int n_shared, int perm[3])
{
    bool used[3] = { false, false, false };
    for (int k = 0; k < n_shared; ++k) { perm[k] = shared_local[k]; used[shared_local[k]] = true; }
    int next = n_shared;
    for (int k = 0; k < 3; ++k) if (!used[k]) perm[next++] = k;
}

// -----------------------------------------------------------------------------
// Kern eines Dreieckspaares
// -----------------------------------------------------------------------------
// 'mode' waehlt den Kern: 0 = Einfachschicht 1/(4 pi R),
//                         1 = Doppelschicht (r-r').n'/(4 pi R^3)
// Die Beitraege werden auf 'block' AUFADDIERT, nicht gesetzt -- der Aufrufer
// summiert ueber die vier Dreieckspaare.
__host__ __device__ inline void sqp_accumulate(
    const double3 nodes_p[4], const double3 nodes_q[4],
    int tri_p, int tri_q, const int perm_p[3], const int perm_q[3],
    int n_shared, int mode, double block[4][4])
{
    // Zahl der Teilgebiete und Transformation nach Beruehrungsart
    const int n_cases = (n_shared == 2) ? 5 : ((n_shared == 1) ? 2 : 1);

    for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
    for (int k = 0; k < 4; ++k) {
    for (int l = 0; l < 4; ++l) {
        const double xi   = GL_PT(i);
        const double eta1 = GL_PT(j);
        const double eta2 = GL_PT(k);
        const double eta3 = GL_PT(l);
        const double gw   = GL_WG(i) * GL_WG(j) * GL_WG(k) * GL_WG(l);

        for (int c = 0; c < n_cases; ++c) {
            double x1, x2, y1, y2, jac;

            if (n_shared == 2) {
                get_common_edge_coords(c, xi, eta1, eta2, eta3, x1, x2, y1, y2, jac);
            } else if (n_shared == 1) {
                get_common_vertex_coords(c, xi, eta1, eta2, eta3, x1, x2, y1, y2, jac);
            } else {
                // Getrennte Dreiecke: einfache Produktregel. x1 = xi, x2 = xi*eta1
                // parametrisiert das Referenzdreieck 0 <= x2 <= x1 <= 1 mit
                // Jacobi xi; entsprechend fuer y.
                x1 = xi;   x2 = xi * eta1;
                y1 = eta2; y2 = eta2 * eta3;
                jac = xi * eta2;
            }

            // Baryzentrisch: lambda = (1-x1, x1-x2, x2)
            double u_Rp, v_Rp, u_Rq, v_Rq;
            sqp_map_to_quad(x1 - x2, x2, tri_p, perm_p, u_Rp, v_Rp);
            sqp_map_to_quad(y1 - y2, y2, tri_q, perm_q, u_Rq, v_Rq);

            double detJ_p, detJ_q;
            double3 Wp[4], Wq[4];
            evaluate_rooftop_basis(u_Rp, v_Rp, nodes_p, Wp, detJ_p);
            evaluate_rooftop_basis(u_Rq, v_Rq, nodes_q, Wq, detJ_q);

            const double3 rp = evaluate_quad_position(u_Rp, v_Rp, nodes_p);
            const double3 rq = evaluate_quad_position(u_Rq, v_Rq, nodes_q);

            const double dx = rp.x - rq.x, dy = rp.y - rq.y, dz = rp.z - rq.z;
            const double R2 = dx*dx + dy*dy + dz*dz;
            if (R2 < 1e-30) continue;
            const double R = sqrt(R2);

            // Faktor 16: je Dreieck bringt die Abbildung auf das
            // Quad-Referenzgebiet [-1,1]^2 eine Determinante von 4.
            double f = gw * 16.0 * jac * detJ_p * detJ_q / (4.0 * M_PI);

            if (mode == 0) {
                f /= R;
            } else {
                const double3 n_q = evaluate_quad_normal(u_Rq, v_Rq, nodes_q);
                f *= (dx*n_q.x + dy*n_q.y + dz*n_q.z) / (R2 * R);
            }

            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    block[a][b] += f * (Wp[a].x*Wq[b].x + Wp[a].y*Wq[b].y + Wp[a].z*Wq[b].z);
        }
    }}}}
}

// -----------------------------------------------------------------------------
// Treiber fuer ein Quad-Paar
// -----------------------------------------------------------------------------
/// 'gid_p' und 'gid_q' sind die GLOBALEN Knotennummern der beiden Quads. Sie
/// werden gebraucht, um die geteilten Ecken zu erkennen -- ein Vergleich der
/// Koordinaten waere von Rundung abhaengig und bei zusammenfallenden, aber
/// topologisch getrennten Knoten falsch.
///
/// mode: 0 = Einfachschicht, 1 = Doppelschicht.
__host__ __device__ inline void integrate_singular_quad_pair(
    const double3 nodes_p[4], const int gid_p[4],
    const double3 nodes_q[4], const int gid_q[4],
    int mode, double block[4][4])
{
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) block[a][b] = 0.0;

    for (int tp = 0; tp < 2; ++tp) {
    for (int tq = 0; tq < 2; ++tq) {
        int lp[3], lq[3];
        sqp_triangle_nodes(tp, lp);
        sqp_triangle_nodes(tq, lq);

        // Geteilte Ecken suchen. Die Reihenfolge muss in BEIDEN Dreiecken
        // dieselbe sein, deshalb wird ueber die Ecken von p gelaufen und die
        // Partnerposition in q dazu notiert.
        int sp[2], sq[2], n_shared = 0;
        for (int a = 0; a < 3 && n_shared < 2; ++a) {
            for (int b = 0; b < 3; ++b) {
                if (gid_p[lp[a]] == gid_q[lq[b]]) {
                    sp[n_shared] = a; sq[n_shared] = b; ++n_shared;
                    break;
                }
            }
        }

        int perm_p[3], perm_q[3];
        sqp_build_permutation(sp, n_shared, perm_p);
        sqp_build_permutation(sq, n_shared, perm_q);

        sqp_accumulate(nodes_p, nodes_q, tp, tq, perm_p, perm_q, n_shared, mode, block);
    }}
}

#endif // SINGULAR_QUAD_PAIR_CUH
