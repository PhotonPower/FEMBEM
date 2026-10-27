#ifndef BEM_INTEGRATION_CUH
#define BEM_INTEGRATION_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "Mesh.cuh"

// -----------------------------------------------------------------------------
// Hilfsstrukturen für 3D Geometrie auf der GPU
// -----------------------------------------------------------------------------
__host__ __device__ inline double norm3(const double3& v) {
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

__host__ __device__ inline double dot3(const double3& a, const double3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline double3 cross3(const double3& a, const double3& b) {
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__host__ __device__ inline double3 sub3(const double3& a, const double3& b) {
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline double3 add3(const double3& a, const double3& b) {
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline double3 scale3(double s, const double3& a) {
    return make_double3(s*a.x, s*a.y, s*a.z);
}

// -----------------------------------------------------------------------------
// Rooftop-Basis (Analogon zu RWG für Vierecke/Quads)
// Auswertung im lokalen Parameterraum (u, v) in [-1, 1]^2
// Liefert die 4 Basisvektoren auf dem Quad
// -----------------------------------------------------------------------------
__host__ __device__ inline void evaluate_rooftop_basis(double u, double v, const double3 nodes[4], double3 W[4], double& detJ) {
    // 1. Tangentialvektoren (Ableitungen des bilinearen Mappings)
    // N0 = 0.25*(1-u)*(1-v), N1 = 0.25*(1+u)*(1-v), ...
    double3 dr_du = make_double3(
        0.25 * (-(1-v)*nodes[0].x + (1-v)*nodes[1].x + (1+v)*nodes[2].x - (1+v)*nodes[3].x),
        0.25 * (-(1-v)*nodes[0].y + (1-v)*nodes[1].y + (1+v)*nodes[2].y - (1+v)*nodes[3].y),
        0.25 * (-(1-v)*nodes[0].z + (1-v)*nodes[1].z + (1+v)*nodes[2].z - (1+v)*nodes[3].z)
    );
    
    double3 dr_dv = make_double3(
        0.25 * (-(1-u)*nodes[0].x - (1+u)*nodes[1].x + (1+u)*nodes[2].x + (1-u)*nodes[3].x),
        0.25 * (-(1-u)*nodes[0].y - (1+u)*nodes[1].y + (1+u)*nodes[2].y + (1-u)*nodes[3].y),
        0.25 * (-(1-u)*nodes[0].z - (1+u)*nodes[1].z + (1+u)*nodes[2].z + (1-u)*nodes[3].z)
    );

    double3 normal = cross3(dr_du, dr_dv);
    detJ = norm3(normal); // Flächenelement dS = detJ * du * dv

    // Kovariante Basis für Rooftop (Tangentialkomponente)
    // Division durch detJ stellt Piola-Transformation für 2D sicher (Erhalt des Normalenflusses)
    //
    // ORIENTIERUNG (FIX): Die Basisfunktion muss entlang der Kantenrichtung der
    // Topologietabelle zeigen. TopologyExtractor::extract_boundary_faces
    // verwendet zyklisch {0,1}, {1,2}, {2,3}, {3,0}, und
    // mesh.boundary_edge_signs korrigiert nur lokale gegen globale Orientierung
    // (v_min -> v_max), nicht lokale Kante gegen Referenzachse.
    //
    // Mit dem bilinearen Mapping (Knoten 0..3 auf (-1,-1),(+1,-1),(+1,+1),(-1,+1))
    // gilt:
    //   +dr_du bei v=-1  ->  n0 -> n1   passt zu Kante 0 = {0,1}
    //   +dr_dv bei u=+1  ->  n1 -> n2   passt zu Kante 1 = {1,2}
    //   +dr_du bei v=+1  ->  n3 -> n2   Kante 2 = {2,3} laeuft aber n2 -> n3
    //   +dr_dv bei u=-1  ->  n0 -> n3   Kante 3 = {3,0} laeuft aber n3 -> n0
    //
    // Die Kanten 2 und 3 brauchen daher ein Minuszeichen. Vorher fehlte es, was
    // in der Kopplung nicht auffangbar war.
    W[0] = scale3( 0.5 * (1.0 - v) / detJ, dr_du); // Kante {0,1}, n0 -> n1
    W[1] = scale3( 0.5 * (1.0 + u) / detJ, dr_dv); // Kante {1,2}, n1 -> n2
    W[2] = scale3(-0.5 * (1.0 + v) / detJ, dr_du); // Kante {2,3}, n2 -> n3
    W[3] = scale3(-0.5 * (1.0 - u) / detJ, dr_dv); // Kante {3,0}, n3 -> n0
}


// -----------------------------------------------------------------------------
// Standard Gauss-Quadratur (Fernfeld / Nicht-Singulär)
// Integriert W_p * W_q * (1/|r_p - r_q|) über zwei getrennte Quads
// -----------------------------------------------------------------------------
__host__ __device__ inline void integrate_regular_quad_pair(
    const double3 nodes_p[4], const double3 nodes_q[4], 
    double matrix_block[4][4]) 
{
    //const double M_PI = 3.14159265358979323;
    // 4-Punkt Gauss-Regel für 2D [-1,1]^2 pro Quad -> Insgesamt 16 Punkte im 4D
    const double gp = 0.5773502691896257; 
    const double u_pts[4] = {-gp, gp, -gp, gp};
    const double v_pts[4] = {-gp, -gp, gp, gp};
    
    for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) matrix_block[i][j] = 0.0;

    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        double up = u_pts[p_idx];
        double vp = v_pts[p_idx];
        
        double detJ_p;
        double3 Wp[4];
        evaluate_rooftop_basis(up, vp, nodes_p, Wp, detJ_p);
        
        // Punkt r_p im globalen Raum
        double3 rp = make_double3(
            0.25 * ((1-up)*(1-vp)*nodes_p[0].x + (1+up)*(1-vp)*nodes_p[1].x + (1+up)*(1+vp)*nodes_p[2].x + (1-up)*(1+vp)*nodes_p[3].x),
            0.25 * ((1-up)*(1-vp)*nodes_p[0].y + (1+up)*(1-vp)*nodes_p[1].y + (1+up)*(1+vp)*nodes_p[2].y + (1-up)*(1+vp)*nodes_p[3].y),
            0.25 * ((1-up)*(1-vp)*nodes_p[0].z + (1+up)*(1-vp)*nodes_p[1].z + (1+up)*(1+vp)*nodes_p[2].z + (1-up)*(1+vp)*nodes_p[3].z)
        );

        for (int q_idx = 0; q_idx < 4; ++q_idx) {
            double uq = u_pts[q_idx];
            double vq = v_pts[q_idx];
            
            double detJ_q;
            double3 Wq[4];
            evaluate_rooftop_basis(uq, vq, nodes_q, Wq, detJ_q);
            
            double3 rq = make_double3(
                0.25 * ((1-uq)*(1-vq)*nodes_q[0].x + (1+uq)*(1-vq)*nodes_q[1].x + (1+uq)*(1+vq)*nodes_q[2].x + (1-uq)*(1+vq)*nodes_q[3].x),
                0.25 * ((1-uq)*(1-vq)*nodes_q[0].y + (1+uq)*(1-vq)*nodes_q[1].y + (1+uq)*(1+vq)*nodes_q[2].y + (1-uq)*(1+vq)*nodes_q[3].y),
                0.25 * ((1-uq)*(1-vq)*nodes_q[0].z + (1+uq)*(1-vq)*nodes_q[1].z + (1+uq)*(1+vq)*nodes_q[2].z + (1-uq)*(1+vq)*nodes_q[3].z)
            );

            double dist = norm3(sub3(rp, rq));
            double kernel = 1.0 / (4.0 * M_PI * dist); // Laplace-Kern G(r)
            
            double weight = detJ_p * detJ_q; // Gauss-Gewichte sind 1.0
            
            // Aufsummieren
            for(int i=0; i<4; ++i) {
                for(int j=0; j<4; ++j) {
                    matrix_block[i][j] += dot3(Wp[i], Wq[j]) * kernel * weight;
                }
            }
        }
    }
}



// -----------------------------------------------------------------------------
// Normierter Flaechennormalenvektor n' an einer Auswertungsstelle (u, v) im
// Rooftop-Parameterraum. Dupliziert absichtlich nur die Tangentenberechnung
// aus evaluate_rooftop_basis() oben (identische Formeln fuer dr_du/dr_dv),
// statt deren Signatur zu aendern -- so bleibt jede bestehende Aufrufstelle
// (G-Kernel, Sauter-Schwab) unangetastet, und die Orientierung von n' ist per
// Konstruktion konsistent mit der dort bereits etablierten Konvention
// (dieselbe cross3(dr_du, dr_dv), nur normiert statt als detJ verwendet).
// Punktweise statt als konstante Dreiecksnormale ausgewertet: das deckt auch
// nicht perfekt ebene (bilineare) Quads exakt ab, nicht nur den ebenen Fall.
__host__ __device__ inline double3 evaluate_quad_normal(double u, double v, const double3 nodes[4]) {
    double3 dr_du = make_double3(
        0.25 * (-(1-v)*nodes[0].x + (1-v)*nodes[1].x + (1+v)*nodes[2].x - (1+v)*nodes[3].x),
        0.25 * (-(1-v)*nodes[0].y + (1-v)*nodes[1].y + (1+v)*nodes[2].y - (1+v)*nodes[3].y),
        0.25 * (-(1-v)*nodes[0].z + (1-v)*nodes[1].z + (1+v)*nodes[2].z - (1+v)*nodes[3].z)
    );
    double3 dr_dv = make_double3(
        0.25 * (-(1-u)*nodes[0].x - (1+u)*nodes[1].x + (1+u)*nodes[2].x + (1-u)*nodes[3].x),
        0.25 * (-(1-u)*nodes[0].y - (1+u)*nodes[1].y + (1+u)*nodes[2].y + (1-u)*nodes[3].y),
        0.25 * (-(1-u)*nodes[0].z - (1+u)*nodes[1].z + (1+u)*nodes[2].z + (1-u)*nodes[3].z)
    );
    double3 n = cross3(dr_du, dr_dv);
    double len = norm3(n);
    // Ungeschuetzt lieferte ein entartetes Quad (len == 0) eine Normale aus
    // Inf/NaN, die von hier aus in jeden Kernwert und ueber die ACA-Bloecke in
    // den ganzen Operator wandert. Statt dessen die Nullnormale: der DLP-Kern
    // r.n' wird damit null, was fuer ein flaechenloses Element richtig ist.
    if (!(len > 0.0)) return make_double3(0.0, 0.0, 0.0);
    return scale3(1.0 / len, n);
}

// -----------------------------------------------------------------------------
// Double-Layer-Potential (DLP), regulaerer Fall: getrennte, sich nicht
// beruehrende Quads (shared == 0, 1 oder 2 -- siehe Begruendung in
// compute_dense_block_kernel_H). Kern: (r_p - r_q)*n'_q / (4*pi*|r_p-r_q|^3).
// Fuer den exakten Self-Patch (shared == 4) ist das Integral analytisch 0 und
// wird nicht ueber diese Funktion, sondern per Sonderfall behandelt (siehe
// compute_dense_block_kernel_H).
/// Punkt auf dem bilinearen Quad. Die vorhandenen Integratoren schreiben diese
/// Interpolation jeweils aus; hier als Helfer, damit die unterteilte Quadratur
/// unten nicht dieselben zwoelf Zeilen noch zweimal wiederholt.
__host__ __device__ inline double3 evaluate_quad_position(
    double u, double v, const double3 n[4])
{
    const double N0 = 0.25 * (1.0 - u) * (1.0 - v);
    const double N1 = 0.25 * (1.0 + u) * (1.0 - v);
    const double N2 = 0.25 * (1.0 + u) * (1.0 + v);
    const double N3 = 0.25 * (1.0 - u) * (1.0 + v);
    return make_double3(N0*n[0].x + N1*n[1].x + N2*n[2].x + N3*n[3].x,
                        N0*n[0].y + N1*n[1].y + N2*n[2].y + N3*n[3].y,
                        N0*n[0].z + N1*n[1].z + N2*n[2].z + N3*n[3].z);
}


// -----------------------------------------------------------------------------
__host__ __device__ inline void integrate_regular_quad_pair_dlp(
    const double3 nodes_p[4], const double3 nodes_q[4],
    double matrix_block[4][4])
{
    const double gp = 0.5773502691896257;
    const double u_pts[4] = {-gp, gp, -gp, gp};
    const double v_pts[4] = {-gp, -gp, gp, gp};

    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] = 0.0;

    for (int p_idx = 0; p_idx < 4; ++p_idx) {
        double up = u_pts[p_idx];
        double vp = v_pts[p_idx];

        double detJ_p;
        double3 Wp[4];
        evaluate_rooftop_basis(up, vp, nodes_p, Wp, detJ_p);

        double3 rp = make_double3(
            0.25 * ((1-up)*(1-vp)*nodes_p[0].x + (1+up)*(1-vp)*nodes_p[1].x + (1+up)*(1+vp)*nodes_p[2].x + (1-up)*(1+vp)*nodes_p[3].x),
            0.25 * ((1-up)*(1-vp)*nodes_p[0].y + (1+up)*(1-vp)*nodes_p[1].y + (1+up)*(1+vp)*nodes_p[2].y + (1-up)*(1+vp)*nodes_p[3].y),
            0.25 * ((1-up)*(1-vp)*nodes_p[0].z + (1+up)*(1-vp)*nodes_p[1].z + (1+up)*(1+vp)*nodes_p[2].z + (1-up)*(1+vp)*nodes_p[3].z)
        );

        for (int q_idx = 0; q_idx < 4; ++q_idx) {
            double uq = u_pts[q_idx];
            double vq = v_pts[q_idx];

            double detJ_q;
            double3 Wq[4];
            evaluate_rooftop_basis(uq, vq, nodes_q, Wq, detJ_q);

            double3 rq = make_double3(
                0.25 * ((1-uq)*(1-vq)*nodes_q[0].x + (1+uq)*(1-vq)*nodes_q[1].x + (1+uq)*(1+vq)*nodes_q[2].x + (1-uq)*(1+vq)*nodes_q[3].x),
                0.25 * ((1-uq)*(1-vq)*nodes_q[0].y + (1+uq)*(1-vq)*nodes_q[1].y + (1+uq)*(1+vq)*nodes_q[2].y + (1-uq)*(1+vq)*nodes_q[3].y),
                0.25 * ((1-uq)*(1-vq)*nodes_q[0].z + (1+uq)*(1-vq)*nodes_q[1].z + (1+uq)*(1+vq)*nodes_q[2].z + (1-uq)*(1+vq)*nodes_q[3].z)
            );

            // n' des Quell-Quads am Auswertepunkt (u_q, v_q) -- siehe
            // evaluate_quad_normal oben fuer die Orientierungsbegruendung.
            double3 n_q = evaluate_quad_normal(uq, vq, nodes_q);

            double3 r_vec = sub3(rp, rq);
            double dist = norm3(r_vec);
            double dist3 = dist * dist * dist;
            // DLP-Kern: (r_p - r_q)*n'_q / (4*pi*|r_p-r_q|^3)
            double kernel = dot3(r_vec, n_q) / (4.0 * M_PI * dist3);

            double weight = detJ_p * detJ_q;

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    matrix_block[i][j] += dot3(Wp[i], Wq[j]) * kernel * weight;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Lokale Oberflaechen-Massenmatrix T auf einem einzelnen Rand-Quad:
// T_ij = int_Gamma W_i(x) . W_j(x) dGamma_x  (Aufgabe 5.1 aus der Herleitung,
// FEM-BEM Kopplungs-Massenmatrix). Kein Distanz-Kern, keine Singularitaet --
// eine einfache 2x2 Gauss-Regel auf dem Quad reicht.
// -----------------------------------------------------------------------------
__host__ __device__ inline void integrate_boundary_mass_matrix(
    const double3 nodes[4], double matrix_block[4][4])
{
    const double gp = 0.5773502691896257;
    const double u_pts[4] = {-gp, gp, -gp, gp};
    const double v_pts[4] = {-gp, -gp, gp, gp};

    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] = 0.0;

    for (int g = 0; g < 4; ++g) {
        double u = u_pts[g];
        double v = v_pts[g];

        double detJ;
        double3 W[4];
        evaluate_rooftop_basis(u, v, nodes, W, detJ);

        double weight = detJ; // Gauss-Gewicht 1.0 je Punkt
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                matrix_block[i][j] += dot3(W[i], W[j]) * weight;
            }
        }
    }
}


// =============================================================================
// Benachbarte Panelpaare: unterteilte Quadratur
// =============================================================================
//
// Anlass. Bisher liefen ALLE Paare ausser dem exakten Self-Patch (shared == 4)
// durch integrate_regular_quad_pair(_dlp), also durch die glatte
// 4-Punkt-Gauss-Quadratur. Der Kommentar an der Verzweigung nahm an, das sei
// "hinreichend genau". Nachgemessen an zwei Einheitsquadraten, Referenz durch
// 16- bis 20-fache Unterteilung mit 10-Punkt-Gauss:
//
//   Einfachschicht 1/r, gemeinsame Kante        4x4 Gauss:  0.23 % Fehler
//   Einfachschicht 1/r, gemeinsamer Knoten      4x4 Gauss:  0.01 %
//   Doppelschicht, KOPLANARE Nachbarn           exakt 0 -- (r-r').n' = 0
//   Doppelschicht, 90-Grad-Kante                4x4 Gauss: 17.04 %
//
// Fuer die Einfachschicht ist die Annahme also richtig: die Gausspunkte treffen
// den Pol nie, und 1/r ist ueber zwei Flaechen integrierbar. Falsch ist sie fuer
// die Doppelschicht an GEOMETRISCHEN Kanten -- dort ist der Integrand
// (r-r').n'/|r-r'|^3 stark singulaer, und mehr Gausspunkte helfen kaum:
//
//   8x8 Gauss: 15.49 %   16x16: 11.19 %   32x32: 7.15 %
//
// Der Fehler faellt mit etwa n^-0.5. Das ist die Signatur einer fuer
// Gauss-Legendre ungeeigneten Singularitaet; Ordnungserhoehung ist der falsche
// Hebel.
//
// Was hilft, ist UNTERTEILUNG des Parametergebiets (gemessen, 2x2 Gauss je
// Teilquad):
//
//   2x2 Teilquads:  4.95 %      4x4:  2.07 %      8x8:  0.66 %
//
// Bemerkenswert und der Grund gegen den naheliegenden Reflex: mehr Gausspunkte
// JE TEILQUAD verschlechtern das Ergebnis (4x4 Teilquads mit 3x3 Gauss ergeben
// 3.32 % statt 2.07 %), und eine Verdichtung zur gemeinsamen Kante hin ebenso
// (4x4 mit sigma=2 ergibt 13.70 %). Beide bringen Punkte naeher an den Pol.
// Gleichmaessige Unterteilung mit niedriger Ordnung je Zelle ist hier das
// Richtige.
//
// Umfang am TEAM-28-Netz: von 16562 benachbarten Panelpaaren sind 1632 nicht
// koplanar (544 mit gemeinsamer Kante, 1088 mit gemeinsamem Knoten). Sie sitzen
// an den Plattenkanten und Spulenecken -- dort, wo die Kraft entsteht.
//
// Deshalb wird die Unterteilungstiefe nach der Normalenabweichung gewaehlt:
// koplanare Nachbarn brauchen sie nicht (Doppelschicht verschwindet dort
// ohnehin), nicht koplanare bekommen die volle Tiefe. Das haelt die Kosten auf
// den 10 Prozent der Paare, die sie brauchen.
//
// WICHTIG zur Umsetzung: unterteilt wird das PARAMETERgebiet [-1,1]^2, nicht
// die Geometrie. Die Rooftop-Basisfunktionen sind auf dem ganzen Quad definiert
// und werden weiterhin an den originalen (u,v) ausgewertet -- eine Unterteilung
// in echte Teilpanels wuerde andere Basisfunktionen bedeuten und waere falsch.

/// Unterteilungsgrad nach Normalenabweichung. Ueber 18 Grad Knick gilt ein Paar
/// als nicht koplanar.
__host__ __device__ inline int adjacent_subdivision_level(
    const double3 nodes_p[4], const double3 nodes_q[4])
{
    const double3 np = evaluate_quad_normal(0.0, 0.0, nodes_p);
    const double3 nq = evaluate_quad_normal(0.0, 0.0, nodes_q);
    const double c = fabs(np.x*nq.x + np.y*nq.y + np.z*nq.z);
    return (c > 0.95) ? 2 : 8;
}

/// Einfachschicht fuer benachbarte Paare, NSUB x NSUB Teilgebiete mit 2x2 Gauss.
__host__ __device__ inline void integrate_adjacent_quad_pair(
    const double3 nodes_p[4], const double3 nodes_q[4],
    double matrix_block[4][4], int nsub)
{
    const double gp = 0.5773502691896257;
    const double h  = 2.0 / static_cast<double>(nsub);
    const double hw = 0.5 * h;                 // halbe Zellbreite = Gewichtsfaktor

    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] = 0.0;

    for (int cpi = 0; cpi < nsub; ++cpi) {
    for (int cpj = 0; cpj < nsub; ++cpj) {
        const double mpu = -1.0 + h * (cpi + 0.5);
        const double mpv = -1.0 + h * (cpj + 0.5);
        for (int gpi = 0; gpi < 2; ++gpi) {
        for (int gpj = 0; gpj < 2; ++gpj) {
            const double up = mpu + hw * (gpi ? gp : -gp);
            const double vp = mpv + hw * (gpj ? gp : -gp);
            const double wp = hw * hw;

            double detJ_p; double3 Wp[4];
            evaluate_rooftop_basis(up, vp, nodes_p, Wp, detJ_p);
            const double3 rp = evaluate_quad_position(up, vp, nodes_p);

            for (int cqi = 0; cqi < nsub; ++cqi) {
            for (int cqj = 0; cqj < nsub; ++cqj) {
                const double mqu = -1.0 + h * (cqi + 0.5);
                const double mqv = -1.0 + h * (cqj + 0.5);
                for (int gqi = 0; gqi < 2; ++gqi) {
                for (int gqj = 0; gqj < 2; ++gqj) {
                    const double uq = mqu + hw * (gqi ? gp : -gp);
                    const double vq = mqv + hw * (gqj ? gp : -gp);
                    const double wq = hw * hw;

                    double detJ_q; double3 Wq[4];
                    evaluate_rooftop_basis(uq, vq, nodes_q, Wq, detJ_q);
                    const double3 rq = evaluate_quad_position(uq, vq, nodes_q);

                    const double dx = rp.x - rq.x, dy = rp.y - rq.y, dz = rp.z - rq.z;
                    const double R = sqrt(dx*dx + dy*dy + dz*dz);
                    if (R < 1e-14) continue;

                    const double f = wp * wq * detJ_p * detJ_q / (4.0 * M_PI * R);
                    for (int i = 0; i < 4; ++i)
                        for (int j = 0; j < 4; ++j)
                            matrix_block[i][j] += f * (Wp[i].x*Wq[j].x + Wp[i].y*Wq[j].y + Wp[i].z*Wq[j].z);
                }}
            }}
        }}
    }}
}

/// Doppelschicht fuer benachbarte Paare, gleiche Unterteilung.
__host__ __device__ inline void integrate_adjacent_quad_pair_dlp(
    const double3 nodes_p[4], const double3 nodes_q[4],
    double matrix_block[4][4], int nsub)
{
    const double gp = 0.5773502691896257;
    const double h  = 2.0 / static_cast<double>(nsub);
    const double hw = 0.5 * h;

    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] = 0.0;

    for (int cpi = 0; cpi < nsub; ++cpi) {
    for (int cpj = 0; cpj < nsub; ++cpj) {
        const double mpu = -1.0 + h * (cpi + 0.5);
        const double mpv = -1.0 + h * (cpj + 0.5);
        for (int gpi = 0; gpi < 2; ++gpi) {
        for (int gpj = 0; gpj < 2; ++gpj) {
            const double up = mpu + hw * (gpi ? gp : -gp);
            const double vp = mpv + hw * (gpj ? gp : -gp);
            const double wp = hw * hw;

            double detJ_p; double3 Wp[4];
            evaluate_rooftop_basis(up, vp, nodes_p, Wp, detJ_p);
            const double3 rp = evaluate_quad_position(up, vp, nodes_p);

            for (int cqi = 0; cqi < nsub; ++cqi) {
            for (int cqj = 0; cqj < nsub; ++cqj) {
                const double mqu = -1.0 + h * (cqi + 0.5);
                const double mqv = -1.0 + h * (cqj + 0.5);
                for (int gqi = 0; gqi < 2; ++gqi) {
                for (int gqj = 0; gqj < 2; ++gqj) {
                    const double uq = mqu + hw * (gqi ? gp : -gp);
                    const double vq = mqv + hw * (gqj ? gp : -gp);
                    const double wq = hw * hw;

                    double detJ_q; double3 Wq[4];
                    evaluate_rooftop_basis(uq, vq, nodes_q, Wq, detJ_q);
                    const double3 rq = evaluate_quad_position(uq, vq, nodes_q);
                    const double3 n_q = evaluate_quad_normal(uq, vq, nodes_q);

                    const double dx = rp.x - rq.x, dy = rp.y - rq.y, dz = rp.z - rq.z;
                    const double R2 = dx*dx + dy*dy + dz*dz;
                    if (R2 < 1e-28) continue;
                    const double R = sqrt(R2);

                    const double num = dx*n_q.x + dy*n_q.y + dz*n_q.z;
                    const double f = wp * wq * detJ_p * detJ_q * num / (4.0 * M_PI * R2 * R);
                    for (int i = 0; i < 4; ++i)
                        for (int j = 0; j < 4; ++j)
                            matrix_block[i][j] += f * (Wp[i].x*Wq[j].x + Wp[i].y*Wq[j].y + Wp[i].z*Wq[j].z);
                }}
            }}
        }}
    }}
}

#endif // BEM_INTEGRATION_CUH
