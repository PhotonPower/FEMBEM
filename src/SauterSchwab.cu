#include "SauterSchwab.cuh"


// Die Gauss-Legendre-Stuetzstellen liegen jetzt als inline-Funktionen
// GL_PT()/GL_WG() in SauterSchwab.cuh -- sie werden auch vom CPU-Pfad
// gebraucht, und __constant__ ist nur aus Device-Code lesbar. Siehe die
// ausfuehrliche Begruendung dort.


// =================================================================================
// Der Haupt-Integrator für ein koinzidentes Dreieckspaar
// =================================================================================
__host__ __device__ void integrate_triangle_coincident(
    const double3 nodes[4], int tri_p, int tri_q, double matrix_block[4][4])
{
    //const double M_PI = 3.14159265358979323846;
    // Initialisiere die Ausgabematrix mit Null
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrix_block[i][j] = 0.0;
        }
    }

    // Die geschachtelte 4D Gauss-Quadratur (4^4 = 256 Punkte pro Simplex)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 4; ++l) {

                    double xi = GL_PT(i);
                    double eta1 = GL_PT(j);
                    double eta2 = GL_PT(k);
                    double eta3 = GL_PT(l);

                    double gauss_weight = GL_WG(i) * GL_WG(j) * GL_WG(k) * GL_WG(l);

                    // Die Duffy-Jacobi Determinante: xi^3 * eta1^2 * eta2
                    // Multipliziert mit 16.0 durch das Mapping vom Dreieck auf das Quad
                    double duffy_jacobian = 16.0 * (xi * xi * xi) * (eta1 * eta1) * eta2;
                    double total_weight = gauss_weight * duffy_jacobian;

                    // Iteration über die 6 Sub-Simplizes
                    for (int case_id = 0; case_id < 6; ++case_id) {

                        double x1, x2, y1, y2;
                        get_coincident_coords(case_id, xi, eta1, eta2, eta3, x1, x2, y1, y2);

                        // Rückrechnung in die Baryzentrischen Dreieckskoordinaten
                        double u_p = x1 - x2;
                        double v_p = x2;
                        double u_q = y1 - y2;
                        double v_q = y2;

                        // Mapping in das Rooftop-System [-1, 1]^2
                        double u_Rp, v_Rp, u_Rq, v_Rq;
                        map_tri_to_quad(u_p, v_p, tri_p, u_Rp, v_Rp);
                        map_tri_to_quad(u_q, v_q, tri_q, u_Rq, v_Rq);

                        // Auswertung der Basisvektoren und der Quad-Jacobi-Determinante
                        double detJ_p, detJ_q;
                        double3 Wp[4], Wq[4];
                        evaluate_rooftop_basis(u_Rp, v_Rp, nodes, Wp, detJ_p);
                        evaluate_rooftop_basis(u_Rq, v_Rq, nodes, Wq, detJ_q);

                        // Berechnung der physikalischen 3D-Koordinaten auf dem Quad
                        double3 rp = make_double3(
                            0.25 * ((1 - u_Rp) * (1 - v_Rp) * nodes[0].x + (1 + u_Rp) * (1 - v_Rp) * nodes[1].x +
                                (1 + u_Rp) * (1 + v_Rp) * nodes[2].x + (1 - u_Rp) * (1 + v_Rp) * nodes[3].x),
                            0.25 * ((1 - u_Rp) * (1 - v_Rp) * nodes[0].y + (1 + u_Rp) * (1 - v_Rp) * nodes[1].y +
                                (1 + u_Rp) * (1 + v_Rp) * nodes[2].y + (1 - u_Rp) * (1 + v_Rp) * nodes[3].y),
                            0.25 * ((1 - u_Rp) * (1 - v_Rp) * nodes[0].z + (1 + u_Rp) * (1 - v_Rp) * nodes[1].z +
                                (1 + u_Rp) * (1 + v_Rp) * nodes[2].z + (1 - u_Rp) * (1 + v_Rp) * nodes[3].z)
                        );

                        double3 rq = make_double3(
                            0.25 * ((1 - u_Rq) * (1 - v_Rq) * nodes[0].x + (1 + u_Rq) * (1 - v_Rq) * nodes[1].x +
                                (1 + u_Rq) * (1 + v_Rq) * nodes[2].x + (1 - u_Rq) * (1 + v_Rq) * nodes[3].x),
                            0.25 * ((1 - u_Rq) * (1 - v_Rq) * nodes[0].y + (1 + u_Rq) * (1 - v_Rq) * nodes[1].y +
                                (1 + u_Rq) * (1 + v_Rq) * nodes[2].y + (1 - u_Rq) * (1 + v_Rq) * nodes[3].y),
                            0.25 * ((1 - u_Rq) * (1 - v_Rq) * nodes[0].z + (1 + u_Rq) * (1 - v_Rq) * nodes[1].z +
                                (1 + u_Rq) * (1 + v_Rq) * nodes[2].z + (1 - u_Rq) * (1 + v_Rq) * nodes[3].z)
                        );

                        // Physischer euklidischer Abstand (Nenner des Kerns)
                        double dist = norm3(sub3(rp, rq));

                        // Auswertung des Kerns (1 / 4*pi*r)
                        // Da xi durch Gauss-Quadratur nie exakt Null wird, ist dist > 0.
                        double kernel_val = 1.0 / (4.0 * M_PI * dist);

                        double final_weight = total_weight * detJ_p * detJ_q * kernel_val;

                        // Addieren zur lokalen 4x4 Quad-Matrix
                        for (int r = 0; r < 4; ++r) {
                            for (int c = 0; c < 4; ++c) {
                                matrix_block[r][c] += dot3(Wp[r], Wq[c]) * final_weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

// =================================================================================
// Der Integrator für das diagonal geteilte Dreieckspaar im Self-Patch
// =================================================================================
__host__ __device__ void integrate_triangle_common_edge_self_patch(
    const double3 nodes[4], int is_tri_B_p, int is_tri_B_q, double matrix_block[4][4])
{
    //const double M_PI = 3.14159265358979323846;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] = 0.0;

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 4; ++l) {

                    double xi = GL_PT(i);
                    double eta1 = GL_PT(j);
                    double eta2 = GL_PT(k);
                    double eta3 = GL_PT(l);

                    double gauss_weight = GL_WG(i) * GL_WG(j) * GL_WG(k) * GL_WG(l);

                    // Iteration über die 5 Sub-Simplizes
                    for (int case_id = 0; case_id < 5; ++case_id) {

                        double x1, x2, y1, y2, ss_jac;
                        get_common_edge_coords(case_id, xi, eta1, eta2, eta3, x1, x2, y1, y2, ss_jac);

                        // 16.0 = 4.0 (Fläche Quad/Tri_p) * 4.0 (Fläche Quad/Tri_q)
                        double total_weight = gauss_weight * ss_jac * 16.0;

                        // Rückrechnung Baryzentrisch
                        double u_p = x1 - x2;  double v_p = x2;
                        double u_q = y1 - y2;  double v_q = y2;

                        double u_Rp, v_Rp, u_Rq, v_Rq;
                        map_self_patch_edge_to_quad(u_p, v_p, is_tri_B_p, u_Rp, v_Rp);
                        map_self_patch_edge_to_quad(u_q, v_q, is_tri_B_q, u_Rq, v_Rq);

                        double detJ_p, detJ_q;
                        double3 Wp[4], Wq[4];
                        evaluate_rooftop_basis(u_Rp, v_Rp, nodes, Wp, detJ_p);
                        evaluate_rooftop_basis(u_Rq, v_Rq, nodes, Wq, detJ_q);

                        // Physikalische Koordinaten berechnen (gleicher Code wie Coincident)
                        double3 rp = make_double3(
                            0.25 * ((1 - u_Rp) * (1 - v_Rp) * nodes[0].x + (1 + u_Rp) * (1 - v_Rp) * nodes[1].x + (1 + u_Rp) * (1 + v_Rp) * nodes[2].x + (1 - u_Rp) * (1 + v_Rp) * nodes[3].x),
                            0.25 * ((1 - u_Rp) * (1 - v_Rp) * nodes[0].y + (1 + u_Rp) * (1 - v_Rp) * nodes[1].y + (1 + u_Rp) * (1 + v_Rp) * nodes[2].y + (1 - u_Rp) * (1 + v_Rp) * nodes[3].y),
                            0.25 * ((1 - u_Rp) * (1 - v_Rp) * nodes[0].z + (1 + u_Rp) * (1 - v_Rp) * nodes[1].z + (1 + u_Rp) * (1 + v_Rp) * nodes[2].z + (1 - u_Rp) * (1 + v_Rp) * nodes[3].z)
                        );

                        double3 rq = make_double3(
                            0.25 * ((1 - u_Rq) * (1 - v_Rq) * nodes[0].x + (1 + u_Rq) * (1 - v_Rq) * nodes[1].x + (1 + u_Rq) * (1 + v_Rq) * nodes[2].x + (1 - u_Rq) * (1 + v_Rq) * nodes[3].x),
                            0.25 * ((1 - u_Rq) * (1 - v_Rq) * nodes[0].y + (1 + u_Rq) * (1 - v_Rq) * nodes[1].y + (1 + u_Rq) * (1 + v_Rq) * nodes[2].y + (1 - u_Rq) * (1 + v_Rq) * nodes[3].y),
                            0.25 * ((1 - u_Rq) * (1 - v_Rq) * nodes[0].z + (1 + u_Rq) * (1 - v_Rq) * nodes[1].z + (1 + u_Rq) * (1 + v_Rq) * nodes[2].z + (1 - u_Rq) * (1 + v_Rq) * nodes[3].z)
                        );

                        double dist = norm3(sub3(rp, rq));
                        double kernel_val = 1.0 / (4.0 * M_PI * dist);
                        double final_weight = total_weight * detJ_p * detJ_q * kernel_val;

                        for (int r = 0; r < 4; ++r) {
                            for (int c = 0; c < 4; ++c) {
                                matrix_block[r][c] += dot3(Wp[r], Wq[c]) * final_weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

// =================================================================================
// FINALER WRAPPER: Das vollständige, rigorose Self-Patch Integral (Identische Quads)
// =================================================================================
__host__ __device__ void integrate_singular_self_patch(
    const double3 nodes[4], double matrix_block[4][4])
{
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] = 0.0;

    double temp_block[4][4];

    // 1. Identisches Dreieck A x A (Coincident, 6 Simplizes)
    integrate_triangle_coincident(nodes, 0, 0, temp_block);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] += temp_block[i][j];

    // 2. Identisches Dreieck B x B (Coincident, 6 Simplizes)
    integrate_triangle_coincident(nodes, 1, 1, temp_block);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] += temp_block[i][j];

    // 3. Gemeinsame Diagonale A x B (Common Edge, 5 Simplizes)
    integrate_triangle_common_edge_self_patch(nodes, 0, 1, temp_block);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] += temp_block[i][j];

    // 4. Gemeinsame Diagonale B x A (Common Edge, 5 Simplizes)
    integrate_triangle_common_edge_self_patch(nodes, 1, 0, temp_block);
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) matrix_block[i][j] += temp_block[i][j];
}

// =================================================================================
// Topologische Permutation für die Diagonale im Self-Patch
// Mappt das Sauter-Schwab Dreieck auf das Rooftop-Quad [-1,1]^2
// =================================================================================
__host__ __device__ __forceinline__ void map_self_patch_edge_to_quad(
    double u, double v, int is_tri_B, double& u_R, double& v_R)
{
    // In der Standard-Sauter-Schwab Kante liegt die gemeinsame Kante auf v=0.
    if (is_tri_B == 0) {
        // Dreieck A (Knoten 0, 1, 2). Wir ordnen es um als (0, 2, 1), 
        // damit die Diagonale (0-2) auf v=0 liegt.
        double u_orig = v;
        double v_orig = u;
        u_R = -1.0 + 2.0 * u_orig + 2.0 * v_orig;
        v_R = -1.0 + 2.0 * v_orig;
    }
    else {
        // Dreieck B (Knoten 0, 2, 3). Wir ordnen es um als (0, 2, 3), 
        // die Diagonale (0-2) liegt bereits auf v=0.
        double u_orig = u;
        double v_orig = v;
        u_R = -1.0 + 2.0 * u_orig;
        v_R = -1.0 + 2.0 * u_orig + 2.0 * v_orig;
    }
}


// Hinweis: get_common_vertex_coords stand frueher hier. Da es __forceinline__
// ist und inzwischen auch aus SingularQuadPair.cuh benutzt wird, muss die
// Definition im Header sichtbar sein -- sie ist nach SauterSchwab.cuh
// gewandert. Inhaltlich unveraendert.

