#include "SauterSchwab.cuh"
#include "SingularPair.cuh"

// Hilfsfunktion: Wandelt die lokalen Sauter-Schwab Dreieckskoordinaten (u_ss, v_ss) 
// über die ermittelte Permutation zurück in das originale Quad-Referenzsystem [-1,1]^2
__host__ __device__ __forceinline__ void map_ss_to_rooftop(
    double u_ss, double v_ss, int tri_type, const int perm[3], double& u_R, double& v_R) 
{
    // 1. Ent-Permutieren: Sauter-Schwab (0,1,2) -> Original-Dreieck (0,1,2)
    double l_ss[3] = {1.0 - u_ss - v_ss, u_ss, v_ss}; // Baryzentrische Koord. im SS-System
    double l_orig[3];
    
    l_orig[perm[0]] = l_ss[0];
    l_orig[perm[1]] = l_ss[1];
    l_orig[perm[2]] = l_ss[2];
    
    // Die originalen affinen Koordinaten im Dreieck
    double u_orig = l_orig[1];
    double v_orig = l_orig[2];

    // 2. Mapping vom Dreieck auf das Rooftop-Quad (wie gehabt)
    if (tri_type == 0) { // Dreieck A
        u_R = -1.0 + 2.0 * u_orig + 2.0 * v_orig;
        v_R = -1.0 + 2.0 * v_orig;
    } else {             // Dreieck B
        u_R = -1.0 + 2.0 * u_orig;
        v_R = -1.0 + 2.0 * u_orig + 2.0 * v_orig;
    }
}


// =================================================================================
// Kernel zur Berechnung aller Common-Edge Paare auf der GPU
// =================================================================================
__global__ void compute_common_edge_kernel(
    int num_pairs,
    const SingularPair* pairs,
    const Mesh::QuadFace* faces,
    const double3* nodes_current,
    double* out_matrices // Flaches Array der Größe num_pairs * 16
) {
    //const double M_PI = 3.14159265358979323846;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    SingularPair pair = pairs[idx];

    // Original-Knoten der Quads holen
    double3 nodes_p[4], nodes_q[4];
    for (int i = 0; i < 4; ++i) {
        nodes_p[i] = nodes_current[faces[pair.quad_p_idx].v[i]];
        nodes_q[i] = nodes_current[faces[pair.quad_q_idx].v[i]];
    }

    double local_matrix[4][4];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            local_matrix[r][c] = 0.0;
        }
    }

    // 4D Gauss-Quadratur (256 Punkte)
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                for (int l = 0; l < 4; ++l) {

                    double xi = GL_PT(i);
                    double eta1 = GL_PT(j);
                    double eta2 = GL_PT(k);
                    double eta3 = GL_PT(l);

                    double gauss_weight = GL_WG(i) * GL_WG(j) * GL_WG(k) * GL_WG(l);

                    // 5 Sub-Simplizes für Common Edge
                    for (int case_id = 0; case_id < 5; ++case_id) {

                        double x1, x2, y1, y2, ss_jac;
                        get_common_edge_coords(case_id, xi, eta1, eta2, eta3, x1, x2, y1, y2, ss_jac);

                        // 16.0 = 4.0 (Fläche Quad/Tri_p) * 4.0 (Fläche Quad/Tri_q)
                        double total_weight = gauss_weight * ss_jac * 16.0;

                        double u_ss_p = x1 - x2;
                        double v_ss_p = x2;

                        double u_ss_q = y1 - y2;
                        double v_ss_q = y2;

                        // Rücktransformation (SS -> Original Dreieck -> Quad)
                        double u_Rp, v_Rp, u_Rq, v_Rq;
                        map_ss_to_rooftop(u_ss_p, v_ss_p, pair.tri_p_type, pair.p_nodes, u_Rp, v_Rp);
                        map_ss_to_rooftop(u_ss_q, v_ss_q, pair.tri_q_type, pair.q_nodes, u_Rq, v_Rq);

                        // Basisvektoren auf dem Quad auswerten
                        double detJ_p, detJ_q;
                        double3 Wp[4], Wq[4];
                        evaluate_rooftop_basis(u_Rp, v_Rp, nodes_p, Wp, detJ_p);
                        evaluate_rooftop_basis(u_Rq, v_Rq, nodes_q, Wq, detJ_q);

                        // Vollständig ausprogrammierte physikalische Koordinaten r_p
                        double3 rp = make_double3(
                            0.25 * ((1.0 - u_Rp) * (1.0 - v_Rp) * nodes_p[0].x +
                                (1.0 + u_Rp) * (1.0 - v_Rp) * nodes_p[1].x +
                                (1.0 + u_Rp) * (1.0 + v_Rp) * nodes_p[2].x +
                                (1.0 - u_Rp) * (1.0 + v_Rp) * nodes_p[3].x),

                            0.25 * ((1.0 - u_Rp) * (1.0 - v_Rp) * nodes_p[0].y +
                                (1.0 + u_Rp) * (1.0 - v_Rp) * nodes_p[1].y +
                                (1.0 + u_Rp) * (1.0 + v_Rp) * nodes_p[2].y +
                                (1.0 - u_Rp) * (1.0 + v_Rp) * nodes_p[3].y),

                            0.25 * ((1.0 - u_Rp) * (1.0 - v_Rp) * nodes_p[0].z +
                                (1.0 + u_Rp) * (1.0 - v_Rp) * nodes_p[1].z +
                                (1.0 + u_Rp) * (1.0 + v_Rp) * nodes_p[2].z +
                                (1.0 - u_Rp) * (1.0 + v_Rp) * nodes_p[3].z)
                        );

                        // Vollständig ausprogrammierte physikalische Koordinaten r_q
                        double3 rq = make_double3(
                            0.25 * ((1.0 - u_Rq) * (1.0 - v_Rq) * nodes_q[0].x +
                                (1.0 + u_Rq) * (1.0 - v_Rq) * nodes_q[1].x +
                                (1.0 + u_Rq) * (1.0 + v_Rq) * nodes_q[2].x +
                                (1.0 - u_Rq) * (1.0 + v_Rq) * nodes_q[3].x),

                            0.25 * ((1.0 - u_Rq) * (1.0 - v_Rq) * nodes_q[0].y +
                                (1.0 + u_Rq) * (1.0 - v_Rq) * nodes_q[1].y +
                                (1.0 + u_Rq) * (1.0 + v_Rq) * nodes_q[2].y +
                                (1.0 - u_Rq) * (1.0 + v_Rq) * nodes_q[3].y),

                            0.25 * ((1.0 - u_Rq) * (1.0 - v_Rq) * nodes_q[0].z +
                                (1.0 + u_Rq) * (1.0 - v_Rq) * nodes_q[1].z +
                                (1.0 + u_Rq) * (1.0 + v_Rq) * nodes_q[2].z +
                                (1.0 - u_Rq) * (1.0 + v_Rq) * nodes_q[3].z)
                        );

                        double dist = norm3(sub3(rp, rq));
                        double kernel_val = 1.0 / (4.0 * M_PI * dist);
                        double final_weight = total_weight * detJ_p * detJ_q * kernel_val;

                        // Lokale Quad-Matrix (4x4) füllen
                        for (int r = 0; r < 4; ++r) {
                            for (int c = 0; c < 4; ++c) {
                                local_matrix[r][c] += dot3(Wp[r], Wq[c]) * final_weight;
                            }
                        }
                    }
                }
            }
        }
    }

    // Ergebnis in das globale Array schreiben (wird später in cuSPARSE assembliert)
    int offset = idx * 16;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out_matrices[offset + r * 4 + c] = local_matrix[r][c];
        }
    }
}

    