#ifndef CELL_CENTERED_FIELDS_CUH
#define CELL_CENTERED_FIELDS_CUH

// =============================================================================
// Zellmittelwerte A, B, J fuer die VTK-Ausgabe
// =============================================================================
//
// Rumpf von compute_cell_centered_fields_kernel (VtkExporter.cu), unveraendert
// hierher verschoben, damit tests/test_piola_production den PRODUKTIVEN
// Ausgabekoerper auf einem gedrehten und gescherten Element gegen ein
// homogenes B pruefen kann, ohne eine VTK-Datei zu schreiben und
// zurueckzulesen. Kernel und Aufrufer bleiben in VtkExporter.cu.
// =============================================================================

#include "Backend.cuh"
#include "MaterialTypes.cuh"
#include "Mesh.cuh"
#include "NedelecDispatch.cuh"

// =================================================================================
// GPU Kernel: Interpoliert A, B und J vom Kanten- ins physikalische Elementzentrum
// =================================================================================
// Rumpf gemeinsam fuer GPU- und CPU-Pfad. Rennfrei: jeder Hexaeder schreibt
// ausschliesslich out_*_center[hex_id], daher die parallele CPU-Variante.
BF_HD inline void compute_cell_centered_fields_kernel_body(
    int hex_id,
    int num_hexes,
    double dt,
    const Hexahedron* hex_nodes,
    const double3* current_nodes,
    const int* hex_edges,
    const int* hex_edge_signs,
    const int* elem_types,               // null = alles Hexaeder
    const MaterialProperties* materials,
    const double* A_current,
    const double* A_previous,
    double3* out_A_center,
    double3* out_B_center,
    double3* out_J_center
) {
    (void)num_hexes;

    // Elementtyp: ohne ihn wuerde ein Tetraeder mit den Hexaeder-Basis-
    // funktionen ausgewertet. Das gibt keinen Absturz, sondern ein falsches
    // B-Feld -- und zwar eines, das plausibel aussieht.
    const int etype   = element_type_or_hex(elem_types, hex_id);
    const int n_nodes = element_num_nodes(etype);

    Hexahedron hex = hex_nodes[hex_id];
    double3 nodes[8];
    for (int i = 0; i < n_nodes; ++i) nodes[i] = current_nodes[hex.v[i]];
    for (int i = n_nodes; i < 8; ++i) nodes[i] = make_double3(0.0, 0.0, 0.0);

    // Hole lokale Vektorpotentiale (Aktuell und Letzter Zeitschritt) mit Vorzeichenkorrektur
    double A_loc_curr[12];
    double A_loc_prev[12];
    for (int i = 0; i < 12; ++i) {
        int g_edge = hex_edges[hex_id * 12 + i];
        int sign = hex_edge_signs[hex_id * 12 + i];
        if (g_edge < 0) { A_loc_curr[i] = 0.0; A_loc_prev[i] = 0.0; continue; }
        A_loc_curr[i] = A_current[g_edge] * sign;
        A_loc_prev[i] = A_previous[g_edge] * sign;
    }

    // Mittelpunkt des Referenzelements: Hexaeder (0,0,0), Tetraeder
    // (1/4,1/4,1/4), Prisma (1/3,1/3,0).
    double xi, eta, zeta;
    element_reference_center(etype, xi, eta, zeta);

    double J[3][3], invJ[3][3], detJ;
    compute_element_jacobian(etype, nodes, xi, eta, zeta, J, invJ, detJ);

    double3 W[12], curlW[12];
    evaluate_element_basis(etype, xi, eta, zeta, W);
    evaluate_element_curl(etype, xi, eta, zeta, curlW);

    double3 A_vec = make_double3(0.0, 0.0, 0.0);
    double3 B_vec = make_double3(0.0, 0.0, 0.0);
    double3 dA_dt_vec = make_double3(0.0, 0.0, 0.0);

    const int n_edges = element_num_edges(etype);
    for (int i = 0; i < n_edges; ++i) {
        // Piola-Transformation fuer das Vektorpotential A (kovariante 1-Form).
        // Index-Reihenfolge wie in fem.cu korrigiert: compute_jacobian() baut J
        // transponiert auf (J[i][j] = d x_j / d xi_i), also ist invJ = J_std^-T
        // und W_phys = J_std^-T W_ref = invJ * W_ref -- ZEILENweise.
        const double3 real_W = piola_covariant(invJ, W[i]);
        const double real_Wx = real_W.x, real_Wy = real_W.y, real_Wz = real_W.z;

        // Flussdichte B = rot A (kontravariante 2-Form): (1/detJ) J_std rot W_ref,
        // mit J_std = J^T also SPALTENweise ueber das hier vorliegende J.
        const double3 real_curlW = piola_contravariant(J, detJ, curlW[i]);
        const double real_curlWx = real_curlW.x, real_curlWy = real_curlW.y,
                     real_curlWz = real_curlW.z;

        // Aufsummieren über die 12 Basisvektoren
        A_vec.x += A_loc_curr[i] * real_Wx;
        A_vec.y += A_loc_curr[i] * real_Wy;
        A_vec.z += A_loc_curr[i] * real_Wz;

        B_vec.x += A_loc_curr[i] * real_curlWx;
        B_vec.y += A_loc_curr[i] * real_curlWy;
        B_vec.z += A_loc_curr[i] * real_curlWz;

        // Zeitliche Ableitung dA/dt (Impliziter Euler)
        double dA_dt = (A_loc_curr[i] - A_loc_prev[i]) / dt;
        dA_dt_vec.x += dA_dt * real_Wx;
        dA_dt_vec.y += dA_dt * real_Wy;
        dA_dt_vec.z += dA_dt * real_Wz;
    }

    out_A_center[hex_id] = A_vec;
    out_B_center[hex_id] = B_vec;
    
    // Totale Stromdichte = Wirbelstrom (-kappa * dA/dt) + Eingeprägter Spulenstrom
    double kappa = materials[hex_id].kappa;
    out_J_center[hex_id] = make_double3(
        -kappa * dA_dt_vec.x + materials[hex_id].current_density_x,
        -kappa * dA_dt_vec.y + materials[hex_id].current_density_y,
        -kappa * dA_dt_vec.z + materials[hex_id].current_density_z
    );
}

#endif // CELL_CENTERED_FIELDS_CUH
