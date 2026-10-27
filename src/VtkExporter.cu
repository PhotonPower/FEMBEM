#include "VtkExporter.h"
#include "NedelecDispatch.cuh"
#include "Backend.cuh"
#include "NedelecHexahedron.cuh"
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>

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
        double real_Wx = invJ[0][0]*W[i].x + invJ[0][1]*W[i].y + invJ[0][2]*W[i].z;
        double real_Wy = invJ[1][0]*W[i].x + invJ[1][1]*W[i].y + invJ[1][2]*W[i].z;
        double real_Wz = invJ[2][0]*W[i].x + invJ[2][1]*W[i].y + invJ[2][2]*W[i].z;

        // Flussdichte B = rot A (kontravariante 2-Form): (1/detJ) J_std rot W_ref,
        // mit J_std = J^T also SPALTENweise ueber das hier vorliegende J.
        double real_curlWx = (J[0][0]*curlW[i].x + J[1][0]*curlW[i].y + J[2][0]*curlW[i].z) / detJ;
        double real_curlWy = (J[0][1]*curlW[i].x + J[1][1]*curlW[i].y + J[2][1]*curlW[i].z) / detJ;
        double real_curlWz = (J[0][2]*curlW[i].x + J[1][2]*curlW[i].y + J[2][2]*curlW[i].z) / detJ;

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

__global__ void compute_cell_centered_fields_kernel(
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
    int hex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hex_id >= num_hexes) return;
    compute_cell_centered_fields_kernel_body(
        hex_id, num_hexes, dt, hex_nodes, current_nodes, hex_edges, hex_edge_signs,
        elem_types, materials, A_current, A_previous,
        out_A_center, out_B_center, out_J_center);
}

// =================================================================================
// CPU Routine: Host-Controller und VTK ASCII Writer
// =================================================================================
void VtkExporter::export_hexahedral_cell_data(
    const std::string& filename, 
    const Mesh& mesh, 
    const MaterialProperties* d_materials,
    const double* d_A_current, 
    const double* d_A_previous, 
    double dt) 
{
    // 1. Device Arrays für die interpolierten Zentrums-Felder allozieren
    DeviceArray<double3> d_A_center(mesh.num_hexes);
    DeviceArray<double3> d_B_center(mesh.num_hexes);
    DeviceArray<double3> d_J_center(mesh.num_hexes);

    // 2. Felder auswerten -- auf der GPU massiv parallel, im CPU-Modus per
    //    OpenMP ueber dieselbe Rumpffunktion.
    BF_LAUNCH_1D_PAR(compute_cell_centered_fields_kernel, mesh.num_hexes,
        static_cast<int>(mesh.num_hexes), 
        dt, 
        mesh.hex_nodes.data(), 
        mesh.nodes_current.data(),
        mesh.hex_edges.data(), 
        mesh.hex_edge_signs.data(), 
        mesh.elem_types.data(),
        d_materials,
        d_A_current, 
        d_A_previous,
        d_A_center.data(), 
        d_B_center.data(), 
        d_J_center.data()
    );
    bf::sync();

    // 3. Felder und Gitter in den CPU-RAM ziehen
    d_A_center.prefetch_to_host();
    d_B_center.prefetch_to_host();
    d_J_center.prefetch_to_host();
    mesh.nodes_current.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();

    // 4. ASCII VTK Datei schreiben
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("VtkExporter Error: Could not open file " + filename + " for writing.");
    }

    // Wissenschaftliche Notation für hohe Präzision der Text-Ausgabe
    file << std::scientific << std::setprecision(6);

    // --- VTK Header ---
    file << "# vtk DataFile Version 3.0\n";
    file << "FEMBEM 3D Simulation Output\n";
    file << "ASCII\n";
    file << "DATASET UNSTRUCTURED_GRID\n";

    // --- POINTS (Knotenkoordinaten) ---
    file << "POINTS " << mesh.num_nodes << " double\n";
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        file << mesh.nodes_current[i].x << " " 
             << mesh.nodes_current[i].y << " " 
             << mesh.nodes_current[i].z << "\n";
    }

    // --- CELLS (Hexaeder Konnektivität) ---
    // VTK Format: [Anzahl Knoten pro Zelle] [Knoten 0] [Knoten 1] ...
    // Für Hexaeder: 8 n0 n1 n2 n3 n4 n5 n6 n7 (daher 9 Einträge pro Zelle)
    file << "\nCELLS " << mesh.num_hexes << " " << mesh.num_hexes * 9 << "\n";
    for (size_t i = 0; i < mesh.num_hexes; ++i) {
        file << "8 ";
        for (int j = 0; j < 8; ++j) {
            file << mesh.hex_nodes[i].v[j] << (j == 7 ? "" : " ");
        }
        file << "\n";
    }

    // --- CELL_TYPES (Immer Typ 12 für Hexaeder in VTK) ---
    file << "\nCELL_TYPES " << mesh.num_hexes << "\n";
    for (size_t i = 0; i < mesh.num_hexes; ++i) {
        file << "12\n";
    }

    // --- CELL_DATA (Unsere physikalischen Vektorfelder auf den Elementen) ---
    file << "\nCELL_DATA " << mesh.num_hexes << "\n";

    // 1. Magnetische Flussdichte B
    file << "VECTORS Magnetic_Flux_Density_B double\n";
    for (size_t i = 0; i < mesh.num_hexes; ++i) {
        file << d_B_center[i].x << " " << d_B_center[i].y << " " << d_B_center[i].z << "\n";
    }

    // 2. Totale Stromdichte J
    file << "\nVECTORS Total_Current_Density_J double\n";
    for (size_t i = 0; i < mesh.num_hexes; ++i) {
        file << d_J_center[i].x << " " << d_J_center[i].y << " " << d_J_center[i].z << "\n";
    }

    // 3. Magnetisches Vektorpotential A
    file << "\nVECTORS Vector_Potential_A double\n";
    for (size_t i = 0; i < mesh.num_hexes; ++i) {
        file << d_A_center[i].x << " " << d_A_center[i].y << " " << d_A_center[i].z << "\n";
    }

    file.close();
    std::cout << "[VTK Exporter] Successfully wrote simulation data to " << filename << "\n";
}