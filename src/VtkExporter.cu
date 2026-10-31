#include "VtkExporter.h"
#include "NedelecDispatch.cuh"
#include "Backend.cuh"
#include "NedelecHexahedron.cuh"
#include "CellCenteredFields.cuh"
#include <fstream>
#include <vector>
#include <iostream>
#include <iomanip>

// Der Rumpf compute_cell_centered_fields_kernel_body (Interpolation von A,
// B und J ins Elementzentrum) steht in CellCenteredFields.cuh, damit
// tests/test_piola_production ihn direkt aufrufen kann.

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