#ifndef TRANSIENT_SOLVER_CUH
#define TRANSIENT_SOLVER_CUH

#include <cuda_runtime.h>
#include "Mesh.cuh"
#include "DeviceArray.cuh"
#include "CoupledSolver.cuh"
#include "MaterialTypes.cuh"

// Struktur für die resultierenden mechanischen Größen
struct MechanicalLoads {
    double3 force;  // [Newton]
    double3 torque; // [Newton-Meter]
};

class TransientSolver {
public:
    // -------------------------------------------------------------------------
    // Aufgabe 7.1: Impliziter Euler Matrix-Aufbau
    // -------------------------------------------------------------------------
    static void build_implicit_euler_system(
        int num_hexes,
         int total_dofs,
        double dt,
        const double* d_local_K, // Aus Aufgabe 5.1 (144 Einträge pro Hex)
        const double* d_local_C, // Aus Aufgabe 5.1 (144 Einträge pro Hex)
        const double* d_local_F, // Quellterm Integral(W_i . J_s), 12 pro Hex
        const int* d_hex_edges,
        const int* d_hex_edge_signs,
        const double* d_A_prev,  // Vektorpotential des letzten Zeitschritts
        const MaterialProperties* d_materials,
        double* d_local_A_sys,   // Output: Kombinierte Matrix (K + 1/dt C)
        double* d_rhs_vector     // Output: Globaler Vektor der rechten Seite
    );

    // -------------------------------------------------------------------------
    // Aufgabe 7.2: Maxwell'scher Spannungstensor
    // -------------------------------------------------------------------------
    static MechanicalLoads compute_maxwell_loads(
        const Mesh& mesh,
        const double* d_A_current,
        const double3 center_of_mass
    );
};

__global__ void scatter_to_csr_kernel(
    int num_hexes,
    const double* local_A_sys,     // Array der Größe num_hexes * 144
    const int* hex_edges,          // Mapping Hexaeder -> globale Kante
    const int* hex_edge_signs,     // +1 / -1
    const int* csr_row_offsets,    // Startindex jeder Zeile in col_indices
    const int* csr_col_indices,    // Spaltenindizes der Non-Zeros
    double* csr_values             // Ziel-Array für die Werte
);

void assemble_csr_matrix_gpu(
    int num_hexes, const double* d_local_A_sys, const int* d_hex_edges, const int* d_hex_edge_signs,
    const int* d_csr_row_offsets, const int* d_csr_col_indices, double* d_csr_values);

#endif // TRANSIENT_SOLVER_CUH