#pragma once

// =============================================================================
// SolverKernels.cuh - Header für FEM/BEM Solver Kernel & Hilfsfunktionen
// =============================================================================

#include "NedelecHexahedron.cuh"
#include "NedelecTetrahedron.cuh"
#include "NedelecPrism.cuh"
#include "NedelecDispatch.cuh"
#include "MaterialTypes.cuh"
#include "Mesh.cuh"
#include <vector>
#include <set>
#include <iostream>
#include "VtkReader.h"
#include "TopologyExtractor.h"
#include "PhysicalObject.h"
#include "ClusterTree.h"
#include "BemCoupling.h"
#include "TreeCotreeGauge.h"
#include "CohomologyGauge.h"
#include "TransientSolver.cuh"
#include <string>
#include <cstdlib>
#include "DeviceArray.cuh"
#include "BemIntegration.cuh"
#include "AcaPlus.h"
#include "CoupledSolver.cuh"
#include "VtkExporter.h"
#include "SauterSchwab.cuh"
#include "Kinematics.cuh"

// =============================================================================
// CSR-Matrix Struktur für GPU-basierte Assemblierung
// =============================================================================
struct CSRMatrix {
    int num_rows;   ///< Anzahl Zeilen (= globale Kanten-DOFs)
    int nnz;        ///< Anzahl Nicht-Null-Einträge

    DeviceArray<double> values;       ///< Werte [nnz]
    DeviceArray<int>    col_indices;  ///< Spaltenindizes [nnz]
    DeviceArray<int>    row_offsets;  ///< Zeilen-Offsets [num_rows+1]

    __host__ void allocate(int rows, int estimated_nnz);
    __host__ void prefetch_to_device();
    __host__ void prefetch_to_host();
    __host__ void fill_zero();
};

// =============================================================================
// Kernel-Deklarationen
// =============================================================================

/**
 * @brief Berechnet pro Hexaeder die lokalen Steifigkeits- (K) und Dämpfungsmatrizen (C)
 *        für die magnetische Vektorpotential-Formulierung mit Nédélec-Elementen 1. Art.
 */
__global__ void assemble_local_matrices_kernel(
    int num_hexes,
    const Hexahedron* hex_nodes,
    const double3* current_nodes,
    const MaterialProperties* materials,
    const int* hex_edge_signs,
    const int* elem_types,
    double* local_K,
    double* local_C,
    double* local_F
);

// =============================================================================
// CPU-Hilfsfunktionen: CSR-Struktur-Aufbau
// =============================================================================

/**
 * @brief Erstellt das Sparsity-Pattern der globalen Systemmatrix im CSR-Format.
 */
void build_csr_structure(const Mesh& mesh, CSRMatrix& csr);

// =============================================================================
// CPU Evaluator für ACA-Algorithmus (BEM Fernfeld-Kopplung)
// =============================================================================

/**
 * @brief Berechnet einen einzelnen Matrixeintrag der BEM-Kopplungsmatrix
 *        für zwei Randkanten (Regular Part, keine Singularität).
 */
__host__ double cpu_evaluate_bem_regular(
    int row_bem_edge, 
    int col_bem_edge, 
    const Mesh& mesh
);

// =============================================================================
// Inline-Helfer aus NedelecHexahedron.cuh
// =============================================================================
//
// Hier standen frueher Vorwaertsdeklarationen von compute_jacobian,
// evaluate_nedelec_basis und evaluate_nedelec_curl mit dem Qualifier
// "__device__ __forceinline__". Die sind entfallen:
//
//   1. Sie waren ohnehin redundant -- NedelecHexahedron.cuh wird oben
//      eingebunden und definiert alle drei inline.
//   2. Die Definitionen sind fuer den CPU-Modus auf "__host__ __device__"
//      erweitert worden (derselbe Integrand muss auf beiden Pfaden
//      auswertbar sein). Eine Deklaration mit abweichendem Ausfuehrungsraum
//      waere damit ein Uebersetzungsfehler.

// =============================================================================
// Konstanten
// =============================================================================
constexpr double SOLVER_MU_0 = 4.0 * 3.14159265358979323846 * 1e-7;
constexpr double SOLVER_GAUSS_POINT = 0.5773502691896257; // 1/sqrt(3)
constexpr int SOLVER_HEX_EDGES = 12;
constexpr int SOLVER_HEX_NODES = 8;
constexpr int SOLVER_LOCAL_MATRIX_SIZE = 144; // 12x12

// =============================================================================
// CUDA-Fehlermakro (falls nicht bereits definiert)
// =============================================================================
#ifndef CUDA_CHECK_ERROR
#define CUDA_CHECK_ERROR(cmd) do {                                             \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA Error %s at %s:%d\n",                            \
                cudaGetErrorString(e), __FILE__, __LINE__);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while(0)
#endif

// argc/argv werden nur fuer die optionale Begrenzung der Schrittzahl genutzt.
int mainloop(int argc, char** argv);