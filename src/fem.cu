#include"fem.cuh"
#include "Backend.cuh"
#include "ReferenceSolver.cuh"
#include "MeshRefinement.cuh"
#include "SingularQuadPair.cuh"
#include "FieldVerification.cuh"
#include "DenseLinalg.cuh"
#include "BlockPreconditionerStudy.h"
#include "HarmonicSolver.cuh"
#include "HarmonicLorentz.cuh"
#include "HarmonicMaxwell.cuh"
// FEM-direkt-Pfad (--fem-direct), siehe docs/design/fem-direct-plan.md
#include "FemDirectBC.h"
#include "FemDirectRun.h"
#include "HarmonicBlockSystem.h"
#include "SparseDirectSolver.h"

#include <algorithm>
#include <chrono>
#include <cmath>   // std::isfinite in der [RHS]-Diagnose; bisher nur transitiv da
#include <map>
#include <random>
#include <utility>
#include <iomanip>

// Rumpf gemeinsam fuer GPU- und CPU-Pfad (siehe include/Backend.cuh). Jeder
// Hexaeder schreibt ausschliesslich in seinen eigenen 144er- bzw.
// 12er-Abschnitt von local_K/local_C/local_F -- rennfrei, daher darf der
// CPU-Pfad parallelisieren.
BF_HD inline void assemble_local_matrices_kernel_body(
    int hex_id,
    int num_hexes,
    const Hexahedron* hex_nodes,
    const double3* current_nodes,
    const MaterialProperties* materials, // Material pro Hexaeder (mu, kappa)
    const int* hex_edge_signs,           // Array aus Aufgabe 2.2 (+1 oder -1)
    const int* elem_types,               // ELEM_TET/HEX/PRISM, null = alles Hexaeder
    double* local_K,                     // Output: Array der Größe num_hexes * 144
    double* local_C,                     // Output: Array der Größe num_hexes * 144
    double* local_F                      // Output: Array der Größe num_hexes * 12
) {
    (void)num_hexes;
    // hex_edge_signs wird hier absichtlich nicht gelesen, siehe Kommentar
    // weiter unten. Explizit als ungenutzt markiert, damit -Wextra nicht warnt.
    (void)hex_edge_signs;

    // 1. Hole Geometrie und Material
    const int etype   = element_type_or_hex(elem_types, hex_id);
    const int n_nodes = element_num_nodes(etype);
    const int n_edges = element_num_edges(etype);

    Hexahedron hex = hex_nodes[hex_id];
    double3 nodes[8];
    for (int i = 0; i < n_nodes; ++i) nodes[i] = current_nodes[hex.v[i]];
    for (int i = n_nodes; i < 8; ++i) nodes[i] = make_double3(0.0, 0.0, 0.0);

    const MaterialProperties& mat = materials[hex_id];
    double nu = 1.0 / (mat.mu_relative * 4.0 * M_PI * 1e-7); // Reluktivität
    double kappa = mat.kappa; // Leitfähigkeit

    // Kanten-Vorzeichen werden hier absichtlich nicht gelesen: die lokalen
    // Elementmatrizen bleiben unorientiert, die Orientierung wendet erst
    // scatter_to_csr_kernel als sign_r*sign_c an. Der Parameter
    // hex_edge_signs bleibt in der Signatur, weil derselbe Launcher ihn an
    // weitere Kernel durchreicht.

    // Lokale Matrizen initialisieren
    double K_e[12][12] = { 0.0 };
    double C_e[12][12] = { 0.0 };

    // Lokaler Quellterm F_i = Integral W_i . J_s dV. Wie K und C unorientiert;
    // die Kantenorientierung wendet erst der RHS-Scatter im
    // implicit_euler_kernel an.
    double F_e[12] = { 0.0 };

    // 2. Quadratur. Die Regel haengt am Elementtyp:
    //    Hexaeder 2x2x2 Gauss (8 Punkte), Prisma 3x2 (6), Tetraeder 4.
    //    In allen drei Faellen ist die Regel fuer W . W exakt.
    const int n_quad = element_num_quad_points(etype);

    for (int g = 0; g < n_quad; ++g) {
        double xi, eta, zeta, quad_w;
        element_quadrature_point(etype, g, xi, eta, zeta, quad_w);

        double J[3][3], invJ[3][3], detJ;
        compute_element_jacobian(etype, nodes, xi, eta, zeta, J, invJ, detJ);

        double3 W[12], curlW[12];
        evaluate_element_basis(etype, xi, eta, zeta, W);
        evaluate_element_curl(etype, xi, eta, zeta, curlW);

        double weight = quad_w * detJ;

        // Physikalische Lage des Gausspunkts, gebraucht fuer die azimutale
        // Stromrichtung einer Ringspule.
        double3 pos = make_double3(0.0, 0.0, 0.0);
        {
            double N[8];
            element_shape_functions(etype, xi, eta, zeta, N);
            for (int i = 0; i < n_nodes; ++i) {
                pos.x += N[i] * nodes[i].x;
                pos.y += N[i] * nodes[i].y;
                pos.z += N[i] * nodes[i].z;
            }
        }

        // Piola-Transformation: Die Vektoren müssen vom Referenz- ins reale Element transformiert werden!
        double3 real_W[12], real_curlW[12];
        for (int i = 0; i < n_edges; ++i) {
            // W_real = (invJ)^T * W_ref  (Kovariante Transformation für Kantenelemente)
            // KORREKTUR der Piola-Transformation.
            //
            // compute_jacobian() baut J TRANSPONIERT auf; der Kommentar dort
            // sagt es woertlich: "J = [ dx/dxi  dy/dxi  dz/dxi ]", also
            // J[i][j] = d x_j / d xi_i, mithin J = J_std^T. Damit ist
            // invJ = J_std^-T und nicht J_std^-1.
            //
            // Die kovariante Piola-Transformation fuer H(curl) lautet
            //     W_phys = J_std^-T * W_ref = invJ * W_ref
            // also ZEILENweise ueber invJ. Bisher stand hier die spaltenweise
            // Form, die invJ^T * W_ref = J_std^-1 * W_ref ergibt.
            //
            // Nachgewiesen am Hexaeder 1200 der Innenspule: seine lokalen
            // Kanten 0 und 2 laufen azimutal (Endknoten auf r = 41.0 bzw.
            // 47.9 mm), die Kanten 1 und 3 radial. Bei rein azimutalem
            // Quellstrom muss Int W_i . J_s auf 0 und 2 gross und auf 1 und 3
            // nahezu null sein. Der Loeser lieferte genau umgekehrt
            // (-31.7 / +124.9 / +29.2 / -117.3), die Nachrechnung mit
            // invJ * W_ref das Erwartete (-118.7 / +4.1 / +118.7 / +4.1) --
            // und die spaltenweise Form reproduziert den Loeserwert auf
            // 0.0000 relativ. Damit ist die Ursache eindeutig belegt.
            real_W[i].x = invJ[0][0] * W[i].x + invJ[0][1] * W[i].y + invJ[0][2] * W[i].z;
            real_W[i].y = invJ[1][0] * W[i].x + invJ[1][1] * W[i].y + invJ[1][2] * W[i].z;
            real_W[i].z = invJ[2][0] * W[i].x + invJ[2][1] * W[i].y + invJ[2][2] * W[i].z;

            // curlW_real = (1/detJ) * J * curlW_ref (Kontravariante Transformation für Flächenflüsse)
            // Dieselbe Ursache, dieselbe Korrektur. Die kontravariante
            // Piola-Transformation fuer H(div) lautet
            //     rot W_phys = (1/detJ) * J_std * rot W_ref
            // und mit J_std = J^T ist das die SPALTENweise Form ueber das hier
            // vorliegende J. Bisher stand die zeilenweise da.
            real_curlW[i].x = (J[0][0] * curlW[i].x + J[1][0] * curlW[i].y + J[2][0] * curlW[i].z) / detJ;
            real_curlW[i].y = (J[0][1] * curlW[i].x + J[1][1] * curlW[i].y + J[2][1] * curlW[i].z) / detJ;
            real_curlW[i].z = (J[0][2] * curlW[i].x + J[1][2] * curlW[i].y + J[2][2] * curlW[i].z) / detJ;
            // Vorzeichen-Korrektur bewusst NICHT hier: die lokalen Element-
            // matrizen bleiben unorientiert. Die Orientierung wird ausschliesslich
            // beim Gather/Scatter angewendet (scatter_to_csr_kernel: sign_r*sign_c,
            // implicit_euler_kernel: Gather mit s_j, Scatter mit s_i).
        }

        // ---------------------------------------------------------------------
        // Quellterm der eingepraegten Stromdichte: F_i += (W_i . J_s) * detJ
        // ---------------------------------------------------------------------
        // Dieses Integral fehlte vollstaendig. Der RHS bestand nur aus
        // C/dt * A_prev, war mit A_prev = 0 also identisch null -- der BiCGStab
        // meldete "Converged at iteration 0", A blieb null und die
        // Maxwell-Kraft entsprechend (0,0,0) N in jedem Zeitschritt.
        // Nur stromdurchflossene Spulen tragen einen eingepraegten Strom. Der
        // Test macht den Kernel unabhaengig davon, ob andere Regionen ihre
        // Stromdichtefelder initialisieren -- DeviceArray legt seinen Speicher
        // per cudaMallocManaged ohne Konstruktoraufruf an, die Vorbelegungen in
        // MaterialProperties greifen dort also nicht automatisch.
        if (mat.type == RegionType::COIL_CURRENT || mat.type == RegionType::COIL_VOLTAGE) {
            double3 Js;
            if (mat.current_is_azimuthal) {
                // Ringspule: Betrag in current_density_x, Richtung phi_hat um die
                // z-Achse durch current_axis_origin. Nur so ist J_s im
                // Spulengebiet divergenzfrei, was die A-Formulierung voraussetzt.
                double dx = pos.x - mat.current_axis_origin.x;
                double dy = pos.y - mat.current_axis_origin.y;
                double r  = sqrt(dx * dx + dy * dy);
                if (r < 1e-12) {
                    Js = make_double3(0.0, 0.0, 0.0);   // auf der Achse undefiniert
                } else {
                    double mag = mat.current_density_x;
                    Js = make_double3(-mag * dy / r, mag * dx / r, 0.0);
                }
            } else {
                Js = make_double3(mat.current_density_x,
                                  mat.current_density_y,
                                  mat.current_density_z);
            }
            for (int i = 0; i < n_edges; ++i) {
                F_e[i] += (real_W[i].x * Js.x + real_W[i].y * Js.y + real_W[i].z * Js.z) * weight;
            }
        }

        // 3. Matrix-Einträge aufsummieren (K_ij = Int nu * rot W_i * rot W_j)
        for (int i = 0; i < n_edges; ++i) {
            for (int j = 0; j < n_edges; ++j) {
                double dot_curl = real_curlW[i].x * real_curlW[j].x + real_curlW[i].y * real_curlW[j].y + real_curlW[i].z * real_curlW[j].z;
                double dot_W = real_W[i].x * real_W[j].x + real_W[i].y * real_W[j].y + real_W[i].z * real_W[j].z;

                K_e[i][j] += nu * dot_curl * weight;
                C_e[i][j] += kappa * dot_W * weight;
            }
        }
    }

    // 4. In globalen Speicher schreiben (Flaches Array der Länge N_hex * 144)
    int offset = hex_id * 144;
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            local_K[offset + i * 12 + j] = K_e[i][j];
            local_C[offset + i * 12 + j] = C_e[i][j];
        }
    }

    // Quellterm, 12 Eintraege pro Hexaeder
    for (int i = 0; i < 12; ++i) {
        local_F[hex_id * 12 + i] = F_e[i];
    }
}

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
) {
    int hex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hex_id >= num_hexes) return;
    assemble_local_matrices_kernel_body(hex_id, num_hexes, hex_nodes, current_nodes,
                                        materials, hex_edge_signs, elem_types,
                                        local_K, local_C, local_F);
}


// -----------------------------------------------------------------------------
// CPU: Einmaliger Aufbau der CSR-Struktur (Zero-Pattern)
// -----------------------------------------------------------------------------
void build_csr_structure(const Mesh& mesh, CSRMatrix& csr) {
    csr.num_rows = mesh.num_edges; // Die Unbekannten sind die globalen Kanten-Zirkulationen

    // Für jede Kante speichern wir, mit welchen anderen Kanten sie in einem Hexaeder verbunden ist.
    // std::set sorgt automatisch für Sortierung und Eindeutigkeit.
    std::vector<std::set<int>> adjacency(mesh.num_edges);

    mesh.hex_edges.prefetch_to_host();

    for (size_t hex_id = 0; hex_id < mesh.num_hexes; ++hex_id) {
        // Hole die globalen Kanten-IDs dieses Elements. Die Schrittweite ist
        // weiterhin 12; ein Tetraeder belegt davon 6 Plaetze, ein Prisma 9,
        // der Rest steht auf -1 und wird uebersprungen.
        const int n_edges = element_num_edges(mesh.elem_type(hex_id));
        int local_global_map[12];
        for (int i = 0; i < n_edges; ++i) {
            local_global_map[i] = mesh.hex_edges[hex_id * 12 + i];
        }

        // Jede Kante ist an alle anderen des Elements (und sich selbst) gekoppelt
        for (int i = 0; i < n_edges; ++i) {
            int row_edge = local_global_map[i];
            if (row_edge < 0) continue;
            for (int j = 0; j < n_edges; ++j) {
                int col_edge = local_global_map[j];
                if (col_edge < 0) continue;
                adjacency[row_edge].insert(col_edge);
            }
        }
    }

    // Zählen der Non-Zeros und Befüllen von row_offsets
    csr.row_offsets.allocate(csr.num_rows + 1);
    csr.row_offsets.prefetch_to_host();

    int current_nnz = 0;
    for (int r = 0; r < csr.num_rows; ++r) {
        csr.row_offsets[r] = current_nnz;
        current_nnz += adjacency[r].size();
    }
    csr.row_offsets[csr.num_rows] = current_nnz; // Letzter Eintrag (notwendig für CSR)
    csr.nnz = current_nnz;

    // Befüllen von col_indices
    csr.col_indices.allocate(csr.nnz);
    csr.values.allocate(csr.nnz);
    csr.col_indices.prefetch_to_host();

    int val_idx = 0;
    for (int r = 0; r < csr.num_rows; ++r) {
        for (int col : adjacency[r]) {
            csr.col_indices[val_idx++] = col;
        }
    }

    // Speicher asynchron zur GPU schieben, bereit für die Assemblierung
    csr.row_offsets.prefetch_to_device();
    csr.col_indices.prefetch_to_device();
    csr.values.prefetch_to_device();
}


// =================================================================================
// CPU Evaluator für den ACA-Algorithmus (Fernfeld BEM)
// =================================================================================
double cpu_evaluate_bem_regular(int row_bem_edge, int col_bem_edge, const Mesh& mesh) {
    // Ermittle die beteiligten Quads für diese Kanten.
    // In einer hochoptimierten Version würde man dies vorab cachen. 
    // Für die Vollständigkeit iterieren wir hier schnell über die Randflächen.
    int quad_p = -1, quad_q = -1;

    for (size_t i = 0; i < mesh.num_boundary_faces; ++i) {
        // Ein Quad hat 4 Kanten (Index i*4 bis i*4+3)
        for (int e = 0; e < 4; ++e) {
            // Cast, weil i ein size_t und row/col_bem_edge ein int ist (-Wsign-compare)
            if (static_cast<int>(i * 4 + e) == row_bem_edge) quad_p = i;
            if (static_cast<int>(i * 4 + e) == col_bem_edge) quad_q = i;
        }
        if (quad_p != -1 && quad_q != -1) break;
    }

    if (quad_p == -1 || quad_q == -1) return 0.0;

    double3 nodes_p[4], nodes_q[4];
    for (int i = 0; i < 4; ++i) {
        nodes_p[i] = mesh.nodes_current[mesh.boundary_faces[quad_p].v[i]];
        nodes_q[i] = mesh.nodes_current[mesh.boundary_faces[quad_q].v[i]];
    }

    // 4-Punkt Gauss-Quadratur auf CPU
    double matrix_block[4][4] = { 0.0 };

    // (Aufruf des Device-Codes auf der CPU erfordert ggf. eine __host__ __device__ Deklaration 
    // in BemIntegration.cuh. Wir gehen davon aus, dass integrate_regular_quad_pair 
    // als __host__ __device__ markiert wurde).
    integrate_regular_quad_pair(nodes_p, nodes_q, matrix_block);

    // Finde den lokalen Kantenindex (0-3) im Quad
    int local_r = row_bem_edge % 4;
    int local_c = col_bem_edge % 4;

    return matrix_block[local_r][local_c];
}

// =================================================================================
// GPU Kernel: Baut einen dichten BEM-Block für das Nahfeld auf
// =================================================================================
// Rumpf gemeinsam fuer GPU- und CPU-Pfad. Jedes (p,q)-Paar beschreibt einen
// eigenen 4x4-Ausschnitt der Blockmatrix -- rennfrei, der CPU-Pfad darf
// parallelisieren. Dieser Kernel ist im CPU-Modus mit Abstand der teuerste
// Posten (Sauter-Schwab-Selbstpatch = 4^4 Gausspunkte je Quad-Paar), OpenMP
// lohnt hier also am meisten.
BF_HD inline void compute_dense_block_kernel_body(
    int idx_p, int idx_q,
    int num_p, const int* quads_p,  // Quads des Beobachtungs-Clusters (Zeilen)
    int num_q, const int* quads_q,  // Quads des Quell-Clusters (Spalten)
    const Mesh::QuadFace* faces,
    const double3* nodes_current,
    double* d_dense_matrix          // Output Matrix (m x n) in Column-Major
) {
    (void)num_q;

    int global_p = quads_p[idx_p];
    int global_q = quads_q[idx_q];

    Mesh::QuadFace face_p = faces[global_p];
    Mesh::QuadFace face_q = faces[global_q];

    // Koordinaten der Quads laden
    double3 n_p[4], n_q[4];
    for (int i = 0; i < 4; ++i) {
        n_p[i] = nodes_current[face_p.v[i]];
        n_q[i] = nodes_current[face_q.v[i]];
    }


    // Globale Knotennummern des Paares. Sauter-Schwab braucht sie, um die
    // geteilten Ecken zu erkennen; ein Koordinatenvergleich waere von Rundung
    // abhaengig und bei zusammenfallenden, aber topologisch getrennten Knoten
    // schlicht falsch.
    int gid_p[4], gid_q[4];
    for (int i = 0; i < 4; ++i) { gid_p[i] = face_p.v[i]; gid_q[i] = face_q.v[i]; }

    // Topologische Prüfung (Wie viele Knoten teilen sich die Quads?)
    int shared = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (face_p.v[i] == face_q.v[j]) shared++;
        }
    }

    double block[4][4];

    // Rigorose Verzweigung basierend auf der Singularität
    if (shared == 4) {
        // Exakt identisches Quad -> Sauter-Schwab Coincident Transformation
        integrate_singular_self_patch(n_p, block);
    }
    else if (shared == 1 || shared == 2) {
        // Benachbarte Paare: Sauter-Schwab. Die Duffy-artige Transformation
        // hebt die Singularitaet auf, statt sie wie die vorherige Unterteilung
        // nur zu ueberdecken. Trifft die Referenz auf 0.02 %.
        integrate_singular_quad_pair(n_p, gid_p, n_q, gid_q, 0, block);
    }
    else {
        // Disjunktes Quad-Paar -> glatte 4-Punkt-Gauss-Quadratur
        integrate_regular_quad_pair(n_p, n_q, block);
    }

    // Schreiben in die globale Dense-Matrix im cuBLAS Column-Major Format!
    // Dimension der Matrix: (num_p * 4) Zeilen, (num_q * 4) Spalten
    int m_rows = num_p * 4;

    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int global_row = idx_p * 4 + r;
            int global_col = idx_q * 4 + c;

            // Index = Spalte * Zeilen_Gesamt + Zeile
            d_dense_matrix[global_col * m_rows + global_row] = block[r][c];
        }
    }
}

__global__ void compute_dense_block_kernel(
    int num_p, const int* quads_p,
    int num_q, const int* quads_q,
    const Mesh::QuadFace* faces,
    const double3* nodes_current,
    double* d_dense_matrix
) {
    // 2D-Grid: Thread x bearbeitet Quad p, Thread y bearbeitet Quad q
    int idx_p = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_q = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_p >= num_p || idx_q >= num_q) return;
    compute_dense_block_kernel_body(idx_p, idx_q, num_p, quads_p, num_q, quads_q,
                                    faces, nodes_current, d_dense_matrix);
}

// =================================================================================
// GPU Kernel fuer den Double-Layer-Anteil (H_BEM) eines dichten Nahfeld-
// Blocks. Noch NICHT in die Hauptschleife (mainloop) verdrahtet -- das ist
// Schritt 3 (2x2-Operator-Umbau). Diese Funktion ist eigenstaendig testbar.
//
// Fallunterscheidung (siehe "Behandlung des Double-Layer-Integrals (DLP) mit
// Sauter-Schwab" + Ergaenzung):
//   shared == 4 (exakter Self-Patch): Integral analytisch 0 -- (r-r') steht
//     exakt orthogonal auf n', da r und r' in derselben Quad-Ebene liegen.
//     Wird uebersprungen, matrix_block bleibt 0.
//   sonst (shared == 0, 1, 2 -- getrennt, Common-Vertex, Common-Edge): Der
//     Zaehler (r-r')*n' verschwindet am Beruehrpunkt linear (O(xi)), waehrend
//     der Nenner nur mit O(xi^3) divergiert und die Sauter-Schwab-Jacobi-
//     Determinante ebenfalls mit O(xi^3) skaliert -- der transformierte
//     Integrand ist voll regulaer (geht sogar gegen 0 am Beruehrpunkt). Anders
//     als beim schwach singulaeren G-Integral (dort schon heute per einfacher
//     Quadratur approximiert, siehe compute_dense_block_kernel) ist die
//     reguraere Quadratur hier also nicht nur "hinreichend genau", sondern
//     durch die Kuerzung sogar exakt gerechtfertigt.
// =================================================================================
// Rumpf gemeinsam fuer GPU- und CPU-Pfad, rennfrei wie die G-Fassung oben.
BF_HD inline void compute_dense_block_kernel_H_body(
    int idx_p, int idx_q,
    int num_p, const int* quads_p,
    int num_q, const int* quads_q,
    const Mesh::QuadFace* faces,
    const double3* nodes_current,
    double* d_dense_matrix_H
) {
    (void)num_q;

    int global_p = quads_p[idx_p];
    int global_q = quads_q[idx_q];

    Mesh::QuadFace face_p = faces[global_p];
    Mesh::QuadFace face_q = faces[global_q];

    double3 n_p[4], n_q[4];
    for (int i = 0; i < 4; ++i) {
        n_p[i] = nodes_current[face_p.v[i]];
        n_q[i] = nodes_current[face_q.v[i]];
    }


    // Globale Knotennummern des Paares. Sauter-Schwab braucht sie, um die
    // geteilten Ecken zu erkennen; ein Koordinatenvergleich waere von Rundung
    // abhaengig und bei zusammenfallenden, aber topologisch getrennten Knoten
    // schlicht falsch.
    int gid_p[4], gid_q[4];
    for (int i = 0; i < 4; ++i) { gid_p[i] = face_p.v[i]; gid_q[i] = face_q.v[i]; }

    int shared = 0;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            if (face_p.v[i] == face_q.v[j]) shared++;
        }
    }

    double block[4][4];
    if (shared == 4) {
        // Self-Patch: analytisch 0, siehe Kommentar oben.
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) block[r][c] = 0.0;
    } else if (shared == 1 || shared == 2) {
        // An geometrischen Kanten -- Plattenrand, Spulenecken -- ist der
        // Integrand (r-r').n'/|r-r'|^3 stark singulaer; die glatte
        // 4-Punkt-Regel lag dort um 20 % daneben, die Unterteilung noch um
        // 2.7 %. Sauter-Schwab hebt die Singularitaet ueber die
        // Jacobi-Determinante der Transformation auf.
        integrate_singular_quad_pair(n_p, gid_p, n_q, gid_q, 1, block);
    } else {
        integrate_regular_quad_pair_dlp(n_p, n_q, block);
    }

    int m_rows = num_p * 4;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            int global_row = idx_p * 4 + r;
            int global_col = idx_q * 4 + c;
            d_dense_matrix_H[global_col * m_rows + global_row] = block[r][c];
        }
    }
}

__global__ void compute_dense_block_kernel_H(
    int num_p, const int* quads_p,
    int num_q, const int* quads_q,
    const Mesh::QuadFace* faces,
    const double3* nodes_current,
    double* d_dense_matrix_H
) {
    int idx_p = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_q = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx_p >= num_p || idx_q >= num_q) return;
    compute_dense_block_kernel_H_body(idx_p, idx_q, num_p, quads_p, num_q, quads_q,
                                      faces, nodes_current, d_dense_matrix_H);
}

// apply_T_matrix_kernel wurde nach src/CoupledSolver.cu
// verschoben -- CoupledMVOperator::apply() muss ihn aus derselben
// Translation Unit starten koennen (kein -rdc=true / Separable Compilation
// in diesem Projekt, siehe Kommentar dort).


// =============================================================================
// --verify-field im produktiven Pfad: B_z aus Nedelec-Kantengraden
// =============================================================================
// B = rot A mit A = sum_e a_e W_e. Die Basisfunktionen sind im Referenzelement
// definiert, also braucht es die Piola-Transformation:
//     W_physikalisch    = invJ^T * W_ref            (kovariante 1-Form)
//     rot W_physikalisch = (J * rot W_ref) / detJ   (kontravariante 2-Form)
// Nur die zweite wird hier gebraucht. Identisch zu
// compute_cell_centered_fields_kernel in VtkExporter.cu -- bewusst dieselbe
// Rechnung, damit die Probe nicht versehentlich eine andere Auswertung prueft
// als die spaetere Ausgabe.
BF_HD inline void verify_bz_nedelec_body(
    int hex_id, int num_hexes,
    const Hexahedron* hex_nodes, const double3* nodes,
    const int* hex_edges, const int* hex_edge_signs,
    const int* elem_types,                 // null = alles Hexaeder
    const double* A, double* out_bz)
{
    (void)num_hexes;

    const int etype   = element_type_or_hex(elem_types, hex_id);
    const int n_nodes = element_num_nodes(etype);
    const int n_edges = element_num_edges(etype);

    const Hexahedron hex = hex_nodes[hex_id];
    double3 nd[8];
    for (int i = 0; i < n_nodes; ++i) nd[i] = nodes[hex.v[i]];
    for (int i = n_nodes; i < 8; ++i) nd[i] = make_double3(0.0, 0.0, 0.0);

    // Mittelpunkt des jeweiligen Referenzelements, nicht immer (0,0,0).
    double r0, s0, t0;
    element_reference_center(etype, r0, s0, t0);

    double J[3][3], invJ[3][3], detJ;
    compute_element_jacobian(etype, nd, r0, s0, t0, J, invJ, detJ);
    if (detJ <= 0.0) { out_bz[hex_id] = 0.0; return; }

    double3 curlW[12];
    evaluate_element_curl(etype, r0, s0, t0, curlW);

    double bz = 0.0;
    for (int i = 0; i < n_edges; ++i) {
        const int ge = hex_edges[hex_id * 12 + i];
        if (ge < 0) continue;
        const double a = A[ge]
                       * static_cast<double>(hex_edge_signs[hex_id * 12 + i]);
        // Spaltenweise, weil J transponiert aufgebaut ist -- siehe die
        // ausfuehrliche Begruendung an der Assemblierung.
        const double real_curlWz =
            (J[0][2] * curlW[i].x + J[1][2] * curlW[i].y + J[2][2] * curlW[i].z) / detJ;
        bz += a * real_curlWz;
    }
    out_bz[hex_id] = bz;
}

__global__ void verify_bz_nedelec(
    int num_hexes, const Hexahedron* hex_nodes, const double3* nodes,
    const int* hex_edges, const int* hex_edge_signs,
    const int* elem_types,
    const double* A, double* out_bz)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_hexes) return;
    verify_bz_nedelec_body(h, num_hexes, hex_nodes, nodes, hex_edges, hex_edge_signs,
                           elem_types, A, out_bz);
}


int mainloop(int argc, char** argv) {
    try {
        std::cout << "========================================================\n";
        std::cout << " BEM-FEM Transient Solver - TEAM 28 Levitation Problem  \n";
        std::cout << "========================================================\n\n";

        // Welches Rechenwerk laeuft? Wird in main() per --cpu/--gpu bzw.
        // FEMBEM_BACKEND festgelegt, siehe include/Backend.cuh.
        bf::print_backend_banner();
        std::cout << "\n";

        // ---------------------------------------------------------------------
        // 1. Geometrie und Topologie initialisieren
        // ---------------------------------------------------------------------
        Mesh mesh;
        std::cout << "[Setup] Reading VTK Mesh...\n";
        VtkReader::read_hexahedral_mesh(bf::mesh_file(), mesh);

        // ---------------------------------------------------------------------
        // Einheiten: team28.vtk ist in Millimetern, die Physik in SI
        // ---------------------------------------------------------------------
        // Die Netzkoordinaten spannen +/-95 in x und y und -52 bis 6.8 in z --
        // das sind Millimeter (TEAM 28: Platte 106x106 mm, Spulen bei z = -26 mm).
        // Leitfaehigkeit (3.4e7 S/m), mu_0 (4*pi*1e-7 H/m), Masse (0.142 kg) und
        // dt (Sekunden) sind dagegen SI. Ohne Umrechnung skalieren die Bloecke
        // unterschiedlich: W geht mit 1/L, curl W mit 1/L^2, dV mit L^3, also
        //
        //     K = nu * Integral(curlW . curlW) dV   ->  Faktor 1e-3
        //     C = kappa * Integral(W . W) dV        ->  Faktor 1e+3
        //
        // Ihr Verhaeltnis ist damit um 1e6 verschoben. Gemessen an diesem Netz:
        //
        //                        Millimeter        Meter
        //     max|K|             2.5428e+05     2.5428e+08
        //     max|C|/dt          4.8888e+11     4.8888e+08
        //     (C/dt) / K            1.92e+06       1.92e+00
        //
        // In Metern sind beide Terme gleicher Groessenordnung, wie es fuer dieses
        // Wirbelstromproblem sein muss. In Millimetern erdrueckt der
        // Leitfaehigkeitsterm den Curl-Curl-Term um sechs Groessenordnungen, die
        // Matrix ist entsprechend konditioniert und der ungepraekonditionierte
        // BiCGStab divergierte (Residuum nach 249 Iterationen um Faktor 6.5
        // gewachsen, dann "rho == 0").
        {
            const double MM_TO_M = 1e-3;
            mesh.nodes_initial.prefetch_to_host();
            mesh.nodes_current.prefetch_to_host();
            // lo/hi erst NACH der Skalierung des ersten Knotens setzen. Eine
            // Initialisierung mit dem noch unskalierten nodes_current[0]
            // hinterlaesst dessen Extrema im Ergebnis, weil fmin/fmax sie nie
            // wieder unterbieten -- die Kontrollausgabe log dann Millimeter,
            // obwohl korrekt skaliert wurde.
            double3 lo = make_double3(0.0, 0.0, 0.0), hi = make_double3(0.0, 0.0, 0.0);
            for (size_t i = 0; i < mesh.num_nodes; ++i) {
                mesh.nodes_initial[i].x *= MM_TO_M;
                mesh.nodes_initial[i].y *= MM_TO_M;
                mesh.nodes_initial[i].z *= MM_TO_M;
                mesh.nodes_current[i].x *= MM_TO_M;
                mesh.nodes_current[i].y *= MM_TO_M;
                mesh.nodes_current[i].z *= MM_TO_M;
                const double3 p = mesh.nodes_current[i];
                if (i == 0) { lo = p; hi = p; }
                else {
                    lo.x = fmin(lo.x, p.x); hi.x = fmax(hi.x, p.x);
                    lo.y = fmin(lo.y, p.y); hi.y = fmax(hi.y, p.y);
                    lo.z = fmin(lo.z, p.z); hi.z = fmax(hi.z, p.z);
                }
            }
            mesh.nodes_initial.prefetch_to_device();
            mesh.nodes_current.prefetch_to_device();
            std::cout << "[Units] Netz von mm nach m skaliert. Ausdehnung jetzt x ["
                      << lo.x << ", " << hi.x << "] y [" << lo.y << ", " << hi.y
                      << "] z [" << lo.z << ", " << hi.z << "] m\n";
        }

        // Netzverfeinerung (--refine=N). MUSS hier stehen: vor der
        // Topologieextraktion, damit Kanten, Randflaechen, Materialzuweisung
        // und Kinematik-Caches alle auf dem verfeinerten Netz aufgebaut werden.
        // Nach der Einheitenskalierung, damit die neuen Knoten in Metern liegen.
        if (bf::refine_levels() > 0) {
            refine_mesh(mesh, bf::refine_levels());
        }

        TopologyExtractor::extract_edges(mesh);
        TopologyExtractor::extract_boundary_faces(mesh);

        // ---------------------------------------------------------------------
// 2. Materialien und Physik
// ---------------------------------------------------------------------
        std::cout << "[Setup] Assigning physical materials by Component ID...\n";

        // Objekte instanziieren
        // ---------------------------------------------------------------------
        // Physikalische Parameter nach TEAM Workshop Problem 28, Modell A
        // ---------------------------------------------------------------------
        // Quelle: H. Karl, J. Fetzer, S. Kurz, G. Lehner, W. M. Rucker,
        // "Description of TEAM Workshop Problem 28: An Electrodynamic
        // Levitation Device".
        //
        //   Aluminiumplatte : sigma = 3.40e7 S/m, m = 0.107 kg,
        //                     Durchmesser 130 mm, Dicke 3 mm
        //   innere Spule    : w1 = 960 Windungen
        //   aeussere Spule  : w2 = 576 Windungen
        //   Strom           : i(t) = 20 A * sin(2*pi*50*t), beide Spulen in
        //                     Reihe, aber mit entgegengesetztem Wickelsinn
        //   Anfangslage     : z = 3.8 mm, stationaere Schwebehoehe 11.3 mm
        //
        // Die Querschnitte sind am Netz gemessen (team28.vtk), nicht aus der
        // Zeichnung abgelesen -- die Diskretisierung ist fuer J maßgeblich:
        //
        //   innere Spule   r = 27.000 .. 55.000 mm, z = -52 .. 0 mm
        //                  A = 28.0 mm * 52.0 mm = 1.456005e-03 m^2
        //   aeussere Spule r = 80.000 .. 95.000 mm, z = -52 .. 0 mm
        //                  A = 15.0 mm * 52.0 mm = 7.800052e-04 m^2
        //
        // Das Netz bestaetigt die Spezifikation in jeder pruefbaren Groesse:
        // Plattendurchmesser 130 mm, Dicke 3 mm, Anfangsspalt 3.8 mm,
        // Spulenhoehe 52 mm, Spulenbreiten 28 und 15 mm. Aus dem Plattenvolumen
        // von 3.978e-05 m^3 folgt mit rho(Al) = 2700 kg/m^3 eine Masse von
        // 0.1074 kg, in Uebereinstimmung mit den angegebenen 0.107 kg.
        //
        // Die Stromdichte ergibt sich als J = w * i_max / A. Vorher stand hier
        // die Amplitude 100.0, also offenbar der Strom in Ampere anstelle einer
        // Stromdichte in A/m^2 -- rund fuenf Groessenordnungen zu klein, und da
        // die Kraft quadratisch im Feld ist, rund zehn Groessenordnungen in der
        // Kraft. Das passt zu den beobachteten 1e-8 N gegenueber den fuer 0.107 kg
        // benoetigten etwa 1.05 N.
        const double TEAM28_I_MAX   = 20.0;      // A
        const double TEAM28_F0      = 50.0;      // Hz
        const int    TEAM28_W_INNER = 960;
        const int    TEAM28_W_OUTER = 576;
        const double A_INNER        = 1.456005e-03;   // m^2, am Netz gemessen
        const double A_OUTER        = 7.800052e-04;   // m^2, am Netz gemessen
        const double J_INNER = TEAM28_W_INNER * TEAM28_I_MAX / A_INNER;  // 1.3187e7 A/m^2
        const double J_OUTER = TEAM28_W_OUTER * TEAM28_I_MAX / A_OUTER;  // 1.4769e7 A/m^2

        LinearMetal plate("Aluminium_Plate", 3.40e7, 1.0);
        // Traegheitsmomente einer Vollscheibe: I_zz = m*r^2/2,
        // I_xx = I_yy = m*r^2/4 + m*h^2/12. Vorher standen hier 1.5e-3 und
        // 3.0e-3, also rund das Dreizehnfache.
        plate.set_mechanics(0.107, 1.1306e-4, 1.1306e-4, 2.2609e-4);
        // FIX: ohne diesen Aufruf bleibt is_moving() fuer die Platte fuer immer
        // false (siehe include/PhysicalObject.h::enable_free_motion). Die von
        // apply_newtons_law() integrierte Translation waere dann nie auf
        // mesh.nodes_current angewendet worden -- die Platte haette sich
        // geometrisch nie bewegt, unabhaengig von der berechneten Kraft.
        plate.enable_free_motion();

        // TEAM 28 hat zwei koaxiale Ringspulen um die z-Achse. Der Strom laeuft
        // azimutal; nur so ist J_s im Spulengebiet divergenzfrei, was die
        // A-Formulierung voraussetzt (siehe MaterialTypes.cuh). Der frueher
        // uebergebene konstante Richtungsvektor (0,1,0) war physikalisch falsch.
        // Der dritte Konstruktorparameter bleibt als Rueckfallebene erhalten,
        // set_azimuthal() ueberschreibt ihn.
        // Entgegengesetzter Wickelsinn bei Reihenschaltung -> entgegengesetzte
        // Vorzeichen. Welche Spule das positive traegt, legt die Richtung des
        // Feldes fest; die Kraft auf die Platte muss nach +z zeigen. Kehrt sich
        // das Vorzeichen der z-Kraft um, sind die beiden Amplituden zu tauschen.
        CoilCurrentDriven coil_inner("Inner_Coil",  J_INNER, TEAM28_F0, make_double3(0, 1, 0));
        CoilCurrentDriven coil_outer("Outer_Coil", -J_OUTER, TEAM28_F0, make_double3(0, 1, 0));
        coil_inner.set_azimuthal(make_double3(0.0, 0.0, 0.0));
        coil_outer.set_azimuthal(make_double3(0.0, 0.0, 0.0));

        LinearMetal air("Air_Domain", 0.0, 1.0);

        // Robuste Zuweisung via VTK Component ID
        assign_materials_by_component_id(mesh, &plate, 3); // ID 10 = Platte
        assign_materials_by_component_id(mesh, &coil_inner, 2); // ID 20 = Innere Spule
        assign_materials_by_component_id(mesh, &coil_outer, 1); // ID 30 = Äußere Spule

        // Materialgrenzflaeche der Aluminiumplatte extrahieren (Component
        // ID 3) -- fuer eine korrekte Maxwell-Spannungstensor-Integration.
        // mesh.boundary_faces (oben via extract_boundary_faces) ist die
        // AEUSSERE BEM-Kopplungsflaeche des gesamten Simulationsgebiets
        // (Spulen und Platte gemeinsam), NICHT die Oberflaeche der Platte
        // allein -- damit wurde bisher faelschlich integriert, siehe
        // compute_maxwell_loads() weiter unten.
        TopologyExtractor::extract_component_surface_faces(mesh, 3);


        // Materialien in CUDA-lesbares flaches Array packen
        DeviceArray<MaterialProperties> d_materials(mesh.num_hexes);
        d_materials.prefetch_to_host();

        // ZUERST ALLES AUF LUFT. Ohne diese Zeile behalten alle Elemente, die
        // keiner Komponente zugewiesen sind, den Inhalt der frischen
        // Allokation -- und DeviceArray nullt sie. Ein genulltes
        // MaterialProperties hat mu_relative = 0, NICHT 1: die
        // Vorgabeinitialisierer der Struktur laufen nicht, weil
        // cudaMallocManaged keine Konstruktoren aufruft.
        //
        // Damit wird nu = 1 / (mu0 * mu_r) zu Inf, und die Steifigkeitsmatrix
        // besteht ueberwiegend aus Inf. Genau das ist beim ersten Lauf mit
        // Luftgebiet passiert: 52556 Inf und 3658 NaN von 72677 Eintraegen.
        //
        // Vor dem FEM-direkt-Pfad war das folgenlos, weil jedes Element des
        // alten Netzes zu Platte oder Spule gehoerte. Mit einem mitvernetzten
        // Luftgebiet gilt das nicht mehr.
        {
            const MaterialProperties air_props = air.get_properties();
            for (size_t h = 0; h < mesh.num_hexes; ++h) d_materials[h] = air_props;
        }

        for (int hex_id : plate.get_hex_indices()) d_materials[hex_id] = plate.get_properties();
        for (int hex_id : coil_inner.get_hex_indices()) d_materials[hex_id] = coil_inner.get_properties();
        for (int hex_id : coil_outer.get_hex_indices()) d_materials[hex_id] = coil_outer.get_properties();

        // ---------------------------------------------------------------------
        // Kontrollausgabe der tatsaechlich im Speicher stehenden Materialdaten
        // ---------------------------------------------------------------------
        // Anlass: die Konsistenzpruefung der rechten Seite meldet
        // ||G^T b||/||b|| = 0.69. Eine unabhaengige Nachrechnung des Quellterms
        // aus dem Netz liefert fuer AZIMUTALEN Spulenstrom 2.1e-04, fuer eine
        // konstante KARTESISCHE Richtung dagegen 0.72 -- der Loeser rechnet
        // also offenbar nicht azimutal, obwohl set_azimuthal() aufgerufen wird
        // und die Reihenfolge stimmt (set_azimuthal vor der Zuweisung).
        //
        // Da Quellterm, Basis, Quadratur, Piola-Transformation und
        // Kantenkonvention Zeile fuer Zeile mit der Nachrechnung
        // uebereinstimmen, bleibt nur der Inhalt von d_materials. DeviceArray
        // legt seinen Speicher per cudaMallocManaged OHNE Konstruktoraufruf an;
        // was dort wirklich steht, ist damit nicht aus dem Quelltext ablesbar,
        // sondern nur durch Nachsehen.
        d_materials.prefetch_to_host();
        bf::sync();
        {
            auto dump = [&](const char* name, const std::vector<int>& hexes) {
                if (hexes.empty()) return;
                const MaterialProperties& m = d_materials[hexes.front()];
                std::cout << "[Material] " << name << " (Hexaeder " << hexes.front() << "):"
                          << "  type=" << static_cast<int>(m.type)
                          << "  kappa=" << m.kappa
                          << "  mu_r=" << m.mu_relative
                          << "\n[Material]   current_is_azimuthal="
                          << (m.current_is_azimuthal ? "TRUE" : "FALSE")
                          << "  axis_origin=(" << m.current_axis_origin.x << ", "
                          << m.current_axis_origin.y << ", " << m.current_axis_origin.z << ")"
                          << "  J=(" << m.current_density_x << ", "
                          << m.current_density_y << ", " << m.current_density_z << ")\n";
            };
            dump("Aluminium_Plate", plate.get_hex_indices());
            dump("Inner_Coil     ", coil_inner.get_hex_indices());
            dump("Outer_Coil     ", coil_outer.get_hex_indices());

            bool all_az = true;
            for (int h : coil_inner.get_hex_indices()) if (!d_materials[h].current_is_azimuthal) all_az = false;
            for (int h : coil_outer.get_hex_indices()) if (!d_materials[h].current_is_azimuthal) all_az = false;
            if (!all_az) {
                std::cout << "[Material] WARNUNG: mindestens ein Spulenelement hat "
                             "current_is_azimuthal = FALSE. Eine konstante kartesische "
                             "Stromdichte ist im Ringgebiet NICHT divergenzfrei; die "
                             "rechte Seite liegt dann ausserhalb des Bildraums des "
                             "curl-curl-Operators und ein Krylov-Verfahren stagniert "
                             "auf genau diesem Anteil.\n";
            }
        }
        d_materials.prefetch_to_device();
        d_materials.prefetch_to_device();

        // ---------------------------------------------------------------------
        // 2.5 Cache für bewegte Geometrien aufbauen
        // ---------------------------------------------------------------------
        std::cout << "[Setup] Building kinematic node caches...\n";
        // Auflage aktivieren: die Platte startet auf der Wickelform (3.8 mm
        // ueber der Spulenoberkante) und kann nicht tiefer. Ohne das faellt sie
        // im Modell durch die Form, sobald die Kraft einmal unter das Gewicht
        // faellt. Bezug ist die Startlage, also translation.z >= 0.
        plate.set_lower_stop(0.0);

        plate.build_node_cache(mesh);

        // ---------------------------------------------------------------------
        // --plate-z: Platte auf eine vorgegebene Schwebehoehe setzen
        // ---------------------------------------------------------------------
        //
        // Die Option gab es bisher nur im ReferenceSolver. Der Hauptpfad hat
        // sie ignoriert -- und damit war die Kraft-ueber-Hoehe-Kurve, die der
        // harmonische Loeser liefern soll, gar nicht abfahrbar.
        //
        // Bezug ist die UNTERKANTE der Platte ueber der Spulenoberkante, denn
        // das ist die Groesse, gegen die TEAM 28 vergleicht.
        // props.translation.z waere nur die Verschiebung gegenueber der
        // Netzlage und mit der Aufgabenstellung nicht vergleichbar.
        //
        // VOR dem Aufbau von Randtopologie und ACA-Baum, damit
        // Panelorientierung und Clusterung gleich fuer die endgueltige Lage
        // bestimmt werden.
        if (bf::plate_z_mm() > 0.0) {
            mesh.nodes_initial.prefetch_to_host();
            mesh.hex_nodes.prefetch_to_host();
            bf::sync();
            double plate_z0 = 0.0;
            bool first = true;
            for (int h : plate.get_hex_indices()) {
                const Hexahedron H = mesh.hex_nodes[h];
                const int nn = element_num_nodes(mesh.elem_type(h));
                for (int n = 0; n < nn; ++n) {
                    const double z = mesh.nodes_initial[H.v[n]].z;
                    if (first || z < plate_z0) { plate_z0 = z; first = false; }
                }
            }
            const double target = bf::plate_z_mm() * 1e-3;
            plate.set_translation_z(target - plate_z0);
            apply_kinematics(mesh, plate);
            bf::sync();
            std::cout << "[Setup] --plate-z: Unterkante von " << (plate_z0 * 1000.0)
                      << " mm auf " << bf::plate_z_mm() << " mm gesetzt (Verschiebung "
                      << ((target - plate_z0) * 1000.0) << " mm).\n";
        }

        // (Optional: Wenn sich Spulen theoretisch auch bewegen, hier ebenfalls aufrufen)
        // coil_inner.build_node_cache(mesh);
        
        // ---------------------------------------------------------------------
        // 3. ACA Baum für BEM aufbauen
        // ---------------------------------------------------------------------
        // Ob BEM ueberhaupt in Frage kommt, entscheidet das NETZ: Sauter-Schwab
        // arbeitet auf Vierecks-Panels, ein Tetraeder- oder Prismennetz hat
        // Dreiecke am Rand.
        //
        // Diese Pruefung muss HIER stehen und nicht bei der Auswertung von
        // 'nobem' weiter unten. Der ACA-Baum und die Kopplungsabbildung werden
        // vor der Argumentauswertung gebaut; nobem kommt zu spaet und wuerde
        // eine bereits aufgebaute (und auf einem Dreiecksnetz sinnlose)
        // Kopplung nur nachtraeglich ignorieren. --fem-direct ist dagegen schon
        // bekannt, weil Backend die '--'-Optionen vorab auswertet.
        const bool bem_possible = mesh.is_pure_hex();
        const bool skip_bem_setup = bf::want_fem_direct() || !bem_possible;

        if (skip_bem_setup) {
            std::cout << "[Setup] BEM-Aufbau uebersprungen ("
                      << (bem_possible ? "--fem-direct"
                                       : "Netz enthaelt Nicht-Hexaeder, "
                                         "Sauter-Schwab braucht Vierecks-Panels")
                      << ").\n";
        }

        ClusterTree tree(30);
        if (!skip_bem_setup) {
            std::cout << "[Setup] Building ACA Hierarchical Tree...\n";
            tree.build_tree(mesh);
        }

        // ---------------------------------------------------------------------
        // 3.1 Kopplung BEM-Elementraum <-> globale FEM-Kanten aufbauen
        // ---------------------------------------------------------------------
        // Die Blaetter des Baums definieren eine Permutation der Randquads. In
        // dieser Numerierung ist jedes Blatt ein zusammenhaengender
        // Indexbereich, die Blockoffsets unten also gueltig. Herleitung und
        // Begruendung in include/BemCoupling.h.
        // Bei uebersprungenem Aufbau bleibt die Abbildung leer, num_bem_dofs
        // ist damit 0. Das ist genau der Zustand, den der FEM-direkt-Pfad
        // erwartet -- nd_h = total_dofs + 0.
        const BemCouplingMap bem_map = skip_bem_setup
                                     ? BemCouplingMap{}
                                     : build_bem_coupling_map(mesh, tree);
        if (!skip_bem_setup) {
            std::cout << "[BemCoupling] " << bem_map.num_bem_dofs
                      << " BEM-Freiheitsgrade auf " << mesh.num_edges
                      << " FEM-Kanten abgebildet.\n";
        }

        DeviceArray<int> d_bem_to_fem(static_cast<size_t>(bem_map.num_bem_dofs));
        DeviceArray<int> d_bem_sign(static_cast<size_t>(bem_map.num_bem_dofs));
        d_bem_to_fem.prefetch_to_host();
        d_bem_sign.prefetch_to_host();
        for (int i = 0; i < bem_map.num_bem_dofs; ++i) {
            d_bem_to_fem[i] = bem_map.bem_to_fem[i];
            d_bem_sign[i]   = bem_map.bem_sign[i];
        }
        d_bem_to_fem.prefetch_to_device();
        d_bem_sign.prefetch_to_device();

        // Hierarchische Blockstruktur. Haengt nur an der Baumtopologie und den
        // Zulaessigkeitsradien, nicht an den Knotenkoordinaten der aktuellen
        // Zeitschicht -- deshalb einmalig vor der Zeitschleife.
        // build_block_pairs() startet bei tree.get_root_index(). Auf einem nie
        // gebauten Baum ist das ein Zugriff auf einen Knoten, den es nicht
        // gibt -- ein Absturz ohne Meldung, nicht einmal eine Ausnahme. Bei
        // uebersprungenem BEM-Aufbau bleibt die Liste leer.
        const std::vector<BemBlockPair> block_pairs =
            skip_bem_setup ? std::vector<BemBlockPair>{}
                           : build_block_pairs(tree, 1.0);
        if (!skip_bem_setup) {
            size_t n_aca = 0;
            for (const BemBlockPair& bp : block_pairs) if (bp.admissible) ++n_aca;
            std::cout << "[BemCoupling] " << block_pairs.size() << " Blockpaare ("
                      << n_aca << " ACA, " << (block_pairs.size() - n_aca)
                      << " dense).\n";
        }

        // ---------------------------------------------------------------------
        // 3.5 CPU-Host-Kopien für den asynchronen ACA-Evaluator erstellen
        // Verhindert Page-Faults, wenn GPU (Nahfeld) und CPU (Fernfeld) gleichzeitig arbeiten
        // ---------------------------------------------------------------------
        std::vector<double3> host_nodes_current(mesh.num_nodes);
        std::vector<Mesh::QuadFace> host_boundary_faces(mesh.num_boundary_faces);

        for (size_t i = 0; i < mesh.num_nodes; ++i) {
            host_nodes_current[i] = mesh.nodes_current[i];
        }
        for (size_t i = 0; i < mesh.num_boundary_faces; ++i) {
            host_boundary_faces[i] = mesh.boundary_faces[i];
        }

        // ---------------------------------------------------------------------
        // 4. GPU Speicher für Zeitschleife reservieren
        // ---------------------------------------------------------------------
        int total_dofs = mesh.num_edges;

        DeviceArray<double> d_A_current(total_dofs);
        DeviceArray<double> d_A_previous(total_dofs);
        DeviceArray<double> d_rhs(total_dofs);

        // gekoppelter Loesungsvektor (A + K) fuer das
        // 2x2-BEM-FEM-System. Einmal vor der Zeitschleife angelegt (nicht pro
        // Zeitschritt), damit BiCGStab wie bisher mit dem Feld des
        // Vorschritts warmstartet -- d_A_current wird nach jedem Solve aus
        // den ersten total_dofs Eintraegen synchronisiert (siehe unten).
        DeviceArray<double> d_x_coupled(static_cast<size_t>(total_dofs) + static_cast<size_t>(bem_map.num_bem_dofs));
        d_x_coupled.fill_zero();

        DeviceArray<double> d_local_K(mesh.num_hexes * 144);
        DeviceArray<double> d_local_F(mesh.num_hexes * 12);   // Quellterm pro Hexaeder
        DeviceArray<double> d_local_C(mesh.num_hexes * 144);
        DeviceArray<double> d_local_A_sys(mesh.num_hexes * 144);

        d_A_current.fill_zero();
        d_A_previous.fill_zero();

        // ---------------------------------------------------------------------
        // 4.5 Aufbau der CSR-Struktur (Sparsity Pattern) für cuSPARSE
        // ---------------------------------------------------------------------
        std::cout << "[Setup] Building CSR Sparsity Pattern...\n";
        CSRMatrix csr_matrix;
        build_csr_structure(mesh, csr_matrix);
        std::cout << "[Setup] Matrix has " << csr_matrix.num_rows << " rows and "
            << csr_matrix.nnz << " non-zero elements.\n";

        // ---------------------------------------------------------------------
        // 5. Die Transiente Simulationsschleife
        // ---------------------------------------------------------------------
        // Zeitschrittweite. War fest eincodiert; jetzt ueber --dt=X, weil die
        // Konditionszahl direkt daran haengt (kappa_2 = 1.07e+06 bei einem
        // Materialkontrast sigma/dt zu nu von 4.3e+05).
        double dt = bf::time_step();   // Standard 1e-4, also 0.1 ms

        // Koeffizientenpuffer fuer den AMS-Alpha-Poisson-Operator. AUSSERHALB
        // der Zeitschleife, weil AmsPreconditioner die Zeiger speichert und
        // erst in setup() liest -- ein Puffer mit kuerzerer Lebensdauer waere
        // ein stiller Fehler.
        std::vector<double> ams_nu_per_hex, ams_beta_per_hex;
        double t_end = 0.1; // 100 ms Endzeit (Vgl. Kurz et al. Fig 4)
        int max_steps = static_cast<int>(t_end / dt);

        // Argumentauswertung. Positional sind weiterhin moeglich:
        //   <schritte>   Kurztest mit begrenzter Schrittzahl
        //   nobem        BEM-Kopplung komplett abschalten (reines FEM-System)
        //
        // Fuer --fem-direct ist nobem die Regel, nicht die Ausnahme: ein Netz
        // mit Luftgebiet braucht keine BEM, und ein Tetraedernetz kann sie
        // ohnehin nicht (Sauter-Schwab arbeitet auf Vierecks-Panels).
        //   noilu        ILU(0)-Preconditioner umgehen (Identitaet statt L*U)
        //
        // Alles, was mit '-' beginnt, ist eine Option und wurde bereits in
        // main() von bf::select_backend_from_args() ausgewertet (--cpu, --gpu,
        // --threads=N, --help). Hier wird es uebersprungen. Vorher wurde starr
        // argv[1] als Schrittzahl und argv[2..] als Diagnoseschalter gelesen --
        // damit haette "fembem --cpu 3" die 3 nicht mehr gefunden. Die
        // Reihenfolge ist jetzt beliebig.
        bool disable_bem_for_debug = false;
        bool disable_ilu_for_debug = false;
        bool steps_from_args = false;

        for (int ai = 1; ai < argc; ++ai) {
            const std::string arg = argv[ai];
            if (bf::is_option(argv[ai])) continue;   // gehoert dem Backend-Parser

            if (arg == "nobem") { disable_bem_for_debug = true; continue; }
            if (arg == "noilu") { disable_ilu_for_debug = true; continue; }

            if (!steps_from_args) {
                const int requested = std::atoi(arg.c_str());
                // Frueher stand hier "requested > 0 && requested < max_steps".
                // Damit war die Schrittzahl still bei 1000 gedeckelt (t_end =
                // 100 ms / dt): jeder groessere Wert fiel durch und landete in
                // der "nicht erkannt"-Meldung. Fuer TEAM 28 ist das zu wenig --
                // die Platte schwingt mit rund 297 ms Periode und erreicht ihre
                // Ruhelage von 11.3 mm erst nach etwa 1700 ms, also 17000
                // Schritten. Jede positive Zahl ist jetzt zulaessig.
                if (requested > 0) {
                    max_steps = requested;
                    steps_from_args = true;
                    std::cout << "[Solver] Schrittzahl per Argument auf " << max_steps
                              << " gesetzt (t_end = " << max_steps * dt * 1000.0
                              << " ms).\n";
                    continue;
                }
            }
            std::cout << "[Solver] Argument \"" << arg << "\" nicht erkannt, ignoriert. "
                      << "(\"--help\" zeigt die Liste.)\n";
        }
        // --fem-direct impliziert nobem. Der BEM-Aufbau ist oben schon
        // uebersprungen worden (skip_bem_setup); wuerde hier weiterhin
        // disable_bem_for_debug == false stehen, liefe der Loeser gegen eine
        // leere Kopplungsabbildung, statt sie zu ignorieren. Die beiden Schalter
        // duerfen nicht auseinanderlaufen, also wird es erzwungen und nicht
        // vom Nutzer verlangt.
        if (bf::want_fem_direct() && !disable_bem_for_debug) {
            disable_bem_for_debug = true;
            std::cout << "[Solver] --fem-direct impliziert nobem.\n";
        }
        if (!bem_possible && !disable_bem_for_debug) {
            std::cout << "[Solver] Netz enthaelt Nicht-Hexaeder, BEM ist damit "
                         "ausgeschlossen -- nobem wird erzwungen.\n";
            disable_bem_for_debug = true;
        }

        if (disable_bem_for_debug) {
            std::cout << "[Solver] DIAGNOSE: BEM-Kopplung deaktiviert (nobem) "
                         "-- reines FEM-System zur Fehlerisolierung.\n";
        }
        if (disable_ilu_for_debug) {
            std::cout << "[Solver] DIAGNOSE: ILU(0)-Preconditioner umgangen (noilu) "
                         "-- Identitaets-Vorkonditionierer zur Fehlerisolierung.\n";
        }

        // ---------------------------------------------------------------------
        // Baum-Kobaum-Eichung aufbauen
        // ---------------------------------------------------------------------
        // A_sys = K + C/dt ist dort singulaer, wo kappa = 0 ist: uebrig bleibt der
        // Curl-Curl-Operator, und der annihiliert jeden diskreten Gradienten.
        // Die Eichung fixiert die Gradientenfreiheitsgrade auf einem Spannwald
        // ueber die nichtleitenden Kanten. Details in include/TreeCotreeGauge.h.
        //
        // Die Leitfaehigkeitsverteilung aendert sich ueber die Zeitschritte nicht,
        // die Eichung wird daher einmalig gebaut.
        // Fuer --fem-direct wird NICHT geeicht. Der Curl-Curl-Nullraum wird dort
        // durch eine kleine Leitfaehigkeit in der Luft entfernt
        // (fem_direct::regularize_nonconducting_regions), und das erledigt
        // Gradienten- UND Kohomologieanteil auf einen Schlag. Fuer eine
        // Zerlegung ist die dadurch verschlechterte Kondition gleichgueltig --
        // genau darin liegt der Vorteil des Pfades.
        //
        // Baum-Kobaum und Regularisierung zugleich waeren nicht falsch, aber
        // ueberfluessig; und die Eichung als Projektion im Operator hilft einer
        // Matrix, die cuDSS uebergeben wird, ohnehin nicht.
        d_materials.prefetch_to_host();
        const GaugeInfo gauge = bf::want_fem_direct()
                              ? GaugeInfo{}
                              : build_tree_cotree_gauge(mesh, d_materials.data(), 0.0);
        d_materials.prefetch_to_device();
        if (bf::want_fem_direct()) {
            std::cout << "[Gauge] uebersprungen (--fem-direct regularisiert "
                         "stattdessen).\n";
        } else
        std::cout << "[Gauge] " << gauge.num_gauged << " von " << total_dofs
                  << " Freiheitsgraden geeicht (Nullraumdimension). "
                  << gauge.num_conductor_edges << " leitende Kanten bleiben frei.\n";

        DeviceArray<char> d_is_gauged(static_cast<size_t>(total_dofs));
        d_is_gauged.prefetch_to_host();
        // --nogauge: LEERE Maske statt uebersprungenem setup_gauge(). Ein erster
        // Anlauf hat den Aufruf einfach weggelassen -- dann bricht
        // add_cohomology_projector ab, denn die Projektoren setzen auf der
        // Baum-Koordinatenmaske auf. Mit einer Maske aus lauter false wird
        // k_project_gauge zur Identitaet, die Kohomologie-Projektion bleibt aber
        // vollstaendig erhalten. Genau das ist gewollt: AMS deckt die
        // Gradientenmoden ab, die harmonischen Felder der zwei Volltori (Ringspulen) nicht.
        for (int i = 0; i < total_dofs; ++i)
            d_is_gauged[i] = bf::no_gauge() ? 0 : gauge.is_gauged[i];
        d_is_gauged.prefetch_to_device();

        // ---------------------------------------------------------------------
        // Kohomologie-Generatoren: die Baum-Eichung entfernt nur den
        // Gradientenanteil. Jede Ringspule ist topologisch ein Volltorus
        // (erste Bettizahl b1 = 1), es bleibt also pro Spule ein zusaetzlicher
        // harmonischer Kozykel im Nullraum von K -- unabhaengig davon, wie die
        // rechte Seite spaeter aussieht (stromeingepraegt, spannungsgetrieben,
        // beliebig). Siehe include/CohomologyGauge.h fuer die Herleitung ueber
        // eine diskrete Helmholtz-Hodge-Zerlegung.
        // Bei --fem-direct (und bei --nogauge) uebersprungen: ohne Eichung gibt
        // es keine Baum-Koordinatenmaske, auf der die Projektoren aufsetzen
        // koennten, und die Regularisierung entfernt die harmonischen Moden
        // ohnehin mit.
        const CohomologyGaugeResult cohomology = bf::no_gauge()
                              ? CohomologyGaugeResult{}
                              : find_cohomology_generators(
            mesh, gauge, d_materials.data(), 0.0);
        // Erwartet fuer TEAM 28: im BEM-FEM-Pfad 2 (je Ringspule ein
        // Volltorus-Loch), im FEM-direkt-Pfad mit Luft 0 (leitend ist nur die
        // Platte, und die ist eine Vollscheibe). Ein Loch in der Platte gaebe
        // einen mehr -- und entspraeche nicht dem Benchmark.
        std::cout << "[Gauge] " << cohomology.generators.size()
                  << " Kohomologie-Generatoren gefunden (TEAM 28: BEM-FEM 2, "
                     "einer je Ringspule; FEM-direkt mit Luft 0).\n";

        // Auf den Baum-Koordinaten null setzen (siehe CohomologyGauge.h,
        // "Anwendung als Eichung") -- sonst waere die Gesamtprojektion
        // (Baum gefolgt von Kohomologie) nicht idempotent.
        // Host-seitige reduzierte Vektoren (auf Baum-Koordinaten null), einmal
        // gebaut und sowohl fuer den Device-Upload als auch fuer die
        // RHS-Projektion weiter unten verwendet.
        std::vector<std::vector<double>> host_cohomology_reduced(cohomology.generators.size());
        std::vector<DeviceArray<double>> d_cohomology_vectors(cohomology.generators.size());
        std::vector<double> cohomology_norm_sq(cohomology.generators.size());
        // ---------------------------------------------------------------------
        // Generatoren eichkonform machen -- durch Gradientenaddition, nicht
        // durch Abschneiden
        // ---------------------------------------------------------------------
        // Bisher wurden die Generatoren auf den Baumkanten schlicht auf null
        // gesetzt. Das ist falsch, und die Messung zeigt es deutlich:
        //
        //   ||A c||/||c|| relativ zu einem Zufallsvektor
        //     rohe Generatoren (--nogauge)   3.1e-09  und  1.6e-09
        //     nach dem Abschneiden           4.9e-02  und  7.2e-02
        //
        // Sieben Groessenordnungen. Die ROHEN Generatoren liegen also im
        // Nullraum von A, das Abschneiden zerstoert diese Eigenschaft. Damit
        // war die Projektion mit ihnen nicht legitim -- sie entfernte 58 % der
        // rechten Seite entlang von Richtungen, auf die der Operator wirkt.
        //
        // Ein Vektor wird nicht dadurch eichkonform, dass man ihn auf den
        // Baumkanten abschneidet: dabei geht eine Komponente verloren, die kein
        // Gradient ist. Richtig ist, einen GRADIENTEN zu addieren, denn
        // curl(grad) = 0 laesst die harmonische Eigenschaft unberuehrt. Gesucht
        // ist also phi mit
        //
        //     (c + G phi)_e = 0   fuer jede Baumkante e
        //
        // Auf einem Spannbaum ist das eindeutig und in einem Durchlauf loesbar:
        // Wurzel auf null setzen und die Bedingung entlang der Baumkanten
        // fortpflanzen. Fuer eine Kante (v_min, v_max) ist
        // (G phi)_e = phi_vmax - phi_vmin, woraus sich der Wert am Kindknoten
        // ergibt.
        //
        // Die Baumstruktur wird aus gauge.is_gauged rekonstruiert -- die
        // geeichten Kanten bilden gerade den Spannwald.
        std::vector<double> phi(mesh.num_nodes, 0.0);
        std::vector<char>   visited(mesh.num_nodes, 0);
        std::vector<std::vector<int>> tree_adj(mesh.num_nodes);
        if (!bf::no_gauge()) {
            mesh.edges.prefetch_to_host();
            bf::sync();
            for (int e = 0; e < total_dofs; ++e) {
                if (!gauge.is_gauged[e]) continue;
                const int2 ed = mesh.edges[e];
                tree_adj[ed.x].push_back(e);
                tree_adj[ed.y].push_back(e);
            }
        }

        auto gauge_generator = [&](const std::vector<double>& raw,
                                   std::vector<double>& out, int& conflicts) {
            out = raw;
            conflicts = 0;
            if (bf::no_gauge()) return;

            std::fill(phi.begin(), phi.end(), 0.0);
            std::fill(visited.begin(), visited.end(), 0);
            std::vector<int> stack;

            for (size_t root = 0; root < mesh.num_nodes; ++root) {
                if (visited[root]) continue;
                visited[root] = 1; phi[root] = 0.0;
                stack.assign(1, static_cast<int>(root));
                while (!stack.empty()) {
                    const int n = stack.back(); stack.pop_back();
                    for (int e : tree_adj[n]) {
                        const int2 ed = mesh.edges[e];
                        const int other = (ed.x == n) ? ed.y : ed.x;
                        if (visited[other]) continue;
                        // (G phi)_e = phi[v_max] - phi[v_min], gefordert:
                        // raw[e] + (G phi)_e = 0
                        if (other == ed.y) phi[other] = phi[n] - raw[e];   // n = v_min
                        else               phi[other] = phi[n] + raw[e];   // n = v_max
                        visited[other] = 1;
                        stack.push_back(other);
                    }
                }
            }

            // c + G phi bilden
            for (int e = 0; e < total_dofs; ++e) {
                const int2 ed = mesh.edges[e];
                out[e] = raw[e] + (phi[ed.y] - phi[ed.x]);
            }
            // Kontrolle: auf Baumkanten muss jetzt null stehen. Bleibt ein Rest,
            // enthielten die geeichten Kanten einen Zyklus -- dann ist es kein
            // Spannwald und die Eichung ist nicht exakt erfuellbar.
            for (int e = 0; e < total_dofs; ++e) {
                if (gauge.is_gauged[e] && std::fabs(out[e]) > 1e-12) ++conflicts;
            }
        };

        for (size_t g = 0; g < cohomology.generators.size(); ++g) {
            const auto& gen = cohomology.generators[g];
            std::vector<double>& reduced = host_cohomology_reduced[g];
            int conflicts = 0;
            gauge_generator(gen.vector, reduced, conflicts);
            double norm_sq = 0.0;
            for (int i = 0; i < total_dofs; ++i) norm_sq += reduced[i] * reduced[i];
            if (conflicts > 0) {
                std::cout << "  [Kohomologie] WARNUNG: Generator " << g << " laesst sich auf "
                          << conflicts << " Baumkanten nicht auf null bringen -- die "
                             "geeichten Kanten bilden dort keinen Baum.\n";
            }
            cohomology_norm_sq[g] = norm_sq;

            d_cohomology_vectors[g].allocate(static_cast<size_t>(total_dofs));
            d_cohomology_vectors[g].prefetch_to_host();
            for (int i = 0; i < total_dofs; ++i) d_cohomology_vectors[g][i] = reduced[i];
            d_cohomology_vectors[g].prefetch_to_device();

            std::cout << "  Generator " << g << " (Komponente " << gen.component_id
                      << "): ||c|| = " << std::sqrt(norm_sq)
                      << ", Kozykel-Residuum = " << gen.residual_cocycle << "\n";
        }

        // ---------------------------------------------------------------------
        // Abzweig: Referenzloeser (--reference)
        // ---------------------------------------------------------------------
        // Bewusst HIER, nach der kompletten Setup-Phase: Netz, Topologie,
        // Materialien, Kinematik-Caches und Randflaechen sind identisch mit dem
        // produktiven Pfad. Nur die Diskretisierung und der Loeser
        // unterscheiden sich -- genau das soll verglichen werden. Der
        // produktive Pfad bleibt unberuehrt.
        if (bf::use_reference_solver()) {
            const std::vector<PhysicalObject*> sources = { &coil_inner, &coil_outer };
            return run_reference_solver(mesh, d_materials, plate, sources, max_steps, dt);
        }

        if (bf::verify_field()) {
            // -----------------------------------------------------------------
            // Magnetostatische Probe per GLEICHSTROM, nicht per sigma = 0.
            // -----------------------------------------------------------------
            // Der erste Versuch nullte sigma. Das war falsch: die Eichung eicht
            // dann korrekt ALLE Kanten (7010 = 7013 Knoten - 3 Koerper), aber
            // find_cohomology_generators baut seine Komponenten aus LEITENDEN
            // Kanten. Ohne Leiter findet es die drei Netzkoerper und sucht in
            // jedem ein Loch -- und bricht ab. sigma = 0 macht die Eichung also
            // nicht einfacher, es verletzt die Annahmen der Kohomologie.
            //
            // Stattdessen bleibt sigma unangetastet und der Spulenstrom wird
            // KONSTANT gehalten: t wird in jedem Schritt auf 5 ms gesetzt, wo
            // sin(2*pi*50*0.005) = 1 ist. Nach dem Abklingen der L/R-Transiente
            // ist dA/dt = 0, also J = 0, und das Feld ist exakt das statische
            // Biot-Savart-Feld. Eichung und Kohomologie sehen genau die
            // Situation, fuer die sie gebaut sind.
            //
            // Zeitkonstante tau = 1.92 ms (aus Q_eff = 0.6047 kalibriert).
            // 300 Schritte = 30 ms = 15.7 tau, Resttransiente 1.6e-7.
            if (max_steps < 300) max_steps = 300;
            std::cout << "\n[Verify] Produktiver Pfad: Gleichstrom 20 A (t auf 5 ms "
                         "festgehalten), " << max_steps << " Schritte.\n"
                      << "[Verify] Nach Abklingen der L/R-Transiente (tau = 1.92 ms) "
                         "ist J = 0 und das Feld rein statisch.\n";
        }

        std::cout << "\n[Solver] Starting Transient Time Loop (" << max_steps << " steps)...\n";

        for (int step = 1; step <= max_steps; ++step) {
            // Im Pruefmodus die Viertelperiode ansteuern, damit der Spulenstrom
            // auf seinem Maximum von 20 A steht; sonst der normale Zeitschritt.
            double t = bf::verify_field() ? 0.005 : step * dt;
            std::cout << "\n--- Time Step " << step << " (t = " << t * 1000.0 << " ms) ---\n";

            // a) Physik-Zustände updaten (Quellströme der Spulen berechnen)
            coil_inner.update_state(t, dt);
            coil_outer.update_state(t, dt);

            // Material-Array auf der GPU aktualisieren (wg. J_source in Spulen)
            // (In einer optimierten Version überträgt man nur die geänderten Spulen-Hexaeder)
            for (int hex_id : coil_inner.get_hex_indices()) d_materials[hex_id] = coil_inner.get_properties();
            for (int hex_id : coil_outer.get_hex_indices()) d_materials[hex_id] = coil_outer.get_properties();

            // b) Lokale FEM-Matrizen aufbauen (K und C)
            BF_LAUNCH_1D_PAR(assemble_local_matrices_kernel, mesh.num_hexes,
                static_cast<int>(mesh.num_hexes), mesh.hex_nodes.data(),
                mesh.nodes_current.data(),
                d_materials.data(), mesh.hex_edge_signs.data(),
                mesh.elem_types.data(),
                d_local_K.data(), d_local_C.data(), d_local_F.data());
            bf::sync();

            // -----------------------------------------------------------------
            // Elementweiser Abgleich des Quellterms
            // -----------------------------------------------------------------
            // Normvergleiche sind an ihre Grenze gekommen: die gesamte Kette --
            // Quadratur, Basis, Piola, Quellterm, Scatter, Kantenkonvention --
            // stimmt Zeile fuer Zeile mit einer unabhaengigen Nachrechnung
            // ueberein, und current_is_azimuthal ist zur Laufzeit TRUE. Trotzdem
            // weichen ||b|| um 13.7 % und ||G^T b||/||b|| um Faktor 3300 ab.
            //
            // Der naechste sinnvolle Schritt ist deshalb kein weiterer
            // aggregierter Vergleich, sondern EIN Element: acht Knoten, zwoelf
            // Zahlen. Damit laesst sich punktgenau sagen, ob die Abweichung in
            // der Elementmatrix oder erst im Scatter entsteht.
            if (bf::verify_field() && step == 1) {
                d_local_F.prefetch_to_host();
                mesh.hex_nodes.prefetch_to_host();
                mesh.nodes_current.prefetch_to_host();
                mesh.hex_edges.prefetch_to_host();
                mesh.hex_edge_signs.prefetch_to_host();
                d_materials.prefetch_to_host();
                bf::sync();

                const auto& ci = coil_inner.get_hex_indices();
                if (!ci.empty()) {
                    const int h = ci.front();
                    const Hexahedron H = mesh.hex_nodes[h];
                    const MaterialProperties& m = d_materials[h];
                    std::cout << "\n[Elementabgleich] Hexaeder " << h
                              << " (Inner_Coil), current_is_azimuthal="
                              << (m.current_is_azimuthal ? "TRUE" : "FALSE")
                              << ", J=(" << std::scientific << std::setprecision(8)
                              << m.current_density_x << ", " << m.current_density_y
                              << ", " << m.current_density_z << ")\n";
                    std::cout << "[Elementabgleich] Knoten (Index, x, y, z in m):\n";
                    for (int i = 0; i < element_num_nodes(mesh.elem_type(h)); ++i) {
                        const double3 p = mesh.nodes_current[H.v[i]];
                        std::cout << "[Elementabgleich]   v[" << i << "] = " << H.v[i]
                                  << "  (" << p.x << ", " << p.y << ", " << p.z << ")\n";
                    }
                    std::cout << "[Elementabgleich] local_F[0..11], globale Kante, Vorzeichen:\n";
                    for (int i = 0; i < 12; ++i) {
                        const int ge = mesh.hex_edges[h * 12 + i];
                        std::cout << "[Elementabgleich]   F[" << i << "] = "
                                  << d_local_F[h * 12 + i]
                                  << "   Kante " << ge
                                  << " = (" << mesh.edges[ge].x << "," << mesh.edges[ge].y << ")"
                                  << "   sign " << mesh.hex_edge_signs[h * 12 + i] << "\n";
                    }
                    std::cout << std::defaultfloat;
                }
            }
            bf::sync();

            // c) Impliziter Euler (A_sys = K + C/dt)
            TransientSolver::build_implicit_euler_system(
                mesh.num_hexes, total_dofs, dt, d_local_K.data(), d_local_C.data(),
                d_local_F.data(),
                mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                d_A_previous.data(), d_materials.data(),
                d_local_A_sys.data(), d_rhs.data()
            );

            // Diagnose der rechten Seite. Billig (ein Durchlauf ueber
            // total_dofs) und der einzige Weg, einen Loeserabbruch von einer
            // fehlenden Anregung zu unterscheiden: ist ||b|| null, fehlt der
            // Quellterm; ist es plausibel gross und der Loeser bricht trotzdem
            // ab, liegt es an der Matrix.
            {
                d_rhs.prefetch_to_host();
                bf::sync();

                // -----------------------------------------------------------------
                // Konsistenz der eingepraegten Stromdichte: G^T b muss null sein
                // -----------------------------------------------------------------
                // G ist der diskrete Gradient (Knoten -> Kante). Fuer die
                // Whitney-Kantenbasis gilt grad(phi) = sum_e (phi_n2 - phi_n1) W_e
                // mit ganzzahligen Koeffizienten -- die Normierung Int_e W.dl = 1
                // ist hier erfuellt (W[0] = 0.125(1-eta)(1-zeta) ergibt auf der
                // Kante den Wert 0.5, integriert ueber xi in [-1,1] also 1), und
                // die kovariante Piola-Transformation erhaelt Linienintegrale.
                //
                // Damit ist b bei divergenzfreiem J_s mit J_s.n = 0 auf dem Rand
                // orthogonal zum Gradientenunterraum, also G^T b = 0. Da der
                // curl-curl-Operator ebenfalls in dieses Orthogonalkomplement
                // abbildet, waere eine Komponente von b entlang der Gradienten
                // ausserhalb des Bildraums -- das System unloesbar und ein
                // Krylov-Verfahren auf genau diesem Anteil stagnierend.
                //
                // WICHTIG: das muss VOR der Eichprojektion gemessen werden. Ein
                // erster Anlauf hat d_rhs erst nach der Projektion gelesen; das
                // Nullen der Baumkanten zerstoert die Divergenzfreiheit
                // (G^T(P b) != G^T b) und lieferte ein Verhaeltnis von 0.886, das
                // nichts bedeutete. Nachgerechnet betraegt der wahre Wert 2e-4.
                {
                    mesh.edges.prefetch_to_host();
                    bf::sync();
                    std::vector<double> gtb(mesh.num_nodes, 0.0);
                    double bmax = 0.0, b2 = 0.0;
                    for (size_t e = 0; e < mesh.num_edges; ++e) {
                        const double be = d_rhs[e];
                        bmax = std::fmax(bmax, std::fabs(be));
                        b2  += be * be;
                        const int2 ed = mesh.edges[e];
                        gtb[ed.y] += be;
                        gtb[ed.x] -= be;
                    }
                    double g2 = 0.0, gmax = 0.0;
                    for (double v : gtb) { g2 += v * v; gmax = std::fmax(gmax, std::fabs(v)); }
                    int nz = 0, nonfinite = 0;
                    for (size_t e = 0; e < mesh.num_edges; ++e) {
                        if (!std::isfinite(d_rhs[e])) ++nonfinite;
                        if (d_rhs[e] != 0.0) ++nz;
                    }
                    // NaN-Falle. Der frueher hier stehende Ausdruck war
                    //     rel = (b2 > 0.0) ? sqrt(g2/b2) : 0.0
                    // und JEDER Vergleich mit NaN ist falsch: bei vergifteter
                    // rechter Seite fiel er in den else-Zweig, setzte rel = 0
                    // und die Meldung unten gab ausdrueckliche ENTWARNUNG --
                    // "divergenzfrei, Inkonsistenz ausgeschlossen" -- genau
                    // dann, wenn b unbrauchbar war. std::fmax verschweigt
                    // gmax auf dieselbe Weise, weil es NaN ueberspringt und
                    // still das Maximum der gesunden Eintraege liefert.
                    // Solange nonfinite > 0 ist, bedeutet keine der drei
                    // Kennzahlen etwas und darf nicht bewertet werden.
                    const bool measurable = (nonfinite == 0) && std::isfinite(b2) && (b2 > 0.0);
                    const double rel = measurable ? std::sqrt(g2 / b2) : -1.0;
                    // Die Kennzahlen entscheiden, ob hier wirklich der ROHE
                    // Quellterm steht -- beim ersten Anlauf stand der bereits
                    // projizierte Vektor darin.
                    //
                    // Hier standen die Vergleichswerte "10315 Eintraege" und
                    // "||b|| = 2.3434e+04" als feste Literale. Sie gehoerten zu
                    // team28.vtk: 4805 Hexaeder, 18690 Kanten, davon 5100 an
                    // Bauteil 1 und 5215 an Bauteil 2, Summe genau 10315. Auf
                    // dem BDF-Netz mit 4709 Kanten ist 10315 nicht nur falsch,
                    // sondern unerreichbar -- nz laeuft ueber num_edges. Der
                    // Abgleich sah damit wie ein Widerspruch zweier Messungen
                    // aus und war der einer Messung mit einer fremden Aufgabe.
                    // Netzabhaengige Referenzen gehoeren nicht in eine
                    // Diagnose, die auf mehreren Netzen laeuft; das
                    // Verhaeltnis ||G^T b||/||b|| ist die einzige der drei
                    // Kennzahlen, die netzunabhaengig eine Aussage traegt.
                    std::cout << "[RHS] Konsistenz G^T b (VOR der Eichung, roher Quellterm):\n"
                              << "[RHS]   Eintraege != 0 = " << nz << " / " << mesh.num_edges
                              << ",   nicht endlich = " << nonfinite
                              << ",   ||b|| = "
                              << std::scientific << std::setprecision(4) << std::sqrt(b2)
                              << std::defaultfloat << "\n";
                    if (!measurable) {
                        std::cout << "[RHS]   -> NICHT MESSBAR: " << nonfinite
                                  << " nicht endliche Eintraege in b. ||G^T b||/||b|| und "
                                     "max|(G^T b)_n| sind bedeutungslos, solange das so ist. "
                                     "Erst die Quelle der NaN/Inf beheben.\n";
                    } else {
                        std::cout << "[RHS]   ||G^T b||/||b|| = "
                                  << std::scientific << std::setprecision(3) << rel
                                  << "   max|(G^T b)_n|/max|b_e| = "
                                  << gmax / (bmax > 0.0 ? bmax : 1.0)
                                  << std::defaultfloat << "\n";
                        std::cout << "[RHS]   " << (rel < 1e-2
                            ? "-> divergenzfrei; b liegt im Bildraum, Inkonsistenz als "
                              "Stagnationsursache ausgeschlossen.\n"
                            : "-> NICHT divergenzfrei; b hat eine Komponente ausserhalb "
                              "des Bildraums.\n");
                    }
                }

                // Endlichkeits-Audit der rohen rechten Seite. Die Kennzahlen
                // oben (||b||, max|b_i|, Anzahl != 0) koennen NaN nicht
                // zuverlaessig anzeigen: min/max-Vergleiche sind fuer NaN immer
                // falsch, und "!= 0" ist fuer NaN wahr -- ein NaN sieht in
                // dieser Statistik aus wie ein gewoehnlicher Eintrag.
                {
                    int rhs_nan = 0, rhs_inf = 0, rhs_first = -1;
                    for (int i = 0; i < total_dofs; ++i) {
                        if (std::isnan(d_rhs[i]))      { ++rhs_nan; if (rhs_first < 0) rhs_first = i; }
                        else if (std::isinf(d_rhs[i])) { ++rhs_inf; if (rhs_first < 0) rhs_first = i; }
                    }
                    if (rhs_nan || rhs_inf) {
                        std::cout << "  [AUDIT] rechte Seite b (roh): " << rhs_nan
                                  << " NaN und " << rhs_inf << " Inf von " << total_dofs
                                  << " Eintraegen, erster bei Index " << rhs_first << "\n";
                    }
                }

                // Eichung auf die rechte Seite anwenden: b_i = 0 fuer geeichte i.
                // Der entfernte Anteil ist genau die nicht loesbare
                // Nullraumkomponente, seine Norm daher eine Kennzahl dafuer, wie
                // inkonsistent die Anregung war -- bei exakter Divergenzfreiheit
                // von J_s waere sie null. Beobachtet: Quadraturfehler der
                // 2x2x2-Regel und facettierte Spulenoberflaeche.
                double removed_sq_tree = 0.0;
                for (int i = 0; i < total_dofs; ++i) {
                    if (!d_is_gauged[i]) continue;
                    removed_sq_tree += d_rhs[i] * d_rhs[i];
                    d_rhs[i] = 0.0;
                }
                // Kohomologie-Anteil ebenfalls herausprojizieren, dieselbe
                // Reihenfolge (erst Baum, dann Kohomologie) wie in
                // CoupledMVOperator::apply, sonst waere die Projektion der
                // rechten Seite nicht dieselbe wie die des Operators.
                //
                // --nocohomology muss BEIDE Seiten treffen. Ein erster Anlauf
                // hat nur den Projektor im Operator weggelassen und die rechte
                // Seite weiter projiziert -- damit war b projiziert und A
                // nicht, also ein inkonsistentes System. Der Lauf sah wie ein
                // Gegentest aus, war aber keiner.
                double removed_sq_cohom = 0.0;
                for (const auto& c : (bf::no_cohomology()
                                      ? std::vector<std::vector<double>>{}
                                      : host_cohomology_reduced)) {
                    double dot = 0.0, norm_sq = 0.0;
                    for (int i = 0; i < total_dofs; ++i) { dot += c[i] * d_rhs[i]; norm_sq += c[i]*c[i]; }
                    const double f = dot / norm_sq;
                    for (int i = 0; i < total_dofs; ++i) d_rhs[i] -= f * c[i];
                    removed_sq_cohom += f * f * norm_sq; // ||f*c||^2, der entfernte Anteil
                }

                double b_sq = 0.0, b_max = 0.0; long b_nnz = 0;
                for (int i = 0; i < total_dofs; ++i) {
                    const double v = d_rhs[i];
                    b_sq += v * v;
                    if (fabs(v) > b_max) b_max = fabs(v);
                    if (v != 0.0) ++b_nnz;
                }
                const double b_norm = sqrt(b_sq);
                const double removed = sqrt(removed_sq_tree + removed_sq_cohom);
                std::cout << "[RHS] ||P b|| = " << b_norm
                          << "   max|b_i| = " << b_max
                          << "   Eintraege != 0 : " << b_nnz << " / " << total_dofs << "\n";
                std::cout << "[RHS] durch die Eichung entfernt: ||(I-P) b|| = " << removed
                          << "   relativ = " << (b_norm > 0.0 ? removed / b_norm : 0.0)
                          << "  (davon Kohomologie: " << std::sqrt(removed_sq_cohom) << ")\n";
                d_rhs.prefetch_to_device();
            }

            // d) Gekoppelten Operator aufbauen
            CoupledMVOperator op(total_dofs);
            op.set_ilu_bypass(disable_ilu_for_debug);

            // 1. Das CSR Values-Array für diesen Zeitschritt auf 0 setzen
            bf::memset_d(csr_matrix.values.data(), 0,
                         static_cast<size_t>(csr_matrix.nnz) * sizeof(double));

            // 2. Lokale Elementmatrizen in die globale CSR-Matrix aufaddieren (Scatter)
            assemble_csr_matrix_gpu(
                mesh.num_hexes,
                d_local_A_sys.data(),
                mesh.hex_edges.data(),
                mesh.hex_edge_signs.data(),
                csr_matrix.row_offsets.data(),
                csr_matrix.col_indices.data(),
                csr_matrix.values.data()
            );

            // 3. Dem cuSPARSE Operator die fertigen Arrays übergeben
            op.setup_fem_matrix(
                csr_matrix.num_rows,
                csr_matrix.nnz,
                csr_matrix.row_offsets.data(),
                csr_matrix.col_indices.data(),
                csr_matrix.values.data()
            );

            // Eichung MUSS vor compute_ilu() gesetzt werden, damit die
            // ILU-Faktorisierung weiss, welche Kanten (Baum) sie als
            // Identitaet isolieren muss (Gauge-Aware ILU, siehe compute_ilu()).
            // --nogauge: die Baum-Kobaum-Eichung wird NICHT installiert. Ohne
            // sie ist das curl-curl-System singulaer, aber konsistent -- die
            // rechte Seite liegt im Bildraum (nachgemessen: G^T b/||b|| =
            // 2.1e-4). Ein Krylov-Verfahren kommt damit zurecht, sofern der
            // Vorkonditionierer den Nullraum nicht verstaerkt; genau dafuer ist
            // AMS gebaut, ILU(0) dagegen konvergiert auf dem ungeeichten System
            // nachweislich in keinem Fall.
            //
            // Die Kohomologie-Projektoren bleiben in BEIDEN Faellen aktiv: AMS
            // deckt die Gradientenmoden ab, nicht die harmonischen Felder, und
            // die gibt es hier wegen der zwei Volltori (die beiden Ringspulen;
            // die Platte ist eine Vollscheibe ohne Loch und traegt keinen bei --
            // der Loeser findet entsprechend genau 2 Kohomologie-Generatoren).
            if (bf::no_gauge()) {
                std::cout << "[Gauge] --nogauge: Maske ist leer, die "
                             "Baum-Kobaum-Eichung wirkt damit nicht. "
                             "Kohomologie-Projektoren bleiben aktiv.\n";
                if (!bf::use_ams()) {
                    std::cout << "[Gauge] WARNUNG: --nogauge ohne --ams. ILU(0) "
                                 "konvergiert auf dem ungeeichten System nicht.\n";
                }
            }
            op.setup_gauge(d_is_gauged.data());

            // Der compute_ilu()-Aufruf wandert von hier weiter nach
            // unten (siehe dort) -- er muss jetzt NACH op.setup_equilibration()
            // laufen, weil er die per SDE skalierte Matrix faktorisiert.
            // setup_equilibration() wiederum braucht die vollstaendig
            // extrahierte BEM-Diagonale (d_bem_inv_diag), die erst nach dem
            // Block-Assemblierungs-Loop unten vorliegt.

            // BEM-Bloecke an die FEM-Kanten anbinden. Ohne diesen Aufruf wirft
            // apply() eine Ausnahme, sobald Bloecke registriert sind.
            // bei disable_bem_for_debug komplett uebersprungen -- op
            // bleibt dann bei num_bem_dofs == 0, apply() faellt automatisch
            // auf das reine FEM-System zurueck (siehe dortiger have_bem-Zweig).
            if (!disable_bem_for_debug) {
                op.setup_bem_coupling(bem_map.num_bem_dofs, d_bem_to_fem.data(), d_bem_sign.data());
                // Geometrie fuer die T-Massenmatrix-Anwendung.
                // nodes_current zeigt auf lebenden, sich zwischen Zeitschritten
                // bewegenden Speicher -- der Zeiger selbst bleibt aber gueltig,
                // daher genuegt ein Aufruf pro (neu konstruiertem) op.
                op.setup_bem_mesh(static_cast<int>(mesh.num_boundary_faces),
                                  mesh.boundary_faces.data(), mesh.nodes_current.data());
            }
            for (size_t g = 0; g < d_cohomology_vectors.size(); ++g) {
                // --nocohomology: Projektor nicht installieren. Begruendung und
                // Messwerte an der Option in Backend.cuh; kurz: die Generatoren
                // liegen nachweislich nicht im Nullraum von A, und die
                // Projektion entfernt 58 % der rechten Seite.
                if (!bf::no_cohomology()) {
                    op.add_cohomology_projector(d_cohomology_vectors[g].data(),
                                                cohomology_norm_sq[g]);
                }
            }

            // BEM-ACA Dynamische Assemblierung
        // Liste für das Speichermanagement (void*, damit double und int Zeiger passen)
            std::vector<void*> allocated_gpu_mem;

            // Hierarchische Blockstruktur: zulaessige Clusterpaare werden als EIN
            // Niedrigrangblock gefuehrt, statt in ihre Blaetter zersplittert zu
            // werden. Da die ACA-Kosten O(k*(m+n)) und nicht O(m*n) sind, senkt
            // das das ACA-Kostenmass sum(m+n) von 7.97e6 auf 1.67e6, Faktor 4.78,
            // bei identischer Abdeckung. Siehe include/BemCoupling.h.
            // bei disable_bem_for_debug komplett uebersprungen -- op
            // bleibt dann ohne jegliche G/H-Bloecke (reines FEM-System).
            // -----------------------------------------------------------------
            // Symmetrie von G_BEM per Konstruktion
            // -----------------------------------------------------------------
            //
            // Der Single-Layer-Operator ist analytisch symmetrisch:
            // G(x,y) = G(y,x), und mit denselben Basisfunktionen gilt
            // G_block(J,I) = G_block(I,J)^T exakt. Bisher wurden beide Bloecke
            // UNABHAENGIG integriert bzw. ACA-komprimiert -- das Ergebnis war
            // nur bis auf Quadratur- und Approximationsfehler symmetrisch.
            //
            // Fuer das innere CG des blocktriangulaeren Vorkonditionierers ist
            // das fatal: CG setzt SPD voraus. Gemessen erreichte es relative
            // Residuen von 1.7 bis 2.1 bei einem Ziel von 0.01 -- es
            // VERGROESSERTE das Residuum, statt es zu senken.
            //
            // Deshalb wird jetzt nur noch die eine Haelfte gerechnet und die
            // andere als Transponierte aufgesetzt. Drei Wirkungen:
            //   * G_BEM ist exakt symmetrisch, unabhaengig von jedem
            //     Approximationsfehler
            //   * der Speicher der Fernfeldbloecke halbiert sich (ACA-Bloecke
            //     teilen sich U/V, dichte Bloecke die Matrix)
            //   * die halbe Integrationsarbeit entfaellt
            //
            // Die Paarliste ist symmetrisch: build_block_pairs() startet bei
            // (root, root) und zerlegt in alle vier Kindkombinationen, und die
            // Zulaessigkeitspruefung ist in i,j symmetrisch. Zu jedem (i,j) mit
            // i != j existiert also (j,i).
            //
            // NUR fuer G_BEM. D_BEM (Double-Layer) ist NICHT symmetrisch --
            // dort wird weiterhin jeder Block einzeln gerechnet.
            struct MirrorEntry { double* d_U; double* d_V; int rank; int m; int n; bool aca; };
            std::map<std::pair<int,int>, MirrorEntry> g_mirror;
            int n_mirrored_aca = 0, n_mirrored_dense = 0;

            if (!disable_bem_for_debug)
            for (const BemBlockPair& bp : block_pairs) {
                const int i = bp.row_node;
                const int j = bp.col_node;
                {
                    int m = bem_map.node_size[i] * 4;
                    int n = bem_map.node_size[j] * 4;

                    if (bp.admissible) {
                        // -------------------------------------------------------------
                        // Fernfeld: Kompression mit ACA+
                        // -------------------------------------------------------------
                        // HIGH-PERFORMANCE EVALUATOR (O(1) Lookup)
                        auto evaluator = [&](int row, int col) {
                            // row und col sind lokale Indizes (0 bis m-1 bzw. n-1).
                            // Ein Quad hat 4 Kanten. 
                            int local_q_i = row / 4; // Welches Quad im Cluster i?
                            int local_e_i = row % 4; // Welche Kante im Quad?

                            int local_q_j = col / 4;
                            int local_e_j = col % 4;

                            // O(1) direkter Lookup über den Cluster-Baum!
                            // Quads eines Clusters ueber die Permutation, weil ein
                            // Block jetzt auch zu einem inneren Knoten gehoeren
                            // kann und quad_indices dort leer ist.
                            int global_quad_p = bem_map.perm[bem_map.node_offset[i] + local_q_i];
                            int global_quad_q = bem_map.perm[bem_map.node_offset[j] + local_q_j];

                            // Koordinaten laden - AUS DEN HOST VEKTOREN!
                            double3 nodes_p[4], nodes_q[4];
                            for (int k = 0; k < 4; ++k) {
                                nodes_p[k] = host_nodes_current[host_boundary_faces[global_quad_p].v[k]];
                                nodes_q[k] = host_nodes_current[host_boundary_faces[global_quad_q].v[k]];
                            }

                            // Reguläre Integration auf der CPU
                            double matrix_block[4][4] = { 0.0 };
                            integrate_regular_quad_pair(nodes_p, nodes_q, matrix_block);

                            return matrix_block[local_e_i][local_e_j];
                            };

                        // Offsets im permutierten BEM-Elementraum. quad_indices[0]
                        // waere falsch: die Quad-Indizes eines Clusters sind
                        // verstreut, weil ClusterTree nach Schwerpunkt sortiert.
                        int row_off = bem_map.node_offset[i] * 4;
                        int col_off = bem_map.node_offset[j] * 4;

                        // Liegt der Spiegelpartner (j,i) schon vor? Dann diesen
                        // Block als dessen Transponierte aufsetzen:
                        // (U V^T)^T = V U^T, also dieselben Puffer mit
                        // vertauschten Rollen. Kein zusaetzlicher Speicher, und
                        // die Symmetrie ist exakt statt approximativ.
                        //
                        // NUR fuer G_BEM -- der H-Block (Double-Layer) ist nicht
                        // symmetrisch und wird unten in jedem Fall gerechnet.
                        const auto mirror_key = std::make_pair(std::min(i, j), std::max(i, j));
                        auto mirror_it = g_mirror.find(mirror_key);
                        const bool g_from_mirror =
                            (i != j) && (mirror_it != g_mirror.end()) && mirror_it->second.aca;

                        if (g_from_mirror) {
                            const MirrorEntry& me = mirror_it->second;
                            op.add_aca_block_mirrored(m, n, me.rank, row_off, col_off,
                                                      me.d_U, me.d_V);
                            ++n_mirrored_aca;
                        } else {
                            CompressedBlock aca = AcaPlus::compress(m, n, evaluator, 1e-4);

                            // Reines Device-Memory (Vermeidet Unified Memory Page-Faults)
                            double* d_U, * d_V;
                            bf::malloc_d(&d_U, aca.U.size() * sizeof(double));
                            bf::malloc_d(&d_V, aca.V.size() * sizeof(double));

                            bf::memcpy_d(d_U, aca.U.data(), aca.U.size() * sizeof(double), cudaMemcpyHostToDevice);
                            bf::memcpy_d(d_V, aca.V.data(), aca.V.size() * sizeof(double), cudaMemcpyHostToDevice);

                            allocated_gpu_mem.push_back((void*)d_U);
                            allocated_gpu_mem.push_back((void*)d_V);

                            op.add_aca_block(m, n, aca.rank, row_off, col_off, d_U, d_V);

                            if (i != j) {
                                g_mirror[mirror_key] =
                                    MirrorEntry{d_U, d_V, aca.rank, m, n, /*aca=*/true};
                            }
                        }

                        // dasselbe fuer H_BEM (Double-Layer,
                        // D_BEM-Anteil) -- gleicher Cluster, gleiche
                        // Offsets, nur integrate_regular_quad_pair_dlp statt
                        // integrate_regular_quad_pair als Kern.
                        auto evaluator_H = [&](int row, int col) {
                            int local_q_i = row / 4;
                            int local_e_i = row % 4;
                            int local_q_j = col / 4;
                            int local_e_j = col % 4;

                            int global_quad_p = bem_map.perm[bem_map.node_offset[i] + local_q_i];
                            int global_quad_q = bem_map.perm[bem_map.node_offset[j] + local_q_j];

                            double3 nodes_p[4], nodes_q[4];
                            for (int k = 0; k < 4; ++k) {
                                nodes_p[k] = host_nodes_current[host_boundary_faces[global_quad_p].v[k]];
                                nodes_q[k] = host_nodes_current[host_boundary_faces[global_quad_q].v[k]];
                            }

                            double matrix_block[4][4] = { 0.0 };
                            integrate_regular_quad_pair_dlp(nodes_p, nodes_q, matrix_block);

                            return matrix_block[local_e_i][local_e_j];
                            };

                        CompressedBlock aca_H = AcaPlus::compress(m, n, evaluator_H, 1e-4);

                        double* d_U_H, * d_V_H;
                        bf::malloc_d(&d_U_H, aca_H.U.size() * sizeof(double));
                        bf::malloc_d(&d_V_H, aca_H.V.size() * sizeof(double));
                        bf::memcpy_d(d_U_H, aca_H.U.data(), aca_H.U.size() * sizeof(double), cudaMemcpyHostToDevice);
                        bf::memcpy_d(d_V_H, aca_H.V.data(), aca_H.V.size() * sizeof(double), cudaMemcpyHostToDevice);
                        allocated_gpu_mem.push_back((void*)d_U_H);
                        allocated_gpu_mem.push_back((void*)d_V_H);

                        op.add_aca_block_H(m, n, aca_H.rank, row_off, col_off, d_U_H, d_V_H);
                    }
                    else {
                        // -------------------------------------------------------------
                        // Nahfeld: Dense Block (Exakte Integration auf der GPU)
                        // -------------------------------------------------------------
                        int num_p = bem_map.node_size[i];
                        int num_q = bem_map.node_size[j];

                        // Reines Device-Memory verwenden (EXTREM wichtig für Performance in der Schleife!)
                        int* d_quads_i, * d_quads_j;
                        bf::malloc_d(&d_quads_i, num_p * sizeof(int));
                        bf::malloc_d(&d_quads_j, num_q * sizeof(int));

                        // Harte Kopie direkt vom std::vector zum VRAM
                        // Auch hier die Quads aus der Permutation nehmen.
                        bf::memcpy_d(d_quads_i, bem_map.perm.data() + bem_map.node_offset[i], num_p * sizeof(int), cudaMemcpyHostToDevice);
                        bf::memcpy_d(d_quads_j, bem_map.perm.data() + bem_map.node_offset[j], num_q * sizeof(int), cudaMemcpyHostToDevice);

                        double* d_dense_mat;
                        bf::malloc_d(&d_dense_mat, static_cast<size_t>(m) * n * sizeof(double));

                        allocated_gpu_mem.push_back((void*)d_quads_i);
                        allocated_gpu_mem.push_back((void*)d_quads_j);
                        allocated_gpu_mem.push_back((void*)d_dense_mat);

                        // Kernel starten (Jetzt ohne Unified Memory Latenzen!)
                        // Im GPU-Modus asynchron in die Queue; im CPU-Modus
                        // laeuft die (OpenMP-)Doppelschleife hier direkt durch.
                        BF_LAUNCH_2D(compute_dense_block_kernel, num_p, num_q,
                            num_p, d_quads_i,
                            num_q, d_quads_j,
                            mesh.boundary_faces.data(),
                            mesh.nodes_current.data(),
                            d_dense_mat);
                        // Kein Sync hier! Wir lassen alle Kernels asynchron
                        // in die Queue laufen, was die Auslastung der GPU maximiert!

                        // Offsets im permutierten BEM-Elementraum. quad_indices[0]
                        // waere falsch: die Quad-Indizes eines Clusters sind
                        // verstreut, weil ClusterTree nach Schwerpunkt sortiert.
                        int row_off = bem_map.node_offset[i] * 4;
                        int col_off = bem_map.node_offset[j] * 4;

                        // Spiegelpartner wie im ACA-Zweig: liegt (j,i) schon
                        // vor, wird dieser Block als dessen Transponierte
                        // aufgesetzt -- derselbe Speicher, dgemv mit
                        // transpose=true. Damit ist auch das Nahfeld von G_BEM
                        // exakt symmetrisch.
                        //
                        // Der Kernel oben laeuft in diesem Fall umsonst; das
                        // liesse sich noch sparen, ist aber eine Optimierung
                        // und keine Korrektheitsfrage -- deshalb hier bewusst
                        // nicht mit der Umstellung vermischt.
                        const auto dmirror_key = std::make_pair(std::min(i, j), std::max(i, j));
                        auto dmirror_it = g_mirror.find(dmirror_key);
                        if (i != j && dmirror_it != g_mirror.end() && !dmirror_it->second.aca) {
                            op.add_dense_block_transposed(m, n, row_off, col_off,
                                                          dmirror_it->second.d_U);
                            ++n_mirrored_dense;
                        } else {
                            op.add_dense_block(m, n, row_off, col_off, d_dense_mat);
                            if (i != j) {
                                g_mirror[dmirror_key] =
                                    MirrorEntry{d_dense_mat, nullptr, 0, m, n, /*aca=*/false};
                            }
                        }

                        // H_BEM-Dense-Block, dieselben Quads
                        // (d_quads_i/d_quads_j), nur mit dem DLP-Kern statt
                        // dem Single-Layer-Kern (siehe compute_dense_block_kernel_H:
                        // shared==4 -> analytisch 0, sonst regulaere Quadratur,
                        // durch die Sauter-Schwab-Kuerzung exakt gerechtfertigt).
                        double* d_dense_mat_H;
                        bf::malloc_d(&d_dense_mat_H, static_cast<size_t>(m) * n * sizeof(double));
                        allocated_gpu_mem.push_back((void*)d_dense_mat_H);

                        BF_LAUNCH_2D(compute_dense_block_kernel_H, num_p, num_q,
                            num_p, d_quads_i,
                            num_q, d_quads_j,
                            mesh.boundary_faces.data(),
                            mesh.nodes_current.data(),
                            d_dense_mat_H);

                        op.add_dense_block_H(m, n, row_off, col_off, d_dense_mat_H);
                    }
                }
            }

            if (n_mirrored_aca + n_mirrored_dense > 0) {
                std::cout << "[BemCoupling] G_BEM symmetrisch aufgesetzt: "
                          << n_mirrored_aca << " ACA- und " << n_mirrored_dense
                          << " dichte Bloecke als Transponierte ihres Spiegelpartners "
                             "(exakte Symmetrie, halber Speicher im Fernfeld).\n";
            }

            // Ein einziger Sync, nachdem ALLE Blöcke an die GPU gesendet wurden
            // (im CPU-Modus ein No-Op -- die Schleifen oben sind bereits fertig).
            bf::sync();

            // Globale Skalierungsmatrix D aufbauen.
            // WICHTIG: Muss aufgerufen werden, NACHDEM alle BEM-Bloecke da
            // sind (d_bem_inv_diag muss vollstaendig extrahiert sein).
            // -----------------------------------------------------------------
            // Liegen die Kohomologie-Generatoren im Nullraum von A?
            // -----------------------------------------------------------------
            // Die Projektion entfernt die Komponente von b entlang der
            // Generatoren. Nach der Piola-Korrektur sind das 58 % der
            // rechten Seite (vorher 4 %), entlang von zwei Richtungen in
            // einem 18690-dimensionalen Raum. Ob das zulaessig ist, haengt
            // an einer einzigen Frage: gilt A c = 0?
            //
            // Fuer ein harmonisches Feld ist curl c = 0 und damit S c = 0.
            // Nur dann ist c unbestimmt, nur dann verlangt die Loesbarkeit
            // b senkrecht zu c, und nur dann aendert die Projektion die
            // Loesung nicht. Liegt ||A c|| dagegen in der Groessenordnung,
            // die ein beliebiger Vektor liefert, gehoert c NICHT in den
            // Nullraum -- und die Projektion wirft Physik weg, naemlich
            // gerade die Verkettung der Spulenloecher durch den
            // eingepraegten Strom.
            //
            // Verglichen wird gegen einen Zufallsvektor gleicher Norm. Eine
            // absolute Schranke fuer ||A c|| haette keine Aussagekraft, weil
            // A hier Eintraege bis 1e9 enthaelt.
            if (bf::verify_field() && step == 1 && !d_cohomology_vectors.empty()) {
                csr_matrix.values.prefetch_to_host();
                csr_matrix.row_offsets.prefetch_to_host();
                csr_matrix.col_indices.prefetch_to_host();
                bf::sync();

                std::vector<double> y(static_cast<size_t>(total_dofs));
                auto apply_A = [&](const double* x) {
                    for (int r = 0; r < total_dofs; ++r) {
                        double acc = 0.0;
                        for (int k = csr_matrix.row_offsets[r];
                             k < csr_matrix.row_offsets[r + 1]; ++k)
                            acc += csr_matrix.values[k] * x[csr_matrix.col_indices[k]];
                        y[r] = acc;
                    }
                    double s = 0.0;
                    for (double q : y) s += q * q;
                    return std::sqrt(s);
                };

                std::vector<double> rnd(static_cast<size_t>(total_dofs));
                unsigned seed = 123456789u;
                double rn2 = 0.0;
                for (int i = 0; i < total_dofs; ++i) {
                    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                    rnd[i] = ((seed >> 8) & 0xffff) / 32767.5 - 1.0;
                    rn2 += rnd[i] * rnd[i];
                }
                const double baseline = apply_A(rnd.data()) / std::sqrt(rn2);

                // Ein Bezugswert von null bedeutet immer, dass die Pruefung
                // selbst an der falschen Stelle steht. Genau das ist beim
                // ersten Anlauf passiert: der Test lief vor der Assemblierung,
                // die CSR war leer, und alle Quotienten kamen als 0 bzw. nan
                // heraus -- eine Ausgabe, die auf den ersten Blick nach einem
                // Befund aussieht.
                if (baseline <= 0.0) {
                    std::cout << "\n  [Kohomologie] ABBRUCH: die Systemmatrix ist an "
                                 "dieser Stelle leer, der Test waere bedeutungslos.\n\n";
                } else {
                    std::cout << "\n  [Kohomologie] Liegen die Generatoren im Nullraum von A?\n"
                              << "  [Kohomologie]   Bezug: ||A r||/||r|| fuer einen Zufallsvektor = "
                              << std::scientific << std::setprecision(3) << baseline << "\n";

                    for (size_t g = 0; g < d_cohomology_vectors.size(); ++g) {
                        d_cohomology_vectors[g].prefetch_to_host();
                        bf::sync();
                        const double cn = std::sqrt(cohomology_norm_sq[g]);
                        const double q  = (cn > 0.0)
                            ? apply_A(d_cohomology_vectors[g].data()) / cn : 0.0;
                        std::cout << "  [Kohomologie]   Generator " << g
                                  << ": ||A c||/||c|| = " << q
                                  << "   relativ zum Bezug " << q / baseline << "\n";
                    }
                    std::cout << std::defaultfloat
                              << "  [Kohomologie]   nahe null  -> im Nullraum, Projektion zulaessig\n"
                                 "  [Kohomologie]   um den Bezug -> NICHT im Nullraum, die\n"
                                 "  [Kohomologie]                   Projektion entfernt dann Physik\n\n";
                }
            }


            op.setup_equilibration();

            // ILU-Faktorisierung der jetzt *skalierten*
            // FEM-Matrix. Von weiter oben hierher verschoben (siehe dortiger
            // Kommentar) -- muss nach setup_equilibration() laufen.
            op.compute_ilu();
            // --ams: Vorkonditionierer des FEM-Blocks aufbauen. Muss NACH der
            // Assemblierung stehen -- die Hilfsmatrizen A_G = G^T A G und
            // A_Pi = Pi^T A Pi haengen von den Werten ab.
            if (bf::use_ams()) {
                // Materialkoeffizienten fuer den Alpha-Poisson-Operator des
                // vektoriellen Hilfsraums. Ohne BEIDE ist er nicht spektral
                // aequivalent zu Pi^T A Pi -- gemessen hat das den aeusseren
                // GMRES von Rel. Res. 0.025 auf 0.99 gebracht.
                //
                // nu = 1/mu (Steifigkeit, ueberall positiv), beta = kappa/dt
                // (gelumpte Masse, nur im Leiter). Genau die Groessen, mit
                // denen assemble_local_matrices_kernel_body auch die
                // Kantenmatrix bildet.
                d_materials.prefetch_to_host();
                bf::sync();
                ams_nu_per_hex.resize(mesh.num_hexes);
                ams_beta_per_hex.resize(mesh.num_hexes);
                for (size_t h = 0; h < mesh.num_hexes; ++h) {
                    const MaterialProperties& m = d_materials[static_cast<int>(h)];
                    ams_nu_per_hex[h]   = 1.0 / (m.mu_relative * 4.0 * M_PI * 1e-7);
                    ams_beta_per_hex[h] = m.kappa / dt;
                }
                op.set_ams_alpha_coefficients(ams_nu_per_hex.data(),
                                              ams_beta_per_hex.data());
                op.setup_ams(mesh);
            }

            // Blocktriangulaerer Vorkonditionierer statt des
            // blockdiagonalen diag(A~, I). Die Reihenfolge ist hier nicht
            // beliebig -- der Setter prueft sie und wirft sonst:
            //   * nach setup_bem_coupling()  -> num_bem_dofs steht fest (Puffer)
            //   * nach allen add_dense_block()/add_aca_block() -> S~ = mu_0*G_BEM
            //     braucht die G-Bloecke
            //   * nach setup_equilibration() -> D_K wird in apply_schur_tilde()
            //     und apply_coupling_T_tilde() mitbenutzt
            //   * nach compute_ilu()/setup_ams() -> der A~^-1-Schritt der
            //     Rueckwaertssubstitution ist genau dieser Vorkonditionierer
            // Mit "nobem" gibt es kein 2x2-System, der Schalter waere dann
            // gegenstandslos (der Setter meldet das, statt still nichts zu tun).
            // Kantenliste fuer die Divergenzdiagnose des Kopplungsterms.
            op.setup_divergence_check(mesh.edges.data(),
                                      static_cast<int>(mesh.num_nodes),
                                      static_cast<int>(mesh.num_edges));

            if (bf::use_block_tri_pc() && !disable_bem_for_debug) {
                op.set_block_triangular_preconditioner(
                    true, bf::pc_inner_iters(), bf::pc_inner_tol());
            }


            // Diagnose -- pruefen, ob die extrahierte BEM-Diagonale
            // (Jacobi-Preconditioner fuer K) pathologische Ausreisser hat.
            if (!disable_bem_for_debug) {
                op.debug_print_bem_diag_stats();
            }

            // e) System Lösen (BiCGStab)
            // gekoppelte RHS (A-Block = bisheriges d_rhs,
            // K-Block = 0, siehe Gl. 11/12) und gekoppelter Loesungsvektor.
            // d_rhs_coupled wird bewusst JEDEN Zeitschritt neu gebaut (die
            // RHS aendert sich ja), d_x_coupled dagegen lebt ausserhalb der
            // Schleife und behaelt den Loesungs-Warmstart vom Vorschritt.
            DeviceArray<double> d_rhs_coupled(static_cast<size_t>(total_dofs) + static_cast<size_t>(bem_map.num_bem_dofs));
            bf::memcpy_d(d_rhs_coupled.data(), d_rhs.data(),
                         static_cast<size_t>(total_dofs) * sizeof(double),
                         cudaMemcpyDeviceToDevice);
            bf::memset_d(d_rhs_coupled.data() + total_dofs, 0,
                         static_cast<size_t>(bem_map.num_bem_dofs) * sizeof(double));

            // Wir loesen \tilde{M} \tilde{x} = \tilde{b} im
            // aequilibrierten Raum. Der Solver selbst bleibt komplett
            // ahnungslos von D -- er sieht nur b_tilde/x_tilde. d_D_scale ist
            // privat, daher ueber die oeffentlichen scale_by_D()/
            // scale_by_D_inverse()-Helfer (No-Op, falls setup_equilibration()
            // nicht aktiv war).

            // 1. RHS skalieren: b_tilde = D * b
            op.scale_by_D(d_rhs_coupled.data());

            // 2. Startvektor skalieren: x_tilde_0 = D^-1 * x_0
            op.scale_by_D_inverse(d_x_coupled.data());
            bf::sync();

            // GMRES(50) statt BiCGStab testweise -- siehe GMRESSolver in
            // CoupledSolver.cuh/.cu. Garantiert (in exakter Arithmetik) streng
            // monoton fallendes Residuum, keine Near-Breakdowns/Spikes wie bei
            // BiCGStab. Gleiches Interface (solve(op, rhs, x, tol, max_iter)).
            // -----------------------------------------------------------------
            // Ist der Operator ueberhaupt eine feste lineare Abbildung?
            // -----------------------------------------------------------------
            // Anlass ist ein Widerspruch, den die Ritz-Messung aufgedeckt hat.
            // Das Spektrum des vorkonditionierten Operators auf dem
            // Krylov-Unterraum ist gutartig:
            //
            //     |lambda| von 1.49 bis 244, kappa = 164, kein Wert nahe null,
            //     alle reell
            //
            // Fuer ein solches Spektrum lautet die GMRES-Schranke
            // ((sqrt(k)-1)/(sqrt(k)+1))^k = 0.855^k, nach 50 Iterationen also
            // 4e-04. Beobachtet werden 0.431 -- drei Groessenordnungen daneben,
            // und zwar INNERHALB des ersten Zyklus, also genau dort, wo die
            // Ritz-Werte herkommen.
            //
            // Ein Krylov-Verfahren kann nicht schlechter sein als das Spektrum
            // des Operators, den es sieht. Es sei denn, es sieht bei jeder
            // Anwendung einen anderen. Genau dieser Fehler steckte zu Beginn im
            // Referenzloeser: dort las apply() host-seitig Daten, die ein
            // asynchroner Kernel noch schrieb, der Operator war damit
            // nichtdeterministisch, die Arnoldi-Relation gebrochen und GMRES
            // divergierte.
            //
            // Zwei Eigenschaften werden geprueft, beide notwendig:
            //   Determinismus  A x zweimal hintereinander muss identisch sein
            //   Linearitaet    A(2x) muss 2 A(x) sein
            // Die zweite faengt einen Zustand, der zwischen Aufrufen
            // stehenbleibt oder addiert wird -- der waere deterministisch, aber
            // trotzdem kein linearer Operator.
            if (bf::verify_field() && step == 1) {
                const int n_all = op.get_total_dofs();
                DeviceArray<double> tx(static_cast<size_t>(n_all));
                DeviceArray<double> t2(static_cast<size_t>(n_all));
                DeviceArray<double> y1(static_cast<size_t>(n_all));
                DeviceArray<double> y2(static_cast<size_t>(n_all));
                tx.prefetch_to_host(); t2.prefetch_to_host(); bf::sync();

                unsigned seed = 88172645u;
                for (int i = 0; i < n_all; ++i) {
                    seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                    tx[i] = ((seed >> 8) & 0xffff) / 32767.5 - 1.0;
                    t2[i] = 2.0 * tx[i];
                }
                tx.prefetch_to_device(); t2.prefetch_to_device();

                op.apply(tx.data(), y1.data()); bf::sync();
                op.apply(tx.data(), y2.data()); bf::sync();

                double n1 = 0.0, dd = 0.0;
                bf::dnrm2(op.get_cublas_handle(), n_all, y1.data(), &n1);
                y1.prefetch_to_host(); y2.prefetch_to_host(); bf::sync();
                int n_diff = 0;
                for (int i = 0; i < n_all; ++i) {
                    const double d = y1[i] - y2[i];
                    dd += d * d;
                    if (d != 0.0) ++n_diff;
                }
                dd = std::sqrt(dd);

                // Linearitaet: A(2x) - 2 A(x)
                y1.prefetch_to_device(); bf::sync();
                op.apply(t2.data(), y2.data()); bf::sync();
                y1.prefetch_to_host(); y2.prefetch_to_host(); bf::sync();
                double lin = 0.0;
                for (int i = 0; i < n_all; ++i) {
                    const double d = y2[i] - 2.0 * y1[i];
                    lin += d * d;
                }
                lin = std::sqrt(lin);

                std::cout << "\n  [Operatortest] Determinismus und Linearitaet\n"
                          << "  [Operatortest]   ||A x|| = " << std::scientific
                          << std::setprecision(4) << n1 << "\n"
                          << "  [Operatortest]   zweimal A x, Abweichung ||y1-y2|| = " << dd
                          << "   relativ " << (n1 > 0 ? dd / n1 : 0.0)
                          << "   abweichende Eintraege: " << n_diff << " von " << n_all << "\n"
                          << "  [Operatortest]   ||A(2x) - 2 A(x)|| = " << lin
                          << "   relativ " << (n1 > 0 ? lin / (2.0 * n1) : 0.0) << "\n"
                          << std::defaultfloat;
                // Schwelle: 100 Maschinengenauigkeiten. Ein erster Anlauf
                // verlangte EXAKTE Gleichheit und meldete prompt "nicht
                // deterministisch" bei einer relativen Abweichung von 4.5e-18 --
                // dem 49-fachen UNTERSCHREITEN von eps = 2.2e-16. Das ist die
                // uebliche Nichtassoziativitaet von atomicAdd: dieselbe Summe,
                // andere Reihenfolge, letztes Bit anders. Fuer GPU-Reduktionen
                // ist exakte Reproduzierbarkeit die falsche Erwartung.
                const double tol_det = 100.0 * 2.220446049250313e-16;
                if (n1 > 0.0 && dd / n1 > tol_det)
                    std::cout << "  [Operatortest]   -> NICHT DETERMINISTISCH. Die "
                                 "Arnoldi-Relation ist damit gebrochen und GMRES kann\n"
                                 "  [Operatortest]      nicht konvergieren, egal wie gut das "
                                 "Spektrum ist. Typische Ursache: host-seitiger\n"
                                 "  [Operatortest]      Zugriff auf Daten, die ein asynchroner "
                                 "Kernel noch schreibt.\n";
                else if (lin / (2.0 * (n1 > 0 ? n1 : 1.0)) > tol_det)
                    std::cout << "  [Operatortest]   -> deterministisch, aber NICHT LINEAR. "
                                 "Es bleibt Zustand zwischen den Aufrufen stehen.\n";
                else
                    std::cout << "  [Operatortest]   -> deterministisch und linear. Als "
                                 "Ursache ausgeschlossen.\n";

                // -------------------------------------------------------------
                // Wie unsymmetrisch ist der Operator? -- gemittelt
                // -------------------------------------------------------------
                // Das ist die Groesse, die den Widerspruch aufloest. Die
                // Ritz-Messung ergab ein gutartiges Spektrum (kappa = 164, alle
                // Werte reell, kein Nahnullraum), und daraus folgte die
                // Schranke 0.855^k -- nach 50 Iterationen also 4e-04, beobachtet
                // 0.431. Diese Schranke gilt aber nur fuer NORMALE Operatoren;
                // sie ist im Kern die CG-Schranke. Bei nichtnormalen Operatoren
                // sagen die Eigenwerte ueber die GMRES-Konvergenz nichts aus --
                // nach Greenbaum, Ptak und Strakos laesst sich zu jedem Spektrum
                // jede Konvergenzkurve konstruieren.
                //
                // ZUR MESSUNG. Ein erster Anlauf verglich y^T A x mit x^T A y
                // fuer EIN Zufallspaar und meldete ein Verhaeltnis von 11.5.
                // Diese Zahl ist wertlos: der symmetrische Anteil y^T S x ist
                // selbst eine Zufallsgroesse und kann nahe null liegen, was den
                // Quotienten beliebig gross macht. Nachgestellt an einer Matrix
                // mit BEKANNTER Unsymmetrie von 0.0997 streuen Einzelpaare von
                // 0.007 bis 25.4.
                //
                // Belastbar wird es durch Mittelung. Fuer x, y mit unabhaengigen
                // Eintraegen der Varianz 1 gilt
                //
                //     y^T A x - x^T A y = y^T (A - A^T) x
                //     E[(y^T M x)^2]    = ||M||_F^2
                //
                // Der Mittelwert der Quadrate ueber mehrere Paare liefert also
                // direkt ||A - A^T||_F und ||A + A^T||_F, und ihr Verhaeltnis ist
                // das gesuchte Mass -- unabhaengig vom Zufall der Ziehung.
                {
                    // 16 Paare = 32 Operatoranwendungen. Der Schaetzer mittelt
                    // Quadrate von Zufallsgroessen, sein relativer Fehler faellt
                    // also nur mit 1/sqrt(n_pairs) und liegt hier bei rund
                    // 30 bis 50 %. Das genuegt: gefragt ist, ob die Unsymmetrie
                    // bei null oder bei 0.1 bis 1 liegt, nicht ihre dritte
                    // Stelle. Mehr Paare waeren teuer, weil jede Anwendung die
                    // ACA-BEM durchlaeuft.
                    const int n_pairs = 16;
                    const int n_fem = n_all - static_cast<int>(bem_map.num_bem_dofs);
                    DeviceArray<double> px(static_cast<size_t>(n_all));
                    DeviceArray<double> py(static_cast<size_t>(n_all));
                    DeviceArray<double> qx(static_cast<size_t>(n_all));
                    DeviceArray<double> qy(static_cast<size_t>(n_all));

                    double sum_d2 = 0.0, sum_s2 = 0.0;      // ganzer Operator
                    double cpl_d2 = 0.0, cpl_s2 = 0.0;      // nur die Kopplung
                    unsigned sd = 20250811u;

                    for (int rep = 0; rep < n_pairs; ++rep) {
                        px.prefetch_to_host(); py.prefetch_to_host(); bf::sync();
                        for (int i = 0; i < n_all; ++i) {
                            sd ^= sd << 13; sd ^= sd >> 17; sd ^= sd << 5;
                            px[i] = ((sd >> 8) & 0xffff) / 32767.5 - 1.0;
                            sd ^= sd << 13; sd ^= sd >> 17; sd ^= sd << 5;
                            py[i] = ((sd >> 8) & 0xffff) / 32767.5 - 1.0;
                        }
                        px.prefetch_to_device(); py.prefetch_to_device();
                        op.apply(px.data(), qx.data());
                        op.apply(py.data(), qy.data());
                        bf::sync();

                        double aa = 0.0, bb = 0.0;
                        bf::ddot(op.get_cublas_handle(), n_all, py.data(), qx.data(), &aa);
                        bf::ddot(op.get_cublas_handle(), n_all, px.data(), qy.data(), &bb);
                        sum_d2 += (aa - bb) * (aa - bb);
                        sum_s2 += (aa + bb) * (aa + bb);

                        // Kopplung isolieren: nur FEM-Zeilen der einen, nur
                        // BEM-Zeilen der anderen Anwendung eingehen lassen. Das
                        // entspricht x im FEM-, y im BEM-Block.
                        double t1 = 0.0, t2 = 0.0;
                        bf::ddot(op.get_cublas_handle(),
                                 static_cast<int>(bem_map.num_bem_dofs),
                                 py.data() + n_fem, qx.data() + n_fem, &t1);
                        bf::ddot(op.get_cublas_handle(), n_fem, px.data(), qy.data(), &t2);
                        cpl_d2 += (t1 - t2) * (t1 - t2);
                        cpl_s2 += (t1 + t2) * (t1 + t2);
                    }

                    const double nd  = std::sqrt(sum_d2 / n_pairs);
                    const double ns  = std::sqrt(sum_s2 / n_pairs);
                    const double cnd = std::sqrt(cpl_d2 / n_pairs);
                    const double cns = std::sqrt(cpl_s2 / n_pairs);
                    const double r_all = nd  / std::fmax(ns,  1e-300);
                    const double r_cpl = cnd / std::fmax(cns, 1e-300);

                    std::cout << "  [Operatortest]   Unsymmetrie, gemittelt ueber "
                              << n_pairs << " Zufallspaare:\n"
                              << "  [Operatortest]     ganzer Operator  ||A-A^T||_F/||A+A^T||_F = "
                              << std::scientific << std::setprecision(3) << r_all << "\n"
                              << "  [Operatortest]     nur die Kopplung ||T'-T^T||_F/||T'+T^T||_F = "
                              << r_cpl << std::defaultfloat << "\n";
                    if (r_all < 1e-8)
                        std::cout << "  [Operatortest]   -> praktisch symmetrisch. Dann gilt die "
                                     "Spektralschranke, und die beobachtete\n"
                                     "  [Operatortest]      Konvergenz ist damit unvereinbar.\n";
                    else {
                        std::cout << "  [Operatortest]   -> unsymmetrisch. Die Eigenwerte sagen "
                                     "dann ueber die GMRES-Konvergenz nichts\n"
                                     "  [Operatortest]      aus (Greenbaum, Ptak, Strakos); "
                                     "massgeblich ist der Wertebereich.\n";
                        if (r_cpl > 0.1 * r_all)
                            std::cout << "  [Operatortest]      Die Kopplungsbloecke tragen den "
                                         "wesentlichen Teil davon.\n";
                        else
                            std::cout << "  [Operatortest]      Die Kopplung ist dabei nicht die "
                                         "Hauptquelle -- der Rest steckt in den\n"
                                         "  [Operatortest]      Diagonalbloecken.\n";
                    }
                }
                std::cout << "\n";
            }

            // -----------------------------------------------------------------
            // Blockweiser Selbsttest des AEQUILIBRIERTEN Operators
            // -----------------------------------------------------------------
            // Die vorhandene Diagnose misst apply_physical, also den
            // UNskalierten Operator -- dort ist ein Groessenunterschied von
            // vielen Dekaden zwischen FEM- und BEM-Block zu erwarten und sagt
            // nichts aus. Entscheidend ist der Operator, den GMRES sieht: nach
            // der Aequilibrierung sollten beide Bloecke eine Diagonale um 1
            // haben, die "Verstaerkung" ||A x||/||x|| also fuer beide etwa 1
            // sein.
            //
            // Gemessen wird mit zwei Testvektoren, die jeweils nur in EINEM
            // Block besetzt sind. Faellt die Verstaerkung des BEM-Blocks um
            // Groessenordnungen ab, ist dieser Block im skalierten System
            // nahezu singulaer -- dann sind die BEM-Unbekannten praktisch
            // unbestimmt, und genau darauf stagniert GMRES.
            if (bf::verify_field() && step == 1) {
                // ACHTUNG: get_total_dofs() liefert bereits die GEKOPPELTE
                // Groesse (total_dofs + num_bem_dofs, siehe CoupledSolver.cuh
                // Z.422). Ein erster Anlauf hat sie als FEM-Groesse genommen und
                // num_bem_dofs nochmals addiert -- die "BEM-Anregung" fuellte
                // dann nur ueberschuessiges Padding, das der Operator nie
                // anfasst, und der Test meldete exakte Nullen, die nichts mit
                // dem Operator zu tun hatten.
                const int n_all = op.get_total_dofs();
                const int n_fem = n_all - static_cast<int>(bem_map.num_bem_dofs);
                DeviceArray<double> tx(static_cast<size_t>(n_all));
                DeviceArray<double> ty(static_cast<size_t>(n_all));
                tx.prefetch_to_host(); bf::sync();

                auto probe = [&](bool fem_part, const char* name) {
                    unsigned seed = 2463534242u;
                    for (int i = 0; i < n_all; ++i) {
                        const bool in_block = fem_part ? (i < n_fem) : (i >= n_fem);
                        seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
                        tx[i] = in_block ? (((seed >> 8) & 0xffff) / 32767.5 - 1.0) : 0.0;
                    }
                    tx.prefetch_to_device();
                    op.apply(tx.data(), ty.data());
                    bf::sync();

                    double nx = 0.0, ny_fem = 0.0, ny_bem = 0.0, quad = 0.0;
                    bf::dnrm2(op.get_cublas_handle(), n_all, tx.data(), &nx);
                    bf::dnrm2(op.get_cublas_handle(), n_fem, ty.data(), &ny_fem);
                    bf::dnrm2(op.get_cublas_handle(),
                              static_cast<int>(bem_map.num_bem_dofs),
                              ty.data() + n_fem, &ny_bem);
                    bf::ddot(op.get_cublas_handle(), n_all, tx.data(), ty.data(), &quad);

                    std::cout << "  [Blocktest] Anregung nur im " << name
                              << "-Block:  ||A x||/||x||  FEM-Zeilen "
                              << std::scientific << std::setprecision(3) << ny_fem / nx
                              << ",  BEM-Zeilen " << ny_bem / nx
                              << "   x^T A x/||x||^2 = " << quad / (nx * nx)
                              << std::defaultfloat << "\n";
                };

                std::cout << "\n  [Blocktest] Operator NACH Aequilibrierung, "
                          << n_fem << " FEM- + " << bem_map.num_bem_dofs
                          << " BEM-Freiheitsgrade = " << n_all << " gesamt.\n"
                          << "  [Blocktest] Beide Bloecke sollten eine Verstaerkung "
                             "um 1 zeigen; ein Abfall um Groessenordnungen bedeutet "
                             "einen im skalierten System nahezu singulaeren Block.\n";
                probe(true,  "FEM");
                probe(false, "BEM");
                std::cout << "\n";
            }

            GMRESSolver solver(op.get_total_dofs(), bf::gmres_restart());
            // Im Pruefmodus einmalig das Spektrum melden -- die Frage, ob 1000
            // Iterationen reichen KOENNEN, ist vor jedem weiteren Einzelfix zu
            // klaeren.
            // Der Ritz-Bericht war bisher an --verify-field gekoppelt, also an
            // eine voellig andere Pruefung. Jetzt auch direkt ueber --ritz.
            // Blockweise Residuumsaufschluesselung: sie beantwortet, ob das
            // verbleibende Residuum im FEM- oder im BEM-Block sitzt. Bei b_K = 0
            // ist das die einzige Groesse, aus der man das ablesen kann.
            solver.enable_block_residual_report(op.get_num_bem_dofs());
            solver.enable_ritz_report((bf::verify_field() || bf::want_ritz()) && step == 1);
            // -----------------------------------------------------------------
            // --direct: das gekoppelte System dicht zerlegen und vergleichen
            // -----------------------------------------------------------------
            //
            // Anlass ist die Netzstudie. Auf vier Ersatznetzen mit divergenz-
            // freiem Quellterm (||G^T b||/||b|| = 2.4e-08 bis 3.6e-08) und
            // korrekter Topologie stagniert der Loeser bei praktisch demselben
            // Residuum:
            //
            //   n=12   1791 Zeilen   0.0113
            //   n=16   3447 Zeilen   0.0107
            //   n=24   9139 Zeilen   0.0128
            //   n=32  17199 Zeilen   0.0194
            //
            // Ueber einen Faktor 10 in der Problemgroesse praktisch
            // netzunabhaengig. Ein Konditionierungsproblem muesste mit dem Netz
            // wachsen, und bei 1791 Unbekannten muesste jeder brauchbare
            // Vorkonditionierer auf 1e-6 kommen. Das Problem sitzt also nicht
            // im Loeser.
            //
            // Diese Option beantwortet die verbleibende Frage direkt: sie baut
            // den Operator spaltenweise auf (eine Matrix-Vektor-Multiplikation
            // je Spalte), zerlegt ihn dicht und vergleicht.
            //
            //   * Bleibt das Residuum der DIREKTEN Loesung ebenfalls bei ~1e-2,
            //     ist das System inkonsistent -- ein Modellierungs- oder
            //     Assemblierungsfehler, kein Loeserproblem.
            //   * Faellt es auf ~1e-15, ist das System loesbar und GMRES
            //     scheitert aus einem anderen Grund. Dann ist der Vergleich der
            //     beiden Loesungsvektoren der naechste Schritt.
            //
            // Nur fuer kleine Netze gedacht: der Speicher waechst quadratisch
            // (bei n=12 sind es 4287^2 * 8 Byte, also rund 147 MB), der Aufwand
            // kubisch.
            if (bf::use_direct_solve()) {
                const int nd = op.get_total_dofs();
                const double gib = static_cast<double>(nd) * nd * 8.0 / (1024.0*1024.0*1024.0);
                std::cout << "\n[Direct] Dichte Zerlegung des gekoppelten Systems: "
                          << nd << " x " << nd << " (" << gib << " GiB)\n";
                if (gib > 8.0) {
                    std::cout << "[Direct] ABBRUCH: ueber 8 GiB. --direct ist fuer kleine "
                                 "Netze gedacht; tools/make_coarse_team28.py erzeugt sie.\n";
                } else {
                    std::vector<double> dense(static_cast<size_t>(nd) * nd, 0.0);
                    std::vector<double> e(nd, 0.0), col(nd, 0.0);
                    // op.apply() startet Kernel und erwartet GERAETEZEIGER.
                    // Host-Zeiger direkt hineinzugeben funktioniert nur im
                    // CPU-Modus, wo "Geraetespeicher" gewoehnlicher
                    // Host-Speicher ist; auf der GPU dereferenziert der Kernel
                    // dann Host-Adressen -> illegal memory access, gemeldet
                    // erst am naechsten Synchronisationspunkt und damit an
                    // einer voellig anderen Stelle.
                    DeviceArray<double> d_e(static_cast<size_t>(nd));
                    DeviceArray<double> d_col(static_cast<size_t>(nd));
                    for (int j = 0; j < nd; ++j) {
                        std::fill(e.begin(), e.end(), 0.0);
                        e[j] = 1.0;
                        bf::memcpy_d(d_e.data(), e.data(),
                                     static_cast<size_t>(nd) * sizeof(double),
                                     cudaMemcpyHostToDevice);
                        op.apply(d_e.data(), d_col.data());
                        bf::memcpy_d(col.data(), d_col.data(),
                                     static_cast<size_t>(nd) * sizeof(double),
                                     cudaMemcpyDeviceToHost);
                        // Column-major, wie DenseLU es erwartet.
                        std::copy(col.begin(), col.end(),
                                  dense.begin() + static_cast<size_t>(j) * nd);
                        if (nd >= 2000 && (j % 1000) == 0)
                            std::cout << "[Direct]   Spalte " << j << " von " << nd << "\n";
                    }

                    std::vector<double> xd(d_rhs_coupled.data(), d_rhs_coupled.data() + nd);
                    // Referenz VOR der Zerlegung sichern: factorize() arbeitet
                    // in place und ueberschreibt dense.
                    std::vector<double> A_copy(dense);

                    // -------------------------------------------------------------
                    // Ist apply() ueberhaupt eine feste lineare Abbildung?
                    // -------------------------------------------------------------
                    //
                    // Das ist nach den bisherigen Messungen die entscheidende Frage.
                    // Fest steht: das System ist loesbar (die direkte Loesung erreicht
                    // ein Residuum um 1e-15), und es liegt nicht an der
                    // Konditionierung (1791 Unbekannte stagnieren wie 18690). Ein
                    // loesbares, kleines System, an dem GMRES scheitert -- dafuer
                    // bleibt im Wesentlichen eine Erklaerung: der Operator, den GMRES
                    // sieht, ist nicht derselbe, den die spaltenweise Abtastung oben
                    // gesehen hat.
                    //
                    // GMRES setzt einen FESTEN linearen Operator voraus. Haengt
                    // apply() vom Aufrufverlauf ab -- etwa ueber einen gemeinsam
                    // benutzten Zwischenpuffer --, bricht der Arnoldi-Prozess
                    // zusammen, waehrend die spaltenweise Abtastung nichts merkt:
                    // sie ruft jede Spalte einzeln ab.
                    //
                    // Drei Pruefungen, aufsteigend schaerfer:
                    //   (1) Wiederholbarkeit: zweimal dasselbe x -> bitgleich?
                    //   (2) Linearitaet: apply(a x + b y) == a apply(x) + b apply(y)?
                    //   (3) Uebereinstimmung mit der abgetasteten Matrix: A_dense * v
                    //       gegen apply(v) fuer einen Zufallsvektor.
                    //
                    // (3) ist die schaerfste: sie vergleicht genau das, was --direct
                    // gesehen hat, mit dem, was GMRES sieht.
                    {
                        std::mt19937 rng(20240814u);
                        std::uniform_real_distribution<double> uni(-1.0, 1.0);
                        std::vector<double> vx(nd), vy(nd), r1(nd), r2(nd), r3(nd), comb(nd);
                        for (int i = 0; i < nd; ++i) { vx[i] = uni(rng); vy[i] = uni(rng); }

                        auto relnorm = [nd](const std::vector<double>& a,
                                            const std::vector<double>& b) {
                            double d = 0.0, s = 0.0;
                            for (int i = 0; i < nd; ++i) {
                                d += (a[i]-b[i]) * (a[i]-b[i]);
                                s += a[i]*a[i];
                            }
                            return std::sqrt(d) / std::sqrt(s > 0.0 ? s : 1.0);
                        };

                        // Auch hier Geraetepuffer: op.apply() startet Kernel.
                        // Ausgerechnet die Wiederholbarkeitspruefung -- die
                        // Zahl, an der der ganze GPU-Pfad haengt -- lief mit
                        // Host-Zeigern und konnte auf der GPU gar nicht
                        // gueltig sein.
                        DeviceArray<double> d_in(static_cast<size_t>(nd));
                        DeviceArray<double> d_out(static_cast<size_t>(nd));
                        auto apply_host = [&](const std::vector<double>& in,
                                              std::vector<double>& out) {
                            bf::memcpy_d(d_in.data(), in.data(),
                                         static_cast<size_t>(nd) * sizeof(double),
                                         cudaMemcpyHostToDevice);
                            op.apply(d_in.data(), d_out.data());
                            bf::memcpy_d(out.data(), d_out.data(),
                                         static_cast<size_t>(nd) * sizeof(double),
                                         cudaMemcpyDeviceToHost);
                        };

                        apply_host(vx, r1);
                        apply_host(vx, r2);
                        std::cout << "[Direct] Operatorpruefung:\n";
                        std::cout << "[Direct]   (1) Wiederholbarkeit  |apply(x) - apply(x)| / |apply(x)| = "
                                  << relnorm(r1, r2) << "   (muss exakt 0 sein)\n";

                        const double ca = 0.7, cb = -1.3;
                        for (int i = 0; i < nd; ++i) comb[i] = ca * vx[i] + cb * vy[i];
                        apply_host(comb, r3);
                        apply_host(vy, r2);
                        for (int i = 0; i < nd; ++i) comb[i] = ca * r1[i] + cb * r2[i];
                        std::cout << "[Direct]   (2) Linearitaet       |apply(ax+by) - (a apply(x) + b apply(y))| / |.| = "
                                  << relnorm(r3, comb) << "\n";

                        // A_dense * vx, column-major
                        std::fill(comb.begin(), comb.end(), 0.0);
                        for (int j = 0; j < nd; ++j) {
                            const double xj = vx[j];
                            if (xj == 0.0) continue;
                            const double* colj = A_copy.data() + static_cast<size_t>(j) * nd;
                            for (int i = 0; i < nd; ++i) comb[i] += colj[i] * xj;
                        }
                        const double mismatch = relnorm(r1, comb);
                        std::cout << "[Direct]   (3) A_dense * v gegen apply(v): " << mismatch << "\n";
                        if (mismatch > 1e-10) {
                            std::cout << "[Direct]   -> ABWEICHUNG. Der Operator, den GMRES sieht, ist NICHT\n"
                                         "[Direct]      der, den die spaltenweise Abtastung gesehen hat. Damit ist\n"
                                         "[Direct]      apply() keine feste lineare Abbildung, und GMRES kann\n"
                                         "[Direct]      bauartbedingt nicht konvergieren. Das ist die Ursache --\n"
                                         "[Direct]      kein Vorkonditionierer und keine Iterationszahl aendern daran\n"
                                         "[Direct]      etwas.\n";
                        } else {
                            std::cout << "[Direct]   -> apply() stimmt mit der abgetasteten Matrix ueberein.\n"
                                         "[Direct]      Der Operator ist also fest und linear; die Ursache liegt\n"
                                         "[Direct]      woanders. Naechster Schritt waere der Vergleich der\n"
                                         "[Direct]      LOESUNGSVEKTOREN (direkt gegen GMRES).\n";
                        }
                    }

                    // bf::DenseLU ruft im GPU-Modus cuSOLVER und erwartet
                    // GERAETEZEIGER -- genau wie op.apply(). dense und xd sind
                    // Host-Vektoren; sie muessen also aufs Geraet.
                    //
                    // Die Puffer leben bis zum Ende dieses Blocks: DenseLU
                    // merkt sich in factorize() den ZEIGER auf die zerlegte
                    // Matrix (siehe DenseLinalg.cuh), solve() greift spaeter
                    // wieder darauf zu. Ein lokaler Puffer waere dann laengst
                    // freigegeben, und das faellt erst beim spaeteren solve()
                    // auf.
                    //
                    // dense selbst bleibt auf dem Host unveraendert (die
                    // Zerlegung laeuft in-place auf der Geraetekopie), was der
                    // nachfolgende Residuumstest ohnehin voraussetzt.
                    DeviceArray<double> d_dense(static_cast<size_t>(nd) * nd);
                    DeviceArray<double> d_rhs_dense(static_cast<size_t>(nd));
                    bf::memcpy_d(d_dense.data(), dense.data(),
                                 static_cast<size_t>(nd) * nd * sizeof(double),
                                 cudaMemcpyHostToDevice);
                    bf::memcpy_d(d_rhs_dense.data(), xd.data(),
                                 static_cast<size_t>(nd) * sizeof(double),
                                 cudaMemcpyHostToDevice);

                    bf::DenseLU lu;
                    bool ok = true;
                    try {
                        lu.factorize(d_dense.data(), nd);
                        lu.solve(d_rhs_dense.data(), 1);
                        bf::memcpy_d(xd.data(), d_rhs_dense.data(),
                                     static_cast<size_t>(nd) * sizeof(double),
                                     cudaMemcpyDeviceToHost);
                    } catch (const std::exception& ex) {
                        ok = false;
                        std::cout << "[Direct] Zerlegung fehlgeschlagen: " << ex.what() << "\n";
                    }

                    if (ok) {
                        // Residuum der direkten Loesung, mit der GESICHERTEN Matrix.
                        double rn = 0.0, bn = 0.0;
                        for (int i = 0; i < nd; ++i) {
                            double ax = 0.0;
                            for (int j = 0; j < nd; ++j)
                                ax += A_copy[static_cast<size_t>(j) * nd + i] * xd[j];
                            const double ri = d_rhs_coupled[i] - ax;
                            rn += ri * ri;
                            bn += d_rhs_coupled[i] * d_rhs_coupled[i];
                        }
                        rn = std::sqrt(rn); bn = std::sqrt(bn);
                        std::cout << "[Direct] Residuum der direkten Loesung: "
                                  << (bn > 0.0 ? rn / bn : rn) << "\n";

                        // -----------------------------------------------------
                        // Die WAHRE Konditionszahl
                        // -----------------------------------------------------
                        //
                        // Der Ritz-Bericht liefert nur eine UNTERE Schranke: Arnoldi
                        // naehert zuerst die aeusseren Eigenwerte an, ganze Cluster
                        // kleiner Eigenwerte koennen unentdeckt bleiben. Er meldete
                        // kappa >= 1.9e+03 und daraus "44 Iterationen je Dekade" --
                        // gemessen waren es rund 2500.
                        //
                        // Hier wird kappa_2 = sigma_max / sigma_min tatsaechlich
                        // gemessen. Beide ueber Potenzmethoden auf A^T A bzw.
                        // (A^T A)^-1. Letzteres braucht eine TRANSPONIERTE Loesung,
                        // die DenseLU nicht anbietet -- deshalb wird A^T einmal
                        // explizit aufgestellt und mitzerlegt. Kostet eine zweite
                        // Zerlegung und nochmals so viel Speicher; bei den Netzen,
                        // fuer die --direct gedacht ist, ist das vertretbar.
                        //
                        // Ergebnis am Netz n=12: kappa_2 = 1.07e+06, sigma_max = 155,
                        // sigma_min = 1.45e-04.
                        //
                        // Zur Deutung, mit einer widerlegten Zwischenvermutung: der
                        // Materialkontrast sigma/dt zu nu betraegt 4.3e+05, und die
                        // Naehe zu 1.07e+06 sah nach Bestaetigung aus. Der
                        // dt-Durchlauf hat das widerlegt. sigma_max bleibt konstant
                        // bei 155, nur sigma_min bewegt sich, und zwar gegenlaeufig:
                        //
                        //   dt=1e-4  sigma_min=1.45e-04  kappa=1.07e+06
                        //   dt=1e-3  sigma_min=5.89e-05  kappa=2.63e+06
                        //   dt=1e-2  sigma_min=7.25e-06  kappa=2.14e+07
                        //
                        // Der Massenterm sigma/dt ist also nicht die URSACHE der
                        // schlechten Konditionierung, sondern das Einzige, was sie
                        // begrenzt. Je groesser dt, desto kleiner der Massenterm,
                        // desto naeher kommt der Operator dem reinen Curl-Curl -- und
                        // der ist singulaer.
                        //
                        // sigma_min wird also vom nichtleitenden Gebiet bestimmt, wo
                        // es keinen Massenterm gibt. Das ist genau die Konstellation,
                        // fuer die AMS gebaut ist.
                        {
                            std::vector<double> At(static_cast<size_t>(nd) * nd);
                            for (int j = 0; j < nd; ++j)
                                for (int i = 0; i < nd; ++i)
                                    At[static_cast<size_t>(i) * nd + j] =
                                        A_copy[static_cast<size_t>(j) * nd + i];

                            // Wie oben: Geraetekopie, und sie muss leben,
                            // solange luT benutzt wird.
                            DeviceArray<double> d_At(static_cast<size_t>(nd) * nd);
                            bf::memcpy_d(d_At.data(), At.data(),
                                         static_cast<size_t>(nd) * nd * sizeof(double),
                                         cudaMemcpyHostToDevice);
                            bf::DenseLU luT;
                            bool okT = true;
                            try { luT.factorize(d_At.data(), nd); }
                            catch (const std::exception& ex) {
                                okT = false;
                                std::cout << "[Direct] Zerlegung von A^T fehlgeschlagen: "
                                          << ex.what() << "\n";
                            }

                            if (okT) {
                                std::mt19937 rng(4242u);
                                std::uniform_real_distribution<double> uni(-1.0, 1.0);
                                std::vector<double> v(nd), w(nd);
                                DeviceArray<double> d_w_dev(static_cast<size_t>(nd));

                                auto normalize = [&](std::vector<double>& u) {
                                    double sq = 0.0;
                                    for (double t : u) sq += t * t;
                                    sq = std::sqrt(sq);
                                    if (sq > 0.0) for (double& t : u) t /= sq;
                                    return sq;
                                };
                                auto mul = [&](const std::vector<double>& in,
                                               std::vector<double>& out) {
                                    std::fill(out.begin(), out.end(), 0.0);
                                    for (int j = 0; j < nd; ++j) {
                                        const double xj = in[j];
                                        if (xj == 0.0) continue;
                                        const double* cj = A_copy.data() + static_cast<size_t>(j) * nd;
                                        for (int i = 0; i < nd; ++i) out[i] += cj[i] * xj;
                                    }
                                };
                                auto mulT = [&](const std::vector<double>& in,
                                                std::vector<double>& out) {
                                    for (int j = 0; j < nd; ++j) {
                                        const double* cj = A_copy.data() + static_cast<size_t>(j) * nd;
                                        double sum = 0.0;
                                        for (int i = 0; i < nd; ++i) sum += cj[i] * in[i];
                                        out[j] = sum;
                                    }
                                };

                                // sigma_max: Potenzmethode auf A^T A
                                for (int i = 0; i < nd; ++i) v[i] = uni(rng);
                                normalize(v);
                                double smax = 0.0;
                                for (int it = 0; it < 80; ++it) {
                                    mul(v, w); mulT(w, v);
                                    smax = std::sqrt(normalize(v));
                                }

                                // sigma_min: Potenzmethode auf (A^T A)^-1 = A^-1 A^-T
                                for (int i = 0; i < nd; ++i) v[i] = uni(rng);
                                normalize(v);
                                double smin = 0.0;
                                for (int it = 0; it < 80; ++it) {
                                    w = v;
                                    // w ist ein Host-Vektor; solve() erwartet
                                    // im GPU-Modus einen Geraetezeiger.
                                    bf::memcpy_d(d_w_dev.data(), w.data(),
                                                 static_cast<size_t>(nd) * sizeof(double),
                                                 cudaMemcpyHostToDevice);
                                    luT.solve(d_w_dev.data(), 1);   // A^-T v
                                    lu.solve(d_w_dev.data(), 1);    // A^-1 (A^-T v)
                                    bf::memcpy_d(w.data(), d_w_dev.data(),
                                                 static_cast<size_t>(nd) * sizeof(double),
                                                 cudaMemcpyDeviceToHost);
                                    v = w;
                                    const double g = normalize(v);
                                    smin = (g > 0.0) ? 1.0 / std::sqrt(g) : 0.0;
                                }

                                std::cout << "[Direct] Konditionszahl (GEMESSEN, nicht geschaetzt):\n"
                                          << "[Direct]   sigma_max = " << smax
                                          << ", sigma_min = " << smin
                                          << ", kappa_2 = " << (smin > 0.0 ? smax / smin : 0.0) << "\n"
                                          << "[Direct]   Der Ritz-Bericht meldete kappa >= 1.9e+03 als UNTERE\n"
                                          << "[Direct]   Schranke. Liegt der gemessene Wert um Groessenordnungen\n"
                                          << "[Direct]   darueber, ist die Stagnation erklaert -- und zwar\n"
                                          << "[Direct]   netzunabhaengig, denn der Materialkontrast sigma/dt zu nu\n"
                                          << "[Direct]   (4.3e+05) steckt in jedem Netz gleichermassen.\n";
                            }
                        }
                        std::cout << "[Direct]   Deutung: ~1e-15 heisst, das System ist "
                                     "loesbar und GMRES scheitert aus einem anderen Grund.\n"
                                     "[Direct]   Bleibt es bei ~1e-2, ist das System "
                                     "INKONSISTENT -- dann ist es kein Loeserproblem.\n";
                    }

                    // ---------------------------------------------------------
                    // --harmonic-study: A(dt) = K + C/dt aufspalten
                    // ---------------------------------------------------------
                    //
                    // TEAM 28 ist vollstaendig linear (mu_r = 1, sigma konstant),
                    // und gesucht ist die Schwebehoehe -- also die zeitgemittelte
                    // Kraft im eingeschwungenen Zustand bei RUHENDER Platte. Ein
                    // harmonischer Ansatz ist damit keine Naeherung, sondern
                    // derselbe Zustand, direkt gerechnet.
                    //
                    // Dafuer werden K und C getrennt. Es braucht drei Abtastungen:
                    // zwei fuer die Aufspaltung, eine DRITTE nur zur Kontrolle.
                    // Mit zwei Matrizen und zwei Unbekannten je Eintrag ist das
                    // System immer exakt loesbar -- ein Rueckbau von A1 und A2
                    // koennte gar nicht fehlschlagen und wuerde insbesondere nicht
                    // merken, wenn die Abtastungen aus verschiedenen Raeumen
                    // stammen.
                    //
                    // WICHTIG: zwischen den Abtastungen darf setup_equilibration()
                    // NICHT erneut laufen. Bleibt D gleich, gilt die Aufspaltung
                    // direkt im Tilde-Raum, denn D(K + C/dt)D = DKD + (DCD)/dt.
                    // Deshalb wird unten nur die FEM-Matrix neu assembliert und
                    // uebergeben, sonst nichts.
                    if (bf::want_harmonic_study()) {
                        const int n_bem = op.get_num_bem_dofs();
                        const int n_fem = nd - n_bem;

                        // Getrennt in Assemblierung und Abtastung: die
                        // Wiederherstellung am Ende braucht nur die
                        // Assemblierung, und die Abtastung bei dt1 gibt es
                        // schon (A_copy). Zusammengefasst waeren das zwei
                        // ueberfluessige Durchlaeufe ueber 4511 Spalten.
                        //
                        // build_implicit_euler_system() schreibt dabei auch
                        // d_rhs neu -- deshalb wird ganz am Ende noch einmal
                        // mit dem ORIGINAL-dt assembliert.
                        auto assemble_at_dt = [&](double dt_x) {
                            TransientSolver::build_implicit_euler_system(
                                mesh.num_hexes, total_dofs, dt_x,
                                d_local_K.data(), d_local_C.data(), d_local_F.data(),
                                mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                                d_A_previous.data(), d_materials.data(),
                                d_local_A_sys.data(), d_rhs.data());
                            bf::memset_d(csr_matrix.values.data(), 0,
                                         static_cast<size_t>(csr_matrix.nnz) * sizeof(double));
                            assemble_csr_matrix_gpu(
                                mesh.num_hexes, d_local_A_sys.data(),
                                mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                                csr_matrix.row_offsets.data(),
                                csr_matrix.col_indices.data(),
                                csr_matrix.values.data());
                            op.setup_fem_matrix(csr_matrix.num_rows, csr_matrix.nnz,
                                                csr_matrix.row_offsets.data(),
                                                csr_matrix.col_indices.data(),
                                                csr_matrix.values.data());
                            bf::sync();
                        };
                        auto sample_dense = [&](std::vector<double>& out) {
                            out.assign(static_cast<size_t>(nd) * nd, 0.0);
                            std::vector<double> ee(nd, 0.0), cc(nd, 0.0);
                            // Wie bei --direct: op.apply() will Geraetezeiger.
                            DeviceArray<double> d_ee(static_cast<size_t>(nd));
                            DeviceArray<double> d_cc(static_cast<size_t>(nd));
                            for (int j = 0; j < nd; ++j) {
                                std::fill(ee.begin(), ee.end(), 0.0);
                                ee[j] = 1.0;
                                bf::memcpy_d(d_ee.data(), ee.data(),
                                             static_cast<size_t>(nd) * sizeof(double),
                                             cudaMemcpyHostToDevice);
                                op.apply(d_ee.data(), d_cc.data());
                                bf::memcpy_d(cc.data(), d_cc.data(),
                                             static_cast<size_t>(nd) * sizeof(double),
                                             cudaMemcpyDeviceToHost);
                                std::copy(cc.begin(), cc.end(),
                                          out.begin() + static_cast<size_t>(j) * nd);
                                if (nd >= 2000 && (j % 1000) == 0)
                                    std::cout << "[Harmonisch]     Spalte " << j
                                              << " von " << nd << "\n";
                            }
                        };

                        // dt2 GROSS: K = A - C/dt ist eine Differenz zweier grosser
                        // Zahlen, und der Stellenverlust ist ~log10(||C/dt||/||K||).
                        // Bei sigma/dt = 3.4e+11 gegen nu = 8e+05 waeren das aus der
                        // Produktivabtastung fuenf bis sechs Stellen; bei dt = 1 s
                        // ist C/dt verschwindend und A praktisch schon K selbst.
                        const double dt1 = dt;
                        const double dt2 = 1.0;
                        const double dt3 = 10.0 * dt;   // nur Kontrolle
                        std::vector<double> H2, H3;

                        // Abtastung 1 liegt bereits vor: A_copy ist genau die
                        // Matrix bei dt1, im selben Raum abgetastet.
                        const std::vector<double>& H1 = A_copy;

                        std::cout << "\n[Harmonisch] Abtastung 1 von 3 (dt = " << dt1
                                  << "): uebernommen aus --direct.\n";
                        std::cout << "[Harmonisch] Abtastung 2 von 3 (dt = " << dt2 << ") ...\n";
                        assemble_at_dt(dt2); sample_dense(H2);
                        std::cout << "[Harmonisch] Abtastung 3 von 3 (dt = " << dt3
                                  << ", nur Kontrolle) ...\n";
                        assemble_at_dt(dt3); sample_dense(H3);

                        const bf::KCSplit split =
                            bf::split_K_and_C(H1, dt1, H2, dt2, H3, dt3, nd, n_fem);
                        std::cout << "[Harmonisch] Vorhersagefehler fuer die Kontrollabtastung: "
                                  << split.rel_error_prediction << "\n"
                                  << "[Harmonisch] ||C|| ausserhalb des FEM-Blocks (relativ): "
                                  << split.c_outside_fem_block << "\n"
                                  << "[Harmonisch] ||C/dt||/||K|| (Stellenverlust): "
                                  << split.split_conditioning << "\n";

                        if (!split.valid) {
                            // Notbremse. Ohne sie kaeme hier eine plausibel
                            // aussehende Zahl heraus, die nichts bedeutet.
                            std::cout << "[Harmonisch] ABBRUCH: das Modell A(dt) = K + C/dt "
                                         "wird von der Kontrollabtastung nicht bestaetigt.\n"
                                         "[Harmonisch]   Wahrscheinlichste Ursache: zwischen den "
                                         "Abtastungen hat sich die\n"
                                         "[Harmonisch]   Aequilibrierung geaendert, die Matrizen "
                                         "liegen also in verschiedenen\n"
                                         "[Harmonisch]   Raeumen. Die Aufspaltung waere dann "
                                         "bedeutungslos.\n";
                        } else {
                            const double omega = 2.0 * 3.14159265358979323846
                                               * bf::harmonic_frequency();
                            bf::BlockPcStudyOptions hopt;
                            hopt.gmres_tol      = 1e-8;
                            hopt.gmres_max_iter = std::min(4000, 4 * nd);
                            hopt.gmres_restart  = std::max(200, bf::gmres_restart());
                            hopt.measure_kappa  = (nd <= 6000);
                            try {
                                bf::study_harmonic_system(H1, split, omega, dt1,
                                                          nd, n_fem, hopt, std::cout);
                            } catch (const std::exception& ex) {
                                std::cout << "[Harmonisch] Abgebrochen: " << ex.what() << "\n";
                            }

                            // -------------------------------------------------
                            // --pc-study zusammen mit --harmonic-study:
                            // die Deckenmessung auf (W+T) = K + omega*C
                            // -------------------------------------------------
                            //
                            // Das ist der Operator, den der C-to-R-Zweig INNEN
                            // loesen muss, und an ihm haengt der ganze Ansatz:
                            // die Genauigkeitsreihe verlangt dort 1e-03, und der
                            // heutige Loeser erreicht das auf einem vergleichbaren
                            // System nicht.
                            //
                            // (W+T) ist ein gekoppeltes FEM-BEM-System derselben
                            // Bauart wie A(dt) -- dieselbe Blockaufteilung, nur
                            // mit dt = 1/omega. Die Studie ist deshalb unveraendert
                            // anwendbar und direkt mit dem transienten Ergebnis
                            // vergleichbar.
                            //
                            // Und hier lohnt sich mehr als dort: je Hoehe ist (W+T)
                            // FEST und wird rund 160-fach benutzt. Eine Variante
                            // mit teurem Aufbau, die in der Transiente ausschied
                            // (dort aendert sich die Matrix in jedem der 1000
                            // Schritte), ist hier vertretbar.
                            if (bf::want_pc_study()) {
                                std::vector<double> WpT(static_cast<size_t>(nd) * nd);
                                for (size_t i = 0; i < WpT.size(); ++i)
                                    WpT[i] = split.K[i] + omega * split.C[i];

                                std::cout << "\n[PC-Studie] Angewandt auf (W+T) = K + omega*C, "
                                             "den INNEREN Operator des\n"
                                             "[PC-Studie] C-to-R-Zweigs (transiente Bauart mit "
                                             "dt = 1/omega = "
                                          << (1.0 / omega) << " s).\n";

                                std::vector<double> rhs_w(d_rhs_coupled.data(),
                                                          d_rhs_coupled.data() + nd);
                                try {
                                    const bf::BlockPcStudyResult wres =
                                        bf::study_block_preconditioners(WpT, nd, n_fem,
                                                                        rhs_w, hopt, std::cout);
                                    bf::report_block_pc_study(wres, std::cout);
                                } catch (const std::exception& ex) {
                                    std::cout << "[PC-Studie] Abgebrochen: " << ex.what() << "\n";
                                }
                            } else {
                                std::cout << "\n[Harmonisch] Hinweis: zusammen mit --pc-study "
                                             "laeuft die Deckenmessung\n"
                                             "[Harmonisch] auf (W+T) = K + omega*C -- dem Operator, "
                                             "an dem der ganze Ansatz haengt.\n";
                            }
                        }

                        // Zustand wiederherstellen: Matrix UND d_rhs gehoeren
                        // zum Original-dt, und setup_fem_matrix() hat die
                        // ILU-Faktorisierung fuer ungueltig erklaert (ilu_ready
                        // = false). Ohne beides rechnete der Produktivlauf
                        // danach mit dt = 10*dt weiter -- und zwar ohne jede
                        // Meldung.
                        assemble_at_dt(dt1);
                        op.compute_ilu();
                        bf::sync();
                        std::cout << "[Harmonisch] Zustand auf dt = " << dt1
                                  << " zurueckgesetzt, ILU neu berechnet.\n";

                        // ---------------------------------------------------
                        // Und hier ist Schluss.
                        // ---------------------------------------------------
                        //
                        // --harmonic-study ist eine DIAGNOSE, kein Zeitschritt.
                        // Sie haengt am --direct-Zweig, und der liegt innerhalb
                        // der Zeitschleife -- ohne diesen Ausstieg liefe danach
                        // der vollstaendige transiente Lauf ueber 1000 Schritte
                        // weiter. Genau das, was der harmonische Ansatz ersetzen
                        // soll, und Stunden fuer nichts.
                        //
                        // Der Zustand ist oben trotzdem wiederhergestellt: die
                        // Studie soll auch dann nichts hinterlassen, wenn dieser
                        // Ausstieg spaeter einmal entfaellt.
                        std::cout << "\n[Harmonisch] Studie abgeschlossen -- kein transienter "
                                     "Lauf.\n"
                                  << "[Harmonisch] --harmonic-study ist eine Diagnose. Fuer den "
                                     "Produktivlauf die\n"
                                  << "[Harmonisch] Option weglassen.\n";
                        return 0;
                    }

                    // ---------------------------------------------------------
                    // --pc-study: die Deckenmessung fuer den BEM-Block
                    // ---------------------------------------------------------
                    //
                    // Laeuft auf A_copy, also der Matrix, die oben ohnehin schon
                    // spaltenweise abgetastet wurde -- kein zweiter Abtastlauf.
                    // A_copy ist im Tilde-Raum (aequilibriert), falls
                    // setup_equilibration() gelaufen ist; das ist genau der Raum,
                    // in dem GMRES rechnet, und damit der richtige.
                    //
                    // Zur Blockgrenze: op.apply() legt die ersten
                    // op.get_total_dofs() - op.get_num_bem_dofs() Eintraege als
                    // FEM-Block ab, den Rest als BEM-Block. Die Grenze wird
                    // deshalb aus dem Operator abgeleitet und nicht aus einer
                    // zweiten Quelle -- eine um eins verschobene Blockgrenze
                    // waere hier nicht als Fehler zu erkennen, sondern nur als
                    // ein etwas anderes Ergebnis.
                    if (bf::want_pc_study()) {
                        const int n_bem = op.get_num_bem_dofs();
                        if (n_bem <= 0) {
                            std::cout << "[PC-Studie] Keine BEM-Kopplung aktiv -- die Studie "
                                         "vergleicht Bloecke, von denen es hier nur einen "
                                         "gibt. Uebersprungen.\n";
                        } else {
                            const int n_fem = nd - n_bem;
                            std::vector<double> rhs(d_rhs_coupled.data(),
                                                    d_rhs_coupled.data() + nd);
                            try {
                                bf::BlockPcStudyOptions pcopt;
                                pcopt.gmres_tol      = 1e-8;
                                pcopt.gmres_max_iter = std::min(4000, 4 * nd);
                                // Restart-Laenge: mindestens 200, auch wenn der
                                // Produktivlauf kuerzer restartet. Die Studie misst
                                // eine DECKE -- was ein Vorkonditionierer hoechstens
                                // hergibt --, und convergence-findings.tex zeigt, dass die
                                // Restart-Laenge allein den Faktor 2,6 ausmacht.
                                // Mit Restart 50 misst man die Restart-Laenge mit
                                // und nicht mehr den Vorkonditionierer. Volles GMRES
                                // waere sauberer, ist aber bei nd = 4511 und 4000
                                // Iterationen nicht speicherbar (144 GiB Basis).
                                pcopt.gmres_restart  = std::max(200, bf::gmres_restart());
                                // kappa_2(M^-1 A) je Variante kostet eine weitere
                                // dichte Zerlegung PLUS eine n^3-Anwendung von M^-1
                                // auf A. Ab etwa 6000 Unbekannten ueberwiegt der
                                // Aufwand den Erkenntnisgewinn -- die
                                // Iterationszahlen sagen dann dasselbe.
                                pcopt.measure_kappa  = (nd <= 6000);

                                const bf::BlockPcStudyResult pcres =
                                    bf::study_block_preconditioners(A_copy, nd, n_fem,
                                                                    rhs, pcopt, std::cout);
                                bf::report_block_pc_study(pcres, std::cout);
                            } catch (const std::exception& ex) {
                                std::cout << "[PC-Studie] Abgebrochen: " << ex.what() << "\n";
                            }
                        }
                    }
                }
            }

            // =================================================================
            // --harmonic: den eingeschwungenen Zustand loesen
            // =================================================================
            //
            // TEAM 28 ist vollstaendig linear (mu_r = 1 ueberall, sigma
            // konstant). Gesucht ist die Schwebehoehe, also die
            // ZEITGEMITTELTE Kraft im eingeschwungenen Zustand bei ruhender
            // Platte. Der harmonische Ansatz ist deshalb keine Naeherung an
            // die Transiente, sondern derselbe Zustand -- eine Loesung statt
            // 1000 Zeitschritten.
            if (bf::want_harmonic_solve()) {
                const double omega = 2.0 * 3.14159265358979323846
                                   * bf::harmonic_frequency();
                const int    nd_h  = total_dofs + static_cast<int>(bem_map.num_bem_dofs);

                // (W+T) = K + omega*C ist GENAU das, was
                // build_implicit_euler_system() bei dt = 1/omega liefert, denn
                // dort ist A_sys = K + C/dt. Keine Differenzbildung noetig.
                //
                // Fuer C allein braucht es eine zweite Assemblierung mit sehr
                // grossem dt (dort ist C/dt vernachlaessigbar, A_sys also
                // praktisch K) und die Differenz auf den CSR-WERTEN. Das
                // Sparsity-Muster ist in beiden Faellen dasselbe, die Differenz
                // also eintragsweise.
                auto assemble_at = [&](double dt_x) {
                    // Die lokalen Elementmatrizen MUESSEN hier neu berechnet
                    // werden. Vorher stand dieser Aufruf nicht da, und
                    // assemble_at hat d_local_K/d_local_C weiterverwendet, die
                    // einmal weiter oben (Schritt b des Zeitschritts) entstanden
                    // sind. Jede Materialaenderung danach war damit wirkungslos
                    // -- und das traf genau die Regularisierung sigma_Luft, die
                    // nach Abschnitt 1 des Plans den Curl-Curl-Nullraum
                    // beseitigen soll. Sie schrieb in d_materials, und danach
                    // las das niemand mehr.
                    //
                    // Folge war nicht ein Abbruch, sondern eine singulaere
                    // Matrix: K ohne sigma_Luft hat auf den Gradienten der 2665
                    // Luft- und 768 Spulenelemente einen Nullraum. cuDSS
                    // zerlegt das mit CUDSS_DATA_INFO = 0, also ohne
                    // Beanstandung, und liefert eine vom Nullraum bestimmte
                    // Loesung -- beobachtet max|A| = 2e+08 statt der erwarteten
                    // Groessenordnung 1e-05, bei einem relativen Residuum von
                    // 0.228, weil der Anteil von b ausserhalb des Bildraums
                    // stehenbleibt.
                    //
                    // Der Kommentar an der Regularisierung warnte wortwoertlich
                    // davor ("Danach waere es wirkungslos -- und genau das
                    // faellt nicht auf"). Die Reihenfolge war richtig, nur hat
                    // assemble_at die Materialien gar nicht mehr gelesen.
                    BF_LAUNCH_1D_PAR(assemble_local_matrices_kernel, mesh.num_hexes,
                        static_cast<int>(mesh.num_hexes), mesh.hex_nodes.data(),
                        mesh.nodes_current.data(),
                        d_materials.data(), mesh.hex_edge_signs.data(),
                        mesh.elem_types.data(),
                        d_local_K.data(), d_local_C.data(), d_local_F.data());
                    bf::sync();

                    TransientSolver::build_implicit_euler_system(
                        mesh.num_hexes, total_dofs, dt_x,
                        d_local_K.data(), d_local_C.data(), d_local_F.data(),
                        mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                        d_A_previous.data(), d_materials.data(),
                        d_local_A_sys.data(), d_rhs.data());
                    bf::memset_d(csr_matrix.values.data(), 0,
                                 static_cast<size_t>(csr_matrix.nnz) * sizeof(double));
                    assemble_csr_matrix_gpu(
                        mesh.num_hexes, d_local_A_sys.data(),
                        mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                        csr_matrix.row_offsets.data(), csr_matrix.col_indices.data(),
                        csr_matrix.values.data());
                    bf::sync();
                };

                // CSRMatrix::allocate() ist in fem.cuh zwar DEKLARIERT, aber
                // nirgends definiert -- eine tote Deklaration. Das Muster
                // entsteht deshalb ueber build_csr_structure(), also genau so
                // wie beim Hauptsystem: gleiches Netz, gleiches Muster, und
                // damit ist die eintragsweise Differenz unten zulaessig.
                CSRMatrix csr_C;
                build_csr_structure(mesh, csr_C);
                if (csr_C.nnz != csr_matrix.nnz || csr_C.num_rows != csr_matrix.num_rows) {
                    // Kann nicht passieren, solange beide aus demselben Netz
                    // stammen -- aber wenn doch, waere die eintragsweise
                    // Differenz stillschweigend falsch.
                    std::cout << "[Harmonisch] ABBRUCH: CSR-Muster weichen ab ("
                              << csr_C.num_rows << "/" << csr_C.nnz << " gegen "
                              << csr_matrix.num_rows << "/" << csr_matrix.nnz << ").\n";
                    return 1;
                }

                const double dt_w = 1.0 / omega;

                // ===========================================================
                // FEM-direkt (--fem-direct): reines FEM + Sparse-Direktloeser
                // ===========================================================
                //
                // AP8 aus docs/design/fem-direct-plan.md. Dieser Zweig verlaesst die
                // Funktion; er teilt mit dem iterativen harmonischen Pfad nur
                // die Assemblierung.
                //
                // WARUM MATRIX UND RECHTE SEITE AUS DEMSELBEN AUFRUF KOMMEN
                //
                // Der iterative Pfad rechnet im aequilibrierten Tilde-Raum:
                // op.setup_equilibration() lief weiter oben, D bleibt stehen,
                // und die rechte Seite wurde dort passend skaliert. Eine
                // skalierte rechte Seite mit einer unskalierten Matrix zu
                // mischen waere lautlos falsch.
                //
                // Dieser Zweig umgeht die Frage, statt sie zu beantworten: er
                // nimmt csr_matrix UND d_rhs aus EINEM assemble_at()-Aufruf.
                // build_implicit_euler_system() schreibt beide frisch und
                // unskaliert, sie sind also konsistent, ganz unabhaengig
                // davon, was op mit D macht. Eine Aequilibrierung braucht der
                // Direktloeser ohnehin nicht.
                if (bf::want_fem_direct()) {
                    std::cout << "\n[FEM-direkt] f = " << bf::harmonic_frequency()
                              << " Hz, omega = " << omega << "\n";

                    if (!fem_direct::SparseDirectSolver::available()) {
                        std::cout << "[FEM-direkt] ABBRUCH: ohne cuDSS gebaut. "
                                     "Neu konfigurieren mit -DFEMBEM_WITH_CUDSS=ON, "
                                     "siehe BUILD.md.\n";
                        return 1;
                    }
                    if (bf::on_cpu()) {
                        std::cout << "[FEM-direkt] ABBRUCH: cuDSS gibt es nicht fuer "
                                     "den CPU-Pfad. --cpu und --fem-direct sind "
                                     "unvertraeglich.\n";
                        return 1;
                    }
                    if (!mesh.is_pure_hex() && !disable_bem_for_debug) {
                        // Kann eigentlich nicht auftreten, weil der BEM-Aufbau
                        // weiter oben schon abbricht -- aber wenn doch, soll
                        // die Meldung hier stehen und nicht dort.
                        std::cout << "[FEM-direkt] ABBRUCH: BEM ist auf einem "
                                     "gemischten Netz nicht moeglich. "
                                     "--fem-direct zusammen mit nobem verwenden.\n";
                        return 1;
                    }

                    // --- 0. Quellterm auf die AMPLITUDE bringen --------------
                    // Die rechte Seite war eine Zeitprobe. update_state(t)
                    // setzt i(t) = I * sin(2*pi*f*t), und assemble_at wertet
                    // den Quellterm damit zum LAUFENDEN Zeitpunkt aus -- beim
                    // ersten Schritt also mit sin(2*pi*50*1e-4) = 0.0314.
                    // Gemessen: ||b|| = 819 statt 23434. Ein Phasorsystem
                    // K + j*omega*C braucht dort die Amplitude; sonst haengt
                    // die Loesung und damit jede Kraft am gewaehlten dt.
                    //
                    // Ausgewertet wird deshalb im Maximum, t = T/4. Das ist
                    // eine Verschiebung des Zeitnullpunkts um T/4, also die
                    // Konvention i(t) = I * cos(omega*t) mit reeller rechter
                    // Seite und b_i = 0. Fuer die zeitgemittelte Kraft ist das
                    // gleichgueltig: sie ist unter einer gemeinsamen
                    // Phasendrehung aller Phasoren invariant.
                    //
                    // REIHENFOLGE: das muss VOR der Regularisierung stehen.
                    // Danach wuerde d_materials[h] = coil.get_properties()
                    // das kappa der Spulen wieder auf 0 setzen -- die Spulen
                    // sind stromgetrieben und tragen kappa = 0 -- und damit
                    // waere der Nullraum auf 768 Elementen zurueck.
                    {
                        if (std::fabs(TEAM28_F0 - bf::harmonic_frequency()) > 1e-9) {
                            std::cout << "[FEM-direkt] ABBRUCH: die Quellfrequenz der "
                                         "Spulen (" << TEAM28_F0 << " Hz) weicht von der "
                                         "harmonischen Frequenz (" << bf::harmonic_frequency()
                                      << " Hz) ab. Dann passt omega*C nicht zur Anregung, "
                                         "und t = T/4 traefe das Maximum nicht.\n";
                            return 1;
                        }
                        const double t_peak = 0.25 / TEAM28_F0;
                        coil_inner.update_state(t_peak, dt);
                        coil_outer.update_state(t_peak, dt);
                        d_materials.prefetch_to_host();
                        for (int h : coil_inner.get_hex_indices())
                            d_materials[h] = coil_inner.get_properties();
                        for (int h : coil_outer.get_hex_indices())
                            d_materials[h] = coil_outer.get_properties();
                        std::cout << "[FEM-direkt] Quellterm auf die Amplitude gesetzt "
                                     "(t = T/4 = " << t_peak << " s): J_innen = "
                                  << coil_inner.get_properties().current_density_x
                                  << ", J_aussen = "
                                  << coil_outer.get_properties().current_density_x
                                  << " A/m^2\n";
                    }

                    // --- 1. Regularisierung, VOR Pruefung und Assemblierung --
                    // Vorher festhalten, WELCHE Elemente kuenstlich leitend
                    // gemacht werden. Danach ist das nicht mehr feststellbar,
                    // weil dann jedes Element kappa > 0 hat -- und genau
                    // darauf beruht die Nichtleiter-Abfrage in
                    // harmonic_lorentz_body. Ohne diese Maske liefen Luft und
                    // Spulen ins Kraftintegral mit, und der Beitrag des
                    // Luftgebiets ist EPS-UNABHAENGIG (sigma_Luft faellt mit
                    // eps, der Scheingradient in A_i waechst mit 1/eps, das
                    // Produkt bleibt) und liegt bei etwa 1,9 N gegen eine
                    // gesuchte Kraft von rund 1 N.
                    DeviceArray<unsigned char> d_regularized(mesh.num_hexes);
                    for (size_t h = 0; h < mesh.num_hexes; ++h)
                        d_regularized[h] = (d_materials[h].kappa > 0.0) ? 0 : 1;

                    // sigma_Luft geht ueber die Massenmatrix C in das System
                    // ein, muss also vor assemble_at() stehen. Danach waere es
                    // wirkungslos -- und genau das faellt nicht auf.
                    //
                    // Sie muss auch vor preflight_fem_direct() stehen. Vorher
                    // lief die Pruefung zuerst und beanstandete kappa = 0 auf
                    // 3433 Elementen -- also genau die Bedingung, die dieser
                    // Block beseitigt. Damit war der Aufruf unerreichbar und
                    // --fem-direct brach immer ab, obwohl beide Teile
                    // vorhanden und richtig waren.
                    //
                    // In dieser Reihenfolge wird die kappa-Pruefung im
                    // Preflight zur Nachbedingung: beanstandet sie danach
                    // noch etwas, hat die Regularisierung ein Gebiet nicht
                    // erfasst. Das ist die nuetzlichere Aussage.
                    // --- 1. Nullraum beseitigen: Regularisierung ODER Eichung
                    // Beides zugleich waere sinnlos: die Eichung waehlt ihren
                    // Baum ueber kappa == 0, und nach der Regularisierung hat
                    // kein Element mehr kappa == 0 -- der Baum bliebe leer.
                    std::vector<char> gauged_edges;   // leer = keine Eichung

                    if (bf::fem_direct_gauge()) {
                        // Mit std::endl, nicht "\n": stuerzt der Baumbau oder
                        // die Generatorensuche ab, sagt das Protokoll sonst
                        // nichts darueber, weil PowerShell die Ausgabe eines
                        // nativen Programms immer durch eine Pipe leitet und
                        // damit vollpuffert.
                        std::cout << "[FEM-direkt] Eichmodus: Baum-Kobaum ueber "
                                  << mesh.num_edges << " Kanten, "
                                  << mesh.num_nodes << " Knoten" << std::endl;
                        d_materials.prefetch_to_host();
                        const GaugeInfo g = build_tree_cotree_gauge(
                            mesh, d_materials.data(), 0.0);
                        gauged_edges = g.is_gauged;

                        size_t n_gauged = 0;
                        for (char c : gauged_edges) if (c) ++n_gauged;
                        std::cout << "[FEM-direkt] Baum-Kobaum-Eichung statt "
                                     "Regularisierung: " << n_gauged
                                  << " von " << mesh.num_edges
                                  << " Kanten geeicht" << std::endl;

                        // Die Generatorensuche laeuft hier ERSTMALS auf diesem
                        // Pfad, und ihre Zahl ist eine Messung, keine Zierde.
                        //
                        // Im BEM-Fall ist das nichtleitende Gebiet die Spule
                        // selbst -- ein Volltorus, eine nicht zusammenziehbare
                        // Schleife, ein Generator je Spule. Mit mitvernetzter
                        // Luft ist es Kasten minus Platte. Ist die Platte eine
                        // Vollscheibe, ist das einfach zusammenhaengend und die
                        // Zahl ist null. Hat das Plattennetz ein Loch auf der
                        // Achse -- check_map meldet r_min = 0,5 mm --, ist die
                        // Platte topologisch ein Volltorus und es gibt genau
                        // einen Generator. Ein solches Netz entspricht nicht dem
                        // Benchmark: TEAM 28 hat KEIN Loch in der Platte (ein
                        // Loch steht im Paper nur als Ausblick).
                        // Die Generatorensuche laeuft als AUSKUNFT, nicht als
                        // Bedingung -- und sie darf den Lauf nicht abbrechen.
                        //
                        // Sie ist fuer den BEM-Fall gebaut, wo das
                        // nichtleitende Gebiet die Spule ist: eine Komponente,
                        // ein Volltorus, ein Generator. Hier ist es Luft plus
                        // Spulen, das zerfaellt je nach Netz in mehrere
                        // Komponenten, und in kleinen davon konvergiert ihr
                        // inneres CG nicht (gemessen: Kozykel-Rest 0,21 auf
                        // einem synthetischen Netz). Das ist eine Schwaeche der
                        // Suche in diesem Regime, keine Aussage ueber die
                        // Topologie.
                        //
                        // ENTSCHEIDEND ist ohnehin das Residuum unten. Ist b1
                        // irgendwo groesser null, bleibt die geeichte Matrix
                        // singulaer, und das schlaegt dort durch. Auf
                        // CUDSS_DATA_INFO darf man sich dabei NICHT verlassen:
                        // eine singulaere Matrix ist auf diesem Pfad schon
                        // einmal mit info = 0 zerlegt worden.
                        try {
                            const auto coh = find_cohomology_generators(
                                mesh, g, d_materials.data(), 0.0, true);
                            std::cout << "[FEM-direkt] Kohomologie: "
                                      << coh.generators.size() << " Generator(en), "
                                      << coh.num_simply_connected
                                      << " einfach zusammenhaengende Komponente(n)"
                                      << std::endl;
                            if (!coh.generators.empty()) {
                                std::cout << "[FEM-direkt] WARNUNG: das nichtleitende "
                                             "Gebiet ist NICHT einfach "
                                             "zusammenhaengend. Die Baumeichung allein "
                                             "genuegt dann nicht, und die Generatoren "
                                             "sind DICHTE Vektoren -- als "
                                             "Koordinatenmaske in eine CSR-Matrix "
                                             "einarbeiten laesst sich das nicht. "
                                             "Erwarte ein schlechtes Residuum."
                                          << std::endl;
                            }
                        } catch (const std::exception& e) {
                            std::cout << "[FEM-direkt] Generatorensuche nicht "
                                         "verwertbar: " << e.what() << std::endl;
                            std::cout << "[FEM-direkt]   Kein Abbruch -- die Suche ist "
                                         "fuer den BEM-Fall gebaut (eine Komponente, "
                                         "ein Volltorus) und in diesem Regime "
                                         "unzuverlaessig. Das Residuum unten "
                                         "entscheidet: bleibt es bei 1e-14, war die "
                                         "Baumeichung ausreichend."
                                      << std::endl;
                        }
                    } else {
                        const double sigma_ref = plate.get_properties().kappa;
                        const double eps = bf::fem_direct_eps();
                        const size_t changed = fem_direct::regularize_nonconducting_regions(
                            d_materials.data(), mesh, sigma_ref, eps);
                        std::cout << "[FEM-direkt] Regularisierung: " << changed
                                  << " Elemente auf sigma = " << eps << " * " << sigma_ref
                                  << " = " << eps * sigma_ref << " gesetzt"
                                  << " (--fem-direct-eps)\n";
                    }

                    // --- 2. Vorbedingungen -----------------------------------
                    double truncation_min = 0.0;
                    {
                        // Der letzte Parameter sagt, dass der Nullraum
                        // ANDERS beseitigt wird als durch Regularisierung --
                        // sonst beanstandet der Preflight im Eichmodus jedes
                        // Luftelement, richtigerweise fuer den regularisierten
                        // Weg und falsch fuer diesen.
                        const auto pre = fem_direct::preflight_fem_direct(
                            mesh, d_materials.data(), 0, 2.0,
                            bf::fem_direct_gauge());
                        std::cout << pre.to_string();
                        truncation_min = pre.truncation_factor_min;
                        if (!pre.ok()) {
                            std::cout << "[FEM-direkt] Nichts gerechnet.\n";
                            return 1;
                        }
                    }
                    d_materials.prefetch_to_device();
                    bf::sync();

                    // --- 3. K und C aus zwei Assemblierungen ----------------
                    // Erst K + omega*C (das ist auch der Aufruf, dessen d_rhs
                    // wir behalten), dann K allein.
                    assemble_at(dt_w);
                    std::vector<double> b_r(static_cast<size_t>(total_dofs));
                    bf::memcpy_d(b_r.data(), d_rhs.data(),
                                 b_r.size() * sizeof(double), cudaMemcpyDeviceToHost);
                    std::vector<double> h_wt(static_cast<size_t>(csr_matrix.nnz));
                    bf::memcpy_d(h_wt.data(), csr_matrix.values.data(),
                                 h_wt.size() * sizeof(double), cudaMemcpyDeviceToHost);

                    // A_previous muss null sein, sonst enthaelt d_rhs den Term
                    // (C/dt)*A_prev und ist nicht die harmonische rechte Seite.
                    {
                        std::vector<double> h_prev(static_cast<size_t>(total_dofs));
                        bf::memcpy_d(h_prev.data(), d_A_previous.data(),
                                     h_prev.size() * sizeof(double), cudaMemcpyDeviceToHost);
                        double m = 0.0;
                        for (double v : h_prev) m = std::max(m, std::fabs(v));
                        if (m > 0.0) {
                            std::cout << "[FEM-direkt] ABBRUCH: A_previous ist nicht null "
                                         "(max " << m << "). Dann enthaelt die rechte Seite "
                                         "den Term (C/dt)*A_prev und ist nicht die "
                                         "harmonische Quelle.\n";
                            return 1;
                        }
                    }

                    assemble_at(1e30);
                    std::vector<double> h_K(static_cast<size_t>(csr_matrix.nnz));
                    bf::memcpy_d(h_K.data(), csr_matrix.values.data(),
                                 h_K.size() * sizeof(double), cudaMemcpyDeviceToHost);

                    std::vector<double> h_C(h_K.size());
                    for (size_t i = 0; i < h_K.size(); ++i)
                        h_C[i] = (h_wt[i] - h_K[i]) / omega;

                    // Wirkt die Regularisierung? Ohne sigma_Luft tragen nur die
                    // 259 Plattenelemente zu C bei, mit ihr alle 3692. Der
                    // Quotient max|omega*C| / max|K| ist deshalb die billigste
                    // Kennzahl dafuer, dass sigma_Luft wirklich in der Matrix
                    // steht und nicht bloss in d_materials -- der Fehler, der
                    // hier eine singulaere Matrix erzeugt hat, ohne dass
                    // irgendeine Ausgabe ihn zeigte.
                    {
                        double kmax = 0.0, cmax = 0.0;
                        for (size_t i = 0; i < h_K.size(); ++i) {
                            kmax = std::max(kmax, std::fabs(h_K[i]));
                            cmax = std::max(cmax, std::fabs(h_C[i]));
                        }
                        std::cout << "[FEM-direkt] max|K| = " << kmax
                                  << ", max|omega*C| = " << omega * cmax
                                  << ", Verhaeltnis " << (kmax > 0.0 ? omega * cmax / kmax : 0.0)
                                  << "\n";
                        if (!(cmax > 0.0)) {
                            std::cout << "[FEM-direkt] ABBRUCH: C ist identisch null. "
                                         "Dann ist K der ganze Operator und auf den "
                                         "Gradienten der nichtleitenden Gebiete "
                                         "singulaer. Die Regularisierung hat die "
                                         "Assemblierung nicht erreicht.\n";
                            return 1;
                        }
                    }

                    // --- 4. Muster und Werte der 2n-Blockform ---------------
                    std::vector<int> ro_h(static_cast<size_t>(csr_matrix.num_rows) + 1);
                    std::vector<int> ci_h(static_cast<size_t>(csr_matrix.nnz));
                    bf::memcpy_d(ro_h.data(), csr_matrix.row_offsets.data(),
                                 ro_h.size() * sizeof(int), cudaMemcpyDeviceToHost);
                    bf::memcpy_d(ci_h.data(), csr_matrix.col_indices.data(),
                                 ci_h.size() * sizeof(int), cudaMemcpyDeviceToHost);

                    std::vector<int> ro2, ci2;
                    fem_direct::build_block_structure(csr_matrix.num_rows,
                                                      ro_h.data(), ci_h.data(),
                                                      ro2, ci2);
                    std::vector<double> v2(ci2.size());
                    fem_direct::fill_block_values(csr_matrix.num_rows,
                                                  ro_h.data(), ci_h.data(),
                                                  h_K.data(), h_C.data(), omega,
                                                  fem_direct::BlockForm::General,
                                                  v2.data());
                    std::vector<double> rhs2(ro2.size() - 1);
                    // --- 4b. Gradientenanteil aus dem Quellterm entfernen ---
                    // Muss VOR fill_block_rhs stehen: danach liegt b in der
                    // 2n-Form, und der diskrete Gradient ist dort je Haelfte
                    // getrennt anzuwenden. Auf dem FEM-Kantenraum ist es ein
                    // Aufruf.
                    //
                    // Der Dirichlet-Kantensatz wird HIER schon gebraucht,
                    // obwohl Schritt 5 ihn erst anwendet: ohne ihn laeuft die
                    // Projektion ueber alle Kanten, b wird durch b - G phi
                    // ueberall besetzt (phi loest ein globales
                    // Poisson-Problem), und Schritt 5 setzt die Randkanten
                    // dann auf null zurueck. Genau das brachte die Divergenz
                    // an den Randknoten wieder ein -- gemessen 1,61e-05 in
                    // der 2n-Form, wo die Projektion 5,45e-16 gemeldet hatte.
                    // Zwei Masken, und der Unterschied ist beabsichtigt.
                    //
                    // dirichlet_edges  -- Randkanten. Diese gehen in die
                    //   Projektion, weil dort b vor der Projektion exakt null
                    //   war und danach genullt wird; ohne sie kommt die
                    //   Divergenz zurueck (gemessen 1,6e-05 statt 6e-16).
                    //
                    // fixed_edges      -- Randkanten PLUS Baumkanten. Nur fuer
                    //   die Randbehandlung der Matrix.
                    //
                    // Die Baumkanten sind bewusst NICHT in der
                    // Projektionsmaske. Der Baum spannt das nichtleitende
                    // Gebiet, und dazu gehoeren die stromgetriebenen Spulen --
                    // Baumkanten laufen also mitten durch den Bereich, in dem
                    // b besetzt ist. Sie zu sperren wuerde die Anregung dort
                    // auf null setzen, BEVOR projiziert wird, und die
                    // Projektion verteilte anschliessend einen Rest um.
                    //
                    // Das klassische Baum-Kobaum-Verfahren tut das nicht: es
                    // laesst b unangetastet und loest das reduzierte System
                    // auf den Kobaumkanten; die b-Eintraege auf Baumkanten
                    // werden nicht verwendet. Genau so ist es hier gehalten.
                    //
                    // OFFEN: welche der beiden Varianten das bessere Feld
                    // liefert, ist nicht durch Nachdenken entschieden. Deshalb
                    // meldet der Lauf unten, wieviel Divergenz die Eichmaske
                    // wieder einbringt -- und F_z gegen die 1,02 N des
                    // regularisierten Weges ist die Kontrolle.
                    const auto dirichlet_edges =
                        fem_direct::collect_dirichlet_edges(mesh);
                    std::vector<char> fixed_edges = dirichlet_edges;
                    if (!gauged_edges.empty()) {
                        for (size_t e = 0; e < fixed_edges.size(); ++e)
                            if (gauged_edges[e]) fixed_edges[e] = 1;
                    }

                    if (bf::fem_direct_project()) {
                        const auto pr = fem_direct::project_out_gradient(
                            mesh, b_r.data(), b_r.size(),
                            dirichlet_edges.data(), 1e-13, 20000);
                        size_t n_fixed = 0;
                        for (char c : dirichlet_edges) if (c) ++n_fixed;
                        std::cout << "[FEM-direkt] Quellterm projiziert (frei ausser "
                                  << n_fixed << " Randkanten): ||G^T b||/||b|| "
                                  << pr.rel_before
                                  << " -> " << pr.rel_after
                                  << ",  ||b|| " << pr.b_norm_before
                                  << " -> " << pr.b_norm_after
                                  << ",  " << pr.cg_iters << " CG-Iterationen\n";
                        if (!pr.converged) {
                            std::cout << "[FEM-direkt]   WARNUNG: das CG auf dem "
                                         "Knoten-Laplace ist nicht konvergiert "
                                         "(relatives Residuum " << pr.cg_rel_res
                                      << "). Der Quellterm ist dann nur teilweise "
                                         "bereinigt.\n";
                        }
                        std::cout << "[FEM-direkt]   Die Loesung aendert sich dadurch "
                                     "um genau den Gradientenanteil -- ohne Wirkung "
                                     "auf B (rot grad = 0) und auf die Kraft in der "
                                     "Platte, gemessen ueber vier eps-Dekaden. Mit "
                                     "--fem-direct-no-project abschaltbar, so sind "
                                     "die Messreihen im Plan entstanden.\n";
                    } else {
                        std::cout << "[FEM-direkt] Quellterm NICHT projiziert "
                                     "(--fem-direct-no-project). ||G^T b||/||b|| "
                                     "bleibt bei seinem Wert aus der [RHS]-Diagnose, "
                                     "und der Scheingradient in A_i entsteht wie "
                                     "bisher.\n";
                    }

                    // Was kostet die Eichmaske? b auf den Baumkanten wird von
                    // der Randbehandlung genullt; das bringt Divergenz zurueck,
                    // und zwar genau an den Knoten, an denen Baumkanten haengen.
                    // Die Zahl steht hier, damit sie eine Messung ist und keine
                    // Fussnote.
                    if (!gauged_edges.empty()) {
                        std::vector<double> probe(b_r.begin(), b_r.end());
                        for (size_t e = 0; e < probe.size(); ++e)
                            if (gauged_edges[e]) probe[e] = 0.0;
                        std::vector<double> gtb(mesh.num_nodes, 0.0);
                        double nb = 0.0, lost = 0.0;
                        for (size_t e = 0; e < mesh.num_edges; ++e) {
                            const int2 ed = mesh.edges[e];
                            gtb[ed.y] += probe[e];
                            gtb[ed.x] -= probe[e];
                            nb += probe[e] * probe[e];
                            if (gauged_edges[e]) lost += b_r[e] * b_r[e];
                        }
                        double g2 = 0.0;
                        for (double v : gtb) g2 += v * v;
                        std::cout << "[FEM-direkt] Preis der Eichmaske: nach dem "
                                     "Nullen der Baumkanten ist ||G^T b||/||b|| = "
                                  << (nb > 0.0 ? std::sqrt(g2 / nb) : 0.0)
                                  << ", und " << std::sqrt(lost)
                                  << " von " << std::sqrt(nb + lost)
                                  << " der Quelltermnorm liegt auf Baumkanten "
                                     "und wird verworfen." << std::endl;
                    }

                    fem_direct::fill_block_rhs(csr_matrix.num_rows, b_r.data(),
                                               nullptr,
                                               fem_direct::BlockForm::General,
                                               rhs2.data());

                    // --- 5. Dirichlet A x n = 0 -----------------------------
                    // Der Kantensatz gilt fuer n Kanten; im 2n-System sind
                    // BEIDE Haelften betroffen (Real- und Imaginaerteil
                    // derselben Kante).
                    {
                        // Derselbe Satz, den die Projektion oben schon
                        // bekommen hat. Nicht erneut sammeln: liefen die
                        // beiden Aufrufe je auseinander, waere b auf Kanten
                        // bereinigt, die hier doch gesperrt werden -- und
                        // genau der Fehler war der Grund fuer die Maske.
                        //
                        // Im Eichmodus kommen die Baumkanten dazu. Fuer eine
                        // CSR-Matrix ist die Eichung nichts anderes als eine
                        // homogene Dirichlet-Bedingung: die Gleichung fuer
                        // eine Baumkante lautet x_e = 0, also Zeile und Spalte
                        // nullen und die Diagonale setzen. Der Operatorpfad
                        // wendet sie stattdessen als Projektion P A P + (I-P)
                        // an -- das ist fuer GMRES richtig und fuer eine
                        // Matrix, die cuDSS bekommt, wirkungslos.
                        const auto& fixed_n = fixed_edges;
                        std::vector<char> fixed_2n(2 * fixed_n.size(), 0);
                        for (size_t i = 0; i < fixed_n.size(); ++i) {
                            fixed_2n[i] = fixed_n[i];
                            fixed_2n[fixed_n.size() + i] = fixed_n[i];
                        }
                        const auto dr = fem_direct::apply_homogeneous_dirichlet(
                            static_cast<int>(ro2.size() - 1),
                            ro2.data(), ci2.data(), v2.data(), rhs2.data(),
                            fixed_2n);
                        std::cout << "[FEM-direkt] Dirichlet: " << dr.num_constrained
                                  << " von " << (ro2.size() - 1)
                                  << " Zeilen festgehalten, Diagonalwert "
                                  << dr.diag_value << " (freie Diagonale "
                                  << dr.diag_min_free << " .. " << dr.diag_max_free
                                  << ")\n";
                    }

                    // --- 6. Zerlegen und loesen -----------------------------
                    // Die Werte muessen fuer cuDSS auf dem Geraet liegen.
                    const int N2 = static_cast<int>(ro2.size() - 1);
                    DeviceArray<int>    d_ro2(ro2.size());
                    DeviceArray<int>    d_ci2(ci2.size());
                    DeviceArray<double> d_v2(v2.size());
                    DeviceArray<double> d_b2(rhs2.size());
                    DeviceArray<double> d_x2(rhs2.size());
                    for (size_t i = 0; i < ro2.size(); ++i) d_ro2[i] = ro2[i];
                    for (size_t i = 0; i < ci2.size(); ++i) d_ci2[i] = ci2[i];
                    for (size_t i = 0; i < v2.size();  ++i) d_v2[i]  = v2[i];
                    for (size_t i = 0; i < rhs2.size(); ++i) { d_b2[i] = rhs2[i]; d_x2[i] = 0.0; }
                    d_ro2.prefetch_to_device(); d_ci2.prefetch_to_device();
                    d_v2.prefetch_to_device();  d_b2.prefetch_to_device();
                    d_x2.prefetch_to_device();  bf::sync();

                    if (bf::fem_direct_iterative()) {
                        // --- 6a. Iterativ: GMRES(m) auf DERSELBEN Matrix ----
                        // Der Vergleich ist nur etwas wert, wenn die einzige
                        // Variable der Loeser ist. GMRES bekommt deshalb
                        // d_ro2/d_ci2/d_v2/d_b2 -- dieselben Arrays, die
                        // cuDSS bekaeme, nach der Dirichlet-Behandlung.
                        //
                        // CoupledMVOperator wird hier NUR als CSR-Matvec mit
                        // ILU(0) verwendet: keine BEM-Kopplung, keine
                        // Baum-Kobaum-Eichung, keine
                        // Kohomologie-Projektoren. Nichts davon wird
                        // aufgesetzt, num_bem_dofs bleibt 0 und die
                        // Eichmaske null -- apply() ist dann der reine
                        // Matvec. total_dofs ist 2n, nicht n.
                        //
                        // ILU(0) auf der Blockform ist die eigentliche
                        // Unbekannte. Die Matrix ist NICHT symmetrisch
                        // ([[K, -wC], [wC, K]]), und das Gebiet ist ungeeicht
                        // -- --fem-direct regularisiert stattdessen ueber
                        // sigma_Luft. HarmonicSolver.cuh haelt fest, dass
                        // GMRES auf den frueheren, geeichten Problemen bei
                        // 4000 Iterationen stagnierte. Ob die Regularisierung
                        // reicht, wo die Eichung fehlt, sagt dieser Lauf.
                        CoupledMVOperator op_it(N2);
                        op_it.setup_fem_matrix(N2, static_cast<int>(ci2.size()),
                                               d_ro2.data(), d_ci2.data(), d_v2.data());
                        op_it.compute_ilu();

                        const int    restart  = bf::gmres_restart();
                        const int    cap      = bf::max_iter();
                        const double tol      = 1e-10;

                        std::cout << "[FEM-direkt] ITERATIV: GMRES(" << restart
                                  << "), Obergrenze " << cap
                                  << " Iterationen, Toleranz " << tol
                                  << ", ILU(0) auf der 2n-Blockform\n";
                        std::cout << "[FEM-direkt]   dieselbe Matrix und rechte Seite "
                                     "wie der Direktloeser -- die einzige Variable "
                                     "ist der Loeser.\n";

                        GMRESSolver it_solver(N2, restart);
                        const auto t0 = std::chrono::steady_clock::now();
                        it_solver.solve(op_it, d_b2.data(), d_x2.data(), tol, cap);
                        const auto t1 = std::chrono::steady_clock::now();
                        const double secs =
                            std::chrono::duration<double>(t1 - t0).count();

                        std::cout << "[FEM-direkt]   " << it_solver.last_iterations()
                                  << " Iterationen, relatives Residuum "
                                  << it_solver.last_relative_residual()
                                  << ", " << secs << " s\n";
                        if (it_solver.last_relative_residual() > tol) {
                            std::cout << "[FEM-direkt]   NICHT konvergiert.\n";

                            // WO sitzt das uebriggebliebene Residuum? Das
                            // entscheidet, ob die Nichtkonvergenz das
                            // Ergebnis beruehrt.
                            //
                            // Auf Gradientenfeldern ist K exakt null, die
                            // Blockform reduziert sich dort auf den mit 1/eps
                            // verstaerkten Scheingradienten -- und genau der
                            // ist physikalisch traege: rot grad = 0, und in
                            // die Platte kommt er nicht (gemessene eps-Reihe
                            // im Plan). Bleibt das Residuum ueberwiegend in
                            // diesem Unterraum, ist die Kraft trotz
                            // Nichtkonvergenz brauchbar; sitzt es im
                            // rotationsbehafteten Anteil, nicht.
                            //
                            // Gemessen wird an r = b - Hx mit demselben
                            // diskreten Gradienten G wie in der
                            // [RHS]-Diagnose: (G^T r)_n = Summe der r_e der
                            // an n einlaufenden minus der auslaufenden
                            // Kanten. Je Haelfte getrennt, weil Real- und
                            // Imaginaerteil verschiedene Rollen haben.
                            d_x2.prefetch_to_host();
                            bf::sync();
                            const int n_half = N2 / 2;
                            std::vector<double> r2(rhs2.size());
                            for (int i = 0; i < N2; ++i) {
                                double ax = 0.0;
                                for (int k = ro2[i]; k < ro2[i + 1]; ++k)
                                    ax += v2[k] * d_x2[ci2[k]];
                                r2[i] = rhs2[i] - ax;
                            }
                            double gr2 = 0.0, r_sq = 0.0;
                            for (int half = 0; half < 2; ++half) {
                                std::vector<double> gtr(mesh.num_nodes, 0.0);
                                for (size_t e = 0; e < mesh.num_edges; ++e) {
                                    const int2 ed = mesh.edges[e];
                                    const double re = r2[half * n_half + e];
                                    gtr[ed.y] += re;
                                    gtr[ed.x] -= re;
                                }
                                for (double v : gtr) gr2 += v * v;
                            }
                            for (int i = 0; i < N2; ++i) r_sq += r2[i] * r2[i];

                            // Bezug: dieselbe Groesse auf b selbst. Das
                            // Verhaeltnis der beiden sagt, ob das Residuum
                            // GRADIENTENLASTIGER ist als die rechte Seite.
                            double gb2 = 0.0, b_sq = 0.0;
                            for (int half = 0; half < 2; ++half) {
                                std::vector<double> gtb(mesh.num_nodes, 0.0);
                                for (size_t e = 0; e < mesh.num_edges; ++e) {
                                    const int2 ed = mesh.edges[e];
                                    const double be = rhs2[half * n_half + e];
                                    gtb[ed.y] += be;
                                    gtb[ed.x] -= be;
                                }
                                for (double v : gtb) gb2 += v * v;
                            }
                            for (int i = 0; i < N2; ++i) b_sq += rhs2[i] * rhs2[i];

                            const double rel_r = (r_sq > 0.0) ? std::sqrt(gr2 / r_sq) : 0.0;
                            const double rel_b = (b_sq > 0.0) ? std::sqrt(gb2 / b_sq) : 0.0;
                            std::cout << "[FEM-direkt]   Wo sitzt das Restresiduum? "
                                         "||G^T r||/||r|| = " << rel_r
                                      << "  gegen ||G^T b||/||b|| = " << rel_b
                                      << ",  Verhaeltnis " << (rel_b > 0.0 ? rel_r / rel_b : 0.0)
                                      << "\n";
                            if (rel_r > 3.0 * rel_b) {
                                std::cout << "[FEM-direkt]   -> das Residuum ist deutlich "
                                             "gradientenlastiger als die rechte Seite. "
                                             "Stehen geblieben ist also der Anteil, den "
                                             "rot A vernichtet und der die Platte nicht "
                                             "erreicht. Die Kraft kann dann trotz "
                                             "Nichtkonvergenz stimmen -- gegen einen "
                                             "Lauf ohne --fem-direct-iterative pruefen, "
                                             "nicht darauf verlassen.\n";
                            } else {
                                std::cout << "[FEM-direkt]   -> das Residuum ist NICHT "
                                             "vorwiegend Gradient. Dann sitzt es im "
                                             "rotationsbehafteten Anteil, und Feld, B "
                                             "und Kraft unten sind unbrauchbar.\n";
                            }
                        }
                        d_x2.prefetch_to_host();
                        bf::sync();
                    } else {
                    fem_direct::SparseDirectSolver solver;
                    fem_direct::SolverOptions sopt;
                    sopt.reordering   = fem_direct::Reordering::NestedDissection;
                    sopt.deterministic = true;
                    solver.set_options(sopt);

                    std::cout << "[FEM-direkt] cuDSS "
                              << fem_direct::SparseDirectSolver::version()
                              << ", System " << N2 << " x " << N2
                              << ", nnz " << ci2.size() << "\n";

                    solver.analyze(N2, static_cast<int>(ci2.size()),
                                   d_ro2.data(), d_ci2.data(), d_v2.data(),
                                   fem_direct::MatrixKind::General);
                    const auto frep = solver.factorize();
                    std::cout << "[FEM-direkt] Faktorisierung: nnz(LU) = "
                              << frep.lu_nnz << ", Fuellfaktor "
                              << frep.fill_ratio << ", Pivotierungen "
                              << frep.npivots << "\n";

                    solver.solve(d_b2.data(), d_x2.data());
                    d_x2.prefetch_to_host();
                    bf::sync();
                    }

                    // --- 7. Residuum auf der ORIGINALmatrix -----------------
                    // Gegen v2/rhs2 nach dem Einarbeiten -- das ist das
                    // System, das tatsaechlich geloest wurde. Ein kleines
                    // Residuum hier ist die einzige Zusicherung, die ein
                    // Direktloeser ueberhaupt geben kann.
                    {
                        double num = 0.0, den = 0.0;
                        for (int r = 0; r < N2; ++r) {
                            double ax = 0.0;
                            for (int k = ro2[r]; k < ro2[r + 1]; ++k)
                                ax += v2[k] * d_x2[ci2[k]];
                            const double d = rhs2[r] - ax;
                            num += d * d;
                            den += rhs2[r] * rhs2[r];
                        }
                        const double rel = (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
                        std::cout << "[FEM-direkt] relatives Residuum ||b - Ax|| / ||b|| = "
                                  << rel << "\n";
                        if (!(rel < 1e-8)) {
                            std::cout << "[FEM-direkt] WARNUNG: das Residuum ist fuer eine "
                                         "Zerlegung zu gross. Erwartet ist 1e-12 oder "
                                         "besser; groessere Werte deuten auf eine "
                                         "fehlerhafte Matrix hin, nicht auf den Loeser.\n";
                        }
                    }

                    // --- 8. Loesung aufteilen -------------------------------
                    std::vector<double> xr(static_cast<size_t>(total_dofs));
                    std::vector<double> xi(static_cast<size_t>(total_dofs));
                    fem_direct::split_block_solution(total_dofs, d_x2.data(),
                                                     xr.data(), xi.data());
                    double max_r = 0.0, max_i = 0.0;
                    for (int i = 0; i < total_dofs; ++i) {
                        max_r = std::max(max_r, std::fabs(xr[i]));
                        max_i = std::max(max_i, std::fabs(xi[i]));
                    }
                    std::cout << "[FEM-direkt] Loesung: max|A_r| = " << max_r
                              << ", max|A_i| = " << max_i << "\n";

                    // --- 8b. Wie viel vom Scheingradienten kommt in der
                    //         Platte an? ----------------------------------
                    // Der Imaginaerteil traegt einen Scheingradienten: auf
                    // Gradientenfeldern ist K exakt null, die Blockform
                    // reduziert sich dort auf x_i = -b_g/(omega*C), und der
                    // Gradientenanteil von b wird um 1/(omega*sigma)
                    // verstaerkt. Gemessen skaliert max|A_i| exakt mit
                    // 1/eps ueber vier Dekaden (Reihe in
                    // docs/design/fem-direct-plan.md).
                    //
                    // Fuer die Kraft ist aber nicht der globale Wert
                    // entscheidend, sondern der in der PLATTE. Dort ist
                    // sigma = 3,4e7 statt eps*3,4e7, die Verstaerkung also
                    // um 1/eps kleiner. Ob davon etwas uebrig bleibt,
                    // entscheidet, ob AP7 mit dieser Quellterm-Vorgabe
                    // ueberhaupt sinnvoll ist -- der Lorentzpfad bildet
                    // J = -sigma*j*omega*A aus A SELBST, nicht aus rot A.
                    //
                    // Zusaetzlich B = rot A in der Platte. Weil rot grad = 0
                    // ist, ist B gegen den Scheingradienten IMMUN. Damit
                    // laesst sich die Abnahme aus Abschnitt 1 (Ergebnis
                    // bleibt zwischen eps = 1e-6 und 1e-10 stehen) schon
                    // JETZT fahren, vor der Kraftauswertung: "das Ergebnis"
                    // ist B, nicht max|A|.
                    {
                        std::vector<char> touches_plate(mesh.num_edges, 0);
                        std::vector<char> touches_other(mesh.num_edges, 0);
                        std::vector<char> is_plate_elem(mesh.num_hexes, 0);
                        for (int h : plate.get_hex_indices())
                            is_plate_elem[h] = 1;

                        for (size_t h = 0; h < mesh.num_hexes; ++h) {
                            for (int i = 0; i < 12; ++i) {
                                const int ge = mesh.hex_edges[h * 12 + i];
                                if (ge < 0) continue;
                                if (is_plate_elem[h]) touches_plate[ge] = 1;
                                else                  touches_other[ge] = 1;
                            }
                        }

                        double p_r = 0.0, p_i = 0.0, inner_i = 0.0;
                        long n_plate = 0, n_inner = 0;
                        for (size_t e = 0; e < mesh.num_edges; ++e) {
                            if (!touches_plate[e]) continue;
                            ++n_plate;
                            p_r = std::max(p_r, std::fabs(xr[e]));
                            p_i = std::max(p_i, std::fabs(xi[e]));
                            if (!touches_other[e]) {
                                ++n_inner;
                                inner_i = std::max(inner_i, std::fabs(xi[e]));
                            }
                        }

                        std::cout << "[FEM-direkt] Platte (" << n_plate
                                  << " Kanten, davon " << n_inner
                                  << " nur in der Platte): max|A_r| = " << p_r
                                  << ", max|A_i| = " << p_i
                                  << ", nur innen max|A_i| = " << inner_i << "\n";

                        // Der Hinweis auf die eps-Skalierung gilt nur OHNE
                        // Projektion. Mit ihr ist der Scheingradient gar nicht
                        // erst da, der Quotient liegt dann bei einer
                        // Groessenordnung um 1 -- und das ist das GUTE
                        // Ergebnis, nicht das schlechte. Ohne diese
                        // Unterscheidung liest der Text sich nach der
                        // Projektion genau falsch.
                        const double share = (max_i > 0.0) ? p_i / max_i : 0.0;
                        std::cout << "[FEM-direkt] Anteil des Imaginaerteils in der "
                                     "Platte: " << share;
                        if (bf::fem_direct_project()) {
                            std::cout << " -- mit Projektion ist ein Wert um 1 erwartet:"
                                         " ohne Scheingradient ist das globale Maximum"
                                         " nicht mehr um 1/eps groesser als das der"
                                         " Platte.\n";
                        } else {
                            std::cout << " -- ohne Projektion muss dieser Wert MIT eps"
                                         " skalieren. Tut er das, bleibt der"
                                         " Scheingradient in der Luft und die Platte"
                                         " ist sauber; bleibt er stehen, nicht.\n";
                        }

                        // B = rot A an den Quadraturpunkten der
                        // Plattenelemente. Dieselbe Piola-Form wie
                        // assemble_local_matrices_kernel: kontravariant
                        // SPALTENweise ueber J, geteilt durch detJ.
                        double bmax_r = 0.0, bmax_i = 0.0;
                        for (int h : plate.get_hex_indices()) {
                            const int etype = element_type_or_hex(
                                mesh.elem_types.data(), h);
                            const int n_edges = element_num_edges(etype);
                            const int n_nodes = element_num_nodes(etype);

                            const Hexahedron H = mesh.hex_nodes[h];
                            double3 nodes[8];
                            for (int i = 0; i < n_nodes; ++i)
                                nodes[i] = mesh.nodes_current[H.v[i]];
                            for (int i = n_nodes; i < 8; ++i)
                                nodes[i] = make_double3(0.0, 0.0, 0.0);

                            double cr[12] = {0.0}, ci[12] = {0.0};
                            for (int i = 0; i < n_edges; ++i) {
                                const int ge = mesh.hex_edges[h * 12 + i];
                                if (ge < 0) continue;
                                const double sg = mesh.hex_edge_signs[h * 12 + i];
                                cr[i] = sg * xr[ge];
                                ci[i] = sg * xi[ge];
                            }

                            const int nq = element_num_quad_points(etype);
                            for (int q = 0; q < nq; ++q) {
                                double r, s, t, w;
                                element_quadrature_point(etype, q, r, s, t, w);
                                double J[3][3], invJ[3][3], detJ;
                                compute_element_jacobian(etype, nodes, r, s, t,
                                                         J, invJ, detJ);
                                double3 C[12];
                                evaluate_element_curl(etype, r, s, t, C);
                                double3 br = make_double3(0, 0, 0);
                                double3 bi = make_double3(0, 0, 0);
                                for (int i = 0; i < n_edges; ++i) {
                                    const double cx =
                                        (J[0][0]*C[i].x + J[1][0]*C[i].y + J[2][0]*C[i].z) / detJ;
                                    const double cy =
                                        (J[0][1]*C[i].x + J[1][1]*C[i].y + J[2][1]*C[i].z) / detJ;
                                    const double cz =
                                        (J[0][2]*C[i].x + J[1][2]*C[i].y + J[2][2]*C[i].z) / detJ;
                                    br.x += cr[i]*cx; br.y += cr[i]*cy; br.z += cr[i]*cz;
                                    bi.x += ci[i]*cx; bi.y += ci[i]*cy; bi.z += ci[i]*cz;
                                }
                                bmax_r = std::fmax(bmax_r, std::sqrt(
                                    br.x*br.x + br.y*br.y + br.z*br.z));
                                bmax_i = std::fmax(bmax_i, std::sqrt(
                                    bi.x*bi.x + bi.y*bi.y + bi.z*bi.z));
                            }
                        }
                        std::cout << "[FEM-direkt] B = rot A in der Platte: max|B_r| = "
                                  << bmax_r << " T, max|B_i| = " << bmax_i
                                  << " T  <== DIESE Zahlen muessen ueber die "
                                     "eps-Reihe stehenbleiben (rot grad = 0)\n";
                    }

                    // --- 9. AP7: zeitgemittelte Lorentzkraft ---------------
                    // <J x B> = (sigma*omega/2) (A_i x rot A_r - A_r x rot A_i),
                    // Volumenintegral ueber die Leiter. Herleitung und der
                    // Grund gegen den Maxwell-Spannungstensor stehen im Kopf
                    // von HarmonicLorentz.cuh.
                    {
                        DeviceArray<double> d_ar(static_cast<size_t>(total_dofs));
                        DeviceArray<double> d_ai(static_cast<size_t>(total_dofs));
                        for (int i = 0; i < total_dofs; ++i) {
                            d_ar[i] = xr[i];
                            d_ai[i] = xi[i];
                        }
                        d_ar.prefetch_to_device();
                        d_ai.prefetch_to_device();
                        d_regularized.prefetch_to_device();
                        d_materials.prefetch_to_device();
                        bf::sync();

                        const double3 com = plate.get_properties().center_of_mass;

                        // Zweimal gerechnet: einmal mit Maske, einmal ohne.
                        // Die zweite Zahl ist kein Ergebnis, sondern die
                        // MESSUNG der Scheinkraft aus dem regularisierten
                        // Luftgebiet. Sie steht hier, damit der Fehler eine
                        // Zahl im Protokoll ist und keine Abschaetzung in
                        // einem Kommentar -- und weil die eps-Abnahme ihn
                        // nicht aufdecken kann.
                        double res[2][6];
                        for (int pass = 0; pass < 2; ++pass) {
                            DeviceArray<double> acc(6);
                            bf::memset_d(acc.data(), 0, 6 * sizeof(double));
                            bf::sync();
                            BF_LAUNCH_1D(harmonic_lorentz, mesh.num_hexes,
                                static_cast<int>(mesh.num_hexes),
                                mesh.hex_nodes.data(), mesh.nodes_current.data(),
                                mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                                mesh.elem_types.data(),
                                d_materials.data(),
                                pass == 0 ? d_regularized.data() : nullptr,
                                d_ar.data(), d_ai.data(), omega,
                                com,
                                acc.data() + 0, acc.data() + 1, acc.data() + 2,
                                acc.data() + 3, acc.data() + 4, acc.data() + 5);
                            bf::sync();
                            bf::memcpy_d(res[pass], acc.data(), 6 * sizeof(double),
                                         cudaMemcpyDeviceToHost);
                        }

                        std::cout << "\n[FEM-direkt] Zeitgemittelte Lorentzkraft "
                                     "(Volumenintegral ueber die Leiter):\n";
                        std::cout << "[FEM-direkt]   F = (" << res[0][0] << ", "
                                  << res[0][1] << ", " << res[0][2] << ") N\n";
                        std::cout << "[FEM-direkt]   M = (" << res[0][3] << ", "
                                  << res[0][4] << ", " << res[0][5]
                                  << ") Nm um den Plattenschwerpunkt\n";
                        std::cout << "[FEM-direkt]   F_z = " << res[0][2]
                                  << " N  <== die Levitationskraft\n";

                        // Gegenprobe. Die Differenz ist der Beitrag der
                        // kuenstlich leitenden Gebiete. Sie MUSS klein gegen
                        // F_z sein; ist sie es nicht, taeuscht das Ergebnis
                        // eine Genauigkeit vor, die es nicht hat.
                        const double dz = res[1][2] - res[0][2];
                        std::cout << "[FEM-direkt]   ohne Maske waere F_z = "
                                  << res[1][2] << " N, also " << dz
                                  << " N Scheinkraft aus dem regularisierten "
                                     "Luftgebiet\n";
                        if (std::fabs(res[0][2]) > 0.0 &&
                            std::fabs(dz) > 0.05 * std::fabs(res[0][2])) {
                            std::cout << "[FEM-direkt]   HINWEIS: die Scheinkraft ist "
                                         "nicht klein gegen das Ergebnis. Sie ist "
                                         "eps-UNABHAENGIG, die eps-Reihe deckt sie "
                                         "also nicht auf.\n";
                        }

                    // --- 9b. Gegenprobe ueber den Maxwell-Spannungstensor ---
                    // Unabhaengiger Weg zur selben Groesse: das
                    // Volumenintegral wertet J x B im Leiter aus, dieses hier
                    // integriert Spannungen ueber die Huelle der Platte.
                    // Stimmen beide ueberein, haengt die Kraftzahl an keinem
                    // Formelfehler in einem der Pfade.
                    //
                    // Der Kopf von HarmonicLorentz.cuh nennt gute Gruende,
                    // dem Oberflaechenpfad WENIGER zu trauen -- er bildet die
                    // Nettokraft als Differenz grosser, sich fast
                    // aufhebender Beitraege. Das bleibt richtig. Eine
                    // Gegenprobe muss nicht genauer sein als das Geprueft,
                    // nur unabhaengig.
                    if (mesh.num_component_surface_faces > 0) {
                        // Dreimal gerechnet. Der dritte Lauf ist eine
                        // NULLPROBE, kein Ergebnis.
                        //
                        // Mit A_i = 0 verschwindet der Realteil der
                        // Stromdichte in der Platte: J = -sigma*j*omega*A,
                        // also Re(J) = sigma*omega*A_i = 0. Ohne Strom
                        // innerhalb der Huelle und bei mu_r = 1 MUSS das
                        // Spannungsintegral ueber eine geschlossene Flaeche
                        // exakt null ergeben. Was stattdessen herauskommt,
                        // ist der Ausloeschungsfehler des Oberflaechenpfades
                        // -- gemessen statt geschaetzt.
                        //
                        // Genau diese Rechnung belegt im Kopf von
                        // HarmonicLorentz.cuh die -0,122 N des alten
                        // Hexaedernetzes.
                        DeviceArray<double> d_null(static_cast<size_t>(total_dofs));
                        for (int i = 0; i < total_dofs; ++i) d_null[i] = 0.0;
                        d_null.prefetch_to_device();
                        bf::sync();

                        double m[3][6];
                        for (int pass = 0; pass < 3; ++pass) {
                            // pass 0: beide Anteile -- das Ergebnis
                            // pass 1: nur A_r  -> muss 0 sein (Nullprobe)
                            // pass 2: nur A_i
                            //
                            // Hier stand einmal (pass == 1) fuer pr und
                            // (pass >= 1) fuer pi. Damit waren bei pass 1
                            // BEIDE Felder null, die "Nullprobe" rechnete ein
                            // leeres Feld und lieferte trivialerweise 0 --
                            // und die eigentliche Nullprobe lief unter pass 2
                            // mit dem Etikett "nur aus A_i". Die Zahlen waren
                            // richtig, die Beschriftung nicht, und die
                            // automatische Bewertung unten las daraufhin die
                            // falsche Zahl und gab Entwarnung.
                            const double* pr = (pass == 2) ? d_null.data() : d_ar.data();
                            const double* pi = (pass == 1) ? d_null.data() : d_ai.data();
                            DeviceArray<double> acc(6);
                            bf::memset_d(acc.data(), 0, 6 * sizeof(double));
                            bf::sync();
                            BF_LAUNCH_1D(harmonic_maxwell, mesh.num_component_surface_faces,
                                static_cast<int>(mesh.num_component_surface_faces),
                                mesh.component_surface_faces.data(),
                                mesh.hex_nodes.data(), mesh.nodes_current.data(),
                                mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                                mesh.elem_types.data(),
                                pr, pi,
                                com,
                                acc.data() + 0, acc.data() + 1, acc.data() + 2,
                                acc.data() + 3, acc.data() + 4, acc.data() + 5);
                            bf::sync();
                            bf::memcpy_d(m[pass], acc.data(), 6 * sizeof(double),
                                         cudaMemcpyDeviceToHost);
                        }
                        const double* m6 = m[0];

                        std::cout << "\n[FEM-direkt] Gegenprobe Maxwell-Spannungstensor ("
                                  << mesh.num_component_surface_faces
                                  << " Grenzflaechen):\n";
                        std::cout << "[FEM-direkt]   F = (" << m6[0] << ", " << m6[1]
                                  << ", " << m6[2] << ") N\n";

                        const double lz = res[0][2];
                        const double rel = (std::fabs(lz) > 0.0)
                            ? std::fabs(m6[2] - lz) / std::fabs(lz) : 0.0;
                        std::cout << "[FEM-direkt]   F_z: Lorentz " << lz
                                  << " N gegen Maxwell " << m6[2]
                                  << " N, Abweichung " << 100.0 * rel << " %\n";

                        std::cout << "[FEM-direkt]   NULLPROBE nur aus A_r: F_z = "
                                  << m[1][2] << " N, exakt sind 0 N. "
                                     "Re(J) = sigma*omega*A_i ist ohne A_i null, "
                                     "ohne Strom in der Huelle und bei mu_r = 1 "
                                     "verschwindet das Integral.\n";
                        std::cout << "[FEM-direkt]   nur aus A_i: F_z = " << m[2][2]
                                  << " N, dort steht die gesuchte Kraft, also "
                                  << lz << " N.\n";

                        const double err_r = std::fabs(m[1][2]);
                        const double err_i = std::fabs(m[2][2] - lz);
                        std::cout << "[FEM-direkt]   Fehler des Oberflaechenpfades: "
                                  << err_r << " N im Realteil, " << err_i
                                  << " N im Imaginaerteil, bei einem gesuchten "
                                     "Ergebnis von " << std::fabs(lz) << " N.\n";

                        if (err_r > 0.2 * std::fabs(lz)) {
                            std::cout << "[FEM-direkt]   -> der Oberflaechenpfad traegt "
                                         "hier nicht, und das ist GEMESSEN, nicht "
                                         "vermutet: er verfehlt einen analytisch "
                                         "bekannten Nullwert um mehr als das gesuchte "
                                         "Ergebnis. Die Abweichung oben ist damit "
                                         "erklaert und kein Hinweis auf das "
                                         "Volumenintegral. Ursache ist die Bildung "
                                         "einer kleinen Nettokraft als Differenz von "
                                         "Einzelbeitraegen um |B|^2/(2 mu_0) mal "
                                         "Huellflaeche -- hier rund 200 N.\n";
                        } else {
                            std::cout << "[FEM-direkt]   -> die Nullprobe ist klein. "
                                         "Eine Abweichung zwischen beiden Wegen waere "
                                         "damit NICHT durch Ausloeschung erklaert, und "
                                         "dann steht das Volumenintegral zur Debatte.\n";
                        }
                    }
                    }   // Ende 9/9b -- hier fallen d_ar, d_ai und acc weg

                    std::cout << "\n[FEM-direkt] Einordnung. Zwei Groessen begrenzen "
                                 "diese Kraft, und beide stehen hier:\n";
                    std::cout << "[FEM-direkt]   Abschneidefaktor " << truncation_min
                              << " (verlangt sind 2). A x n = 0 so nah am Leiter "
                                 "verformt das Feld dort, wo integriert wird.\n";
                    std::cout << "[FEM-direkt]   Platte mit "
                              << plate.get_hex_indices().size() << " Elementen und "
                              << mesh.num_component_surface_faces
                              << " Grenzflaechen. Dort sitzen die Wirbelstroeme, und "
                                 "dort wird integriert -- eine Verfeinerung, die nur "
                                 "das Luftgebiet betrifft, aendert an dieser Zahl "
                                 "nichts.\n";
                    return 0;
                }

                std::cout << "\n[Harmonisch] f = " << bf::harmonic_frequency()
                          << " Hz, omega = " << omega
                          << ", dt-Aequivalent 1/omega = " << dt_w << " s\n";

                // 1. K allein (sehr grosses dt) -> in csr_C zwischenlagern
                assemble_at(1e30);
                bf::memcpy_d(csr_C.values.data(), csr_matrix.values.data(),
                             static_cast<size_t>(csr_matrix.nnz) * sizeof(double),
                             cudaMemcpyDeviceToDevice);

                // 2. (W+T) = K + omega*C  -> bleibt in csr_matrix
                assemble_at(dt_w);

                // 3. C = ((W+T) - K) / omega, eintragsweise auf den CSR-Werten
                {
                    std::vector<double> h_wt(static_cast<size_t>(csr_matrix.nnz));
                    std::vector<double> h_k (static_cast<size_t>(csr_matrix.nnz));
                    bf::memcpy_d(h_wt.data(), csr_matrix.values.data(),
                                 h_wt.size() * sizeof(double), cudaMemcpyDeviceToHost);
                    bf::memcpy_d(h_k.data(), csr_C.values.data(),
                                 h_k.size() * sizeof(double), cudaMemcpyDeviceToHost);
                    double max_c = 0.0;
                    int non_finite = 0;
                    for (size_t i = 0; i < h_wt.size(); ++i) {
                        h_k[i] = (h_wt[i] - h_k[i]) / omega;
                        if (!std::isfinite(h_k[i])) { ++non_finite; continue; }
                        max_c = std::max(max_c, std::fabs(h_k[i]));
                    }
                    if (non_finite > 0)
                        std::cout << "[Harmonisch] WARNUNG: " << non_finite
                                  << " nicht endliche Eintraege in C.\n";
                    bf::memcpy_d(csr_C.values.data(), h_k.data(),
                                 h_k.size() * sizeof(double), cudaMemcpyHostToDevice);
                    std::cout << "[Harmonisch] Massenmatrix C aus zwei Assemblierungen "
                                 "gewonnen, max|C_ij| = " << max_c << "\n";
                }

                // Der Operator bekommt (W+T). Die Aequilibrierung wird NICHT
                // erneut aufgesetzt -- D bleibt, was es ist, und der
                // harmonische Loeser rechnet konsistent im selben Tilde-Raum
                // wie die rechte Seite, die oben schon skaliert wurde.
                op.setup_fem_matrix(csr_matrix.num_rows, csr_matrix.nnz,
                                    csr_matrix.row_offsets.data(),
                                    csr_matrix.col_indices.data(),
                                    csr_matrix.values.data());
                op.compute_ilu();
                bf::sync();

                // -----------------------------------------------------------
                // Vorabmessung: traegt ILU auf (W+T) ueberhaupt?
                // -----------------------------------------------------------
                //
                // C-to-R steht und faellt damit. Die Deckenmessung hat 2914
                // Schritte bis 1e-4 gemessen -- aber UNVORKONDITIONIERT. Dass
                // ILU dort besser ist, war eine Annahme, keine Messung, und
                // der erste Lauf sprach eher dagegen (Stagnation bei 0,0448).
                //
                // Diese Messung kostet Minuten und entscheidet, ob der
                // eigentliche Lauf Stunden wert ist. Ohne sie laeuft der
                // aeussere Loeser auf einem Vorkonditionierer, der seine
                // Aufgabe nicht erfuellt -- und die einzige Rueckmeldung ist
                // ein Abbruch von Hand.
                if (bf::want_harmonic_checks()) {
                    const double probe_tol   = bf::harmonic_inner_tol();
                    const int    probe_maxit = 4000;   // > die gemessenen 2914
                    // Die Vorabmessung laeuft mit VOLLEM GMRES, also Restart =
                    // Iterationsdeckel.
                    //
                    // Vorher stand hier max(400, --gmres-restart). Ohne die
                    // Option bedeutete das Restart 400 -- und damit misst die
                    // Vorabmessung nicht (W+T), sondern ihre eigene
                    // Restart-Laenge: bei 400 stagniert sie bei 0,0024, bei
                    // 3000 konvergiert sie in 2590 Schritten auf 9,95e-05. Am
                    // selben Operator.
                    //
                    // Sie hat den Lauf dann mit der Begruendung abgebrochen,
                    // (W+T) sei nicht loesbar. Genau der Fehler, den sie
                    // verhindern soll, steckte in ihr selbst -- und sie ist
                    // damit dieselbe Falle wie beim inneren Loeser einen
                    // Commit zuvor: zwei Groessen an denselben Schalter
                    // gehaengt.
                    //
                    // Bei nd = 4511 kostet das 289 MB Basisspeicher.
                    const int probe_restart = probe_maxit;
                    DeviceArray<double> d_probe(static_cast<size_t>(nd_h));
                    GMRESSolver probe(nd_h, probe_restart);

                    std::cout << "\n[Harmonisch] Vorabmessung: (W+T) bis " << probe_tol
                              << ", Restart " << probe_restart
                              << ", hoechstens " << probe_maxit << " Schritte\n";

                    bf::memset_d(d_probe.data(), 0,
                                 static_cast<size_t>(nd_h) * sizeof(double));
                    bf::sync();
                    probe.solve(op, d_rhs_coupled.data(), d_probe.data(),
                                probe_tol, probe_maxit);
                    const int    it_ilu  = probe.last_iterations();
                    const double res_ilu = probe.last_relative_residual();

                    // Gegenprobe ohne ILU. Ohne sie waere nicht zu
                    // unterscheiden, ob ILU nichts bringt oder sogar schadet
                    // -- und die Studie hat ihren Bezugswert (2914)
                    // unvorkonditioniert gemessen, also ist das der
                    // vergleichbare Fall.
                    op.set_ilu_bypass(true);
                    bf::memset_d(d_probe.data(), 0,
                                 static_cast<size_t>(nd_h) * sizeof(double));
                    bf::sync();
                    probe.solve(op, d_rhs_coupled.data(), d_probe.data(),
                                probe_tol, probe_maxit);
                    const int    it_plain  = probe.last_iterations();
                    const double res_plain = probe.last_relative_residual();
                    op.set_ilu_bypass(false);

                    std::cout << "[Harmonisch]   mit ILU:            " << it_ilu
                              << " Schritte, rel. Residuum " << res_ilu << "\n"
                              << "[Harmonisch]   ohne (ILU-Bypass):  " << it_plain
                              << " Schritte, rel. Residuum " << res_plain << "\n";

                    // -------------------------------------------------------
                    // Die Messung entscheidet -- aber am richtigen Kriterium.
                    // -------------------------------------------------------
                    //
                    // ACHTUNG, hier stand ein Fehler, und zwar zum ZWEITEN
                    // Mal derselbe: verglichen wurde das RESTRESIDUUM. Wenn
                    // beide Varianten konvergieren, sagt das aber nichts --
                    // dann liegen beide knapp unter der Toleranz, und die
                    // Iterationszahl entscheidet. Gemessen wurde 2502 Schritte
                    // mit ILU gegen 2590 ohne, bei Residuen 9,99e-05 und
                    // 9,95e-05: die Residuen sprechen fuer den Bypass, die
                    // Iterationszahl fuer ILU. Richtig ist die Iterationszahl.
                    //
                    // Nur wenn KEINE der beiden konvergiert, ist die
                    // Iterationszahl fuer beide der Deckel und sagt nichts --
                    // dann entscheidet das Residuum.
                    //
                    // Genau diese Regel steht seit laengerem in
                    // compare_accuracy_series(); ich habe sie hier neu und
                    // falsch geschrieben, statt sie zu benutzen.
                    const bool ilu_conv   = (res_ilu   <= probe_tol * 10.0);
                    const bool plain_conv = (res_plain <= probe_tol * 10.0);
                    bool bypass_is_better;
                    if (ilu_conv && plain_conv)      bypass_is_better = (it_plain < it_ilu);
                    else if (plain_conv)             bypass_is_better = true;
                    else if (ilu_conv)               bypass_is_better = false;
                    else                             bypass_is_better = (res_plain < res_ilu);

                    op.set_ilu_bypass(bypass_is_better);
                    const double res_best = bypass_is_better ? res_plain : res_ilu;
                    std::cout << "[Harmonisch]   -> der innere Loeser laeuft "
                              << (bypass_is_better ? "OHNE ILU" : "mit ILU");
                    if (ilu_conv && plain_conv)
                        std::cout << " (weniger Iterationen: "
                                  << (bypass_is_better ? it_plain : it_ilu) << " gegen "
                                  << (bypass_is_better ? it_ilu : it_plain) << ").\n";
                    else
                        std::cout << " (die einzige bzw. bessere der beiden Messungen).\n";

                    // Frueher stand hier die Behauptung, ILU sei
                    // strukturbedingt schlechter, weil es blockdiagonal ist.
                    // Bei Restart 400 sah es so aus (0,0101 gegen 0,00243),
                    // aber beide Laeufe waren STAGNIERT -- das war ein
                    // Restart-Artefakt und keine Eigenschaft von ILU. Bei
                    // Restart 3000 konvergieren beide, und ILU ist sogar
                    // leicht besser. Die Behauptung ist damit
                    // zurueckgezogen.
                    if (ilu_conv && plain_conv) {
                        const double ratio = static_cast<double>(it_plain)
                                           / static_cast<double>(std::max(1, it_ilu));
                        std::cout << "[Harmonisch]      ILU bringt hier Faktor " << ratio
                                  << " an Iterationen -- also praktisch nichts, aber es\n"
                                     "[Harmonisch]      SCHADET auch nicht. Ein frueherer Lauf mit "
                                     "Restart 400 legte das\n"
                                     "[Harmonisch]      Gegenteil nahe; dort waren aber beide "
                                     "Varianten stagniert, und\n"
                                     "[Harmonisch]      der Vergleich stagnierter Residuen sagt "
                                     "nichts ueber den\n"
                                     "[Harmonisch]      Vorkonditionierer, nur ueber die "
                                     "Restart-Laenge.\n";
                    }

                    // -------------------------------------------------------
                    // Gegenprobe: liegt es an der EICHUNG?
                    // -------------------------------------------------------
                    //
                    // Der innere Loeser im C-to-R bekommt keine physikalische
                    // rechte Seite, sondern Krylov-Vektoren des AEUSSEREN
                    // Prozesses. (W+T) ist auf dem geeichten Anteil singulaer
                    // (478 von 1791 Freiheitsgraden plus zwei
                    // Kohomologie-Generatoren), und ein Vektor, der dort
                    // Komponenten hat, liegt nicht im Bildraum.
                    //
                    // Das innere GMRES minimiert dann ueber einen Unterraum,
                    // in dem keine Loesung existiert: das REKURSIV verfolgte
                    // Residuum sinkt, das echte nicht. Beim Restart, wo r =
                    // b - A x neu gerechnet wird, springt es -- im Lauf
                    // beobachtet von 0,436 auf 7,6.
                    //
                    // Diese drei Messungen entscheiden das. Der Zufallsvektor
                    // ist der Ersatz fuer einen Krylov-Vektor: beliebig, und
                    // ohne Grund im Bildraum zu liegen.
                    {
                        std::mt19937 rng(20260822u);
                        std::normal_distribution<double> nd_dist(0.0, 1.0);
                        std::vector<double> h_rand(static_cast<size_t>(nd_h));
                        for (double& v : h_rand) v = nd_dist(rng);

                        DeviceArray<double> d_rand(static_cast<size_t>(nd_h));
                        auto run_probe = [&](const char* label, bool project) {
                            bf::memcpy_d(d_rand.data(), h_rand.data(),
                                         static_cast<size_t>(nd_h) * sizeof(double),
                                         cudaMemcpyHostToDevice);
                            if (project) op.project_gauge_tilde_orthogonal(d_rand.data());
                            bf::memset_d(d_probe.data(), 0,
                                         static_cast<size_t>(nd_h) * sizeof(double));
                            bf::sync();
                            probe.solve(op, d_rand.data(), d_probe.data(),
                                        probe_tol, probe_maxit);
                            std::cout << "[Harmonisch]     " << label << ": "
                                      << probe.last_iterations() << " Schritte, rel. Residuum "
                                      << probe.last_relative_residual() << "\n";
                            return probe.last_relative_residual();
                        };

                        std::cout << "[Harmonisch]\n"
                                     "[Harmonisch]   Gegenprobe zur Eichung -- dieselbe Matrix, "
                                     "drei rechte Seiten:\n";
                        std::cout << "[Harmonisch]     physikalische rechte Seite: "
                                  << (bypass_is_better ? it_plain : it_ilu)
                                  << " Schritte, rel. Residuum " << res_best << "\n";
                        const double res_raw  = run_probe("Zufallsvektor, NICHT projiziert ", false);
                        const double res_proj = run_probe("Zufallsvektor, PROJIZIERT       ", true);

                        std::cout << "[Harmonisch]\n";
                        if (res_proj <= probe_tol * 10.0 && !(res_raw <= probe_tol * 10.0)) {
                            std::cout <<
                                "[Harmonisch]   -> BESTAETIGT: der unprojizierte Vektor traegt "
                                "nicht, der projizierte\n"
                                "[Harmonisch]      schon. (W+T) ist auf dem geeichten Anteil "
                                "singulaer, und die\n"
                                "[Harmonisch]      inneren Loesungen des C-to-R brauchen eine "
                                "Projektion ihrer\n"
                                "[Harmonisch]      Eingabe. Das ist der Fix, und er ist damit "
                                "belegt statt vermutet.\n";
                        } else if (res_proj <= probe_tol * 10.0 && res_raw <= probe_tol * 10.0) {
                            std::cout <<
                                "[Harmonisch]   -> WIDERLEGT: auch der unprojizierte "
                                "Zufallsvektor traegt. Die\n"
                                "[Harmonisch]      Eichung ist nicht die Ursache; der "
                                "Residuensprung im inneren\n"
                                "[Harmonisch]      Loeser kommt woanders her.\n";
                        } else {
                            std::cout <<
                                "[Harmonisch]   -> UNKLAR: auch der projizierte Zufallsvektor "
                                "traegt nicht. Dann ist\n"
                                "[Harmonisch]      entweder die Projektion nicht die richtige, "
                                "oder (W+T) ist fuer\n"
                                "[Harmonisch]      allgemeine rechte Seiten grundsaetzlich zu "
                                "schwer -- und der\n"
                                "[Harmonisch]      C-to-R-Zweig braucht mehr als eine "
                                "Projektion.\n";
                        }
                    }

                    if (!(res_best <= probe_tol * 10.0)) {
                        std::cout <<
                            "[Harmonisch]\n"
                            "[Harmonisch] ABBRUCH -- und das ist ein BEFUND, kein Fehler.\n"
                            "[Harmonisch]   Der innere Loeser erreicht die geforderte Toleranz "
                            "nicht. C-to-R\n"
                            "[Harmonisch]   kann damit nicht wirken, und der aeussere Lauf waere "
                            "Stunden fuer\n"
                            "[Harmonisch]   nichts.\n"
                            "[Harmonisch]\n"
                            "[Harmonisch]   Die harmonische Formulierung ist damit NICHT "
                            "widerlegt: der aeussere\n"
                            "[Harmonisch]   Loeser braucht mit exaktem (W+T) nur 38 Iterationen "
                            "(gemessen mit\n"
                            "[Harmonisch]   --harmonic-study). Was fehlt, ist ein tragfaehiger "
                            "Vorkonditionierer\n"
                            "[Harmonisch]   fuer (W+T) = K + omega*C -- einen REELLEN gekoppelten "
                            "Operator, also\n"
                            "[Harmonisch]   genau das Problem, das der transiente Loeser ohnehin "
                            "schon hat, nur\n"
                            "[Harmonisch]   ohne den Umweg ueber 1000 Zeitschritte.\n"
                            "[Harmonisch]\n"
                            "[Harmonisch]   Naechster Schritt ist deshalb der Vorkonditionierer "
                            "fuer (W+T),\n"
                            "[Harmonisch]   nicht der naechste Umbau der Formulierung.\n"
                            "[Harmonisch]\n"
                            "[Harmonisch]   Diese Vorabmessung lief mit vollem GMRES (Restart = "
                            "Iterationsdeckel),\n"
                            "[Harmonisch]   der Befund haengt also nicht an der Restart-Laenge. "
                            "Der Deckel selbst\n"
                            "[Harmonisch]   koennte aber zu niedrig sein -- die Deckenmessung "
                            "brauchte 2914 Schritte\n"
                            "[Harmonisch]   bis 1e-4. Ein hoeherer Deckel ist der billigste "
                            "naechste Versuch.\n";
                        return 0;
                    }
                    std::cout << "[Harmonisch]   -> traegt. Der aeussere Lauf beginnt.\n";

                } else {
                    // Ohne die Proben faengt die Warnung in
                    // CtoRPreconditioner den Fall ab, dass der innere Loeser
                    // seine Toleranz verfehlt. Sie meldet EINMAL und laesst
                    // weiterlaufen -- schlechter als ein Abbruch vor dem
                    // Rechnen, aber die Proben kosten rund 40000 innere
                    // Schritte, bevor die erste aeussere Iteration beginnt.
                    //
                    // Die ILU-Entscheidung faellt dann nicht per Messung.
                    // Gemessen wurde 2502 Schritte mit gegen 2590 ohne -- der
                    // Unterschied ist marginal, also bleibt ILU an, so wie
                    // compute_ilu() es oben aufgesetzt hat.
                    std::cout << "\n[Harmonisch] Diagnosen uebersprungen. Mit "
                                 "--harmonic-checks laufen Vorabmessung,\n"
                                 "[Harmonisch] Eichungs-Gegenprobe und "
                                 "Linearitaetspruefung -- das kostet rund 40000\n"
                                 "[Harmonisch] innere Schritte vor der ersten aeusseren "
                                 "Iteration.\n";
                }

                // Phasor-rechte-Seite: der Quellterm ist der Spulenstrom als
                // komplexe Amplitude. Bei reeller Amplitude (Kosinus-Anregung)
                // ist der Imaginaerteil null.
                DeviceArray<double> d_br(static_cast<size_t>(nd_h));
                DeviceArray<double> d_bi(static_cast<size_t>(nd_h));
                DeviceArray<double> d_xr(static_cast<size_t>(nd_h));
                DeviceArray<double> d_xi(static_cast<size_t>(nd_h));
                // -----------------------------------------------------------
                // Die rechte Seite muss die VOLLE Amplitude tragen.
                // -----------------------------------------------------------
                //
                // d_rhs_coupled ist der Quellterm des ZEITSCHRITTS, und der
                // Spulenstrom ist sinusfoermig:
                //
                //     i(t) = I_max * sin(2*pi*f*t)     (PhysicalObject.h:245)
                //
                // Bei t = dt = 1e-04 s und f = 50 Hz ist sin(0,0314) = 0,0314
                // -- die rechte Seite tragt also nur 3,1 % der Amplitude. Die
                // Kraft ist quadratisch in den Feldern, es fehlte damit
                // Faktor 1000. Genau das war zu sehen: -6,9e-05 N statt der
                // geforderten 1,0497 N bei 11,3 mm.
                //
                // Der Phasor braucht I_max, nicht i(dt). Skaliert wird
                // deshalb mit 1/sin(w*dt) zurueck.
                //
                // Sauberer waere ein eigener Quellterm-Aufbau mit der
                // Amplitude direkt. Das Zurueckskalieren ist exakt, solange
                // der Quellterm LINEAR im Strom ist -- und das ist er, die
                // Anregung geht nur ueber d_local_F ein. Bei sin(w*dt) nahe
                // null waere es allerdings numerisch unsinnig, deshalb die
                // Pruefung unten.
                {
                    const double s = std::sin(omega * dt);
                    if (!(std::fabs(s) > 1e-3)) {
                        std::cout << "[Harmonisch] ABBRUCH: sin(omega*dt) = " << s
                                  << " ist zu klein, um die Amplitude aus dem\n"
                                  << "[Harmonisch] Zeitschritt-Quellterm zurueckzurechnen. "
                                     "Waehlen Sie --dt so, dass\n"
                                  << "[Harmonisch] omega*dt nicht nahe einem Vielfachen von "
                                     "pi liegt (etwa dt = 1/(4f)).\n";
                        return 1;
                    }
                    const double amp_scale = 1.0 / s;
                    bf::memcpy_d(d_br.data(), d_rhs_coupled.data(),
                                 static_cast<size_t>(nd_h) * sizeof(double),
                                 cudaMemcpyDeviceToDevice);
                    bf::dscal(op.get_cublas_handle(), nd_h, amp_scale, d_br.data());
                    std::cout << "[Harmonisch] Quellterm auf die volle Amplitude skaliert: "
                                 "sin(omega*dt) = " << s
                              << ", Faktor " << amp_scale << "\n";
                }
                bf::memset_d(d_bi.data(), 0, static_cast<size_t>(nd_h) * sizeof(double));
                bf::memset_d(d_xr.data(), 0, static_cast<size_t>(nd_h) * sizeof(double));
                bf::memset_d(d_xi.data(), 0, static_cast<size_t>(nd_h) * sizeof(double));
                bf::sync();

                // inner_tol = 1e-4: die Genauigkeitsreihe aus --harmonic-study
                // hat gezeigt, dass 1e-1 und 1e-2 nicht tragen, 1e-3 mit 81 und
                // 1e-4 mit 41 aeusseren Iterationen -- und dass die
                // GESAMTARBEIT bei 1e-4 am kleinsten ist (41 x 2914 gegen
                // 81 x 2899).
                const bf::HarmonicSolveResult hres =
                    bf::solve_harmonic(op, csr_C, nd_h, total_dofs, omega,
                                       d_br.data(), d_bi.data(),
                                       d_xr.data(), d_xi.data(),
                                       1e-6,
                                       (bf::harmonic_outer_cap() > 0
                                            ? bf::harmonic_outer_cap()
                                            : bf::max_iter()),
                                       // AEUSSERER Restart: bewusst KURZ und
                                       // NICHT --gmres-restart. Der aeussere
                                       // Loeser braucht laut Studie 38 bis 41
                                       // Iterationen; ein Restart von 3000
                                       // legte 4 x 3000 Vektoren an, also
                                       // 433 MB fuer nichts. Der lange Restart
                                       // gehoert zum INNEREN Loeser, und dort
                                       // steht er auch. Beide an denselben
                                       // Schalter zu haengen war falsch.
                                       bf::harmonic_outer_restart(),
                                       bf::harmonic_inner_tol(),
                                       std::max(4000, bf::gmres_restart()));

                // Zurueck in den physikalischen Raum.
                op.scale_by_D(d_xr.data());
                op.scale_by_D(d_xi.data());
                bf::sync();

                if (!hres.converged)
                    std::cout << "[Harmonisch] WARNUNG: nicht konvergiert. Die Kraft unten "
                                 "ist entsprechend unsicher.\n";

                // -----------------------------------------------------------
                // Zeitgemittelte Kraft aus dem Phasor
                // -----------------------------------------------------------
                //
                // Die Loesung ist ein Phasor: das tatsaechliche Feld ist
                // A(t) = Re(A^ e^{jwt}) = A_r cos(wt) - A_i sin(wt).
                //
                // Die Kraft ist QUADRATISCH in den Feldern, ihr Zeitmittel
                // also nicht die Kraft des gemittelten Feldes -- das waere
                // null, weil das Zeitmittel eines Sinus null ist. Gemittelt
                // werden muss die KRAFT, nicht das Feld.
                //
                // Das Mittel ueber N >= 3 gleichverteilte Phasen ist dabei
                // exakt: die quadratische Groesse enthaelt nur einen
                // konstanten und einen 2w-Anteil, und N gleichverteilte
                // Stuetzstellen loeschen den 2w-Anteil vollstaendig aus.
                // Deshalb wird der VORHANDENE, reelle Kraftpfad N-mal
                // aufgerufen statt der Maxwell-Tensor komplex neu geschrieben
                // -- exakt, ohne eine Zeile neuer Physik.
                //
                // N = 8 statt der genuegenden 3, weil der Kraftpfad ueber
                // Flaechenintegration und Interpolation laeuft und dabei
                // Anteile hoeherer Ordnung entstehen koennen.
                {
                    const int n_phases = 8;
                    const std::vector<double> phases = bf::harmonic_phases(n_phases);

                    // Nur der FEM-Anteil geht in den Kraftpfad; der BEM-Anteil
                    // der Loesung wird dort nicht gebraucht.
                    std::vector<double> h_ar(static_cast<size_t>(total_dofs));
                    std::vector<double> h_ai(static_cast<size_t>(total_dofs));
                    bf::memcpy_d(h_ar.data(), d_xr.data(),
                                 static_cast<size_t>(total_dofs) * sizeof(double),
                                 cudaMemcpyDeviceToHost);
                    bf::memcpy_d(h_ai.data(), d_xi.data(),
                                 static_cast<size_t>(total_dofs) * sizeof(double),
                                 cudaMemcpyDeviceToHost);

                    // Wie gross ist der Imaginaerteil ueberhaupt?
                    //
                    // Bei A_i = 0 zerfiele das System in K A_r = b_r und
                    // K A_i = 0: die Loesung waere rein reell, es gaebe keine
                    // Phasenverzoegerung der Wirbelstroeme und damit auch
                    // keine Levitation. Die beiden Normen sagen unmittelbar,
                    // ob omega*C ueberhaupt in den Loeser koppelt.
                    {
                        double nr = 0.0, ni = 0.0;
                        for (int i = 0; i < total_dofs; ++i) {
                            nr += h_ar[static_cast<size_t>(i)] * h_ar[static_cast<size_t>(i)];
                            ni += h_ai[static_cast<size_t>(i)] * h_ai[static_cast<size_t>(i)];
                        }
                        nr = std::sqrt(nr); ni = std::sqrt(ni);
                        std::cout << "[Harmonisch] ||A_r|| = " << nr
                                  << ", ||A_i|| = " << ni
                                  << ",  Verhaeltnis = " << (nr > 0.0 ? ni / nr : -1.0) << "\n";
                        if (nr > 0.0 && ni / nr < 1e-6)
                            std::cout << "[Harmonisch]   -> der Imaginaerteil ist praktisch null. "
                                         "omega*C koppelt nicht,\n"
                                         "[Harmonisch]      das System zerfaellt in zwei "
                                         "unabhaengige Haelften, und ohne\n"
                                         "[Harmonisch]      Phasenverzoegerung ist das "
                                         "Lorentz-Zeitmittel zwangslaeufig null.\n";
                    }

                    DeviceArray<double> d_a_phase(static_cast<size_t>(total_dofs));
                    std::vector<double> h_phase;
                    double3 f_sum{0.0, 0.0, 0.0};
                    double3 t_sum{0.0, 0.0, 0.0};

                    // -------------------------------------------------------
                    // Lorentz-Volumenintegral, geschlossen aus dem Phasor
                    // -------------------------------------------------------
                    //   <J x B> = (sigma*w/2) ( A_i x rot A_r - A_r x rot A_i )
                    //
                    // Ein Durchlauf ueber die Hexaeder, keine Phasenschleife.
                    // Der Maxwell-Oberflaechenpfad daneben rechnet dieselbe
                    // Groesse als Differenz sich fast aufhebender
                    // Flaechenbeitraege -- wie gross dessen Fehler ist, zeigt
                    // F(A_r) = -0,122 N, wo bei mu_r = 1 und stromfreiem
                    // Inneren exakt null stehen muesste.
                    MechanicalLoads Lvol{};
                    {
                        DeviceArray<double> d_xr_fem(static_cast<size_t>(total_dofs));
                        DeviceArray<double> d_xi_fem(static_cast<size_t>(total_dofs));
                        bf::memcpy_d(d_xr_fem.data(), h_ar.data(),
                                     static_cast<size_t>(total_dofs) * sizeof(double),
                                     cudaMemcpyHostToDevice);
                        bf::memcpy_d(d_xi_fem.data(), h_ai.data(),
                                     static_cast<size_t>(total_dofs) * sizeof(double),
                                     cudaMemcpyHostToDevice);
                        DeviceArray<double> acc(6);
                        bf::memset_d(acc.data(), 0, 6 * sizeof(double));
                        bf::sync();
                        BF_LAUNCH_1D(harmonic_lorentz, mesh.num_hexes,
                            static_cast<int>(mesh.num_hexes),
                            mesh.hex_nodes.data(), mesh.nodes_current.data(),
                            mesh.hex_edges.data(), mesh.hex_edge_signs.data(),
                            mesh.elem_types.data(),
                            d_materials.data(),
                            nullptr,   // dieser Pfad regularisiert nicht, kappa trennt noch
                            d_xr_fem.data(), d_xi_fem.data(), omega,
                            plate.get_properties().center_of_mass,
                            acc.data() + 0, acc.data() + 1, acc.data() + 2,
                            acc.data() + 3, acc.data() + 4, acc.data() + 5);
                        bf::sync();
                        double h6[6];
                        bf::memcpy_d(h6, acc.data(), 6 * sizeof(double),
                                     cudaMemcpyDeviceToHost);
                        Lvol.force  = make_double3(h6[0], h6[1], h6[2]);
                        Lvol.torque = make_double3(h6[3], h6[4], h6[5]);
                    }

                    std::cout << "\n[Harmonisch] Zeitgemittelte Kraft aus " << n_phases
                              << " Phasen:\n";
                    for (int k = 0; k < n_phases; ++k) {
                        bf::harmonic_phase_sample(h_ar, h_ai, phases[k], h_phase);
                        bf::memcpy_d(d_a_phase.data(), h_phase.data(),
                                     static_cast<size_t>(total_dofs) * sizeof(double),
                                     cudaMemcpyHostToDevice);
                        bf::sync();
                        const MechanicalLoads L = TransientSolver::compute_maxwell_loads(
                            mesh, d_a_phase.data(), plate.get_properties().center_of_mass);
                        f_sum.x += L.force.x;  f_sum.y += L.force.y;  f_sum.z += L.force.z;
                        t_sum.x += L.torque.x; t_sum.y += L.torque.y; t_sum.z += L.torque.z;

                        std::cout << "[Harmonisch]   Phase " << k << " ("
                                  << (360.0 * k / n_phases) << " Grad): F_z = "
                                  << L.force.z << " N\n";
                    }
                    const double inv = 1.0 / static_cast<double>(n_phases);
                    const double3 f_avg{f_sum.x * inv, f_sum.y * inv, f_sum.z * inv};
                    const double3 t_avg{t_sum.x * inv, t_sum.y * inv, t_sum.z * inv};

                    std::cout << "[Harmonisch]\n"
                              << "[Harmonisch] Hoehe h = "
                              << (bf::plate_z_mm() > 0.0 ? bf::plate_z_mm() : 3.8)
                              << " mm\n"
                              << "[Harmonisch] ZEITGEMITTELTE KRAFT : ("
                              << f_avg.x << ", " << f_avg.y << ", " << f_avg.z << ") N\n"
                              << "[Harmonisch] ZEITGEMITTELTES MOMENT: ("
                              << t_avg.x << ", " << t_avg.y << ", " << t_avg.z << ") Nm\n"
                              << "[Harmonisch] LORENTZ (Volumen)     : ("
                              << Lvol.force.x << ", " << Lvol.force.y << ", "
                              << Lvol.force.z << ") N\n"
                              << "[Harmonisch]   Der Volumenwert ist der belastbarere: das\n"
                                 "[Harmonisch]   Oberflaechenintegral gibt fuer den Realteil "
                                 "allein -0,122 N,\n"
                                 "[Harmonisch]   wo bei mu_r = 1 exakt null stehen muesste.\n";

                    // Gewichtskraft zum Vergleich -- die Schwebehoehe ist die
                    // Hoehe, bei der F_z = m*g gilt.
                    // mass ist ein Feld von PhysicalObject, NICHT von
                    // MaterialProperties -- get_properties().mass gibt es
                    // nicht. Es ist ausserdem protected, deshalb der neue
                    // Getter get_mass(). Ich hatte die Deklaration gelesen und
                    // den Zugriffsmodifizierer drei Zeilen darueber uebersehen.
                    const double mass = plate.get_mass();
                    const double weight = mass * 9.80665;
                    std::cout << "[Harmonisch]\n"
                              << "[Harmonisch] Platte: m = " << mass << " kg, Gewicht = "
                              << weight << " N\n"
                              << "[Harmonisch] F_z / (m g) = " << (f_avg.z / weight) << "\n";
                    if (f_avg.z > weight)
                        std::cout << "[Harmonisch]   -> Auftrieb ueberwiegt: die "
                                     "Gleichgewichtshoehe liegt HOEHER als\n"
                                     "[Harmonisch]      die aktuelle Plattenposition.\n";
                    else
                        std::cout << "[Harmonisch]   -> Gewicht ueberwiegt: die "
                                     "Gleichgewichtshoehe liegt TIEFER als\n"
                                     "[Harmonisch]      die aktuelle Plattenposition.\n";
                    std::cout << "[Harmonisch]   Die Schwebehoehe folgt aus F_z(h) = m g, also "
                                 "aus mehreren Laeufen\n"
                                 "[Harmonisch]   mit verschiedenem --plate-z und einer "
                                 "Nullstellensuche darueber.\n";
                }
                return 0;
            }

            solver.solve(op, d_rhs_coupled.data(), d_x_coupled.data(), 1e-6, bf::max_iter());

            // 3. Ruecktransformation der Loesung: x = D * x_tilde
            op.scale_by_D(d_x_coupled.data());
            bf::sync();

            // A-Feld aus dem gekoppelten Loesungsvektor extrahieren (die
            // ersten total_dofs Eintraege sind exakt A) -- alles Folgende
            // (Kraftberechnung, Bewegung, VTK-Export, d_A_previous) bleibt
            // dadurch unveraendert, da es weiterhin nur d_A_current sieht.
            bf::memcpy_d(d_A_current.data(), d_x_coupled.data(),
                         static_cast<size_t>(total_dofs) * sizeof(double),
                         cudaMemcpyDeviceToDevice);

            // --verify-field: B_z auf der Achse gegen Biot-Savart, dann Schluss.
            // Gerechnet wurde mit sigma = 0 (oben genullt, vor der Eichung), es
            // gibt also keine Wirbelstroeme -- uebrig bleibt das reine
            // Quellfeld der Spulen, und dafuer ist die analytische Loesung
            // bekannt. Damit ist entscheidbar, ob dieser Pfad bereits das FELD
            // verfehlt, ohne dass Wirbelstroeme oder Kraftschaetzer etwas
            // verschleiern.
            // Pruefmodus: nicht 300 Schritte lang auf einem nicht geloesten
            // System weiterrechnen. Die Feldprobe vergleicht B_z gegen eine
            // analytische Loesung -- das ist nur sinnvoll, wenn das System
            // ueberhaupt geloest wurde.
            if (bf::verify_field() && solver.last_relative_residual() > 1e-5) {
                std::cout << "\n========================================================\n"
                          << " FELDPROBE ABGEBROCHEN: GMRES nicht konvergiert         \n"
                          << "========================================================\n"
                          << " Relatives Residuum nach dem Solve: "
                          << solver.last_relative_residual() << "\n\n"
                          << " Gerechnet wurde MAGNETOSTATISCH (Gleichstrom 20 A, kein\n"
                          << " Zeitschritt-Term nach Abklingen). Das ist der einfachste\n"
                          << " Fall dieses Loesers -- konvergiert er hier nicht, liegt es\n"
                          << " NICHT an der Wirbelstromphysik.\n\n"
                          << " Ein Vergleich von B_z gegen Biot-Savart waere jetzt\n"
                          << " bedeutungslos: verglichen wuerde nicht die Diskretisierung,\n"
                          << " sondern der Abbruchzustand des Krylov-Verfahrens.\n\n"
                          << " Naechster Schritt zur Eingrenzung -- dieselbe Rechnung ohne\n"
                          << " BEM-Kopplung, also ein reines geeichtes curl-curl-System:\n"
                          << "     ./fembem --verify-field nobem\n"
                          << " Konvergiert das, sitzt der Fehler in der BEM-Kopplung.\n"
                          << " Stagniert es auch, in Eichung, Operator oder\n"
                          << " Vorkonditionierer. Zusaetzlich \"noilu\" trennt den\n"
                          << " Vorkonditionierer ab.\n"
                          << "========================================================\n";
                return 1;
            }

            if (bf::verify_field() && step == max_steps) {
                DeviceArray<double> d_bz(mesh.num_hexes);
                BF_LAUNCH_1D_PAR(verify_bz_nedelec, mesh.num_hexes,
                    static_cast<int>(mesh.num_hexes), mesh.hex_nodes.data(),
                    mesh.nodes_current.data(), mesh.hex_edges.data(),
                    mesh.hex_edge_signs.data(), mesh.elem_types.data(),
                    d_A_current.data(), d_bz.data());
                bf::sync();

                d_bz.prefetch_to_host();
                mesh.nodes_current.prefetch_to_host();
                mesh.hex_nodes.prefetch_to_host();
                bf::sync();

                // Dieselben Auswertepunkte wie im Referenzpfad: innerster
                // Plattenelementring. Die Auswahl liegt hier und nicht in
                // FieldVerification, weil sie die Plattenelemente kennen muss.
                std::vector<FieldSample> samples;
                double r_min = 1e30;
                for (int h : plate.get_hex_indices()) {
                    const Hexahedron H = mesh.hex_nodes[h];
                    double cx = 0.0, cy = 0.0;
                    // Mittelung ueber die BELEGTEN Knoten. Der frueher feste
                    // Faktor 0.125 gilt nur fuer acht; bei einem Tetraeder
                    // waere der Schwerpunkt sonst um die Haelfte zu klein --
                    // und die Radiusauswahl unten damit stillschweigend falsch.
                    const int nn = element_num_nodes(mesh.elem_type(h));
                    for (int n = 0; n < nn; ++n) {
                        cx += mesh.nodes_current[H.v[n]].x;
                        cy += mesh.nodes_current[H.v[n]].y;
                    }
                    cx /= nn; cy /= nn;
                    r_min = std::fmin(r_min, std::sqrt(cx * cx + cy * cy));
                }
                for (int h : plate.get_hex_indices()) {
                    const Hexahedron H = mesh.hex_nodes[h];
                    double cx = 0.0, cy = 0.0, cz = 0.0;
                    const int nn = element_num_nodes(mesh.elem_type(h));
                    for (int n = 0; n < nn; ++n) {
                        const double3 p = mesh.nodes_current[H.v[n]];
                        cx += p.x; cy += p.y; cz += p.z;
                    }
                    cx /= nn; cy /= nn; cz /= nn;
                    const double rr = std::sqrt(cx * cx + cy * cy);
                    if (rr > r_min * 1.05) continue;
                    samples.push_back({rr, cz, d_bz[h]});
                }

                report_field_verification("Nedelec-Kantenelemente + ACA-BEM",
                                          samples, 20.0, r_min);
                return 0;
            }

            // f) Kräfte berechnen (Maxwell Tensor auf der Aluminiumplatte)
            MechanicalLoads loads = TransientSolver::compute_maxwell_loads(
                mesh, d_A_current.data(), plate.get_properties().center_of_mass
            );

            // g) Platte bewegen (Newton-Euler Gleichungen)
            plate.apply_newtons_law(loads.force, loads.torque, dt);

             // Wenn die Platte sich bewegt hat, Koordinaten updaten
            if (plate.is_moving()) {
                apply_kinematics(mesh, plate);
                tree.update_tree_kinematics(mesh);
                
                // WICHTIG: Die Host-Kopie für den ACA-Evaluator aktualisieren!
                // mesh.nodes_current wurde vom Kernel verschoben, wir müssen es in den RAM laden.
                bf::sync();
                mesh.nodes_current.prefetch_to_host();
                for (size_t i = 0; i < mesh.num_nodes; ++i) {
                    host_nodes_current[i] = mesh.nodes_current[i];
                }
            }

            // h) ParaView Export
            char filename[256];
            snprintf(filename, sizeof(filename), "team28_out_%04d.vtk", step);
            VtkExporter::export_hexahedral_cell_data(filename, mesh, d_materials.data(), d_A_current.data(), d_A_previous.data(), dt);

            // i) State Update für nächsten Zeitschritt
            bf::memcpy_d(d_A_previous.data(), d_A_current.data(),
                         static_cast<size_t>(total_dofs) * sizeof(double),
                         cudaMemcpyDeviceToDevice);
       
        

        // j) Aufräumen des dynamischen GPU-Speichers für diesen Zeitschritt
        for (void* ptr : allocated_gpu_mem) {
            bf::free_d(ptr);
        }
        allocated_gpu_mem.clear();

    }
        std::cout << "\n========================================================\n";
        std::cout << " Simulation Successfully Finished! \n";
        std::cout << "========================================================\n";

    }
    catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}