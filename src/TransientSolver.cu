#include "TransientSolver.cuh"
#include "Backend.cuh"
#include "NedelecHexahedron.cuh"
#include "MaxwellLoads.cuh"
#include <iostream>
#include <stdexcept>


// =================================================================================
// Kernel 7.1: Impliziter Euler System-Assemblierung
// Berechnet A_sys = K + (1/dt)*C und den RHS-Vektor = J_s + (1/dt)*C*A_prev
// =================================================================================
// Rumpf gemeinsam fuer beide Pfade. ACHTUNG: streut per bf::atomic_add in
// rhs_vector, benachbarte Hexaeder teilen sich Kanten. Auf der CPU deshalb
// bewusst die SERIELLE Startvariante (BF_LAUNCH_1D, nicht _PAR) -- dort ist
// bf::atomic_add eine gewoehnliche Addition.
BF_HD inline void implicit_euler_kernel_body(
    int hex_id,
    int num_hexes,
    double dt,
    const double* local_K,
    const double* local_C,
    const double* local_F,
    const int* hex_edges,
    const int* hex_edge_signs,
    const double* A_prev,
    const MaterialProperties* materials,
    double* local_A_sys,
    double* rhs_vector
) {
    (void)num_hexes;
    (void)materials;

    double inv_dt = 1.0 / dt;
    int offset = hex_id * 144;

    // 1. Matrix Addition A_sys = K + (1/dt)*C
    for (int i = 0; i < 144; ++i) {
        local_A_sys[offset + i] = local_K[offset + i] + inv_dt * local_C[offset + i];
    }

    // 2. Rechte Seite berechnen: (1/dt)*C*A_prev
    // Hole lokale A_prev Werte mit Vorzeichenkorrektur
    // Vorbelegung ist auf einem gemischten Netz zwingend, nicht kosmetisch.
    // Bei einem Tetraeder bleiben die Plaetze 6..11 ungeschrieben, die
    // Multiplikation unten laeuft aber ueber alle zwoelf. local_C ist dort
    // sauber null -- und 0.0 * NaN ist NaN. Ein NaN-Bitmuster im Frame
    // vergiftet damit alle sechs echten Zeilen des Elements. Genau so
    // entstanden die 90 NaN in der rechten Seite des BDF-Netzes: eine kleine,
    // unrunde, laufabhaengige Zahl, weil nur ein Teil der Threads ein solches
    // Muster erwischt. Auf einem reinen Hexaedernetz konnte es nicht auffallen.
    double A_loc_prev[12] = {0.0};
    int global_edges[12];
    int signs[12];

    for (int i = 0; i < 12; ++i) {
        global_edges[i] = hex_edges[hex_id * 12 + i];
        signs[i] = hex_edge_signs[hex_id * 12 + i];
        // -1 = ungenutzter Platz eines Tetraeders oder Prismas.
        if (global_edges[i] < 0) { signs[i] = 0; continue; }
        A_loc_prev[i] = A_prev[global_edges[i]] * signs[i];
    }

    // Matrix-Vektor Multiplikation für dieses Element
    double rhs_local[12] = {0.0};
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) {
            rhs_local[i] += local_C[offset + i * 12 + j] * A_loc_prev[j];
        }
        rhs_local[i] *= inv_dt;

        // Eingepraegte Stroeme J_s. Das Integral W_i . J_s dV wird in
        // assemble_local_matrices_kernel je Hexaeder ausgewertet und liegt
        // unorientiert in local_F; die Orientierung kommt unten beim Scatter
        // ueber signs[i] dazu, genau wie bei K und C.
        //
        // Dieser Term fehlte vorher vollstaendig. Der RHS bestand nur aus
        // C/dt * A_prev und war mit A_prev = 0 identisch null: der BiCGStab
        // meldete "Converged at iteration 0", A blieb null, und die
        // Maxwell-Kraft war (0,0,0) N in jedem Zeitschritt.
        rhs_local[i] += local_F[hex_id * 12 + i];
    }

    // 3. Atomares Aufaddieren in den globalen Vektor (RHS Scatter)
    // Beachte die erneute Multiplikation mit signs[i] für das Scatter-Mapping
    // signs[i] = 0 neutralisiert den WERT, nicht den INDEX. Ohne diese
    // Abfrage schreibt jeder Tetraeder sechsmal auf rhs_vector[-1], also vor
    // den Anfang des Feldes -- 22152 Zugriffe beim BDF-Netz.
    for (int i = 0; i < 12; ++i) {
        if (global_edges[i] < 0) continue;
        bf::atomic_add(&rhs_vector[global_edges[i]], rhs_local[i] * signs[i]);
    }
}

__global__ void implicit_euler_kernel(
    int num_hexes,
    double dt,
    const double* local_K,
    const double* local_C,
    const double* local_F,
    const int* hex_edges,
    const int* hex_edge_signs,
    const double* A_prev,
    const MaterialProperties* materials,
    double* local_A_sys,
    double* rhs_vector
) {
    int hex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hex_id >= num_hexes) return;
    implicit_euler_kernel_body(hex_id, num_hexes, dt, local_K, local_C, local_F,
                               hex_edges, hex_edge_signs, A_prev, materials,
                               local_A_sys, rhs_vector);
}

void TransientSolver::build_implicit_euler_system(
    int num_hexes, int total_dofs, double dt,
    const double* d_local_K, const double* d_local_C,
    const double* d_local_F,
    const int* d_hex_edges, const int* d_hex_edge_signs,
    const double* d_A_prev, const MaterialProperties* d_materials,
    double* d_local_A_sys, double* d_rhs_vector) 
{
    bf::memset_d(d_rhs_vector, 0, static_cast<size_t>(total_dofs) * sizeof(double));

    BF_LAUNCH_1D(implicit_euler_kernel, num_hexes,
        num_hexes, dt, d_local_K, d_local_C, d_local_F,
        d_hex_edges, d_hex_edge_signs, d_A_prev, d_materials,
        d_local_A_sys, d_rhs_vector);
    bf::sync();
}


// =================================================================================
// Kernel 7.2: Maxwell Spannungstensor auf der BEM-Oberfläche
// =================================================================================
// Der Rumpf compute_maxwell_loads_kernel_body steht in MaxwellLoads.cuh,
// damit tests/test_piola_production ihn ohne den Rest dieser Datei
// aufrufen kann. Reduziert per bf::atomic_add auf sechs Skalare -- auf
// der CPU deshalb seriell starten (BF_LAUNCH_1D).

__global__ void compute_maxwell_loads_kernel(
    int num_surface_faces,
    const Mesh::QuadFace* faces,
    const Hexahedron* hex_nodes,
    const double3* current_nodes,
    const int* hex_edges,
    const int* hex_edge_signs,
    const double* A_current,
    double3 center_of_mass,
    double* out_force_x, double* out_force_y, double* out_force_z,
    double* out_torque_x, double* out_torque_y, double* out_torque_z
) {
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_surface_faces) return;
    compute_maxwell_loads_kernel_body(
        face_idx, num_surface_faces, faces, hex_nodes, current_nodes, hex_edges,
        hex_edge_signs, A_current, center_of_mass,
        out_force_x, out_force_y, out_force_z,
        out_torque_x, out_torque_y, out_torque_z);
}

MechanicalLoads TransientSolver::compute_maxwell_loads(
    const Mesh& mesh, const double* d_A_current, const double3 center_of_mass) 
{
    // Speicher für die globalen Summen allozieren (Device)
    double* d_f_x; double* d_f_y; double* d_f_z;
    double* d_m_x; double* d_m_y; double* d_m_z;
    
    bf::malloc_d(&d_f_x, sizeof(double)); bf::malloc_d(&d_f_y, sizeof(double)); bf::malloc_d(&d_f_z, sizeof(double));
    bf::malloc_d(&d_m_x, sizeof(double)); bf::malloc_d(&d_m_y, sizeof(double)); bf::malloc_d(&d_m_z, sizeof(double));

    bf::memset_d(d_f_x, 0, sizeof(double)); bf::memset_d(d_f_y, 0, sizeof(double)); bf::memset_d(d_f_z, 0, sizeof(double));
    bf::memset_d(d_m_x, 0, sizeof(double)); bf::memset_d(d_m_y, 0, sizeof(double)); bf::memset_d(d_m_z, 0, sizeof(double));

    // component_surface_faces (Grenzflaeche der Aluminiumplatte, siehe
    // TopologyExtractor::extract_component_surface_faces) statt boundary_faces
    // (aeussere BEM-Kopplungsflaeche des GESAMTEN Gebiets -- Spulen und Platte
    // gemeinsam, dafuer NICHT geeignet, siehe Kommentar in Mesh.cuh). Eine
    // Maxwell-Spannungstensor-Integration fuer "die Kraft auf die Platte"
    // braucht eine geschlossene Flaeche, die NUR die Platte umschliesst.
    if (mesh.num_component_surface_faces == 0) {
        throw std::runtime_error("compute_maxwell_loads: mesh.component_surface_faces ist leer -- "
                                  "TopologyExtractor::extract_component_surface_faces() wurde nicht aufgerufen.");
    }

    // Dieser Kernel integriert ueber VIERECKSflaechen mit HEXAEDER-Nachbarn:
    // die 2x2-Gaussregel, die Flaechennormale aus local_face_to_nodes und die
    // Auswertung der Nachbarelementbasis setzen das alle voraus. Auf einem
    // Tetraeder- oder Prismennetz gibt es Dreiecksflaechen, und dann rechnet
    // er still falsch -- kein NaN, nur eine falsche Kraft.
    //
    // Verallgemeinert wird er absichtlich NICHT: fuer den FEM-direkt-Pfad ist
    // das Lorentz-Volumenintegral (HarmonicLorentz.cuh) der Kraftpfad. Es
    // braucht keine Feldspur von aussen und ist auf allen Elementtypen
    // ausgewertet. Siehe AP7 in docs/design/fem-direct-plan.md.
    if (!mesh.is_pure_hex()) {
        throw std::runtime_error(
            "compute_maxwell_loads: der Maxwell-Spannungstensor ist hier nur fuer "
            "reine Hexaedernetze umgesetzt (Vierecks-Panels, 2x2-Gauss). Auf einem "
            "Tetraeder- oder Prismennetz stattdessen das Lorentz-Volumenintegral "
            "benutzen -- das ist fuer den FEM-direkt-Pfad ohnehin der genauere Weg, "
            "weil es keine Feldspur auf der Luftseite braucht.");
    }
    BF_LAUNCH_1D(compute_maxwell_loads_kernel, mesh.num_component_surface_faces,
        static_cast<int>(mesh.num_component_surface_faces),
        mesh.component_surface_faces.data(),
        mesh.hex_nodes.data(),
        mesh.nodes_current.data(),
        mesh.hex_edges.data(),
        mesh.hex_edge_signs.data(),
        d_A_current,
        center_of_mass,
        d_f_x, d_f_y, d_f_z,
        d_m_x, d_m_y, d_m_z);

    bf::sync();

    MechanicalLoads loads;
    bf::memcpy_d(&loads.force.x, d_f_x, sizeof(double), cudaMemcpyDeviceToHost);
    bf::memcpy_d(&loads.force.y, d_f_y, sizeof(double), cudaMemcpyDeviceToHost);
    bf::memcpy_d(&loads.force.z, d_f_z, sizeof(double), cudaMemcpyDeviceToHost);

    bf::memcpy_d(&loads.torque.x, d_m_x, sizeof(double), cudaMemcpyDeviceToHost);
    bf::memcpy_d(&loads.torque.y, d_m_y, sizeof(double), cudaMemcpyDeviceToHost);
    bf::memcpy_d(&loads.torque.z, d_m_z, sizeof(double), cudaMemcpyDeviceToHost);

    // Cleanup
    bf::free_d(d_f_x); bf::free_d(d_f_y); bf::free_d(d_f_z);
    bf::free_d(d_m_x); bf::free_d(d_m_y); bf::free_d(d_m_z);

    std::cout << "[Maxwell Tensor] Net Force : (" << loads.force.x << ", " << loads.force.y << ", " << loads.force.z << ") N\n";
    std::cout << "[Maxwell Tensor] Net Torque: (" << loads.torque.x << ", " << loads.torque.y << ", " << loads.torque.z << ") Nm\n";

    return loads;
}

// =================================================================================
// GPU Kernel: Scatter lokaler 12x12 Matrizen in das globale CSR Array
// =================================================================================
// Rumpf gemeinsam fuer beide Pfade. Streut per bf::atomic_add in csr_values;
// mehrere Hexaeder treffen denselben Eintrag. Auf der CPU seriell starten.
BF_HD inline void scatter_to_csr_kernel_body(
    int hex_id,
    int num_hexes,
    const double* local_A_sys,     // Array der Größe num_hexes * 144
    const int* hex_edges,          // Mapping Hexaeder -> globale Kante
    const int* hex_edge_signs,     // +1 / -1
    const int* csr_row_offsets,    // Startindex jeder Zeile in col_indices
    const int* csr_col_indices,    // Spaltenindizes der Non-Zeros
    double* csr_values             // Ziel-Array für die Werte
) {
    (void)num_hexes;

    int offset_local = hex_id * 144;
    int offset_mapping = hex_id * 12;

    // Hole die 12 globalen Kanten-IDs und Vorzeichen für dieses Hexaeder
    int global_edges[12];
    int signs[12];
    for (int i = 0; i < 12; ++i) {
        global_edges[i] = hex_edges[offset_mapping + i];
        signs[i] = hex_edge_signs[offset_mapping + i];
        if (global_edges[i] < 0) { signs[i] = 0; continue; }
    }

    // Iteriere über die 12x12 lokale Matrix
    for (int r = 0; r < 12; ++r) {
        int global_row = global_edges[r];
        // Ungenutzter Platz eines Tetraeders oder Prismas. Ohne diese Abfrage
        // liest die Zeile darunter csr_row_offsets[-1] -- und ist der so
        // gelesene Wert negativ, laeuft die Suchschleife von einem negativen k
        // und kann atomic_add(&csr_values[k], ...) mit k < 0 ausfuehren.
        if (global_row < 0) continue;
        int sign_r = signs[r];

        // Suchen der passenden Spalte im CSR-Array für diese Zeile
        int csr_start = csr_row_offsets[global_row];
        int csr_end = csr_row_offsets[global_row + 1];

        for (int c = 0; c < 12; ++c) {
            int global_col = global_edges[c];
            if (global_col < 0) continue;
            int sign_c = signs[c];

            // Vorzeichenkorrektur für den Eintrag K_ij
            double val = local_A_sys[offset_local + r * 12 + c] * sign_r * sign_c;

            // Finde den Index im csr_values Array
            for (int k = csr_start; k < csr_end; ++k) {
                if (csr_col_indices[k] == global_col) {
                    // Da mehrere Hexaeder auf denselben Eintrag (r, c) addieren können,
                    // ist eine atomare Addition auf der GPU zwingend erforderlich!
                    bf::atomic_add(&csr_values[k], val);
                    break;
                }
            }
        }
    }
}

__global__ void scatter_to_csr_kernel(
    int num_hexes,
    const double* local_A_sys,
    const int* hex_edges,
    const int* hex_edge_signs,
    const int* csr_row_offsets,
    const int* csr_col_indices,
    double* csr_values
) {
    int hex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hex_id >= num_hexes) return;
    scatter_to_csr_kernel_body(hex_id, num_hexes, local_A_sys, hex_edges,
                               hex_edge_signs, csr_row_offsets, csr_col_indices,
                               csr_values);
}

// C++ Host-Wrapper
void assemble_csr_matrix_gpu(
    int num_hexes, const double* d_local_A_sys, const int* d_hex_edges, const int* d_hex_edge_signs,
    const int* d_csr_row_offsets, const int* d_csr_col_indices, double* d_csr_values)
{
    BF_LAUNCH_1D(scatter_to_csr_kernel, num_hexes,
        num_hexes, d_local_A_sys, d_hex_edges, d_hex_edge_signs,
        d_csr_row_offsets, d_csr_col_indices, d_csr_values);
    bf::sync();
}
