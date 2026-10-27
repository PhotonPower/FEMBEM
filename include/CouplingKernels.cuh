#ifndef COUPLING_KERNELS_CUH
#define COUPLING_KERNELS_CUH

// =============================================================================
// CouplingKernels.cuh -- die Ruempfe der Kopplungs-Massenmatrix [T]
// =============================================================================
//
// Warum ein eigener Header und nicht (wie bisher) direkt in CoupledSolver.cu?
//
// Die __global__ HUELLEN muessen dort bleiben: das Projekt kompiliert ohne
// Separable Compilation (-rdc=true), ein __global__ Kernel aus einer anderen
// .cu-Datei liesse sich nicht linken. Die RUEMPFE dagegen sind gewoehnliche
// __host__ __device__ inline Funktionen -- und genau sie tragen die
// Vorzeichenkonvention, also den fehleranfaelligsten Teil der Kopplung.
// Solange sie in einer .cu-Datei stehen, kann kein Test sie aufrufen und die
// Konvention bleibt eine Behauptung im Kommentar.
//
// Hier ausgelagert prueft tests/test_coupling_sign_convention.cu sie direkt
// auf dem Host, ohne GPU und ohne Netz.
//
// Die beiden Ruempfe gehoeren strikt zusammen: sie beschreiben dieselbe
// Matrix T, einmal fuer den Operator (Bloecke -T oben rechts, -0.5*T unten
// links) und einmal fuer die Rueckwaertssubstitution des blocktriangulaeren
// Vorkonditionierers (+T). Wer an einem der Vorzeichen dreht, muss das andere
// mitziehen -- der Test haelt genau diese Kopplung fest.
// =============================================================================

#include "Backend.cuh"
#include "BemIntegration.cuh"
#include "Mesh.cuh"

// -----------------------------------------------------------------------------
// Operator-Anteil: y_A -= T*x_K und y_K -= 0.5*T*x_A
// -----------------------------------------------------------------------------
//
// T ist blockdiagonal pro Rand-Quad (Wi und Wj leben auf demselben Quad) --
// daher lohnt sich kein Speichern/Assemblieren, T_loc wird pro Quad direkt in
// der Matrix-Vektor-Anwendung ausgewertet.
//
// Herleitung der Vorzeichen-Konvention (WICHTIG, nicht geraten):
// x_A ist ein globaler FEM-Vektor; sein "Gather" in den BEM-Elementraum
// erfolgt exakt wie in k_gather_bem: x_A_bem[c] = sign[c] * x_A[map[c]].
// x_K (die BEM-Unbekannte) ist in DERSELBEN signierten Konvention definiert
// wie d_x_bem -- konsistent mit allen bestehenden G-Bloecken
// (dense_blocks/aca_blocks), die ja ebenfalls auf d_x_bem wirken.
// Damit folgt fuer die Bilinearform (Testfunktion W_r, lokal/roh):
//   - Beitrag zu y_K (blockdiagonal, keine Kopplung ueber Quad-Grenzen, daher
//     kein atomicAdd noetig): da y_K in derselben signierten Konvention wie
//     x_K gespeichert wird, muss die Testfunktion W_r ebenfalls signiert
//     werden (y_K_signed[r] = sign[r] * (rohe Zeile r)). Zusammen mit dem
//     Gather-Vorzeichen von x_A_bem[c] = sign[c]*x_A[map[c]] ergibt sich der
//     Faktor sign[r]*sign[c] -- ein einziges Mal, nicht doppelt.
//   - Beitrag zu y_A (globaler FEM-Vektor, additiv ueber geteilte Kanten,
//     daher atomicAdd wie in k_scatter_add_bem): dieselbe Rechnung mit
//     vertauschten Rollen liefert denselben Faktor sign[r]*sign[c].
// In beiden Faellen also: T_signed[r][c] = sign[r] * sign[c] * T_loc[r][c],
// und zwar EINMAL angewendet, nicht pro Auftreten von x_A_bem/x_K erneut.
//
// Streut per bf::atomic_add nach y_A (benachbarte Rand-Quads teilen sich
// Kanten) -- auf der CPU deshalb SERIELL starten, also BF_LAUNCH_1D und nicht
// BF_LAUNCH_1D_PAR an der Aufrufstelle.
BF_HD inline void apply_T_matrix_kernel_body(
    int face_idx,
    int num_boundary_faces,
    const Mesh::QuadFace* faces, const double3* nodes,
    const int* bem_to_fem_map, const int* bem_edge_signs,
    const double* x_A, const double* x_K,
    double* y_A, double* y_K)
{
    (void)num_boundary_faces;

    Mesh::QuadFace quad = faces[face_idx];
    double3 q_nodes[4];
    for (int i = 0; i < 4; ++i) q_nodes[i] = nodes[quad.v[i]];

    double T_loc[4][4];
    integrate_boundary_mass_matrix(q_nodes, T_loc);

    int g_idx[4];
    int sgn[4];
    for (int i = 0; i < 4; ++i) {
        g_idx[i] = bem_to_fem_map[face_idx * 4 + i];
        sgn[i]   = bem_edge_signs[face_idx * 4 + i];
    }

    for (int r = 0; r < 4; ++r) {
        double val_A = 0.0; // Beitrag zu y_A[g_idx[r]], global (atomar)
        double val_K = 0.0; // Beitrag zu y_K[face_idx*4+r], lokal (kein atomicAdd noetig)

        for (int c = 0; c < 4; ++c) {
            double T_rc = static_cast<double>(sgn[r]) * static_cast<double>(sgn[c]) * T_loc[r][c];

            // Oberer rechter Block (Gl. 11): y_A -= T * x_K
            val_A -= T_rc * x_K[face_idx * 4 + c];

            // Unterer linker Block, "1/2 T"-Anteil von H (Gl. 12), SYMMETRISIERT:
            // die gesamte K-Gleichung (RHS_K ist immer 0, siehe fem.cu) wird mit
            // -1 multipliziert, um das Vorzeichenmuster der Kopplungsbloecke zu
            // vereinheitlichen (vorher -T oben rechts vs. +0.5T unten links --
            // gegensaetzliche Vorzeichen sind ein klassischer Grund fuer
            // komplexe Eigenwerte in Sattelpunkt-aehnlichen Systemen). Aendert
            // NICHTS an der Loesung (f(x)=0 <=> -f(x)=0), siehe apply_physical()
            // fuer die dazu konsistenten G_BEM/H_BEM-Vorzeichen.
            val_K -= 0.5 * T_rc * x_A[g_idx[c]];
        }

        bf::atomic_add(&y_A[g_idx[r]], val_A);
        y_K[face_idx * 4 + r] += val_K;
    }
}

// -----------------------------------------------------------------------------
// Vorkonditionierer-Anteil: y_A += T*x_K
// -----------------------------------------------------------------------------
//
// Bewusst ein eigener Rumpf und nicht der obige mit x_A = 0: letzteres wuerde
// jedes Mal einen Nullvektor der Laenge total_dofs verlangen und den (dann
// nutzlosen) 0.5*T-Anteil mitrechnen.
//
// Die Vorzeichenkonvention ist Zeile fuer Zeile dieselbe wie oben --
// T_signed[r][c] = sign[r]*sign[c]*T_loc[r][c], EINMAL angewendet -- nur das
// Vorzeichen des Beitrags ist umgekehrt. Grund: der Operator hat oben rechts
// den Block -T; die Rueckwaertssubstitution des blocktriangulaeren
// Vorkonditionierers lautet
//     A~ z_A = r_A - (-T) z_K = r_A + T z_K,
// braucht also +T.
//
// Diese Beziehung ist keine Konvention, sondern eine Invariante: fuer jedes
// x_K muss gelten
//     apply_T_matrix_kernel_body(x_A = 0, x_K)|_{y_A}
//   + apply_T_to_A_kernel_body(x_K)|_{y_A}                = 0.
// Genau das prueft tests/test_coupling_sign_convention.cu.
BF_HD inline void apply_T_to_A_kernel_body(
    int face_idx,
    int num_boundary_faces,
    const Mesh::QuadFace* faces, const double3* nodes,
    const int* bem_to_fem_map, const int* bem_edge_signs,
    const double* x_K, double* y_A)
{
    (void)num_boundary_faces;

    Mesh::QuadFace quad = faces[face_idx];
    double3 q_nodes[4];
    for (int i = 0; i < 4; ++i) q_nodes[i] = nodes[quad.v[i]];

    double T_loc[4][4];
    integrate_boundary_mass_matrix(q_nodes, T_loc);

    int g_idx[4];
    int sgn[4];
    for (int i = 0; i < 4; ++i) {
        g_idx[i] = bem_to_fem_map[face_idx * 4 + i];
        sgn[i]   = bem_edge_signs[face_idx * 4 + i];
    }

    for (int r = 0; r < 4; ++r) {
        double val_A = 0.0;
        for (int c = 0; c < 4; ++c) {
            const double T_rc = static_cast<double>(sgn[r]) * static_cast<double>(sgn[c]) * T_loc[r][c];
            val_A += T_rc * x_K[face_idx * 4 + c];
        }
        bf::atomic_add(&y_A[g_idx[r]], val_A);
    }
}

#endif // COUPLING_KERNELS_CUH
