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
// Matrix T, einmal fuer den Operator (Bloecke -T oben rechts, +0.5*T unten
// links) und einmal fuer die Rueckwaertssubstitution des blocktriangulaeren
// Vorkonditionierers (+T). Wer an einem der Vorzeichen dreht, muss das andere
// mitziehen -- der Test haelt genau diese Kopplung fest.
// =============================================================================

#include "Backend.cuh"
#include "BemIntegration.cuh"
#include "Mesh.cuh"

// -----------------------------------------------------------------------------
// Operator-Anteil: y_A -= T*x_K und y_K += 0.5*T*x_A
// -----------------------------------------------------------------------------
//
// T ist blockdiagonal pro Rand-Quad (Wi und Wj leben auf demselben Quad) --
// daher lohnt sich kein Speichern/Assemblieren, T_loc wird pro Quad direkt in
// der Matrix-Vektor-Anwendung ausgewertet.
//
// Herleitung der Vorzeichen-Konvention (WICHTIG, nicht geraten):
//
// Bezeichnungen. Eine FEM-Kante g tragt den Freiheitsgrad a_g (Kantenintegral,
// global orientiert v_min -> v_max). Ein BEM-Freiheitsgrad i = pos*4 + e gehoert
// zur lokalen Kante e des Quads perm[pos], zeigt entlang der LOKALEN Richtung
// (Rooftop-/Spurfunktion W_e, siehe evaluate_rooftop_basis), und sign[i] = +1,
// wenn die lokale Richtung mit der globalen uebereinstimmt, sonst -1.
//
//   Gather von a (k_gather_bem):  a_bem[i] = sign[i] * a[g_i]
//       Der Koeffizient vor W_e ist das Kantenintegral entlang der lokalen
//       Richtung, also sign * (globales Kantenintegral).
//   K (x_K) ist der ROHE Koeffizient vor W_e, ohne Vorzeichen: so benutzen ihn
//       die G- und D-Bloecke (sie wirken ohne sign auf x_K bzw. auf a_bem).
//
// Zeile 1 (FEM-Kante g): Testfunktion ist die globale Kantenfunktion phi_g. Auf
//   dem Quad ist ihre Spur sign[r] * W_r (lokale Kante r -> g). Der Randterm
//   ist int phi_g . K dS = sign[r] * sum_c T_rc * x_K[c]. NUR das Zeilen-
//   vorzeichen: x_K ist schon roh. Ein zusaetzliches sign[c] (wie frueher) waere
//   eine zweite Vorzeichenumkehr fuer jede Kante mit sign = -1.
//
// Zeile 2 (BEM-Zeile r): Testfunktion ist die rohe W_r, wie bei G und D, also
//   OHNE sign[r]. Die A-Spalte ist a_bem[c] = sign[c] * a[g_c]. Der Beitrag ist
//   sum_c T_rc * sign[c] * a[g_c]. NUR das Spaltenvorzeichen.
//
// Beide Bloecke tragen damit jeweils EIN sign, und die Kopplungsbloecke bleiben
// zueinander transponiert bis auf Faktor und Vorzeichen (Zeile 1: -S T, Zeile 2:
// +0.5 T S, S = diag(sign)).
//
// Herleitung des Vorzeichens des 1/2-T-Terms (K-Zeile): Fuer eine im Aussengebiet
// harmonische, im Unendlichen abklingende Funktion u gilt auf dem Rand mit der
// Normale n nach aussen (aus dem Leiter heraus, ins Gebiet hinein)
//     (1/2 - D) u = -G du/dn,   D u = int u d/dn_y(1/(4 pi R)) dS_y.
// Handrechnung Einheitskugel, u = 1/r: D*1 = -1/2, G du/dn = -1, beide Seiten 1.
// Mit mu_0 k = du/dn und der Testfunktion W_r (roh): (+1/2 T - D) a + mu_0 G k = 0.
// Gemessen: Wuerfeltest ohne FEM (tests/test_bem_calderon.cu) und --bem-row-check.
// Beitrag zu y_K: kein atomicAdd noetig (blockdiagonal pro Quad); zu y_A:
// additiv ueber geteilte Kanten, daher atomicAdd.
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
            // Oberer rechter Block (Gl. 11): y_A -= S T * x_K (nur Zeilenvorzeichen)
            val_A -= static_cast<double>(sgn[r]) * T_loc[r][c] * x_K[face_idx * 4 + c];

            // Unterer linker Block, "1/2 T"-Anteil der K-Zeile (nur Spalten-
            // vorzeichen, siehe Herleitung oben). Die K-Zeile ist die Calderon-
            // Beziehung fuer das AUSSENgebiet (Luft) mit nach aussen zeigender
            // Normale (siehe unten):
            //     (+1/2 T - D) a + mu_0 G k = 0     mit  mu_0 k = du/dn.
            // Das Vorzeichen des 1/2-T-Terms ist also POSITIV (frueher negativ:
            // das war die Innenraumform (1/2 + D) u = G du/dn, Befund d in
            // docs/design/bem-row-check.md). D und G wirken unveraendert mit
            // -D_BEM und +mu_0 G_BEM (apply_bem_blocks).
            val_K += 0.5 * T_loc[r][c] * static_cast<double>(sgn[c]) * x_A[g_idx[c]];
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
// Die Vorzeichenkonvention ist Zeile fuer Zeile dieselbe wie in Zeile 1 oben --
// nur das Zeilenvorzeichen sign[r], EINMAL angewendet -- nur das
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
            // wie Zeile 1 des Operators: nur das Zeilenvorzeichen
            val_A += static_cast<double>(sgn[r]) * T_loc[r][c] * x_K[face_idx * 4 + c];
        }
        bf::atomic_add(&y_A[g_idx[r]], val_A);
    }
}

#endif // COUPLING_KERNELS_CUH
