#ifndef COUPLED_SOLVER_CUH
#define COUPLED_SOLVER_CUH

#include <cuda_runtime.h>

// =================================================================================
// Compiler-Isolierung für NVIDIA CUDA 13 Header (MSVC Workaround)
// Deaktiviert C4996 (Deprecated API Warnungen/Fehler), die beim bloßen Parsen 
// von <cusparse.h> auftreten, obwohl wir ausschließlich die moderne Generic API nutzen.
// =================================================================================
#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include <cusparse.h>
#include <cublas_v2.h>

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <stdexcept>
#include <string>
#include <vector>
#include "AmsPreconditioner.cuh"  // AMS statt ILU(0)
#include "Backend.cuh"      // GPU/CPU-Umschaltung
#include "HostLinalg.cuh"   // CPU-Ersatz fuer cuBLAS/cuSPARSE
#include "DeviceArray.cuh"
#include "Mesh.cuh"

// -----------------------------------------------------------------------------
// Rigorose Fehlerbehandlung für NVIDIA Bibliotheken
// -----------------------------------------------------------------------------
#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        cublasStatus_t status = (call);                                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                                             \
            throw std::runtime_error("cuBLAS Error at " + std::string(__FILE__) +          \
                                     ":" + std::to_string(__LINE__));                      \
        }                                                                                  \
    } while (0)

// Der STATUSCODE gehoert in die Meldung. Vorher stand dort nur "cuSPARSE
// Error at Datei:Zeile" -- damit ist nicht zu unterscheiden, ob ein Puffer zu
// klein war (INVALID_VALUE), die Matrix nicht unterstuetzt wird
// (NOT_SUPPORTED) oder der Speicher ausging (ALLOC_FAILED). Jede dieser
// Ursachen fuehrt woanders hin, und ohne den Code beginnt die Fehlersuche
// beim Raten.
#define CUSPARSE_CHECK(call)                                                               \
    do {                                                                                   \
        cusparseStatus_t status = (call);                                                  \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                           \
            throw std::runtime_error("cuSPARSE Error " +                                   \
                                     std::string(cusparseGetErrorName(status)) + " (" +    \
                                     std::string(cusparseGetErrorString(status)) +         \
                                     ") at " + std::string(__FILE__) +                     \
                                     ":" + std::to_string(__LINE__));                      \
        }                                                                                  \
    } while (0)



// -----------------------------------------------------------------------------
// Rigorose Fehlerbehandlung für NVIDIA Bibliotheken
// -----------------------------------------------------------------------------
#define CUBLAS_CHECK(call)                                                                 \
    do {                                                                                   \
        cublasStatus_t status = (call);                                                    \
        if (status != CUBLAS_STATUS_SUCCESS) {                                             \
            throw std::runtime_error("cuBLAS Error at " + std::string(__FILE__) +          \
                                     ":" + std::to_string(__LINE__));                      \
        }                                                                                  \
    } while (0)

// Der STATUSCODE gehoert in die Meldung. Vorher stand dort nur "cuSPARSE
// Error at Datei:Zeile" -- damit ist nicht zu unterscheiden, ob ein Puffer zu
// klein war (INVALID_VALUE), die Matrix nicht unterstuetzt wird
// (NOT_SUPPORTED) oder der Speicher ausging (ALLOC_FAILED). Jede dieser
// Ursachen fuehrt woanders hin, und ohne den Code beginnt die Fehlersuche
// beim Raten.
#define CUSPARSE_CHECK(call)                                                               \
    do {                                                                                   \
        cusparseStatus_t status = (call);                                                  \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                           \
            throw std::runtime_error("cuSPARSE Error " +                                   \
                                     std::string(cusparseGetErrorName(status)) + " (" +    \
                                     std::string(cusparseGetErrorString(status)) +         \
                                     ") at " + std::string(__FILE__) +                     \
                                     ":" + std::to_string(__LINE__));                      \
        }                                                                                  \
    } while (0)


// -----------------------------------------------------------------------------
// Strukturen für die BEM-Blöcke
// -----------------------------------------------------------------------------
struct DenseBlock {
    int m, n;
    int row_offset, col_offset;
    double* d_matrix; // Column-Major

    // wird der Block als Transponierte seiner gespeicherten Matrix
    // angewandt? Damit teilen sich die Blockpaare (I,J) und (J,I) EINEN
    // Speicher, und G_BEM ist per Konstruktion exakt symmetrisch. Siehe die
    // Begruendung an add_dense_block_transposed().
    //
    // m und n bezeichnen immer die Groesse des ANGEWANDTEN Blocks. Ist
    // transposed gesetzt, hat die gespeicherte Matrix also n Zeilen und m
    // Spalten, und ihr Leading Dimension ist n (nicht m).
    bool transposed = false;
};

struct AcaBlock {
    int m, n, k;
    int row_offset, col_offset;
    double* d_U; // m x k (Column-Major)
    double* d_V; // n x k (Column-Major)

    // Bei ACA braucht es kein Transpositions-Flag: (U V^T)^T = V U^T. Der
    // gespiegelte Block zeigt einfach mit vertauschten Rollen auf DIESELBEN
    // Puffer -- d_U des einen ist d_V des anderen. Kein zusaetzlicher Speicher,
    // und die Symmetrie ist exakt statt approximativ.
};

// -----------------------------------------------------------------------------
// Aufgabe 6.1: Der gekoppelte Matrix-Vektor Operator y = A*x
// -----------------------------------------------------------------------------
class CoupledMVOperator {
private:
    cublasHandle_t cublas;
    cusparseHandle_t cusparse;

    // FEM Sparse Matrix
    cusparseSpMatDescr_t mat_FEM = nullptr;   // initialisiert, damit der
                                              // Destruktor auch dann sicher ist,
                                              // wenn setup_fem_matrix() nie lief.
    size_t cusparse_buffer_size;
    void* d_cusparse_buffer;

    int total_dofs;

    // BEM Blöcke (Single-Layer, G_BEM)
    std::vector<DenseBlock> dense_blocks;
    std::vector<AcaBlock> aca_blocks;

    // BEM Blöcke fuer den Double-Layer-Anteil (H_BEM = 0.5*T + D_BEM).
    // D_BEM wirkt auf die (gegatherte) Randspur von A, genau wie G_BEM auf K
    // wirkt -- daher dieselbe DenseBlock/AcaBlock-Struktur, nur separat
    // gespeichert, weil es ein anderer Integralkern ist (siehe
    // integrate_regular_quad_pair_dlp).
    std::vector<DenseBlock> dense_blocks_H;
    std::vector<AcaBlock> aca_blocks_H;

    // Geometrie fuer die "matrixfreie" T-Anwendung
    // (apply_T_matrix_kernel in CoupledSolver.cu braucht die Rand-Quads samt
    // aktueller Knotenkoordinaten, um T_loc pro Quad zu berechnen).
    int m_num_boundary_faces = 0;
    const Mesh::QuadFace* d_mesh_faces = nullptr; // Eigentum des Aufrufers
    const double3* d_mesh_nodes = nullptr;        // Eigentum des Aufrufers, aktuelle Koordinaten

    // Vakuum-Reluktivitaet 1/mu_0 wird NICHT gebraucht -- fuer den G-Block
    // brauchen wir mu_0 selbst (Gl. 12: H*A = mu_0 * G * K, siehe apply()).
    static constexpr double m_mu_0 = 1.2566370614359172e-06; // 4*pi*1e-7

    // Temporärer Vektor für ACA: z = V^T * x
    // Die Größe muss dem maximalen Rang 'k' aller ACA-Blöcke entsprechen
    DeviceArray<double> d_temp_z; 

    // -------------------------------------------------------------------------
    // Kopplung des BEM-Blockoperators an die FEM-Kanten (siehe BemCoupling.h)
    // -------------------------------------------------------------------------
    // Die Blöcke sind in permutierten BEM-Elementindizes formuliert, der
    // Operator arbeitet auf total_dofs = mesh.num_edges. Dazwischen liegen ein
    // Gather und ein additiver Scatter, jeweils einmal pro apply():
    //     x_bem[i] = s_i * x_fem[map_i]
    //     y_bem    = B * x_bem
    //     y_fem[map_i] += s_i * y_bem[i]      (atomar, 2 BEM-DOFs pro FEM-Kante)
    int num_bem_dofs = 0;
    const int* d_bem_to_fem = nullptr;  // Länge num_bem_dofs, Eigentum des Aufrufers
    const int* d_bem_sign   = nullptr;  // Länge num_bem_dofs, Eigentum des Aufrufers
    DeviceArray<double> d_x_bem;
    DeviceArray<double> d_y_bem;

    // Diagonale von (-mu_0 * G)^-1 fuer einen Jacobi-Vorkonditionierer
    // auf dem K-Block (siehe apply_preconditioner). Default 1.0 (= Identitaet,
    // das bisherige Verhalten), damit ein evtl. nicht abgedeckter BEM-DOF
    // nicht versehentlich auf 0 vorkonditioniert wird (siehe setup_bem_coupling).
    DeviceArray<double> d_bem_inv_diag;

    // -------------------------------------------------------------------------
    // Baum-Kobaum-Eichung (siehe include/TreeCotreeGauge.h)
    // -------------------------------------------------------------------------
    // A_sys = K + C/dt ist in nichtleitenden Gebieten singulaer, Nullraum sind
    // die diskreten Gradienten. Angewendet wird die Eichung als Projektion:
    // mit P = "setze geeichte Eintraege auf null" rechnet apply()
    //     y = P*A*P*x + (I - P)*x
    // Fuer geeichte Zeilen lautet die Gleichung damit x_i = 0. Die Form P*A*P
    // erhaelt die Symmetrie der Matrix.
    const char* d_is_gauged = nullptr;   // Laenge total_dofs, Eigentum des Aufrufers
    DeviceArray<double> d_x_proj;        // Arbeitspuffer fuer P*x

    // -------------------------------------------------------------------------
    // Kohomologie-Generatoren (siehe include/CohomologyGauge.h)
    // -------------------------------------------------------------------------
    // Zusaetzlich zur Baum-Eichung (Koordinatenmaske) koennen dichte
    // Projektionsvektoren registriert werden, einer pro nicht einfach
    // zusammenhaengendem nichtleitenden Gebiet. Jeder Vektor projiziert eine
    // Komponente x -= c*(c^T x)/(c^T c) heraus. Die Vektoren muessen bereits
    // auf den Baum-Koordinaten null sein (siehe find_cohomology_generators
    // und die Reduktion in CohomologyGauge.h), sonst ist die Gesamtprojektion
    // nicht idempotent.
    struct CohomologyProjector {
        const double* d_vector = nullptr; // Laenge total_dofs, Eigentum des Aufrufers
        double norm_sq = 0.0;             // c^T c, host-seitig vorberechnet
    };
    std::vector<CohomologyProjector> cohomology_projectors;

    // -------------------------------------------------------------------------
    // ILU(0)-Vorkonditionierung des FEM-Sparse-Anteils (siehe compute_ilu())
    // -------------------------------------------------------------------------
    // Rohe CSR-Zeiger der FEM-Matrix, Eigentum des Aufrufers (siehe
    // setup_fem_matrix). cusparseCreateCsr haelt selbst keine Kopie, daher
    // werden sie hier gemerkt -- compute_ilu() braucht sie fuer die
    // (Neu-)Faktorisierung bei jedem Zeitschritt.
    int m_ilu_num_rows = 0;

    // Diagnose -- zaehlt apply()-Aufrufe seit dem letzten compute_ilu()
    // (also seit Beginn des aktuellen Zeitschritts). Beim ersten apply()-
    // Aufruf eines Zeitschritts (typischerweise das initiale Residuum in
    // BiCGStab) werden einmalig ||FEM-Anteil|| und ||BEM-Anteil|| von y VOR
    // dem Aufsummieren geloggt, um eine vermutete Groessenordnungs-Differenz
    // zwischen FEM-Steifigkeit (skaliert mit der Reluktivitaet) und dem rein
    // geometrischen BEM-Kern (1/(4*pi*r), ohne sichtbaren nu_0/mu_0-Faktor)
    // mit echten Zahlen zu pruefen, statt sie zu vermuten.
    int m_apply_call_index = 0;
    int m_ilu_nnz = 0;
    int* m_row_offsets = nullptr;
    int* m_col_indices = nullptr;
    double* m_values = nullptr;

    double* d_ilu_values = nullptr;  // L\U in-place, gleiche Sparsity wie FEM
    DeviceArray<double> d_ilu_temp;  // Zwischenergebnis fuer L*y = x
    // Zweiter Puffer allein fuer die SpSV-Analyse. cuSPARSE verlangt fuer
    // SpSV, dass Ein- und Ausgabevektor VERSCHIEDEN sind -- vorher zeigten
    // beide Deskriptoren auf d_ilu_temp. Auf manchen Karten faellt das nicht
    // auf, weil die Analyse die Werte nicht liest; auf anderen schlaegt sie
    // fehl.
    DeviceArray<double> d_ilu_probe;

    // -------------------------------------------------------------------------
    // eigene ILU(0)-Faktorisierung
    // -------------------------------------------------------------------------
    // Im CPU-Modus gibt es kein cusparseDcsrilu02/cusparseSpSV. Die
    // Faktorisierung wird stattdessen von bf::csr_ilu0() gerechnet und liegt
    // hier -- gleiche Sparsity wie die FEM-Matrix, L (Einheitsdiagonale) im
    // strikt unteren, U im oberen Dreieck. m_cpu_ilu_diag haelt je Zeile die
    // Position des Diagonaleintrags im values-Array; die
    // Dreieckssubstitutionen brauchen sie, und sie einmal je Faktorisierung
    // statt je Anwendung zu suchen ist der Unterschied zwischen linearem und
    // quadratischem Aufwand in der Zeilenlaenge.
    //
    // Bewusst std::vector und nicht DeviceArray: dieser Speicher wird
    // ausschliesslich vom Host benutzt, ein Unified-Memory-Puffer waere hier
    // nur Umweg.
    std::vector<double> m_cpu_ilu_values;
    std::vector<int>    m_cpu_ilu_diag;
    std::vector<double> m_cpu_ilu_temp;   // Zwischenergebnis L*y = x

    // AMS-Vorkonditionierer (--ams). Host-seitig; die Portierung auf die GPU
    // ist ein eigener Schritt, hier geht es zuerst um die Iterationszahl.
    AmsPreconditioner   m_ams;
    bool                m_ams_ready = false;
    mutable std::vector<double> m_ams_in, m_ams_out;

    cusparseMatDescr_t ilu_descr = nullptr;
    csrilu02Info_t ilu_info = nullptr;

    cusparseSpMatDescr_t mat_L = nullptr;
    cusparseSpMatDescr_t mat_U = nullptr;
    cusparseSpSVDescr_t spsv_info_L = nullptr;
    cusparseSpSVDescr_t spsv_info_U = nullptr;

    void* d_ilu_buffer = nullptr;
    void* d_spsv_buffer_L = nullptr;
    void* d_spsv_buffer_U = nullptr;

    bool ilu_ready = false;

    // siehe set_ilu_bypass() oben.
    bool m_ilu_bypass = false;

    // -------------------------------------------------------------------------
    // D*A*D ausserhalb des
    // Krylov-Loesers. Siehe setup_equilibration() unten fuer die Herleitung.
    // -------------------------------------------------------------------------
    DeviceArray<double> d_D_scale;        // Laenge total_dofs+num_bem_dofs
    DeviceArray<double> d_apply_in_temp;  // Scratch: D * x_tilde (physikalischer Raum)
    DeviceArray<double> d_apply_out_temp; // Scratch: unskalierter Output vor D*y
    DeviceArray<double> d_tilde_proj_temp; // Scratch: P_tilde(x_tilde), siehe apply()
    bool is_equilibrated = false;

    // waehrend apply() den physikalischen Operator
    // aufruft, muss dessen eigener (I-P)*x_real Nullraum-Penalty-Term
    // unterdrueckt werden -- er wuerde sonst zusammen mit dem Rest von
    // y_real ein zweites Mal mit D skaliert und den Nullraum-Eigenwert von
    // 1.0 auf D_i^2 verzerren. Siehe apply() fuer den korrekten Tilde-
    // Ersatz-Term.
    bool m_skip_nullspace_penalty = false;

    // apply_many(): apply_physical() laesst die G-/H-Bloecke aus und legt
    // stattdessen seine Blockeingaben (x_bem und x_K, physikalisch) in Spalte
    // m_many_col der Sammelpuffer ab. apply_many() rechnet die Bloecke danach
    // fuer alle Spalten gemeinsam.
    bool m_skip_bem_blocks = false;
    int m_many_col = 0;
    DeviceArray<double> m_many_xbem, m_many_xk, m_many_yk, m_many_z;
    void apply_bem_blocks(const double* d_xbem, const double* d_xk, double* d_y_K);
    void apply_bem_blocks_many(int nrhs);

    // die bisherige apply()-Implementierung, jetzt nur noch intern
    // aufgerufen -- arbeitet unveraendert im physikalischen (unskalierten) Raum.
    void apply_physical(const double* d_x, double* d_y);

    // -------------------------------------------------------------------------
    // Blocktriangulaerer Vorkonditionierer (Johnson-Nedelec)
    // -------------------------------------------------------------------------
    //
    // Der bisherige Vorkonditionierer ist blockdiagonal, P = diag(A~, I). Er
    // ignoriert die Kopplungsbloecke vollstaendig -- physikalisch entkoppelt er
    // Leiter und Aussenraum. Weil die Kopplung bauartbedingt UNSYMMETRISCH ist
    // (oben rechts steht -T, unten links -0.5*T - D_BEM; die beiden koennen per
    // Definition nicht zueinander transponiert sein), muss das Krylov-Verfahren
    // die Stetigkeit der Tangentialkomponente allein ueber Matrix-Vektor-
    // Produkte rekonstruieren. Genau dort sitzt das Residuum-Plateau.
    //
    // Standardausweg fuer Johnson-Nedelec ist der blocktriangulaere Ansatz
    //
    //     P = ( A~   -T )        P^-1 r = ( A~^-1 (r_A + T * S~^-1 r_K) )
    //         (  0   S~ )                 (        S~^-1 r_K            )
    //
    // Das exakte Schur-Komplement waere
    //     S = mu_0*G_BEM - (-0.5*T - D_BEM) * A^-1 * (-T).
    // Bewiesen (Hiptmair, Steinbach, Heuer) und in der BEM-FEM-Praxis etabliert
    // ist die spektrale Aequivalenz des Single-Layer-Operators zum exakten
    // Schur-Komplement, also
    //     S~ = mu_0 * G_BEM.
    //
    // G_BEM liegt nur als ACA-komprimierter Operator vor, S~^-1 laesst sich
    // also nicht explizit bilden. Stattdessen loest ein INNERES CG das System
    // S~ z_K = r_K. G_BEM (Single-Layer) ist positiv definit und gutartig; im
    // aequilibrierten Tilde-Raum ist seine Diagonale sogar exakt 1.0 (siehe
    // setup_equilibration()), weshalb typisch 5 bis 10 Iterationen bei einer
    // Toleranz von 1e-2 genuegen. Dass der innere Loeser damit ungenau und von
    // Aufruf zu Aufruf verschieden ist, ist zulaessig: GMRESSolver speichert
    // die Z-Basis (z_j = M^-1 v_j) mit und ist damit ein FLEXIBLES GMRES.
    //
    // Am Modellproblem (n = 1500, 33 % Kopplungsunsymmetrie) gemessen:
    // 62 FGMRES-Iterationen blockdiagonal gegen 20 blocktriangulaer.
    bool   m_block_tri_pc      = false;
    int    m_pc_inner_iters    = 10;
    double m_pc_inner_tol      = 1e-2;
    int    m_pc_call_index     = 0;   // nur fuer die einmalige Diagnoseausgabe

    DeviceArray<double> d_pc_rA;      // r_A + T*z_K            (Laenge total_dofs)
    DeviceArray<double> d_pc_TzK;     // T*z_K                  (Laenge total_dofs)
    DeviceArray<double> d_pc_S_tmp;   // Scratch in apply_schur_tilde (num_bem_dofs)
    DeviceArray<double> d_pc_K_phys;  // D_K .* z_K, physikalisch  (num_bem_dofs)
    DeviceArray<double> d_pc_cg_r;    // CG-Residuum            (num_bem_dofs)
    DeviceArray<double> d_pc_cg_p;    // CG-Suchrichtung        (num_bem_dofs)
    DeviceArray<double> d_pc_cg_Ap;   // CG: S~*p               (num_bem_dofs)
    DeviceArray<double> d_pc_cg_best; // beste Iterierte des inneren CG

    /// Legt die obigen Puffer an, sobald total_dofs und num_bem_dofs feststehen.
    /// Idempotent -- ein zweiter Aufruf mit gleicher Groesse tut nichts.
    void ensure_block_tri_buffers();

    /// Der bisherige FEM-Block-Vorkonditionierer A~^-1 (AMS bzw. ILU(0), GPU
    /// wie CPU), herausgeloest aus apply_preconditioner(), damit ihn beide
    /// Zweige benutzen koennen. Arbeitet auf den ersten m_ilu_num_rows
    /// Eintraegen; d_in_A und d_out_A duerfen sich NICHT ueberlappen.
    void apply_fem_preconditioner(const double* d_in_A, double* d_out_A);


    /// Loest S~ z_K = r_K mit innerem CG (Startvektor 0). Rueckgabe: Anzahl
    /// tatsaechlich ausgefuehrter Iterationen. Bricht ab, sobald das relative
    /// Residuum unter m_pc_inner_tol faellt, spaetestens nach
    /// m_pc_inner_iters Schritten -- oder wenn p^T S~ p <= 0 auftritt (Verlust
    /// der Definitheit durch die ACA-Approximation; das bis dahin erreichte
    /// z_K bleibt dann eine gueltige Naeherung).
    int solve_schur_block(const double* d_r_K, double* d_z_K);

    /// out_A = (D_A .*) T * (D_K .*) z_K -- der Kopplungsblock des
    /// Vorkonditionierers, im Tilde-Raum. Vorzeichen: Block (1,2) des Operators
    /// ist -T, die Rueckwaertssubstitution braucht daher +T (siehe
    /// apply_T_to_A_kernel in CoupledSolver.cu).
    void apply_coupling_T_tilde(const double* d_z_K, double* d_out_A);

    /// true, wenn alle Eintraege von d_out (volle gekoppelte Laenge) endlich
    /// sind. Siehe die Notbremse in apply_preconditioner().
    bool preconditioner_output_is_finite(const double* d_out) const;

    /// Meldet, wie viele Eintraege von v nicht endlich sind, getrennt nach
    /// FEM- und BEM-Block, samt erstem Fundort.
    void report_non_finite(const char* label, const double* v) const;

    // Kantenliste fuer die Divergenzpruefung des Kopplungsterms (siehe
    // relative_gradient_content). Nicht besitzend -- zeigt in das Mesh.
    const int2* m_edges     = nullptr;
    int         m_num_nodes = 0;
    int         m_num_edges = 0;

    /// Prueft die BEM-Blockdaten (dichte Matrizen sowie U/V der ACA-Bloecke,
    /// je fuer D_BEM/H und G_BEM) auf NaN/Inf. Schweigt, wenn alles endlich
    /// ist. Siehe die Begruendung an der Definition.
    void audit_bem_blocks() const;

    /// Damit die Fallback-Warnung nur einmal je Operator erscheint und nicht
    /// tausendfach das Log flutet.
    bool m_block_tri_warned = false;

    /// Relatives Residuum, das das innere CG zuletzt erreicht hat (-1, wenn es
    /// gar nicht lief). Nur Diagnose.
    mutable double m_last_inner_rel_res = -1.0;

public:
    CoupledMVOperator(int N);
    ~CoupledMVOperator();

    /// Anteil des Gradientenunterraums in einem rechte-Seite-artigen Vektor,
    /// gemessen als ||G^T v|| / ||v|| im PHYSIKALISCHEN Raum.
    ///
    /// G ist der diskrete Gradient (Knoten -> Kanten); G^T v ist damit die
    /// diskrete Divergenz. Der Curl-Curl-Operator bildet Gradienten auf null
    /// ab, ein Gradientenanteil in der rechten Seite liegt also NICHT im
    /// Bildraum -- und A~^-1 darauf angewandt ist beliebig gross.
    ///
    /// Vergleichsmassstab liefert fem.cu fuer die physikalische rechte Seite:
    /// dort ist ||G^T b||/||b|| = 2.1e-04, also praktisch divergenzfrei.
    ///
    /// OEFFENTLICH seit dem AMS-Modus des FEM-direkt-Pfades. Dort ist die Frage
    /// nicht diagnostisch, sondern entscheidend: das System ist singulaer, und
    /// eine rechte Seite mit Gradientenanteil ist schlicht unloesbar. Der
    /// Aufrufer prueft damit vor dem Loesen, ob die Aequilibrierung die
    /// physikalisch hergestellte Konsistenz erhalten hat -- sie tut es nur bei
    /// b_tilde = D * b, und die Gegenrichtung sieht bis zum stehenbleibenden
    /// Residuum voellig unauffaellig aus.
    ///
    /// Voraussetzung ist setup_divergence_check(); ohne sie liefert die
    /// Methode -1.
    double relative_gradient_content(const double* d_v_tilde) const;

    // Setup der Sparse Matrix (aus Aufgabe 5.2)
    void setup_fem_matrix(int num_rows, int nnz, int* d_row_offsets, int* d_col_indices, double* d_values);

    // Berechnet (bzw. bei geaenderter FEM-Matrix: aktualisiert) die
    // ILU(0)-Zerlegung M = L*U der FEM-Matrix. Muss nach setup_fem_matrix()
    // und vor der ersten apply_preconditioner() aufgerufen werden; bei einer
    // neu assemblierten Matrix (naechster Zeitschritt) erneut aufrufen. M
    // approximiert nur den FEM-Anteil -- BEM-Kopplung und Eichung bleiben
    // unberuecksichtigt, da ein Vorkonditionierer nur billig annaehern muss.
    void compute_ilu();

    // Wendet M^-1 auf d_in an (M = L*U aus compute_ilu()) und schreibt das
    // Ergebnis nach d_out. Fuer rechts-vorkonditioniertes BiCGStab:
    // M*p_hat = p, v = A*p_hat (analog fuer s_hat/t).
    // Bei aktiver BEM-Kopplung hat d_in/d_out die volle
    // gekoppelte Laenge (FEM+BEM). Nur der FEM-Anteil (die ersten
    // m_ilu_num_rows Eintraege) wird per ILU vorkonditioniert; der BEM-Anteil
    // (K) wird unveraendert 1:1 durchgereicht (Identitaets-Vorkonditionierer
    // -- G_BEM ist als Integraloperator gutartig und braucht dafuer ueblicherweise
    // keine Vorkonditionierung).
    void apply_preconditioner(const double* d_in, double* d_out);

    // Diagnose-Schalter -- wenn aktiv, wird apply_preconditioner() zur
    // reinen Identitaet (d_out = d_in, kein ILU/SpSV), um zu testen, ob der
    // ILU(0)-Preconditioner (inkl. Gershgorin-Shift) selbst die Ursache eines
    // Konvergenzproblems ist, statt es zu vermuten.
    void set_ilu_bypass(bool enable) { m_ilu_bypass = enable; }

    // Blocktriangulaeren Vorkonditionierer ein-/ausschalten (siehe die
    // Herleitung bei m_block_tri_pc oben). Muss NACH setup_bem_coupling() und
    // setup_equilibration() aufgerufen werden -- die Puffergroessen haengen an
    // num_bem_dofs, und die Skalierung D_K wird mitbenutzt. Ohne aktive
    // BEM-Kopplung (num_bem_dofs == 0) bleibt der Schalter wirkungslos, der
    // Vorkonditionierer ist dann ohnehin der reine FEM-Block.
    //
    // inner_iters/inner_tol steuern das innere CG auf S~ = mu_0*G_BEM. Der
    // Vorkonditionierer muss nur billig annaehern -- 1e-2 genuegt, mehr
    // Genauigkeit kostet innere Iterationen ohne die aeussere Iterationszahl
    // weiter zu senken (am Modellproblem gemessen: inner_iters 5 -> 22, 10 ->
    // 20, 20 -> 20 aeussere Iterationen).
    void set_block_triangular_preconditioner(bool enable, int inner_iters = 10,
                                             double inner_tol = 1e-2);

    /// Reicht die Kantenliste des Netzes durch, damit
    /// relative_gradient_content() rechnen kann. Optional -- ohne diesen
    /// Aufruf entfaellt die Divergenzdiagnose des Kopplungsterms schlicht.
    void setup_divergence_check(const int2* edges, int num_nodes, int num_edges) {
        m_edges = edges; m_num_nodes = num_nodes; m_num_edges = num_edges;
    }

    /// y = S~ * x im selben (Tilde-)Raum, in dem der aeussere Loeser rechnet:
    ///     y = D_K * (mu_0 * G_BEM) * (D_K * x)
    /// Ohne aktive Aequilibrierung entfaellt D_K. Beide Vektoren haben Laenge
    /// num_bem_dofs. Nutzt d_temp_z als ACA-Scratch (wie apply_physical()).
    void apply_schur_tilde(const double* d_x_K, double* d_y_K);

    bool block_triangular_preconditioner_active() const {
        return m_block_tri_pc && num_bem_dofs > 0;
    }
    
    // Kopplung der BEM-Blöcke an die FEM-Kanten aktivieren. Muss vor apply()
    // aufgerufen werden, sobald BEM-Blöcke vorhanden sind. Die beiden Arrays
    // müssen die Länge num_bem_dofs haben, in permutierter Reihenfolge vorliegen
    // (siehe build_bem_coupling_map) und für die Lebensdauer des Operators
    // gültig bleiben; der Operator übernimmt kein Eigentum.
    void setup_bem_coupling(int num_bem_dofs_in, const int* d_map, const int* d_sign);

    // Geometrie fuer die T-Massenmatrix-Anwendung (siehe
    // apply_T_matrix_kernel). Muss zusammen mit setup_bem_coupling() vor
    // apply() aufgerufen werden, sobald BEM-Kopplung aktiv ist. d_faces/d_nodes
    // bleiben im Eigentum des Aufrufers; d_nodes darf sich zwischen
    // Zeitschritten aendern (bewegte Geometrie), der Zeiger selbst aber nicht.
    void setup_bem_mesh(int num_boundary_faces, const Mesh::QuadFace* d_faces, const double3* d_nodes);

    // Eichung aktivieren. d_flags muss total_dofs Eintraege haben (1 = geeicht)
    // und fuer die Lebensdauer des Operators gueltig bleiben. Die rechte Seite
    // muss der Aufrufer passend projizieren (b_i = 0 fuer geeichte i).
    void setup_gauge(const char* d_flags);

    // Registriert einen zusaetzlichen Kohomologie-Projektionsvektor. Muss nach
    // setup_gauge() aufgerufen werden. d_vector bleibt im Eigentum des
    // Aufrufers und muss fuer die Lebensdauer des Operators gueltig bleiben.
    void add_cohomology_projector(const double* d_vector, double norm_sq);

    // -------------------------------------------------------------------------
    // AMS als Vorkonditionierer des FEM-Blocks (--ams)
    // -------------------------------------------------------------------------
    /// Baut den AMS aus der PHYSIKALISCHEN FEM-Matrix auf.
    ///
    /// Wichtig ist der Bezugsraum. Aequilibriert wird x_phys = D x_tilde und
    /// A_tilde = D A D. Der Vorkonditionierer im Tilde-Raum ist dann
    ///     B_tilde = D^-1 B D^-1
    /// mit B dem AMS zur physikalischen Matrix. Die Hilfsmatrizen sind dabei
    /// UNVERAENDERT, denn (D^-1 G)^T (D A D) (D^-1 G) = G^T A G -- die
    /// Skalierung kuerzt sich exakt heraus. Deshalb wird der AMS aus der
    /// physikalischen Matrix gebaut und in apply_preconditioner nur mit zwei
    /// Divisionen durch D_scale eingerahmt.
    ///
    /// Muss nach setup_fem_matrix() und nach jeder Neuassemblierung der Werte
    /// aufgerufen werden -- die Hilfsmatrizen haengen von A ab.
    void setup_ams(const Mesh& mesh);

    /// Reicht die Wahl der AMG-Grundlage im vektoriellen Hilfsraum durch
    /// (siehe AmsPreconditioner::set_use_alpha_poisson). Muss VOR setup_ams()
    /// aufgerufen werden. Gebraucht vom End-to-End-Test, der beide Wege
    /// gegeneinander misst.
    void set_ams_use_alpha_poisson(bool on) { m_ams.set_use_alpha_poisson(on); }

    /// Materialkoeffizienten je Hexaeder fuer den Alpha-Poisson-Operator.
    /// Muss VOR setup_ams() aufgerufen werden.
    void set_ams_alpha_coefficients(const double* nu_per_hex, const double* beta_per_hex) {
        m_ams.set_alpha_coefficients(nu_per_hex, beta_per_hex);
    }

    double ams_vector_convergence() const { return m_ams.vector_convergence(); }
    double ams_scalar_convergence() const { return m_ams.scalar_convergence(); }

    bool ams_ready() const { return m_ams_ready; }

    // Wendet die volle Eichungs-Projektion P an (Baum-Koordinatenmaske, dann
    // alle registrierten Kohomologie-Projektoren, in dieser Reihenfolge) und
    // schreibt P(d_in) nach d_out. d_in == d_out ist erlaubt (in-place).
    // Ist keine Eichung aktiv (setup_gauge() nicht aufgerufen), kopiert diese
    // Methode nur d_in nach d_out (bzw. tut nichts, falls d_in == d_out).
    // Von apply() intern genutzt UND vom BiCGStabSolver, um die Ausgabe des
    // ILU(0)-Preconditioners nach-zueichen (siehe apply_preconditioner()).
    // Arbeitet auch bei aktiver BEM-Kopplung unveraendert nur
    // auf den ersten total_dofs (FEM-)Eintraegen von d_in/d_out -- ein evtl.
    // angehaengter BEM/K-Anteil (bei d_in/d_out der Laenge get_total_dofs())
    // wird gar nicht erst gelesen oder geschrieben. Die Eichung wirkt also
    // automatisch ausschliesslich auf den A-Block, ohne dass diese Methode
    // dafuer angepasst werden musste.
    void project_gauge(const double* d_in, double* d_out);

    // project_gauge() arbeitet mit den PHYSIKALISCHEN
    // Kohomologie-Vektoren c (unskaliert). Seit apply()/BiCGStab im
    // aequilibrierten Tilde-Raum rechnen (siehe setup_equilibration()),
    // wuerde ein direkter Aufruf von project_gauge() auf einem Tilde-Vektor
    // c^T x_tilde statt des korrekten c^T x_real bilden -- dimensional
    // inkonsistent, injiziert Rauschen in die Krylov-Suchrichtungen. Diese
    // Methode transformiert stattdessen nach physikalisch (x_real = D*x_tilde),
    // projiziert dort korrekt, und transformiert zurueck (x_tilde = x_real/D).
    // Ohne aktive Aequilibrierung (oder ohne aktive Eichung) identisch zu
    // project_gauge(). d_in_tilde/d_out_tilde muessen Laenge get_total_dofs()
    // haben (nicht nur total_dofs), da der BEM-Anteil mituebertragen wird.
    void project_gauge_tilde(const double* d_in_tilde, double* d_out_tilde);

    // Hilfsfunktion fuer project_gauge_tilde() -- fuehrt die
    // Projektion DIREKT im Tilde-Raum aus, mit dem korrekten Tilde-Nullvektor
    // c_tilde = D^-1 * c (nicht c selbst!). Arbeitet in-place auf d_x_tilde,
    // das Laenge total_dofs (nicht get_total_dofs()) haben darf -- die
    // Kohomologie-Projektoren betreffen ausschliesslich den A-Block, exakt
    // wie project_gauge(). Siehe project_gauge_tilde() fuer die Herleitung,
    // warum die vorherige Transform-Projekt-Rücktransform-Variante keine
    // orthogonale Projektion im Tilde-Raum war.
    void project_gauge_tilde_orthogonal(double* d_x_tilde);

    // Hinzufügen von BEM Blöcken (Single-Layer, G_BEM -- wirkt auf K_BEM)
    void add_dense_block(int m, int n, int row_off, int col_off, double* d_mat);

    // derselbe Speicher, als Transponierte angewandt.
    //
    // G_BEM (Single-Layer) ist analytisch symmetrisch. Bisher wurden die
    // Bloecke (I,J) und (J,I) UNABHAENGIG voneinander integriert bzw.
    // ACA-komprimiert -- das Ergebnis war nur bis auf Quadratur- und
    // Approximationsfehler symmetrisch. Fuer das innere CG des
    // blocktriangulaeren Vorkonditionierers ist das fatal: CG setzt SPD voraus
    // und divergiert sonst. Gemessen erreichte es relative Residuen von 1.7
    // bis 2.1 bei einem Ziel von 0.01, vergroesserte das Residuum also.
    //
    // Wird der gespiegelte Block statt dessen als Transponierte des bereits
    // berechneten aufgesetzt, ist G_BEM per Konstruktion exakt symmetrisch --
    // unabhaengig von jedem Approximationsfehler. Nebeneffekt: der Speicher
    // fuer die Fernfeldbloecke halbiert sich, und die halbe Integrationsarbeit
    // entfaellt.
    //
    // m/n/row_off/col_off beziehen sich auf den ANGEWANDTEN Block; d_mat ist
    // die gespeicherte (n x m) Matrix des Partners.
    void add_dense_block_transposed(int m, int n, int row_off, int col_off, double* d_mat);
    void add_aca_block(int m, int n, int k, int row_off, int col_off, double* d_U, double* d_V);

    /// Gespiegelter ACA-Block: (U V^T)^T = V U^T, also dieselben Puffer mit
    /// vertauschten Rollen. Kein zusaetzlicher Speicher.
    /// Siehe add_dense_block_transposed() fuer die Begruendung.
    void add_aca_block_mirrored(int m, int n, int k, int row_off, int col_off,
                                double* d_U_partner, double* d_V_partner) {
        add_aca_block(m, n, k, row_off, col_off, /*U=*/d_V_partner, /*V=*/d_U_partner);
    }

    // BEM Blöcke fuer den Double-Layer-Anteil D_BEM (wirkt
    // auf die gegatherte Randspur von A, siehe apply()). row_off/col_off sind
    // dieselben BEM-Element-Offsets wie bei add_dense_block/add_aca_block.
    void add_dense_block_H(int m, int n, int row_off, int col_off, double* d_mat);
    void add_aca_block_H(int m, int n, int k, int row_off, int col_off, double* d_U, double* d_V);

    // Diagnose -- gibt Min/Max/Mittelwert von |d_bem_inv_diag| aus, plus
    // die Anzahl Eintraege, die noch beim Default (1.0) stehen (also nie von
    // einem Self-Patch-Dense-Block ueberschrieben wurden). Aufrufen NACHDEM
    // alle add_dense_block()-Aufrufe fuer diesen Zeitschritt abgeschlossen
    // sind. Dient dazu, pathologische (sehr grosse/kleine) Diagonalwerte zu
    // finden, die den Jacobi-K-Preconditioner instabil machen koennten.
    void debug_print_bem_diag_stats() const;

    // siehe Implementierung in CoupledSolver.cu -- zerlegt
    // die Norm eines Residuumvektors in geeichte Kanten / freie FEM-Kanten /
    // BEM-Block, um zu lokalisieren, wo ein Konvergenz-Plateau tatsaechlich
    // sitzt. r muss Laenge get_total_dofs() haben, label ist nur fuer die
    // Ausgabe (z.B. "r", "p_hat").
    void debug_print_gauge_split_norm(const double* r, const char* label) const;

    // Baut D = diag(1/sqrt(|A_ii|)) (FEM-Anteil, aus der rohen
    // FEM-Diagonale) bzw. D = diag(sqrt(|(-mu_0*G_ii)^-1|)) (BEM-Anteil, aus
    // d_bem_inv_diag) auf. Muss aufgerufen werden, NACHDEM alle
    // add_dense_block()/add_aca_block()-Aufrufe fuer diesen Zeitschritt
    // abgeschlossen sind (die BEM-Diagonale muss vollstaendig extrahiert
    // sein), aber VOR compute_ilu() (das die FEM-Kopie damit skaliert) und
    // vor dem ersten apply()-Aufruf.
    void setup_equilibration();

    // v *= D bzw. v /= D (elementweise), fuer die Skalierung von
    // RHS und Startvektor ausserhalb des Krylov-Loesers (siehe fem.cu). v
    // muss Laenge get_total_dofs() haben. Ohne aktive Aequilibrierung
    // (setup_equilibration() nicht aufgerufen) ist dies ein No-Op.
    void scale_by_D(double* v);
    void scale_by_D_inverse(double* v);

    // Die Kernoperation loest jetzt das 2x2-Blocksystem
    //     ( A_sys   -T      ) (A_FEM )   (RHS_FEM)
    //     ( 0.5*T+D  -mu_0*G) (K_BEM )  = (   0   )
    // d_x/d_y haben Laenge get_total_dofs() = FEM-DOFs + BEM-DOFs: die ersten
    // FEM-DOFs Eintraege sind A, der Rest ist K. Ohne aktive BEM-Kopplung
    // (setup_bem_coupling() nicht aufgerufen) reduziert sich das auf das reine
    // FEM-System wie zuvor.
    // Ist setup_equilibration() vorher aufgerufen worden, ist dies
    // nur noch ein duenner Wrapper -- er transformiert d_x_tilde nach
    // physikalisch (D*x), ruft apply_physical() auf und transformiert das
    // Ergebnis zurueck (D*y). Ohne aktive Aequilibrierung (setup_equilibration()
    // nicht aufgerufen) faellt apply() auf apply_physical() zurueck (No-Op-D).
    void apply(const double* d_x, double* d_y);

    // Wie apply(), aber fuer nrhs Vektoren auf einmal: Spalte j von d_X liegt
    // bei d_X + j*ld (column-major), Ergebnis entsprechend in d_Y. Ergebnis
    // gleich nrhs Aufrufen von apply() bis auf Rundung.
    //
    // Warum: apply() ruft fuer jeden der einige tausend BEM-Bloecke ein bis
    // zwei Matrix-Vektor-Produkte auf; auf hex_s1 sind das rund 14000
    // cuBLAS-Aufrufe und 0,3 s je Anwendung, fast nur Startkosten. Hier
    // laufen die BEM-Bloecke einmal je Block als Matrix-Matrix-Produkt ueber
    // alle Spalten. Der Rest (duenne FEM-Matrix, T-Kopplung, Eichung) bleibt
    // spaltenweise, er ist billig.
    void apply_many(const double* d_X, double* d_Y, int ld, int nrhs);

    // Liefert jetzt die GEKOPPELTE Systemgroesse (FEM-DOFs +
    // BEM-DOFs), sobald setup_bem_coupling() aufgerufen wurde -- das ist die
    // Vektorlaenge, mit der BiCGStabSolver dimensioniert werden muss. Ohne
    // BEM-Kopplung identisch zum bisherigen (reinen FEM-)Verhalten.
    int get_total_dofs() const { return total_dofs + num_bem_dofs; }
    int get_num_bem_dofs() const { return num_bem_dofs; }
    cublasHandle_t get_cublas_handle() const { return cublas; }
};

// -----------------------------------------------------------------------------
// Aufgabe 6.2: Der BiCGStab Krylov Löser
// -----------------------------------------------------------------------------
class BiCGStabSolver {
private:
    DeviceArray<double> r;
    DeviceArray<double> r_hat;
    DeviceArray<double> p;
    DeviceArray<double> v;
    DeviceArray<double> s;
    DeviceArray<double> t;
    DeviceArray<double> p_hat; // M^-1 * p (rechts-ILU(0)-Vorkonditionierung)
    DeviceArray<double> s_hat; // M^-1 * s
    DeviceArray<double> x_best; // bestes bisher gefundenes x (kleinstes Rel.Res.)

public:
    BiCGStabSolver(int N);
    
    // Löst das System A*x = b. d_x dient als Startvektor (z.B. Nullen)
    // und enthält am Ende die Lösung.
    void solve(CoupledMVOperator& op, const double* d_b, double* d_x, double tol = 1e-6, int max_iter = 1000);
};

// -----------------------------------------------------------------------------
// GMRES(m) mit Rechts-Vorkonditionierung (Flexible-GMRES-Bauart) als
// Alternative zu BiCGStab. Garantiert (in exakter Arithmetik) streng monoton
// fallendes Residuum ueber den aufgebauten Krylov-Raum -- keine Near-
// Breakdowns, keine Spikes wie bei BiCGStab. Kosten: ein Restart-Zyklus
// braucht O(m) Basisvektoren der Laenge N (V_basis, Z_basis), Orthogonalisier-
// ung ist O(m^2) statt O(1) pro Iteration.
// -----------------------------------------------------------------------------
class GMRESSolver {
public:
    GMRESSolver(int N, int m_restart = 50);
    ~GMRESSolver() = default;

    void solve(CoupledMVOperator& op, const double* d_b, double* d_x, double tol = 1e-6, int max_iter = 1000);

    /// Relatives Residuum am Ende des letzten solve(). Gebraucht, damit
    /// Aufrufer erkennen koennen, ob das Ergebnis ueberhaupt eine Loesung ist --
    /// eine Kraftberechnung auf einem nicht konvergierten Feld liefert sonst
    /// Zahlen wie 105 N ohne jeden Hinweis darauf, dass sie bedeutungslos sind.
    double last_relative_residual() const { return m_last_rel_res; }

    /// Iterationen des letzten solve()-Aufrufs. Gebraucht fuer den
    /// End-to-End-Test der Vorkonditionierer, der genau diese Zahl misst --
    /// die Kennzahl, auf die es ankommt und die bisher nirgends abgreifbar war.
    int last_iterations() const { return m_last_iterations; }

    /// Ritz-Spektrum nach dem ersten vollen Arnoldi-Zyklus ausgeben. Liefert
    /// eine untere Schranke fuer die Konditionszahl des VORKONDITIONIERTEN
    /// Operators und damit die Antwort auf die Frage, ob die zugestandene
    /// Iterationszahl ueberhaupt reichen kann.
    /// Blockweise Aufschluesselung des echten Residuums bei jedem Restart.
    /// num_bem ist die Laenge des BEM-Blocks am Ende des Vektors; 0 schaltet
    /// die Ausgabe ab. Siehe die Begruendung an der Ausgabestelle.
    void enable_block_residual_report(int num_bem) {
        m_num_bem_for_report = num_bem;
        m_report_block_residual = (num_bem > 0);
    }

    void enable_ritz_report(bool on) { m_want_ritz = on; }

private:
    double m_last_rel_res = -1.0;
    int    m_last_iterations = 0;
    bool                m_want_ritz = false;
    bool m_report_block_residual = false;
    int  m_num_bem_for_report = 0;
    int                 m_ritz_dim  = 0;
    std::vector<double> m_H_raw;
    int N;
    int m; // Restart-Parameter (typisch 30 bis 50)

    DeviceArray<double> r;
    DeviceArray<double> w;

    // Basisvektoren auf der GPU, spaltenweise flach abgelegt:
    // V_basis speichert (m+1) orthonormale Vektoren der Laenge N (Krylov-
    // Basis des Operators A*M^-1), Z_basis speichert m vorkonditionierte
    // Vektoren M^-1*v_j (fuer die Rueckrechnung x += Z*y).
    DeviceArray<double> V_basis;
    DeviceArray<double> Z_basis;

    // CPU-seitige (m+1 x m) Hessenberg-Matrix, flach abgelegt als H[row*m+col].
    std::vector<double> H;
    std::vector<double> g;
    std::vector<double> cosines;
    std::vector<double> sines;

    static void apply_givens_rotation(double& h1, double& h2, double& cs, double& sn);
    static void generate_givens_rotation(double dx, double dy, double& cs, double& sn);
};

#endif // COUPLED_SOLVER_CUH