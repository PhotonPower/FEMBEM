#ifndef REFERENCE_HARMONIC_CUH
#define REFERENCE_HARMONIC_CUH

// =============================================================================
// ReferenceHarmonic.cuh -- Frequenzbereich fuer den Referenzloeser
// =============================================================================
//
// Start mit:  ./fembem --reference --harmonic[=HZ] --fixed-plate [--plate-z=MM]
//
// Gehoert zum Referenzloeser (include/ReferenceSolver.cuh) und teilt KEINEN
// Code mit dem harmonischen Nedelec-Pfad (HarmonicSolver, HarmonicBlockSystem).
// Hier stehen nur die netzunabhaengigen Bausteine -- Blockoperator, flexibles
// GMRES und der blockdiagonale Vorkonditionierer --, damit
// tests/test_reference_harmonic sie an kleinen Matrizen gegen eine komplexe
// Direktloesung pruefen kann. Assemblierung und Ablauf stehen in
// src/ReferenceSolver.cu (run_reference_harmonic).
//
// -----------------------------------------------------------------------------
// Gleichung
// -----------------------------------------------------------------------------
// TEAM 28 ist linear. Mit A(t) = Re(A^ e^{jwt}), A^ = A_r + j A_i und der
// transienten Gleichung des Referenzloesers
//
//     (K + G) A + C dA/dt = b(t),    C = sigma * M  (Leitermassenmatrix)
//
// wird im eingeschwungenen Zustand (K + G + jwC) A^ = b^, reell geschrieben
//
//     [ K+G   -wC  ] [A_r]   [b]
//     [ wC    K+G  ] [A_i] = [0]
//
// b ist reell: die Quelle wird im Scheitelwert angesetzt (Spulenstrom 20 A,
// t = T/4 wie beim transienten --verify-field). Das legt nur die globale
// Phase fest; das Zyklusmittel der Kraft ist dagegen invariant.
//
// G = -(1/mu_0) T D ist der DtN-Block aus ReferenceOperator::apply, K der
// Sparse-Anteil ohne Leitwertterm. K und K + wC kommen aus derselben
// Assemblierung (reference_assemble mit dt = 1e30 bzw. dt = 1/w) auf
// identischem CSR-Muster, C ist die Differenz geteilt durch w.
//
// -----------------------------------------------------------------------------
// Loeser
// -----------------------------------------------------------------------------
// Aeusserer Loeser FGMRES (rechtsvorkonditioniert, flexibel: der
// Vorkonditionierer darf sich je Iteration aendern, weil er selbst ein
// iteratives Verfahren mit Toleranz ist). Vorkonditionierer blockdiagonal
//
//     P = diag(M, M),   M = K + G + wC.
//
// Fuer symmetrisches K+G > 0 und C >= 0 (G ist nur naeherungsweise
// symmetrisch) hat P^-1 A die Eigenwerte (1 +- j mu)/(1 + mu), mu >= 0 die
// verallgemeinerten Eigenwerte von wC v = mu (K+G) v. Sie liegen auf den
// beiden Strecken Re + |Im| = 1 von 1 nach +-j, mit |lambda| >= 1/sqrt(2).
// Netzunabhaengig ist das, aber kein Haufen: je groesser mu_max (also w*tau
// der langsamsten Wirbelstrommode), desto naeher reicht das Spektrum an +-j
// und desto mehr aeussere Iterationen braucht GMRES. Am Modellsystem in
// tests/test_reference_harmonic sind es bei mu_max ~ 2.5 rund 30 bis 1e-11.
// M ist genau die Systemmatrix eines transienten Schritts mit dt = 1/w; die
// innere Loesung uebernimmt deshalb der vorhandene reference_gmres mit
// ILU(0), unveraendert.
// =============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <functional>
#include <ostream>
#include <vector>

#include "Backend.cuh"
#include "DeviceArray.cuh"

/// Aeussere Toleranz (echtes relatives Residuum ||b - A x|| / ||b||).
constexpr double REFERENCE_HARMONIC_TOL      = 1e-8;
/// Restart und Deckel des aeusseren FGMRES.
constexpr int    REFERENCE_HARMONIC_RESTART  = 30;
constexpr int    REFERENCE_HARMONIC_MAX_ITER = 200;
/// Toleranz der inneren M-Loesungen. Lockerer als aussen: FGMRES verkraftet
/// einen ungenauen, sich aendernden Vorkonditionierer, die aeussere Toleranz
/// wird am echten Residuum geprueft.
constexpr double REFERENCE_HARMONIC_INNER_TOL = 1e-6;

/// y = A x fuer den reellen 2n-Block. x und y haben 2n Eintraege, erst
/// Realteil, dann Imaginaerteil.
struct ReferenceHarmonicOperator {
    int n = 0;                                   // 3 * num_nodes
    /// y = (K + G) x auf n Eintraegen (ReferenceOperator::apply mit K).
    std::function<void(const double*, double*)> apply_kg;
    /// C = sigma*M als CSR auf n Zeilen.
    const int*    c_row_offsets = nullptr;
    const int*    c_col_indices = nullptr;
    const double* c_values      = nullptr;
    double        omega         = 0.0;
    cublasHandle_t cublas       = nullptr;

    mutable DeviceArray<double> tmp;             // n

    void allocate_buffers() { if (n > 0) tmp.allocate(static_cast<size_t>(n)); }
    void apply(const double* x, double* y) const;
};

/// Optionen und Protokoll des aeusseren Loesers.
struct ReferenceFgmresOptions {
    double tol      = REFERENCE_HARMONIC_TOL;
    int    restart  = REFERENCE_HARMONIC_RESTART;
    int    max_iter = REFERENCE_HARMONIC_MAX_ITER;
};

struct ReferenceFgmresResult {
    bool   converged     = false;
    int    iterations    = 0;
    double rel_residual  = -1.0;       // echtes Residuum am Ende
    /// Je aeusserer Iteration: geschaetztes relatives Residuum (Givens).
    std::vector<double> history;
    /// Je aeusserer Iteration: Summe der inneren Iterationen des
    /// Vorkonditionierers (beide Bloecke).
    std::vector<int>    inner_iterations;
    /// Je Restart: echtes relatives Residuum.
    std::vector<double> restart_residuals;
};

/// Anwendung des Vorkonditionierers: out = P^-1 in auf N Eintraegen,
/// Rueckgabe die Zahl der inneren Iterationen (0 bei direkter Loesung).
using ReferencePrecond = std::function<int(const double*, double*)>;
using ReferenceApply   = std::function<void(const double*, double*)>;

/// Flexibles GMRES(m), rechtsvorkonditioniert. x ist Startwert und Ergebnis
/// (N Eintraege, Managed Memory). Bricht bei NaN/Inf mit std::runtime_error
/// ab. log == nullptr: keine Ausgabe.
ReferenceFgmresResult reference_harmonic_fgmres(
    const ReferenceApply& A, const ReferencePrecond& P,
    int N, const double* b, double* x,
    const ReferenceFgmresOptions& opt, cublasHandle_t cublas,
    std::ostream* log);

/// Blockdiagonaler Vorkonditionierer diag(M, M) auf 2n Eintraegen.
/// solve_m(rhs, x) loest M x = rhs fuer n Eintraege (x kommt mit null
/// belegt an) und gibt die inneren Iterationen zurueck.
inline ReferencePrecond reference_harmonic_block_precond(
    int n, std::function<int(const double*, double*)> solve_m)
{
    return [n, solve_m](const double* in, double* out) {
        bf::memset_d(out, 0, static_cast<size_t>(2) * n * sizeof(double));
        int it = solve_m(in, out);
        it += solve_m(in + n, out + n);
        return it;
    };
}

#endif // REFERENCE_HARMONIC_CUH
