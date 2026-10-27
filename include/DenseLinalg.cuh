#ifndef DENSE_LINALG_CUH
#define DENSE_LINALG_CUH

// =============================================================================
// DenseLinalg.cuh -- LU-Zerlegung und Loesen fuer VOLLBESETZTE Matrizen
// =============================================================================
//
// Gebraucht vom Referenzloeser (--reference): dort wird die Einfachschicht-
// matrix V vollbesetzt aufgebaut und der Dirichlet-zu-Neumann-Operator
//
//     D = V^-1 (K - 1/2 I)
//
// explizit gebildet. Das ist eine LU-Zerlegung von V plus N_bem
// Dreiecksloesungen. Bei N_bem = 4138 sind das 2/3*N^3 = 4.7e10 FLOP fuer die
// Zerlegung und 2*N^3 = 1.4e11 fuer die Loesungen -- auf der GPU Bruchteile
// einer Sekunde, und weil sich die Randgeometrie in jedem Zeitschritt aendert
// (die Platte bewegt sich), fallen sie in jedem Schritt neu an.
//
// GPU: cuSOLVER (Dgetrf/Dgetrs). CPU: eigene LU mit Spaltenpivotisierung.
// Die CPU-Fassung ist NICHT fuer Produktionslaeufe gedacht -- bei N = 4138
// dauert sie einkernig Minuten. Sie existiert, damit der Referenzloeser unter
// --cpu ueberhaupt lauffaehig und damit auf kleinen Netzen testbar ist.
//
// Speicherlayout durchgehend COLUMN-MAJOR (A(i,j) == A[j*lda + i]), wie es
// LAPACK und cuSOLVER erwarten.
// =============================================================================

#include "Backend.cuh"

#include <vector>

namespace bf {

/// Deckel fuer die Zerlegung: haelt im GPU-Fall den cuSOLVER-Handle und den
/// Arbeitspuffer, im CPU-Fall nur die Pivotfolge. Nicht kopierbar, damit die
/// Ressourcen genau einmal freigegeben werden.
class DenseLU {
public:
    DenseLU() = default;
    ~DenseLU();

    DenseLU(const DenseLU&)            = delete;
    DenseLU& operator=(const DenseLU&) = delete;

    /// Zerlegt die n x n Matrix A (column-major, lda = n) IN PLACE.
    /// Wirft bei singulaerer Matrix (exakt verschwindender Pivot).
    void factorize(double* A, int n);

    /// Loest A X = B in place, B ist n x nrhs (column-major, ldb = n).
    /// factorize() muss vorher gelaufen sein und A darf seither nicht
    /// verandert worden sein -- der Zeiger wird gemerkt, nicht die Matrix
    /// kopiert.
    void solve(double* B, int nrhs) const;

    int size() const { return m_n; }

private:
    void  release();

    int      m_n     = 0;
    double*  m_A     = nullptr;   // nicht besitzend: zeigt auf die Matrix des Aufrufers
    // GPU
    void*    m_solver_handle = nullptr;   // cusolverDnHandle_t, als void* um den
                                          // cusolver-Header nicht zu exportieren
    double*  m_work  = nullptr;
    int      m_lwork = 0;
    int*     m_ipiv  = nullptr;
    int*     m_info  = nullptr;
    // CPU
    mutable std::vector<int> m_host_ipiv;
};

} // namespace bf

#endif // DENSE_LINALG_CUH
