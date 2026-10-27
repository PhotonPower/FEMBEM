#ifndef FEMBEM_HOST_LINALG_CUH
#define FEMBEM_HOST_LINALG_CUH

// =============================================================================
// HostLinalg.cuh -- backendneutrale Huelle um die benutzten cuBLAS- und
// cuSPARSE-Operationen
// =============================================================================
//
// CoupledSolver.cu benutzt genau sechs BLAS-1/2-Operationen (Dnrm2, Ddot,
// Daxpy, Dscal, Dcopy, Dgemv), ein Sparse-Matrix-Vektor-Produkt, eine
// ILU(0)-Faktorisierung und zwei Dreieckssubstitutionen. Jede davon bekommt
// hier eine Fassung, die im GPU-Modus an cuBLAS/cuSPARSE weiterreicht und im
// CPU-Modus selbst rechnet.
//
// Die Signaturen sind absichtlich an die cuBLAS-Originale angelehnt (Handle
// als erstes Argument, Ergebnis per Zeiger), damit die Aufrufstellen in
// CoupledSolver.cu eine reine Umbenennung bleiben und beim Lesen weiterhin
// die BLAS-Semantik erkennbar ist. Im CPU-Modus ist der Handle nullptr und
// wird ignoriert.
//
// Genauigkeit: die CPU-Reduktionen (nrm2, dot) summieren seriell in
// double. cuBLAS summiert baumfoermig, das Ergebnis kann sich also in den
// letzten Bits unterscheiden -- das ist bei einer Reduktion ueber ~1e5
// Elemente unvermeidlich und fuer die Abbruchkriterien des Krylov-Loesers
// unerheblich.
// =============================================================================

#include "Backend.cuh"

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4996)
#endif

#include <cublas_v2.h>
#include <cusparse.h>

#if defined(_MSC_VER)
    #pragma warning(pop)
#endif

#include <vector>

namespace bf {

// -----------------------------------------------------------------------------
// BLAS Level 1
// -----------------------------------------------------------------------------
void dnrm2(cublasHandle_t h, int n, const double* x, double* result);
void ddot (cublasHandle_t h, int n, const double* x, const double* y, double* result);
void daxpy(cublasHandle_t h, int n, double alpha, const double* x, double* y);
void dscal(cublasHandle_t h, int n, double alpha, double* x);
void dcopy(cublasHandle_t h, int n, const double* x, double* y);

// -----------------------------------------------------------------------------
// BLAS Level 2: y = alpha * op(A) * x + beta * y, A ist column-major (m x n)
// mit fuehrender Dimension lda. transpose == false entspricht CUBLAS_OP_N.
// -----------------------------------------------------------------------------
void dgemv(cublasHandle_t h, bool transpose, int m, int n,
           double alpha, const double* A, int lda,
           const double* x, double beta, double* y);

// -----------------------------------------------------------------------------
// Duennbesetzte Operationen (CSR, nullbasiert, Spaltenindizes je Zeile
// aufsteigend sortiert -- build_csr_structure() erzeugt sie aus einem
// std::set, das ist also gegeben und wird von csr_ilu0() vorausgesetzt).
// -----------------------------------------------------------------------------

/// y = A * x (beta = 0). Backendneutral: im GPU-Modus ein Kernel, im CPU-Modus
/// eine (OpenMP-)Schleife. Bewusst ein eigener Kernel statt cusparseSpMV --
/// hier wird kein Deskriptor wiederverwendet, und ein zeilenweiser Kernel ist
/// fuer diese Matrix genauso schnell wie cuSPARSE, aber ohne den
/// Verwaltungsaufwand. CoupledSolver benutzt weiterhin cusparseSpMV, dort
/// lohnt die Deskriptor-Wiederverwendung.
void csr_spmv(int num_rows, const int* row_offsets, const int* col_indices,
              const double* values, const double* x, double* y);

/// ILU(0) in-place auf 'values' (gleiche Sparsity wie die Eingabematrix),
/// IKJ-Variante. Ergebnis: L (Einheitsdiagonale, implizit) im strikt unteren,
/// U im oberen Dreieck einschliesslich Diagonale.
///
/// Rueckgabe: Zeilenindex des ersten Nullpivots, oder -1 bei Erfolg. Der
/// Aufrufer behandelt das genau wie cusparseXcsrilu02_zeroPivot.
int csr_ilu0(int num_rows, const int* row_offsets, const int* col_indices,
             double* values, std::vector<int>& diag_index_out);

/// Loest (L*U) * x = b mit der von csr_ilu0() erzeugten Faktorisierung.
/// 'temp' braucht num_rows Eintraege und dient dem Vorwaertseinsetzen.
/// b und x duerfen NICHT auf denselben Speicher zeigen.
void csr_lu_solve(int num_rows, const int* row_offsets, const int* col_indices,
                  const double* lu_values, const std::vector<int>& diag_index,
                  const double* b, double* x, double* temp);

} // namespace bf

#endif // FEMBEM_HOST_LINALG_CUH
