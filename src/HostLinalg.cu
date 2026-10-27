#include "HostLinalg.cuh"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// Fehlerpruefung fuer die NVIDIA-Bibliotheken. Bewusst lokal dupliziert statt
// aus CoupledSolver.cuh geholt: dieser Header soll unabhaengig von der
// Loeserklasse benutzbar bleiben.
// -----------------------------------------------------------------------------
#define BF_CUBLAS_CHECK(call)                                                       \
    do {                                                                            \
        cublasStatus_t st_ = (call);                                                \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
            throw std::runtime_error("cuBLAS Error at " + std::string(__FILE__) +   \
                                     ":" + std::to_string(__LINE__));               \
        }                                                                           \
    } while (0)

namespace bf {

// -----------------------------------------------------------------------------
// Schutz gegen einen Fehler, der einmal teuer war
// -----------------------------------------------------------------------------
// Diese Wrapper entscheiden nach dem BACKEND, nicht nach dem Handle. Ein
// nullptr-Handle ist im CPU-Modus richtig (er wird ignoriert), im GPU-Modus
// aber ein Programmfehler des Aufrufers -- cuBLAS meldet dann nur ein
// nichtssagendes CUBLAS_STATUS_NOT_INITIALIZED. Genau so ist der
// Referenzloeser beim ersten Lauf gescheitert ("cuBLAS Error at
// HostLinalg.cu:53"), und die Meldung liess nicht erkennen, dass ein Handle
// fehlte. Deshalb hier eine Diagnose, die die Ursache benennt.
static inline void require_handle(const void* h, const char* fn) {
    if (on_gpu() && h == nullptr) {
        throw std::runtime_error(
            std::string("bf::") + fn + ": im GPU-Modus wurde ein nullptr-cuBLAS-Handle "
            "uebergeben. Diese Wrapper entscheiden nach bf::g_backend, nicht nach dem "
            "Handle -- der Aufrufer muss im GPU-Modus einen mit cublasCreate() "
            "erzeugten Handle durchreichen.");
    }
}

// =============================================================================
// BLAS Level 1
// =============================================================================

void dnrm2(cublasHandle_t h, int n, const double* x, double* result) {
    require_handle(h, "dnrm2");
    if (n <= 0) { *result = 0.0; return; }

    if (on_cpu()) {
        // Skalierte Summation (wie im Referenz-BLAS): verhindert Ueber-/
        // Unterlauf, wenn einzelne Komponenten extrem gross oder klein sind.
        // Bei diesem Loeser durchaus relevant -- die Diagnoseausgaben zeigen
        // Komponenten von ~1e-8 bis ~1e9 im selben Vektor.
        //
        // ACHTUNG, hier steckte ein Fehler mit erheblicher Wirkung: NaN und Inf
        // muessen ausdruecklich behandelt werden. Jeder Vergleich mit NaN ist
        // falsch, also war "scale < a" fuer ein NaN nie erfuellt und scale blieb
        // 0.0 -- ein Vektor aus LAUTER NaN lieferte damit die Norm 0.0 statt
        // NaN. (Ein Vektor mit gemischten Eintraegen lieferte korrekt NaN, weil
        // die endlichen Werte scale gesetzt haben. Deshalb fiel es nie auf.)
        //
        // Die Folgen waren keine Kleinigkeit: GMRES prueft die Konvergenz mit
        // r_norm/b_norm. Wurde der Loesungsvektor irgendwo NaN, wurde auch das
        // Residuum durchgehend NaN -- und meldete sich als 0.0 zurueck. Der
        // Loeser gab "Converged | Rel. Res: 0" aus und lieferte NaN-Kraefte.
        // Ebenso alle Diagnosenormen: "||FEM-Anteil von y|| = 0" bedeutete
        // nicht "der Vektor ist null", sondern "der Vektor ist komplett NaN".
        double scale = 0.0;
        double ssq   = 1.0;
        bool saw_nan = false;
        bool saw_inf = false;
        for (int i = 0; i < n; ++i) {
            const double a = std::fabs(x[i]);
            if (a == 0.0) continue;
            if (std::isnan(a)) { saw_nan = true; continue; }
            if (std::isinf(a)) { saw_inf = true; continue; }
            if (scale < a) {
                const double q = scale / a;
                ssq   = 1.0 + ssq * q * q;
                scale = a;
            } else {
                const double q = a / scale;
                ssq += q * q;
            }
        }
        // NaN schlaegt Inf: ist auch nur ein Eintrag NaN, ist die Norm NaN.
        if (saw_nan)      *result = std::numeric_limits<double>::quiet_NaN();
        else if (saw_inf) *result = std::numeric_limits<double>::infinity();
        else              *result = (scale == 0.0) ? 0.0 : scale * std::sqrt(ssq);
        return;
    }
    BF_CUBLAS_CHECK(cublasDnrm2(h, n, x, 1, result));
}

void ddot(cublasHandle_t h, int n, const double* x, const double* y, double* result) {
    require_handle(h, "ddot");
    if (n <= 0) { *result = 0.0; return; }

    if (on_cpu()) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += x[i] * y[i];
        *result = sum;
        return;
    }
    BF_CUBLAS_CHECK(cublasDdot(h, n, x, 1, y, 1, result));
}

void daxpy(cublasHandle_t h, int n, double alpha, const double* x, double* y) {
    require_handle(h, "daxpy");
    if (n <= 0) return;

    if (on_cpu()) {
        BF_OMP_PARALLEL_FOR
        for (int i = 0; i < n; ++i) y[i] += alpha * x[i];
        return;
    }
    BF_CUBLAS_CHECK(cublasDaxpy(h, n, &alpha, x, 1, y, 1));
}

void dscal(cublasHandle_t h, int n, double alpha, double* x) {
    require_handle(h, "dscal");
    if (n <= 0) return;

    if (on_cpu()) {
        BF_OMP_PARALLEL_FOR
        for (int i = 0; i < n; ++i) x[i] *= alpha;
        return;
    }
    BF_CUBLAS_CHECK(cublasDscal(h, n, &alpha, x, 1));
}

void dcopy(cublasHandle_t h, int n, const double* x, double* y) {
    require_handle(h, "dcopy");
    if (n <= 0 || x == y) return;

    if (on_cpu()) {
        std::memcpy(y, x, static_cast<size_t>(n) * sizeof(double));
        return;
    }
    BF_CUBLAS_CHECK(cublasDcopy(h, n, x, 1, y, 1));
}

// =============================================================================
// BLAS Level 2
// =============================================================================

void dgemv(cublasHandle_t h, bool transpose, int m, int n,
           double alpha, const double* A, int lda,
           const double* x, double beta, double* y)
{
    require_handle(h, "dgemv");
    if (m <= 0 || n <= 0) return;

    if (on_cpu()) {
        // WICHTIG: beta == 0 bedeutet, dass y NICHT GELESEN werden darf.
        //
        // Das ist keine Feinheit, sondern ausdruecklicher Teil der
        // BLAS-Spezifikation, und der Aufrufer verlaesst sich darauf: bei
        // beta == 0 darf y uninitialisiert sein. Die vorherige Fassung
        // rechnete unbedingt
        //     y[i] = alpha * acc + beta * y[i];
        // und las y damit auch bei beta == 0. Enthaelt der uninitialisierte
        // Puffer ein NaN-Bitmuster, ist 0 * NaN = NaN -- das Ergebnis ist
        // vollstaendig NaN, obwohl A und x endlich sind.
        //
        // Genau so sind die NaN-Kraefte entstanden. apply_physical() benutzt
        // d_temp_z als Zwischenpuffer fuer z = V^T*x der ACA-Bloecke, mit
        // beta = 0. DeviceArray::allocate() initialisiert nicht (fill_zero()
        // ist eine eigene Methode), d_temp_z enthaelt also rohen
        // malloc-Speicher. Ein NaN darin machte z zu NaN, und der
        // anschliessende U*z-Schritt trug es in ALLE Zeilen des Blocks. Weil
        // G- und H-Bloecke denselben Puffer benutzen, waren beide betroffen --
        // 16544 von 16544 Eintraegen, bei sauberen Blockdaten und sauberer
        // Eingabe.
        //
        // A ist column-major: A(i,j) == A[j * lda + i], i in [0,m), j in [0,n).
        const bool overwrite = (beta == 0.0);
        if (!transpose) {
            // y (Laenge m) = alpha * A * x (x Laenge n) + beta * y
            BF_OMP_PARALLEL_FOR
            for (int i = 0; i < m; ++i) {
                double acc = 0.0;
                for (int j = 0; j < n; ++j) {
                    acc += A[static_cast<size_t>(j) * lda + i] * x[j];
                }
                y[i] = overwrite ? (alpha * acc) : (alpha * acc + beta * y[i]);
            }
        } else {
            // y (Laenge n) = alpha * A^T * x (x Laenge m) + beta * y
            BF_OMP_PARALLEL_FOR
            for (int j = 0; j < n; ++j) {
                double acc = 0.0;
                const double* col = A + static_cast<size_t>(j) * lda;
                for (int i = 0; i < m; ++i) {
                    acc += col[i] * x[i];
                }
                y[j] = overwrite ? (alpha * acc) : (alpha * acc + beta * y[j]);
            }
        }
        return;
    }

    BF_CUBLAS_CHECK(cublasDgemv(h, transpose ? CUBLAS_OP_T : CUBLAS_OP_N,
                                m, n, &alpha, A, lda, x, 1, &beta, y, 1));
}

// =============================================================================
// Duennbesetzt
// =============================================================================

BF_HD inline void csr_spmv_kernel_body(int row, int num_rows, const int* row_offsets,
                                       const int* col_indices, const double* values,
                                       const double* x, double* y)
{
    (void)num_rows;
    double acc = 0.0;
    const int start = row_offsets[row];
    const int end   = row_offsets[row + 1];
    for (int k = start; k < end; ++k) acc += values[k] * x[col_indices[k]];
    y[row] = acc;
}

} // namespace bf

__global__ void csr_spmv_kernel(int num_rows, const int* row_offsets,
                                const int* col_indices, const double* values,
                                const double* x, double* y)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    bf::csr_spmv_kernel_body(row, num_rows, row_offsets, col_indices, values, x, y);
}

namespace bf {

void csr_spmv(int num_rows, const int* row_offsets, const int* col_indices,
              const double* values, const double* x, double* y)
{
    // Jede Zeile schreibt nur y[row] -- rennfrei, also die parallele Variante.
    BF_LAUNCH_1D_PAR(csr_spmv_kernel, num_rows,
                     num_rows, row_offsets, col_indices, values, x, y);
}

int csr_ilu0(int num_rows, const int* row_offsets, const int* col_indices,
             double* values, std::vector<int>& diag_index_out)
{
    // Position des Diagonaleintrags je Zeile einmalig bestimmen. Die
    // FEM-Sparsity aus build_csr_structure() enthaelt die Diagonale immer
    // (jede Kante ist mit sich selbst gekoppelt), ein fehlender Eintrag waere
    // ein struktureller Nullpivot -- genau das, was cusparse hier meldet.
    diag_index_out.assign(static_cast<size_t>(num_rows), -1);
    for (int row = 0; row < num_rows; ++row) {
        for (int k = row_offsets[row]; k < row_offsets[row + 1]; ++k) {
            if (col_indices[k] == row) { diag_index_out[row] = k; break; }
        }
        if (diag_index_out[row] < 0) return row;   // struktureller Nullpivot
    }

    // Streuvektor: bildet Spaltenindex -> Position im values-Array der
    // aktuell bearbeiteten Zeile ab. Ohne ihn waere der innere Abgleich
    // quadratisch in der Zeilenlaenge.
    std::vector<int> pos(static_cast<size_t>(num_rows), -1);

    for (int i = 0; i < num_rows; ++i) {
        const int row_start = row_offsets[i];
        const int row_end   = row_offsets[i + 1];

        for (int k = row_start; k < row_end; ++k) pos[col_indices[k]] = k;

        // IKJ-Variante: fuer jedes k < i mit A(i,k) != 0 eliminieren.
        for (int kk = row_start; kk < row_end; ++kk) {
            const int k = col_indices[kk];
            if (k >= i) break;   // Spalten sind aufsteigend sortiert

            const double pivot = values[diag_index_out[k]];
            if (pivot == 0.0) {
                for (int t = row_start; t < row_end; ++t) pos[col_indices[t]] = -1;
                return k;        // numerischer Nullpivot
            }

            const double mult = values[kk] / pivot;
            values[kk] = mult;

            // Zeile i -= mult * Zeile k, aber nur auf Positionen, die im
            // Muster von Zeile i schon existieren (das ist gerade ILU(0):
            // kein Fill-in).
            for (int j = diag_index_out[k] + 1; j < row_offsets[k + 1]; ++j) {
                const int col = col_indices[j];
                const int p   = pos[col];
                if (p >= 0) values[p] -= mult * values[j];
            }
        }

        if (values[diag_index_out[i]] == 0.0) {
            for (int t = row_start; t < row_end; ++t) pos[col_indices[t]] = -1;
            return i;
        }

        for (int k = row_start; k < row_end; ++k) pos[col_indices[k]] = -1;
    }

    return -1;
}

void csr_lu_solve(int num_rows, const int* row_offsets, const int* col_indices,
                  const double* lu_values, const std::vector<int>& diag_index,
                  const double* b, double* x, double* temp)
{
    // 1. Vorwaerts: L * temp = b, L hat implizite Einheitsdiagonale.
    for (int i = 0; i < num_rows; ++i) {
        double acc = b[i];
        for (int k = row_offsets[i]; k < diag_index[i]; ++k) {
            acc -= lu_values[k] * temp[col_indices[k]];
        }
        temp[i] = acc;
    }

    // 2. Rueckwaerts: U * x = temp, U mit echter Diagonale.
    for (int i = num_rows - 1; i >= 0; --i) {
        double acc = temp[i];
        for (int k = diag_index[i] + 1; k < row_offsets[i + 1]; ++k) {
            acc -= lu_values[k] * x[col_indices[k]];
        }
        x[i] = acc / lu_values[diag_index[i]];
    }
}

} // namespace bf
