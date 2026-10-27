#include "DenseLinalg.cuh"

#include <cusolverDn.h>

#include <cmath>
#include <stdexcept>
#include <string>

#define BF_CUSOLVER_CHECK(call)                                                    \
    do {                                                                           \
        cusolverStatus_t st_ = (call);                                              \
        if (st_ != CUSOLVER_STATUS_SUCCESS) {                                       \
            throw std::runtime_error("cuSOLVER Error " + std::to_string((int)st_) + \
                                     " at " + __FILE__ + ":" +                      \
                                     std::to_string(__LINE__));                     \
        }                                                                           \
    } while (0)

namespace bf {

DenseLU::~DenseLU() { release(); }

void DenseLU::release() {
    if (on_gpu()) {
        if (m_work)  { cudaFree(m_work);  m_work  = nullptr; }
        if (m_ipiv)  { cudaFree(m_ipiv);  m_ipiv  = nullptr; }
        if (m_info)  { cudaFree(m_info);  m_info  = nullptr; }
        if (m_solver_handle) {
            cusolverDnDestroy(static_cast<cusolverDnHandle_t>(m_solver_handle));
            m_solver_handle = nullptr;
        }
    }
    m_host_ipiv.clear();
    m_n = 0;
    m_A = nullptr;
    m_lwork = 0;
}

// =============================================================================
// Zerlegung
// =============================================================================
void DenseLU::factorize(double* A, int n) {
    if (n <= 0 || A == nullptr) {
        throw std::runtime_error("DenseLU::factorize: leere Matrix.");
    }
    release();
    m_n = n;
    m_A = A;

    if (on_cpu()) {
        // ------------------------------------------------------------------
        // Eigene LU mit Spaltenpivotisierung, column-major, in place.
        // Speichert die Pivotfolge im LAPACK-Format (1-basiert), damit
        // solve() denselben Ablauf wie der GPU-Pfad hat.
        // ------------------------------------------------------------------
        m_host_ipiv.assign(static_cast<size_t>(n), 0);
        auto at = [A, n](int i, int j) -> double& {
            return A[static_cast<size_t>(j) * n + i];
        };

        // Massstab fuer die Pivotschranke: der groesste Betrag der Matrix.
        double a_max = 0.0;
        for (size_t t = 0; t < static_cast<size_t>(n) * n; ++t)
            a_max = std::fmax(a_max, std::fabs(A[t]));
        const double pivot_floor = 1e-14 * (a_max > 0.0 ? a_max : 1.0);

        for (int k = 0; k < n; ++k) {
            // Pivotsuche in Spalte k
            int    piv = k;
            double best = std::fabs(at(k, k));
            for (int i = k + 1; i < n; ++i) {
                const double v = std::fabs(at(i, k));
                if (v > best) { best = v; piv = i; }
            }
            // "== 0.0" faengt zu wenig. Ein Pivot von 1e-300 kommt durch und
            // erzeugt 1e+300; ein NaN-Pivot ebenfalls, denn "v > best" ist fuer
            // NaN falsch -- ein NaN wird nie als besseres Pivot gewaehlt, steht
            // aber als Startwert best = |at(k,k)| schon drin und ist ungleich
            // null. Geprueft wird deshalb auf Endlichkeit und auf eine Schranke
            // relativ zum groessten Eintrag der Ausgangsmatrix.
            if (!std::isfinite(best) || best <= pivot_floor) {
                throw std::runtime_error(
                    "DenseLU::factorize (CPU): Matrix ist singulaer oder nicht endlich. "
                    "Pivot in Spalte " + std::to_string(k) + " ist " +
                    std::to_string(best) + " (Schranke " + std::to_string(pivot_floor) + ").");
            }
            m_host_ipiv[k] = piv + 1;   // 1-basiert wie LAPACK

            if (piv != k) {
                for (int j = 0; j < n; ++j) {
                    const double t = at(k, j); at(k, j) = at(piv, j); at(piv, j) = t;
                }
            }

            const double inv_pivot = 1.0 / at(k, k);
            for (int i = k + 1; i < n; ++i) at(i, k) *= inv_pivot;

            // Rang-1-Aktualisierung des restlichen Blocks
            BF_OMP_PARALLEL_FOR
            for (int j = k + 1; j < n; ++j) {
                const double akj = at(k, j);
                if (akj == 0.0) continue;
                for (int i = k + 1; i < n; ++i) at(i, j) -= at(i, k) * akj;
            }
        }
        return;
    }

    // ---------------------------------------------------------------------
    // GPU: cuSOLVER
    // ---------------------------------------------------------------------
    cusolverDnHandle_t h = nullptr;
    BF_CUSOLVER_CHECK(cusolverDnCreate(&h));
    m_solver_handle = static_cast<void*>(h);

    BF_CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(h, n, n, A, n, &m_lwork));
    BF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_work),
                             static_cast<size_t>(m_lwork) * sizeof(double)));
    BF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_ipiv),
                             static_cast<size_t>(n) * sizeof(int)));
    BF_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&m_info), sizeof(int)));

    BF_CUSOLVER_CHECK(cusolverDnDgetrf(h, n, n, A, n, m_work, m_ipiv, m_info));

    int info = 0;
    BF_CUDA_CHECK(cudaMemcpy(&info, m_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        throw std::runtime_error("DenseLU::factorize (GPU): cuSOLVER Dgetrf meldet info = " +
                                 std::to_string(info) +
                                 (info > 0 ? " (Nullpivot in dieser Spalte)"
                                           : " (ungueltiges Argument)"));
    }
}

// =============================================================================
// Loesen
// =============================================================================
void DenseLU::solve(double* B, int nrhs) const {
    if (m_n <= 0 || m_A == nullptr) {
        throw std::runtime_error("DenseLU::solve: factorize() wurde nicht aufgerufen.");
    }
    if (nrhs <= 0 || B == nullptr) return;

    const int n = m_n;

    if (on_cpu()) {
        const double* A = m_A;
        auto at = [A, n](int i, int j) -> double {
            return A[static_cast<size_t>(j) * n + i];
        };

        BF_OMP_PARALLEL_FOR
        for (int c = 0; c < nrhs; ++c) {
            double* b = B + static_cast<size_t>(c) * n;

            // Zeilentausch wie bei der Zerlegung
            for (int k = 0; k < n; ++k) {
                const int piv = m_host_ipiv[k] - 1;
                if (piv != k) { const double t = b[k]; b[k] = b[piv]; b[piv] = t; }
            }
            // Vorwaerts: L y = b, L hat Einheitsdiagonale
            for (int i = 1; i < n; ++i) {
                double s = b[i];
                for (int k = 0; k < i; ++k) s -= at(i, k) * b[k];
                b[i] = s;
            }
            // Rueckwaerts: U x = y
            for (int i = n - 1; i >= 0; --i) {
                double s = b[i];
                for (int k = i + 1; k < n; ++k) s -= at(i, k) * b[k];
                b[i] = s / at(i, i);
            }
        }
        return;
    }

    cusolverDnHandle_t h = static_cast<cusolverDnHandle_t>(m_solver_handle);
    BF_CUSOLVER_CHECK(cusolverDnDgetrs(h, CUBLAS_OP_N, n, nrhs,
                                       m_A, n, m_ipiv, B, n, m_info));
    int info = 0;
    BF_CUDA_CHECK(cudaMemcpy(&info, m_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        throw std::runtime_error("DenseLU::solve (GPU): cuSOLVER Dgetrs meldet info = " +
                                 std::to_string(info));
    }
}

} // namespace bf
