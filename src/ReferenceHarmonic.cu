// =============================================================================
// ReferenceHarmonic.cu -- Blockoperator und FGMRES des harmonischen
// Referenzpfads. Herleitung in include/ReferenceHarmonic.cuh.
// =============================================================================

#include "ReferenceHarmonic.cuh"

#include "HostLinalg.cuh"

#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <string>

void ReferenceHarmonicOperator::apply(const double* x, double* y) const {
    const double* xr = x;
    const double* xi = x + n;
    double* yr = y;
    double* yi = y + n;

    // Diagonalbloecke: (K+G) auf beide Haelften
    apply_kg(xr, yr);
    apply_kg(xi, yi);

    // Kopplung: y_r -= w C x_i,  y_i += w C x_r
    bf::csr_spmv(n, c_row_offsets, c_col_indices, c_values, xi, tmp.data());
    bf::daxpy(cublas, n, -omega, tmp.data(), yr);
    bf::csr_spmv(n, c_row_offsets, c_col_indices, c_values, xr, tmp.data());
    bf::daxpy(cublas, n,  omega, tmp.data(), yi);
}

ReferenceFgmresResult reference_harmonic_fgmres(
    const ReferenceApply& A, const ReferencePrecond& P,
    int N, const double* b, double* x,
    const ReferenceFgmresOptions& opt, cublasHandle_t cublas,
    std::ostream* log)
{
    ReferenceFgmresResult res;
    const int m = opt.restart;

    // Krylov-Basis V und vorkonditionierte Richtungen Z. Z ist der Unterschied
    // zu GMRES: die Loesung wird aus Z y aufgebaut, nicht aus P^-1 (V y) --
    // nur so darf sich P zwischen den Iterationen aendern.
    DeviceArray<double> r(static_cast<size_t>(N));
    DeviceArray<double> w(static_cast<size_t>(N));
    std::vector<DeviceArray<double>> V(static_cast<size_t>(m) + 1);
    std::vector<DeviceArray<double>> Z(static_cast<size_t>(m));
    for (int i = 0; i <= m; ++i) V[i].allocate(static_cast<size_t>(N));
    for (int i = 0; i <  m; ++i) Z[i].allocate(static_cast<size_t>(N));

    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
    std::vector<double> cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0), yv(m, 0.0);

    double bnorm = 0.0;
    bf::dnrm2(cublas, N, b, &bnorm);
    if (!std::isfinite(bnorm)) {
        throw std::runtime_error("Referenz-FGMRES: rechte Seite nicht endlich.");
    }
    if (bnorm == 0.0) {
        bf::memset_d(x, 0, static_cast<size_t>(N) * sizeof(double));
        res.converged = true; res.rel_residual = 0.0;
        return res;
    }

    auto true_residual = [&]() {
        A(x, w.data());
        bf::dcopy(cublas, N, b, r.data());
        bf::daxpy(cublas, N, -1.0, w.data(), r.data());
        double rn = 0.0;
        bf::dnrm2(cublas, N, r.data(), &rn);
        return rn;
    };

    double rn = true_residual();
    res.rel_residual = rn / bnorm;
    res.restart_residuals.push_back(res.rel_residual);
    if (log) {
        *log << "  [Ref-FGMRES] Start: echtes rel. Residuum " << std::scientific
             << std::setprecision(3) << res.rel_residual << std::defaultfloat << "\n";
    }

    while (res.iterations < opt.max_iter && res.rel_residual >= opt.tol) {
        if (!std::isfinite(rn)) {
            throw std::runtime_error("Referenz-FGMRES: Residuum nicht endlich.");
        }
        bf::dcopy(cublas, N, r.data(), V[0].data());
        bf::dscal(cublas, N, 1.0 / rn, V[0].data());
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = rn;

        int k_used = 0;
        for (int k = 0; k < m && res.iterations < opt.max_iter; ++k) {
            ++res.iterations;
            const int inner = P(V[k].data(), Z[k].data());
            A(Z[k].data(), w.data());

            for (int i = 0; i <= k; ++i) {
                bf::ddot(cublas, N, w.data(), V[i].data(), &H[i][k]);
                bf::daxpy(cublas, N, -H[i][k], V[i].data(), w.data());
            }
            bf::dnrm2(cublas, N, w.data(), &H[k + 1][k]);
            if (!std::isfinite(H[k + 1][k])) {
                throw std::runtime_error(
                    "Referenz-FGMRES: NaN/Inf in Iteration " +
                    std::to_string(res.iterations) + " (Operator oder "
                    "Vorkonditionierer liefert nicht endliche Werte).");
            }
            if (H[k + 1][k] > 1e-300) {
                bf::dcopy(cublas, N, w.data(), V[k + 1].data());
                bf::dscal(cublas, N, 1.0 / H[k + 1][k], V[k + 1].data());
            } else {
                V[k + 1].fill_zero();
            }

            for (int i = 0; i < k; ++i) {
                const double t = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
                H[i + 1][k]    = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
                H[i][k]        = t;
            }
            const double d = std::sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
            if (d > 0.0) { cs[k] = H[k][k] / d; sn[k] = H[k+1][k] / d; }
            else         { cs[k] = 1.0; sn[k] = 0.0; }
            H[k][k]     = d;
            H[k + 1][k] = 0.0;
            g[k + 1]    = -sn[k] * g[k];
            g[k]        =  cs[k] * g[k];

            k_used = k + 1;
            const double est = std::fabs(g[k + 1]) / bnorm;
            res.history.push_back(est);
            res.inner_iterations.push_back(inner);
            if (log) {
                *log << "  [Ref-FGMRES] it " << std::setw(3) << res.iterations
                     << "  rel. Residuum (Schaetzung) " << std::scientific
                     << std::setprecision(3) << est << std::defaultfloat
                     << "  innere Iterationen " << inner << "\n";
            }
            if (est < opt.tol) break;
        }

        for (int i = k_used - 1; i >= 0; --i) {
            double s = g[i];
            for (int j = i + 1; j < k_used; ++j) s -= H[i][j] * yv[j];
            yv[i] = (H[i][i] != 0.0) ? (s / H[i][i]) : 0.0;
        }
        for (int i = 0; i < k_used; ++i) {
            bf::daxpy(cublas, N, yv[i], Z[i].data(), x);
        }

        // Echtes Residuum, nicht die Schaetzung: bei ungenauem
        // Vorkonditionierer koennen beide auseinanderlaufen.
        rn = true_residual();
        res.rel_residual = rn / bnorm;
        res.restart_residuals.push_back(res.rel_residual);
        if (log) {
            *log << "  [Ref-FGMRES] nach " << res.iterations
                 << " Iterationen: echtes rel. Residuum " << std::scientific
                 << std::setprecision(3) << res.rel_residual << std::defaultfloat << "\n";
        }
        if (!std::isfinite(res.rel_residual)) {
            throw std::runtime_error("Referenz-FGMRES: echtes Residuum nicht endlich.");
        }
        if (k_used == 0) break;
    }

    res.converged = (res.rel_residual < opt.tol);
    return res;
}
