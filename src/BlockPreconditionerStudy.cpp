#include "BlockPreconditionerStudy.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <vector>   // std::string / std::to_string: MSVC zieht das
                    // NICHT ueber andere Header mit, GCC schon.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <utility>
#include <stdexcept>

namespace bf {

namespace {

// A ist column-major: A(i,j) = A[j*n + i].
inline double& at(std::vector<double>& A, int n, int i, int j) {
    return A[static_cast<size_t>(j) * n + i];
}
inline double at(const std::vector<double>& A, int n, int i, int j) {
    return A[static_cast<size_t>(j) * n + i];
}

// y = A x
void matvec(const std::vector<double>& A, int n, const double* x, double* y) {
    for (int i = 0; i < n; ++i) y[i] = 0.0;
    for (int j = 0; j < n; ++j) {
        const double xj = x[j];
        if (xj == 0.0) continue;
        const double* col = A.data() + static_cast<size_t>(j) * n;
        for (int i = 0; i < n; ++i) y[i] += col[i] * xj;
    }
}

// y = A^T x
void matvec_transposed(const std::vector<double>& A, int n, const double* x, double* y) {
    for (int j = 0; j < n; ++j) {
        const double* col = A.data() + static_cast<size_t>(j) * n;
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += col[i] * x[i];
        y[j] = acc;
    }
}

double norm2(const double* v, int n) {
    // Skalierte Summation, aber MIT Endlichkeitspruefung. Die naive Variante
    // "if (scale < a) scale = a" ist fuer NaN falsch und lieferte im Projekt
    // schon einmal die Norm 0.0 fuer einen Vektor aus lauter NaN
    // (docs/design/nan-and-thresholds.md). Hier wird deshalb gezaehlt statt
    // verglichen.
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(v[i])) return std::numeric_limits<double>::quiet_NaN();
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

// Reproduzierbarer Startvektor. Fest eingestellter Seed: eine Konditionszahl,
// die sich von Lauf zu Lauf aendert, ist als Befund wertlos.
void fill_random(double* v, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int i = 0; i < n; ++i) v[i] = uni(rng);
}

} // namespace

// =============================================================================
// HostDenseLU
// =============================================================================

void HostDenseLU::factorize(std::vector<double>& A, int n) {
    if (n <= 0) throw std::runtime_error("HostDenseLU::factorize: n <= 0.");
    if (A.size() < static_cast<size_t>(n) * n)
        throw std::runtime_error("HostDenseLU::factorize: Matrix zu klein fuer n.");

    m_n = n;
    m_lu.assign(A.begin(), A.begin() + static_cast<size_t>(n) * n);
    m_piv.assign(static_cast<size_t>(n), 0);

    // Massstab fuer die Singularitaetspruefung: groesster Betrag der Matrix.
    // Eine ABSOLUTE Schranke waere hier falsch -- die Eintraege des
    // gekoppelten Systems liegen je nach Block um Groessenordnungen
    // auseinander.
    double scale = 0.0;
    int    non_finite = 0;
    for (size_t i = 0; i < m_lu.size(); ++i) {
        const double a = std::fabs(m_lu[i]);
        if (!std::isfinite(a)) { ++non_finite; continue; }   // zaehlen, nicht vergleichen
        if (a > scale) scale = a;
    }
    if (non_finite > 0)
        throw std::runtime_error("HostDenseLU::factorize: " + std::to_string(non_finite)
                                 + " nicht endliche Eintraege in der Matrix.");
    if (scale == 0.0) throw std::runtime_error("HostDenseLU::factorize: Nullmatrix.");

    const double pivot_floor = scale * 1e-14;

    for (int k = 0; k < n; ++k) {
        int    best_row = k;
        double best     = std::fabs(at(m_lu, n, k, k));
        if (!std::isfinite(best)) best = -1.0;   // NaN darf nie "das beste" bleiben
        for (int i = k + 1; i < n; ++i) {
            const double v = std::fabs(at(m_lu, n, i, k));
            if (std::isfinite(v) && v > best) { best = v; best_row = i; }
        }
        if (!(best > pivot_floor)) {
            throw std::runtime_error(
                "HostDenseLU::factorize: singulaer in Spalte " + std::to_string(k)
                + " (groesstes Pivot " + std::to_string(best) + ", Schranke "
                + std::to_string(pivot_floor) + ").");
        }

        m_piv[static_cast<size_t>(k)] = best_row;
        if (best_row != k) {
            for (int j = 0; j < n; ++j) std::swap(at(m_lu, n, k, j), at(m_lu, n, best_row, j));
        }

        const double pivot = at(m_lu, n, k, k);
        for (int i = k + 1; i < n; ++i) {
            const double f = at(m_lu, n, i, k) / pivot;
            at(m_lu, n, i, k) = f;
            if (f == 0.0) continue;
            for (int j = k + 1; j < n; ++j) at(m_lu, n, i, j) -= f * at(m_lu, n, k, j);
        }
    }
}

void HostDenseLU::solve(double* b) const {
    const int n = m_n;
    for (int k = 0; k < n; ++k) {
        const int p = m_piv[static_cast<size_t>(k)];
        if (p != k) std::swap(b[k], b[p]);
    }
    // L y = Pb  (L hat Einheitsdiagonale)
    for (int i = 1; i < n; ++i) {
        double acc = b[i];
        for (int j = 0; j < i; ++j) acc -= at(m_lu, n, i, j) * b[j];
        b[i] = acc;
    }
    // U x = y
    for (int i = n - 1; i >= 0; --i) {
        double acc = b[i];
        for (int j = i + 1; j < n; ++j) acc -= at(m_lu, n, i, j) * b[j];
        b[i] = acc / at(m_lu, n, i, i);
    }
}

void HostDenseLU::solve_transposed(double* b) const {
    // A = P^T L U  =>  A^T = U^T L^T P.
    // Also: U^T z = b, dann L^T w = z, dann x = P^T w (Tausche rueckwaerts).
    const int n = m_n;
    for (int i = 0; i < n; ++i) {
        double acc = b[i];
        for (int j = 0; j < i; ++j) acc -= at(m_lu, n, j, i) * b[j];
        b[i] = acc / at(m_lu, n, i, i);
    }
    for (int i = n - 1; i >= 0; --i) {
        double acc = b[i];
        for (int j = i + 1; j < n; ++j) acc -= at(m_lu, n, j, i) * b[j];
        b[i] = acc;
    }
    for (int k = n - 1; k >= 0; --k) {
        const int p = m_piv[static_cast<size_t>(k)];
        if (p != k) std::swap(b[k], b[p]);
    }
}

void HostDenseLU::solve_many(double* B, int nrhs) const {
    for (int c = 0; c < nrhs; ++c) solve(B + static_cast<size_t>(c) * m_n);
}

// =============================================================================
// kappa_2
// =============================================================================

ConditionEstimate estimate_condition(const std::vector<double>& A, int n,
                                     int max_power_iters, double power_tol) {
    ConditionEstimate est;
    if (n <= 0) return est;

    std::vector<double> v(static_cast<size_t>(n)), w(static_cast<size_t>(n));

    // --- sigma_max: Potenzmethode auf A^T A -------------------------------
    fill_random(v.data(), n, 987654321u);
    {
        double nv = norm2(v.data(), n);
        for (int i = 0; i < n; ++i) v[i] /= nv;
        double lambda = 0.0, prev = -1.0;
        bool conv = false;
        for (int it = 0; it < max_power_iters; ++it) {
            matvec(A, n, v.data(), w.data());
            matvec_transposed(A, n, w.data(), v.data());
            lambda = norm2(v.data(), n);
            if (!(lambda > 0.0) || !std::isfinite(lambda)) break;
            for (int i = 0; i < n; ++i) v[i] /= lambda;
            if (prev > 0.0 && std::fabs(lambda - prev) <= power_tol * lambda) { conv = true; break; }
            prev = lambda;
        }
        est.sigma_max = (lambda > 0.0) ? std::sqrt(lambda) : 0.0;
        est.converged = conv;
    }

    // --- sigma_min = 1 / sigma_max(A^-1) ----------------------------------
    // Potenzmethode auf A^-T A^-1: v -> A^-1 v -> A^-T (.). Beide Richtungen
    // aus DERSELBEN Zerlegung -- deshalb solve_transposed(). --direct stellt
    // dafuer heute A^T explizit auf und zerlegt ein zweites Mal; das ist eine
    // vollstaendige n^2-Kopie und eine weitere n^3-Zerlegung fuer nichts.
    try {
        std::vector<double> work(A.begin(), A.begin() + static_cast<size_t>(n) * n);
        HostDenseLU lu;
        lu.factorize(work, n);

        fill_random(v.data(), n, 123456789u);
        double nv = norm2(v.data(), n);
        for (int i = 0; i < n; ++i) v[i] /= nv;

        double lambda = 0.0, prev = -1.0;
        bool conv = false;
        for (int it = 0; it < max_power_iters; ++it) {
            lu.solve(v.data());
            lu.solve_transposed(v.data());
            lambda = norm2(v.data(), n);
            if (!(lambda > 0.0) || !std::isfinite(lambda)) break;
            for (int i = 0; i < n; ++i) v[i] /= lambda;
            if (prev > 0.0 && std::fabs(lambda - prev) <= power_tol * lambda) { conv = true; break; }
            prev = lambda;
        }
        est.sigma_min = (lambda > 0.0) ? 1.0 / std::sqrt(lambda) : 0.0;
        est.converged = est.converged && conv;
    } catch (const std::exception&) {
        est.sigma_min = 0.0;
        est.converged = false;
    }

    est.kappa = (est.sigma_min > 0.0) ? est.sigma_max / est.sigma_min : -1.0;
    return est;
}

namespace {

// Eigenwerte einer kleinen SYMMETRISCHEN (k x k)-Matrix, zyklisches
// Jacobi-Verfahren. Fuer k <= ~50 voellig ausreichend und ohne LAPACK.
// Rueckgabe aufsteigend sortiert.
std::vector<double> jacobi_eigenvalues(std::vector<double> M, int k) {
    for (int sweep = 0; sweep < 100; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < k; ++p)
            for (int q = p + 1; q < k; ++q) off += M[static_cast<size_t>(q) * k + p]
                                                 * M[static_cast<size_t>(q) * k + p];
        if (std::sqrt(off) < 1e-14) break;

        for (int p = 0; p < k; ++p) {
            for (int q = p + 1; q < k; ++q) {
                const double apq = M[static_cast<size_t>(q) * k + p];
                if (std::fabs(apq) < 1e-300) continue;
                const double app = M[static_cast<size_t>(p) * k + p];
                const double aqq = M[static_cast<size_t>(q) * k + q];
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0)
                               / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0);
                const double s = t * c;
                for (int i = 0; i < k; ++i) {
                    const double aip = M[static_cast<size_t>(p) * k + i];
                    const double aiq = M[static_cast<size_t>(q) * k + i];
                    M[static_cast<size_t>(p) * k + i] = c * aip - s * aiq;
                    M[static_cast<size_t>(q) * k + i] = s * aip + c * aiq;
                }
                for (int j = 0; j < k; ++j) {
                    const double apj = M[static_cast<size_t>(j) * k + p];
                    const double aqj = M[static_cast<size_t>(j) * k + q];
                    M[static_cast<size_t>(j) * k + p] = c * apj - s * aqj;
                    M[static_cast<size_t>(j) * k + q] = s * apj + c * aqj;
                }
            }
        }
    }
    std::vector<double> ev(static_cast<size_t>(k));
    for (int i = 0; i < k; ++i) ev[static_cast<size_t>(i)] = M[static_cast<size_t>(i) * k + i];
    std::sort(ev.begin(), ev.end());
    return ev;
}

// Modifiziertes Gram-Schmidt auf den Spalten von V (n x k), in place.
void orthonormalize(std::vector<double>& V, int n, int k) {
    for (int j = 0; j < k; ++j) {
        double* vj = V.data() + static_cast<size_t>(j) * n;
        for (int i = 0; i < j; ++i) {
            const double* vi = V.data() + static_cast<size_t>(i) * n;
            double d = 0.0;
            for (int t = 0; t < n; ++t) d += vi[t] * vj[t];
            for (int t = 0; t < n; ++t) vj[t] -= d * vi[t];
        }
        const double nv = norm2(vj, n);
        if (nv > 0.0 && std::isfinite(nv)) {
            for (int t = 0; t < n; ++t) vj[t] /= nv;
        } else {
            // Entartete Richtung durch eine frische ersetzen, sonst bricht die
            // Ritz-Projektion still zusammen und liefert Nullen, die wie echte
            // Singulaerwerte aussehen.
            fill_random(vj, n, 4242u + static_cast<unsigned>(j));
            const double nn = norm2(vj, n);
            for (int t = 0; t < n; ++t) vj[t] /= nn;
        }
    }
}

} // namespace

std::vector<double> smallest_singular_values(const std::vector<double>& A, int n,
                                             int k, int max_iters, double tol) {
    if (n <= 0 || k <= 0) return {};
    k = std::min(k, n);

    HostDenseLU lu;
    try {
        std::vector<double> work(A.begin(), A.begin() + static_cast<size_t>(n) * n);
        lu.factorize(work, n);
    } catch (const std::exception&) {
        return {};
    }

    // Block-Inverse-Iteration auf (A^T A)^-1 = A^-1 A^-T.
    // Der kleinste Singulaerwert von A ist der groesste Eigenwert davon --
    // die Iteration konvergiert also von der richtigen Seite.
    std::vector<double> V(static_cast<size_t>(n) * k);
    for (int j = 0; j < k; ++j)
        fill_random(V.data() + static_cast<size_t>(j) * n, n, 20260819u + static_cast<unsigned>(j));
    orthonormalize(V, n, k);

    std::vector<double> prev;
    std::vector<double> ritz;
    for (int it = 0; it < max_iters; ++it) {
        for (int j = 0; j < k; ++j) {
            double* vj = V.data() + static_cast<size_t>(j) * n;
            lu.solve_transposed(vj);
            lu.solve(vj);
        }
        orthonormalize(V, n, k);

        // Rayleigh-Ritz auf A^T A: M = V^T (A^T A) V = (AV)^T (AV).
        std::vector<double> AV(static_cast<size_t>(n) * k);
        for (int j = 0; j < k; ++j)
            matvec(A, n, V.data() + static_cast<size_t>(j) * n,
                   AV.data() + static_cast<size_t>(j) * n);
        std::vector<double> M(static_cast<size_t>(k) * k, 0.0);
        for (int a = 0; a < k; ++a)
            for (int b = a; b < k; ++b) {
                double d = 0.0;
                const double* ca = AV.data() + static_cast<size_t>(a) * n;
                const double* cb = AV.data() + static_cast<size_t>(b) * n;
                for (int t = 0; t < n; ++t) d += ca[t] * cb[t];
                M[static_cast<size_t>(b) * k + a] = d;
                M[static_cast<size_t>(a) * k + b] = d;
            }

        ritz = jacobi_eigenvalues(M, k);
        for (double& e : ritz) e = (e > 0.0) ? std::sqrt(e) : 0.0;

        if (!prev.empty()) {
            double worst = 0.0;
            for (int i = 0; i < k; ++i) {
                const double s = std::max(ritz[static_cast<size_t>(i)], 1e-300);
                worst = std::max(worst,
                                 std::fabs(ritz[static_cast<size_t>(i)]
                                         - prev[static_cast<size_t>(i)]) / s);
            }
            if (worst < tol) break;
        }
        prev = ritz;
    }
    return ritz;
}

void symmetric_part_extremes(const std::vector<double>& A, int n,
                             double& lambda_min, double& lambda_max,
                             int max_power_iters, double power_tol) {
    lambda_min = lambda_max = 0.0;
    if (n <= 0) return;

    // Symmetrischen Anteil aufstellen: Sym = (A + A^T)/2.
    std::vector<double> Sym(static_cast<size_t>(n) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            Sym[static_cast<size_t>(j) * n + i] =
                0.5 * (at(A, n, i, j) + at(A, n, j, i));

    std::vector<double> v(static_cast<size_t>(n)), w(static_cast<size_t>(n));

    // lambda mit groesstem BETRAG (Potenzmethode, Rayleigh-Quotient).
    auto dominant = [&](const std::vector<double>& M) {
        fill_random(v.data(), n, 55555u);
        double nv = norm2(v.data(), n);
        for (int i = 0; i < n; ++i) v[i] /= nv;
        double lam = 0.0, prev = 0.0;
        for (int it = 0; it < max_power_iters; ++it) {
            matvec(M, n, v.data(), w.data());
            lam = 0.0;
            for (int i = 0; i < n; ++i) lam += v[i] * w[i];   // Rayleigh
            const double nw = norm2(w.data(), n);
            if (!(nw > 0.0) || !std::isfinite(nw)) break;
            for (int i = 0; i < n; ++i) v[i] = w[i] / nw;
            if (it > 0 && std::fabs(lam - prev) <= power_tol * std::fabs(lam)) break;
            prev = lam;
        }
        return lam;
    };

    const double lam1 = dominant(Sym);

    // Verschieben: Sym - lam1*I hat den am weitesten von lam1 entfernten
    // Eigenwert als dominanten. Damit liegt das ganze Spektrum eingeschlossen
    // zwischen beiden -- unabhaengig davon, ob lam1 das Maximum oder das
    // Minimum war.
    std::vector<double> Shift(Sym);
    for (int i = 0; i < n; ++i) Shift[static_cast<size_t>(i) * n + i] -= lam1;
    const double lam2 = dominant(Shift) + lam1;

    lambda_min = std::min(lam1, lam2);
    lambda_max = std::max(lam1, lam2);
}

// =============================================================================
// Dichtes GMRES(m), rechts vorkonditioniert
// =============================================================================

DenseGmresResult dense_gmres(const std::vector<double>& A, int n,
                             const std::vector<double>& b,
                             const std::function<void(double*)>& apply_M_inverse,
                             double tol, int max_iter, int restart) {
    DenseGmresResult res;
    const int m = (restart > 0) ? std::min(restart, max_iter) : max_iter;

    const double b_norm = norm2(b.data(), n);
    if (!(b_norm > 0.0)) { res.converged = true; res.relative_res = 0.0; return res; }

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    std::vector<double> r(static_cast<size_t>(n));
    std::vector<double> w(static_cast<size_t>(n));
    std::vector<double> V(static_cast<size_t>(n) * (m + 1));
    std::vector<double> Z(static_cast<size_t>(n) * m);
    std::vector<double> H(static_cast<size_t>(m + 1) * m, 0.0);
    std::vector<double> g(static_cast<size_t>(m + 1), 0.0);
    std::vector<double> cs(static_cast<size_t>(m), 0.0), sn(static_cast<size_t>(m), 0.0);
    std::vector<double> y(static_cast<size_t>(m), 0.0);

    int total = 0;
    double rel = 1.0;

    while (total < max_iter) {
        matvec(A, n, x.data(), r.data());
        for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];
        double beta = norm2(r.data(), n);
        if (!std::isfinite(beta)) break;
        rel = beta / b_norm;
        if (rel <= tol) break;

        for (int i = 0; i < n; ++i) V[i] = r[i] / beta;
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta;
        std::fill(H.begin(), H.end(), 0.0);

        int k = 0;
        for (; k < m && total < max_iter; ++k, ++total) {
            double* zk = Z.data() + static_cast<size_t>(k) * n;
            std::copy(V.begin() + static_cast<size_t>(k) * n,
                      V.begin() + static_cast<size_t>(k + 1) * n, zk);
            apply_M_inverse(zk);
            matvec(A, n, zk, w.data());

            // Modifiziertes Gram-Schmidt
            for (int i = 0; i <= k; ++i) {
                const double* vi = V.data() + static_cast<size_t>(i) * n;
                double h = 0.0;
                for (int t = 0; t < n; ++t) h += vi[t] * w[t];
                for (int t = 0; t < n; ++t) w[t] -= h * vi[t];
                H[static_cast<size_t>(i) * m + k] = h;
            }
            const double h_next = norm2(w.data(), n);
            H[static_cast<size_t>(k + 1) * m + k] = h_next;

            // Givens-Drehungen der Vorgeschichte anwenden
            for (int i = 0; i < k; ++i) {
                const double h1 = H[static_cast<size_t>(i) * m + k];
                const double h2 = H[static_cast<size_t>(i + 1) * m + k];
                H[static_cast<size_t>(i) * m + k]     =  cs[i] * h1 + sn[i] * h2;
                H[static_cast<size_t>(i + 1) * m + k] = -sn[i] * h1 + cs[i] * h2;
            }
            const double hk  = H[static_cast<size_t>(k) * m + k];
            const double hk1 = H[static_cast<size_t>(k + 1) * m + k];
            const double den = std::hypot(hk, hk1);
            if (!(den > 0.0)) { ++total; ++k; break; }
            cs[k] = hk / den;
            sn[k] = hk1 / den;
            H[static_cast<size_t>(k) * m + k]     = den;
            H[static_cast<size_t>(k + 1) * m + k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k]     =  cs[k] * g[k];

            rel = std::fabs(g[k + 1]) / b_norm;

            if (h_next > 1e-14 * std::max(1.0, den)) {
                double* vn = V.data() + static_cast<size_t>(k + 1) * n;
                for (int t = 0; t < n; ++t) vn[t] = w[t] / h_next;
            } else {
                ++k; ++total;
                break;   // happy breakdown
            }
            if (rel <= tol) { ++k; ++total; break; }
        }

        // Rueckwaertssubstitution mit RELATIVER Pivotschranke (siehe die
        // gleichlautende Stelle in CoupledSolver.cu: ohne sie wird y[i] bei
        // Stagnation unendlich und x komplett NaN, ohne jede Meldung).
        double h_scale = 0.0;
        for (int i = 0; i < k; ++i) h_scale = std::max(h_scale, std::fabs(H[static_cast<size_t>(i) * m + i]));
        for (int i = k - 1; i >= 0; --i) {
            double acc = g[i];
            for (int j = i + 1; j < k; ++j) acc -= H[static_cast<size_t>(i) * m + j] * y[j];
            const double d = H[static_cast<size_t>(i) * m + i];
            y[i] = (std::isfinite(d) && std::fabs(d) > 1e-14 * h_scale) ? acc / d : 0.0;
        }
        for (int j = 0; j < k; ++j) {
            const double* zj = Z.data() + static_cast<size_t>(j) * n;
            const double yj = y[j];
            if (yj == 0.0) continue;
            for (int t = 0; t < n; ++t) x[t] += yj * zj[t];
        }

        if (rel <= tol) break;
        if (k == 0) break;   // kein Fortschritt moeglich
    }

    // Das ECHTE Residuum, nicht das aus den Givens-Drehungen mitgefuehrte.
    matvec(A, n, x.data(), r.data());
    for (int i = 0; i < n; ++i) r[i] = b[i] - r[i];
    const double rn = norm2(r.data(), n);
    res.iterations   = total;
    res.relative_res = rn / b_norm;
    res.converged    = std::isfinite(res.relative_res) && res.relative_res <= tol * 10.0;
    return res;
}

// =============================================================================
// Die Studie
// =============================================================================

namespace {

std::vector<double> extract_block(const std::vector<double>& A, int n,
                                  int r0, int nr, int c0, int nc) {
    std::vector<double> B(static_cast<size_t>(nr) * nc, 0.0);
    for (int j = 0; j < nc; ++j)
        for (int i = 0; i < nr; ++i)
            B[static_cast<size_t>(j) * nr + i] = at(A, n, r0 + i, c0 + j);
    return B;
}

} // namespace

BlockPcStudyResult study_block_preconditioners(const std::vector<double>& A, int n, int n_A,
                                               const std::vector<double>& b,
                                               const BlockPcStudyOptions& opt,
                                               std::ostream& out) {
    BlockPcStudyResult res;
    const int n_K = n - n_A;
    if (n_A <= 0 || n_K <= 0)
        throw std::runtime_error("study_block_preconditioners: leerer FEM- oder BEM-Block "
                                 "(n=" + std::to_string(n) + ", n_A=" + std::to_string(n_A) + ").");
    if (A.size() < static_cast<size_t>(n) * n || b.size() < static_cast<size_t>(n))
        throw std::runtime_error("study_block_preconditioners: A oder b zu klein.");

    out << "\n[PC-Studie] Gekoppeltes System " << n << " x " << n
        << "  (FEM " << n_A << ", BEM " << n_K << ")\n";

    // --- 1. Konditionszahlen der Bloecke -----------------------------------
    //
    // Das ist bereits die halbe Antwort. Die blockweise Residuums-
    // aufschluesselung sagt, WO das Residuum sitzt; diese beiden Zahlen sagen,
    // WELCHER BLOCK SCHLECHT KONDITIONIERT IST. Nur die zweite Frage
    // entscheidet ueber den Vorkonditionierer.
    std::vector<double> A11 = extract_block(A, n, 0,   n_A, 0,   n_A);
    std::vector<double> A12 = extract_block(A, n, 0,   n_A, n_A, n_K);
    std::vector<double> A21 = extract_block(A, n, n_A, n_K, 0,   n_A);
    std::vector<double> A22 = extract_block(A, n, n_A, n_K, n_A, n_K);

    out << "[PC-Studie] kappa_2 der Diagonalbloecke ...\n";
    res.whole   = estimate_condition(A,   n,   opt.power_iters, opt.power_tol);
    res.block_A = estimate_condition(A11, n_A, opt.power_iters, opt.power_tol);
    res.block_K = estimate_condition(A22, n_K, opt.power_iters, opt.power_tol);

    out << "[PC-Studie]   ganzes System  sigma_max = " << res.whole.sigma_max
        << ", sigma_min = " << res.whole.sigma_min
        << ", kappa_2 = " << res.whole.kappa << "\n";
    out << "[PC-Studie]   FEM-Block A11  sigma_max = " << res.block_A.sigma_max
        << ", sigma_min = " << res.block_A.sigma_min
        << ", kappa_2 = " << res.block_A.kappa << "\n";
    out << "[PC-Studie]   BEM-Block A22  sigma_max = " << res.block_K.sigma_max
        << ", sigma_min = " << res.block_K.sigma_min
        << ", kappa_2 = " << res.block_K.kappa << "\n";

    // -----------------------------------------------------------------------
    // Das untere Spektralende des FEM-Blocks
    // -----------------------------------------------------------------------
    // sigma_min(A11) ist praktisch sigma_min des Gesamtsystems, und A11^-1
    // traegt diesen Fastnullraum in das Schur-Komplement hinein. Damit ist es
    // die Wurzel der ganzen Kette -- aber sigma_min allein sagt nicht, ob ein
    // EINZELNER Restmodus dahintersteckt (Eichung unvollstaendig, behebbar)
    // oder ein BUENDEL (echter Fastnullraum, dann ist es keine Eichfrage).
    // Die beiden Faelle verlangen entgegengesetzte Massnahmen.
    {
        const int k = std::min(12, n_A);
        const std::vector<double> sv = smallest_singular_values(A11, n_A, k, 300, 1e-9);
        if (!sv.empty()) {
            res.smallest_sv_A11 = sv;
            out << "[PC-Studie]   " << sv.size() << " kleinste Singulaerwerte von A11:\n"
                << "[PC-Studie]    ";
            for (double s : sv) out << " " << s;
            out << "\n";
            if (sv.size() >= 2 && sv[0] > 0.0) {
                const double gap = sv[1] / sv[0];
                out << "[PC-Studie]     Abstand sigma_2/sigma_1 = " << gap;
                if (gap > 5.0) {
                    out << "  -> EINZELNER abgesetzter Modus\n";
                } else {
                    out << "  -> BUENDEL, kein abgesetzter Einzelmodus\n";
                }
            }
        }
    }

    // --- 2. Zerlegungen der Bloecke ----------------------------------------
    HostDenseLU lu11, lu22, luS, luSdiag;
    bool have11 = false, have22 = false, haveS = false;
    const std::vector<double>& A11_ref = A11;
    try { std::vector<double> t(A11); lu11.factorize(t, n_A); have11 = true; }
    catch (const std::exception& ex) { out << "[PC-Studie]   A11 nicht zerlegbar: " << ex.what() << "\n"; }
    try { std::vector<double> t(A22); lu22.factorize(t, n_K); have22 = true; }
    catch (const std::exception& ex) { out << "[PC-Studie]   A22 nicht zerlegbar: " << ex.what() << "\n"; }

    // Exaktes Schur-Komplement S = A22 - A21 A11^-1 A12.
    // Der Bericht hat S~ = mu_0*G_BEM = A22 widerlegt. Diese Variante trennt
    // die beiden moeglichen Gruende: war die SCHUR-IDEE falsch, oder nur die
    // NAEHERUNG? Ohne diese Zeile bleibt das offen.
    std::vector<double> S;
    std::vector<double> W;        // A11^-1 A12, fuer die Genauigkeitsstudie unten
    std::vector<double> S_diag;   // billige Naeherung, siehe unten
    bool haveSdiag = false;
    if (have11) {
        W.assign(A12.begin(), A12.end());     // n_A x n_K
        lu11.solve_many(W.data(), n_K);       // W = A11^-1 A12
        S = A22;
        double coupling_fro = 0.0, S_fro = 0.0;
        for (int j = 0; j < n_K; ++j)
            for (int i = 0; i < n_K; ++i) {
                double acc = 0.0;
                for (int t = 0; t < n_A; ++t)
                    acc += A21[static_cast<size_t>(t) * n_K + i] * W[static_cast<size_t>(j) * n_A + t];
                S[static_cast<size_t>(j) * n_K + i] -= acc;
                coupling_fro += acc * acc;
            }
        for (double s : S) S_fro += s * s;
        coupling_fro = std::sqrt(coupling_fro);
        S_fro        = std::sqrt(S_fro);

        // -----------------------------------------------------------------
        // Wie falsch ist S~ = mu_0*G_BEM = A22 als Naeherung an S?
        // -----------------------------------------------------------------
        // Der Bericht hat S~ = A22 widerlegt, ohne die Groesse des
        // weggelassenen Terms zu kennen. Genau die steht hier: ist
        // ||A21 A11^-1 A12||_F von derselben Groessenordnung wie ||S||_F oder
        // groesser, dann ist A22 nicht eine grobe Naeherung an S, sondern eine
        // andere Matrix -- und dass ein inneres CG darauf divergiert, ist kein
        // Raetsel mehr.
        res.schur.available    = true;
        res.schur.rel_coupling = (S_fro > 0.0) ? coupling_fro / S_fro : -1.0;

        double asym = 0.0;
        for (int j = 0; j < n_K; ++j)
            for (int i = 0; i < n_K; ++i) {
                const double d = S[static_cast<size_t>(j) * n_K + i]
                               - S[static_cast<size_t>(i) * n_K + j];
                asym += d * d;
            }
        res.schur.rel_asymmetry = (S_fro > 0.0) ? std::sqrt(asym) / S_fro : -1.0;

        const ConditionEstimate cS = estimate_condition(S, n_K, opt.power_iters, opt.power_tol);
        res.schur.kappa = cS.kappa;

        symmetric_part_extremes(S, n_K, res.schur.sym_lambda_min, res.schur.sym_lambda_max,
                                opt.power_iters, opt.power_tol);
        res.schur.positive_definite = (res.schur.sym_lambda_min > 0.0);

        out << "[PC-Studie] Schur-Komplement S = A22 - A21 A11^-1 A12:\n"
            << "[PC-Studie]   kappa_2(S) = " << res.schur.kappa << "\n"
            << "[PC-Studie]   ||A21 A11^-1 A12||_F / ||S||_F = " << res.schur.rel_coupling
            << "   (das ist der Term, den S~ = mu_0*G_BEM weglaesst)\n"
            << "[PC-Studie]   ||S - S^T||_F / ||S||_F = " << res.schur.rel_asymmetry << "\n"
            << "[PC-Studie]   Eigenwerte von (S+S^T)/2: [" << res.schur.sym_lambda_min
            << ", " << res.schur.sym_lambda_max << "]"
            << (res.schur.positive_definite ? "  -> positiv definit"
                                            : "  -> NICHT positiv definit")
            << "\n";

        try { std::vector<double> t(S); luS.factorize(t, n_K); haveS = true; }
        catch (const std::exception& ex) { out << "[PC-Studie]   S nicht zerlegbar: " << ex.what() << "\n"; }

        // -----------------------------------------------------------------
        // Die billige Schur-Naeherung: A11^-1 durch diag(A11)^-1 ersetzt.
        // -----------------------------------------------------------------
        // S selbst ist im Ernstfall nicht bezahlbar -- es aufzustellen kostet
        // n_K Loesungen mit A11. Die Frage ist deshalb nicht, ob S traegt
        // (Variante 4 beantwortet das), sondern ob eine BILLIG auswertbare
        // Naeherung an S noch traegt. diag(A11)^-1 ist die billigste
        // ueberhaupt und damit die richtige Untergrenze: traegt schon sie,
        // ist der Weg frei; traegt sie nicht, sagt der Abstand zu Variante 4,
        // wieviel Genauigkeit in A11^-1 noetig ist.
        S_diag = A22;
        for (int j = 0; j < n_K; ++j)
            for (int i = 0; i < n_K; ++i) {
                double acc = 0.0;
                for (int t = 0; t < n_A; ++t) {
                    const double d = at(A11_ref, n_A, t, t);
                    if (!(std::fabs(d) > 0.0)) continue;
                    acc += A21[static_cast<size_t>(t) * n_K + i]
                         * (A12[static_cast<size_t>(j) * n_A + t] / d);
                }
                S_diag[static_cast<size_t>(j) * n_K + i] -= acc;
            }
        try { std::vector<double> t(S_diag); luSdiag.factorize(t, n_K); haveSdiag = true; }
        catch (const std::exception& ex) {
            out << "[PC-Studie]   S_diag nicht zerlegbar: " << ex.what() << "\n";
        }
    }

    // --- 3. Die Varianten ---------------------------------------------------
    struct Def {
        std::string name;
        bool available;
        std::function<void(double*)> apply;
        // kappa_2(M^-1 A) kostet je Variante eine weitere dichte Zerlegung PLUS
        // eine n^3-Anwendung von M^-1 auf A. Fuer die Epsilon-Reihe ist das
        // verschwendet: dort interessiert allein, wie die Iterationszahl mit
        // der Genauigkeit von A11^-1 waechst.
        bool measure_kappa = true;
    };
    std::vector<Def> defs;

    defs.push_back({"0  M = I                          (Bezug)", true,
                    [](double*) {}});

    defs.push_back({"1  M = blockdiag(A11, I)          exakter FEM-Block", have11,
                    [&](double* v) { if (have11) lu11.solve(v); }});

    defs.push_back({"2  M = blockdiag(I, A22)          EXAKTER BEM-BLOCK", have22,
                    [&](double* v) { if (have22) lu22.solve(v + n_A); }});

    defs.push_back({"3  M = blockdiag(A11, A22)        beide exakt", have11 && have22,
                    [&](double* v) { if (have11) lu11.solve(v); if (have22) lu22.solve(v + n_A); }});

    // Blocktriangulaer mit EXAKTEM Schur-Komplement:
    //   z_K = S^-1 r_K ;  z_A = A11^-1 (r_A - A12 z_K)
    defs.push_back({"4  M = [[A11,A12],[0,S]]          exaktes Schur-Komplement", have11 && haveS,
                    [&](double* v) {
                        if (!have11 || !haveS) return;
                        luS.solve(v + n_A);
                        std::vector<double> corr(static_cast<size_t>(n_A), 0.0);
                        for (int j = 0; j < n_K; ++j) {
                            const double zj = v[n_A + j];
                            if (zj == 0.0) continue;
                            const double* col = A12.data() + static_cast<size_t>(j) * n_A;
                            for (int i = 0; i < n_A; ++i) corr[i] += col[i] * zj;
                        }
                        for (int i = 0; i < n_A; ++i) v[i] -= corr[i];
                        lu11.solve(v);
                    }});

    // Blocktriangulaer mit S~ = A22 = mu_0*G_BEM -- also GENAU der Ansatz aus
    // convergence-findings.tex, aber mit EXAKTER innerer Loesung statt des inneren CG.
    // Trennt sauber, ob dort die Naeherung S~ scheiterte oder das innere CG.
    defs.push_back({"5  M = [[A11,A12],[0,A22]]        S~ = mu_0*G_BEM, exakt geloest", have11 && have22,
                    [&](double* v) {
                        if (!have11 || !have22) return;
                        lu22.solve(v + n_A);
                        std::vector<double> corr(static_cast<size_t>(n_A), 0.0);
                        for (int j = 0; j < n_K; ++j) {
                            const double zj = v[n_A + j];
                            if (zj == 0.0) continue;
                            const double* col = A12.data() + static_cast<size_t>(j) * n_A;
                            for (int i = 0; i < n_A; ++i) corr[i] += col[i] * zj;
                        }
                        for (int i = 0; i < n_A; ++i) v[i] -= corr[i];
                        lu11.solve(v);
                    }});

    // Billige Schur-Naeherung: A11^-1 -> diag(A11)^-1.
    defs.push_back({"6  M = [[A11,A12],[0,S_diag]]     billige Schur-Naeherung", have11 && haveSdiag,
                    [&](double* v) {
                        if (!have11 || !haveSdiag) return;
                        luSdiag.solve(v + n_A);
                        std::vector<double> corr(static_cast<size_t>(n_A), 0.0);
                        for (int j = 0; j < n_K; ++j) {
                            const double zj = v[n_A + j];
                            if (zj == 0.0) continue;
                            const double* col = A12.data() + static_cast<size_t>(j) * n_A;
                            for (int i = 0; i < n_A; ++i) corr[i] += col[i] * zj;
                        }
                        for (int i = 0; i < n_A; ++i) v[i] -= corr[i];
                        lu11.solve(v);
                    }});

    // -----------------------------------------------------------------------
    // Wie genau muss A11^-1 INNERHALB der Schur-Anwendung sein?
    // -----------------------------------------------------------------------
    // Das ist die letzte offene Frage vor dem Produktivbau. Variante 4 nutzt
    // A11^-1 EXAKT und loest in 2 Iterationen; Variante 6 nutzt diag(A11)^-1
    // und bleibt weit von der Konvergenz. Zwischen beiden liegen viele
    // Groessenordnungen, und im Ernstfall steht dort ILU(0) oder AMS -- also
    // etwas dazwischen.
    //
    // Gemessen wird deshalb S_eps = A22 - A21 (A11^-1 A12 + eps * Rauschen),
    // mit relativem Rauschen fester Staerke eps. Das ist ein PESSIMISTISCHES
    // Modell fuer einen ungenauen inneren Loeser: echtes ILU-Restfehler ist
    // strukturiert und nicht zufaellig, zufaelliges Rauschen trifft alle
    // Richtungen gleichmaessig und damit auch die empfindlichsten. Eine
    // Genauigkeit, die hier reicht, reicht im Ernstfall erst recht.
    //
    // Ergebnis der Reihe ist eine Kurve "Genauigkeit von A11^-1 gegen
    // Iterationszahl". Sie sagt, ob ILU(0) genuegt, ob es ein paar innere
    // Krylov-Schritte braucht, oder ob der blocktriangulare Weg zwar
    // theoretisch traegt, aber keine bezahlbare Naeherung zulaesst -- der
    // letzte Fall waere das entscheidende Nein VOR dem CUDA-Code.
    std::vector<HostDenseLU> lu_eps;
    std::vector<bool>        have_eps;
    const std::vector<double> eps_list = {1e-1, 1e-2, 1e-3, 1e-4, 1e-6};
    if (have11 && !W.empty()) {
        lu_eps.resize(eps_list.size());
        have_eps.assign(eps_list.size(), false);

        double W_rms = 0.0;
        for (double w : W) W_rms += w * w;
        W_rms = std::sqrt(W_rms / static_cast<double>(W.size()));

        for (size_t e = 0; e < eps_list.size(); ++e) {
            const double eps = eps_list[e];
            // Fester Seed je eps: eine Kurve, die sich von Lauf zu Lauf
            // aendert, ist als Befund wertlos.
            std::mt19937 rng(1000u + static_cast<unsigned>(e));
            std::normal_distribution<double> nd(0.0, 1.0);

            std::vector<double> Weps(W);
            for (double& w : Weps) w += eps * W_rms * nd(rng);

            std::vector<double> Se = A22;
            for (int j = 0; j < n_K; ++j)
                for (int i = 0; i < n_K; ++i) {
                    double acc = 0.0;
                    for (int t = 0; t < n_A; ++t)
                        acc += A21[static_cast<size_t>(t) * n_K + i]
                             * Weps[static_cast<size_t>(j) * n_A + t];
                    Se[static_cast<size_t>(j) * n_K + i] -= acc;
                }
            try {
                std::vector<double> tmp(Se);
                lu_eps[e].factorize(tmp, n_K);
                have_eps[e] = true;
            } catch (const std::exception&) { have_eps[e] = false; }
        }

        for (size_t e = 0; e < eps_list.size(); ++e) {
            std::ostringstream nm;
            nm << "7." << (e + 1) << "  blocktri, A11^-1 mit rel. Fehler "
               << eps_list[e];
            const size_t idx = e;
            defs.push_back({nm.str(), have_eps[e],
                            [&, idx](double* v) {
                                if (!have11 || !have_eps[idx]) return;
                                lu_eps[idx].solve(v + n_A);
                                std::vector<double> corr(static_cast<size_t>(n_A), 0.0);
                                for (int j = 0; j < n_K; ++j) {
                                    const double zj = v[n_A + j];
                                    if (zj == 0.0) continue;
                                    const double* col = A12.data() + static_cast<size_t>(j) * n_A;
                                    for (int i = 0; i < n_A; ++i) corr[i] += col[i] * zj;
                                }
                                for (int i = 0; i < n_A; ++i) v[i] -= corr[i];
                                lu11.solve(v);
                            },
                            /*measure_kappa=*/false});
        }
    }

    // -----------------------------------------------------------------------
    // Die ANDERE Eliminationsrichtung: S_A = A11 - A12 A22^-1 A21
    // -----------------------------------------------------------------------
    // Bisher wurde ausschliesslich ueber A11 eliminiert. Das ist die
    // ungeschickte Richtung: A11 ist der SCHLECHT konditionierte Block, sein
    // Fastnullraum wird durch A11^-1 in das Schur-Komplement getragen, und
    // genau daher kommt kappa_2(S) = 2e+08 und die Indefinitheit.
    //
    // A22 dagegen ist gut konditioniert und SPD (Single-Layer-Operator).
    // A22^-1 ist damit gutartig, und S_A lebt im FEM-Raum -- also dort, wo
    // AMS zu Hause ist. Das ist die klassische DtN-Formulierung: der
    // FEM-Operator mit exakter Randbedingung. Der Zusatzterm A12 A22^-1 A21
    // ist ein Dirichlet-zu-Neumann-Beitrag auf dem Rand und wirkt dort
    // typischerweise positiv -- er koennte S_A also BESSER konditionieren als
    // A11 selbst, statt schlechter.
    //
    // Ob das so ist, entscheidet nicht die Plausibilitaet, sondern die
    // Messung. Sie kostet hier fast nichts, weil A22 ohnehin schon zerlegt
    // ist.
    std::vector<double> S_A;
    std::vector<double> Y;        // A22^-1 A21, n_K x n_A
    HostDenseLU luSA;
    bool haveSA = false;
    if (have22) {
        Y.assign(A21.begin(), A21.end());     // n_K x n_A
        lu22.solve_many(Y.data(), n_A);       // Y = A22^-1 A21

        S_A = A11;
        double coupling_fro = 0.0, SA_fro = 0.0;
        for (int j = 0; j < n_A; ++j)
            for (int i = 0; i < n_A; ++i) {
                double acc = 0.0;
                for (int t = 0; t < n_K; ++t)
                    acc += A12[static_cast<size_t>(t) * n_A + i] * Y[static_cast<size_t>(j) * n_K + t];
                S_A[static_cast<size_t>(j) * n_A + i] -= acc;
                coupling_fro += acc * acc;
            }
        for (double v : S_A) SA_fro += v * v;
        coupling_fro = std::sqrt(coupling_fro);
        SA_fro       = std::sqrt(SA_fro);

        res.schur_A.available    = true;
        res.schur_A.rel_coupling = (SA_fro > 0.0) ? coupling_fro / SA_fro : -1.0;

        double asym = 0.0;
        for (int j = 0; j < n_A; ++j)
            for (int i = 0; i < n_A; ++i) {
                const double d = S_A[static_cast<size_t>(j) * n_A + i]
                               - S_A[static_cast<size_t>(i) * n_A + j];
                asym += d * d;
            }
        res.schur_A.rel_asymmetry = (SA_fro > 0.0) ? std::sqrt(asym) / SA_fro : -1.0;

        const ConditionEstimate cSA = estimate_condition(S_A, n_A, opt.power_iters, opt.power_tol);
        res.schur_A.kappa = cSA.kappa;
        symmetric_part_extremes(S_A, n_A, res.schur_A.sym_lambda_min, res.schur_A.sym_lambda_max,
                                opt.power_iters, opt.power_tol);
        res.schur_A.positive_definite = (res.schur_A.sym_lambda_min > 0.0);

        out << "[PC-Studie] Schur-Komplement S_A = A11 - A12 A22^-1 A21 (andere Richtung):\n"
            << "[PC-Studie]   kappa_2(S_A) = " << res.schur_A.kappa
            << "   (zum Vergleich: kappa_2(A11) = " << res.block_A.kappa
            << ", kappa_2(S) = " << res.schur.kappa << ")\n"
            << "[PC-Studie]   ||A12 A22^-1 A21||_F / ||S_A||_F = " << res.schur_A.rel_coupling << "\n"
            << "[PC-Studie]   ||S_A - S_A^T||_F / ||S_A||_F = " << res.schur_A.rel_asymmetry << "\n"
            << "[PC-Studie]   Eigenwerte von (S_A+S_A^T)/2: [" << res.schur_A.sym_lambda_min
            << ", " << res.schur_A.sym_lambda_max << "]"
            << (res.schur_A.positive_definite ? "  -> positiv definit"
                                              : "  -> NICHT positiv definit") << "\n";

        try { std::vector<double> t(S_A); luSA.factorize(t, n_A); haveSA = true; }
        catch (const std::exception& ex) { out << "[PC-Studie]   S_A nicht zerlegbar: " << ex.what() << "\n"; }
    }

    // Blockuntere Dreiecksform mit S_A:
    //   z_A = S_A^-1 r_A ;  z_K = A22^-1 (r_K - A21 z_A)
    defs.push_back({"8  M = [[S_A,0],[A21,A22]]        Elimination ueber A22 (DtN)", haveSA && have22,
                    [&](double* v) {
                        if (!haveSA || !have22) return;
                        luSA.solve(v);
                        std::vector<double> corr(static_cast<size_t>(n_K), 0.0);
                        for (int j = 0; j < n_A; ++j) {
                            const double zj = v[j];
                            if (zj == 0.0) continue;
                            const double* col = A21.data() + static_cast<size_t>(j) * n_K;
                            for (int i = 0; i < n_K; ++i) corr[i] += col[i] * zj;
                        }
                        for (int i = 0; i < n_K; ++i) v[n_A + i] -= corr[i];
                        lu22.solve(v + n_A);
                    }});

    // Genauigkeitsreihe fuer die ANDERE Richtung: wie genau muss A22^-1 sein?
    // A22 ist SPD und gut konditioniert; ein inneres CG darauf ist billig.
    // Wenn diese Reihe deutlich toleranter ausfaellt als 7.x, ist die Richtung
    // ueber A22 der praktikable Weg -- und dann ist auch der Aufwand fuer
    // einen A22-Vorkonditionierer wieder gerechtfertigt, diesmal aber an der
    // Stelle, an der er wirklich sitzt.
    std::vector<HostDenseLU> luSA_eps;
    std::vector<bool>        haveSA_eps;
    // Reihe 11 (beide Anteile gestoert) entsteht in derselben Schleife: SAe
    // ist dort ohnehin schon aufgebaut, und ein zweiter Aufbau kostete
    // n_A^2 * n_K Flops fuer nichts.
    std::vector<HostDenseLU> luSA_both(eps_list.size());
    std::vector<bool>        haveSA_both(eps_list.size(), false);
    if (have22 && !Y.empty()) {
        luSA_eps.resize(eps_list.size());
        haveSA_eps.assign(eps_list.size(), false);

        double Y_rms = 0.0;
        for (double y : Y) Y_rms += y * y;
        Y_rms = std::sqrt(Y_rms / static_cast<double>(Y.size()));

        for (size_t e = 0; e < eps_list.size(); ++e) {
            std::mt19937 rng(2000u + static_cast<unsigned>(e));
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> Yeps(Y);
            for (double& y : Yeps) y += eps_list[e] * Y_rms * nd(rng);

            std::vector<double> SAe = A11;
            for (int j = 0; j < n_A; ++j)
                for (int i = 0; i < n_A; ++i) {
                    double acc = 0.0;
                    for (int t = 0; t < n_K; ++t)
                        acc += A12[static_cast<size_t>(t) * n_A + i]
                             * Yeps[static_cast<size_t>(j) * n_K + t];
                    SAe[static_cast<size_t>(j) * n_A + i] -= acc;
                }
            try {
                std::vector<double> tmp(SAe);
                luSA_eps[e].factorize(tmp, n_A);
                haveSA_eps[e] = true;
            } catch (const std::exception&) { haveSA_eps[e] = false; }

            // ... und dieselbe Matrix noch einmal, zusaetzlich als Operator
            // gestoert: das ist Reihe 11.
            double SAe_rms = 0.0;
            for (double v : SAe) SAe_rms += v * v;
            SAe_rms = std::sqrt(SAe_rms / static_cast<double>(SAe.size()));
            std::mt19937 rng2(4000u + static_cast<unsigned>(e));
            std::normal_distribution<double> nd2(0.0, 1.0);
            for (double& v : SAe) v += eps_list[e] * SAe_rms * nd2(rng2);
            try { luSA_both[e].factorize(SAe, n_A); haveSA_both[e] = true; }
            catch (const std::exception&) { haveSA_both[e] = false; }
        }

        for (size_t e = 0; e < eps_list.size(); ++e) {
            std::ostringstream nm;
            nm << "9." << (e + 1) << "  DtN-Zweig, A22^-1 mit rel. Fehler " << eps_list[e];
            const size_t idx = e;
            defs.push_back({nm.str(), haveSA_eps[e] && have22,
                            [&, idx](double* v) {
                                if (!haveSA_eps[idx] || !have22) return;
                                luSA_eps[idx].solve(v);
                                std::vector<double> corr(static_cast<size_t>(n_K), 0.0);
                                for (int j = 0; j < n_A; ++j) {
                                    const double zj = v[j];
                                    if (zj == 0.0) continue;
                                    const double* col = A21.data() + static_cast<size_t>(j) * n_K;
                                    for (int i = 0; i < n_K; ++i) corr[i] += col[i] * zj;
                                }
                                for (int i = 0; i < n_K; ++i) v[n_A + i] -= corr[i];
                                lu22.solve(v + n_A);
                            },
                            /*measure_kappa=*/false});
        }
    }

    // -----------------------------------------------------------------------
    // Wie genau muss der S_A-LOESER sein? (Reihe 10, und kombiniert Reihe 11)
    // -----------------------------------------------------------------------
    //
    // Die Reihe 9.x hat eine Luecke: sie stoert nur A22^-1 und nimmt an, dass
    // S_A EXAKT geloest wird. Im Produktivcode steht dort AMS -- und
    // kappa_2(S_A) ist gross und S_A indefinit. Ohne diese Reihe waere die
    // Empfehlung "nimm den DtN-Zweig" auf einer Annahme gebaut, nicht auf
    // einer Messung.
    //
    // Gestoert wird die MATRIX S_A, nicht die Loesung je Aufruf. Das ist kein
    // Detail: wuerde man dem Ergebnisvektor bei jeder Anwendung frisches
    // Rauschen aufaddieren, waere M kein fester linearer Operator mehr, die
    // Arnoldi-Rekursion verloere ihre Grundlage und GMRES maesse Unsinn. Ein
    // ungenauer, aber deterministischer Loeser entspricht einer gestoerten
    // Matrix -- und genau die wird hier gebaut.
    //
    // Reihe 10: nur S_A gestoert, A22^-1 exakt   -> isoliert den S_A-Loeser.
    // Reihe 11: BEIDE auf derselben Stufe gestoert -> die realistische
    //           Produktivlage, in der weder AMS noch das innere CG exakt sind.
    //           Sie ist die eigentliche Entscheidungsgrundlage; Reihe 10 sagt,
    //           welcher der beiden Anteile den Ausschlag gibt.
    std::vector<HostDenseLU> luSA_only(eps_list.size());
    std::vector<bool>        haveSA_only(eps_list.size(), false);
    if (haveSA) {
        double SA_rms = 0.0;
        for (double v : S_A) SA_rms += v * v;
        SA_rms = std::sqrt(SA_rms / static_cast<double>(S_A.size()));

        for (size_t e = 0; e < eps_list.size(); ++e) {
            std::mt19937 rng(3000u + static_cast<unsigned>(e));
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> M(S_A);
            for (double& v : M) v += eps_list[e] * SA_rms * nd(rng);
            try { luSA_only[e].factorize(M, n_A); haveSA_only[e] = true; }
            catch (const std::exception&) { haveSA_only[e] = false; }
        }

        for (size_t e = 0; e < eps_list.size(); ++e) {
            std::ostringstream nm;
            nm << "10." << (e + 1) << " DtN, S_A-Loeser mit rel. Fehler " << eps_list[e];
            const size_t idx = e;
            defs.push_back({nm.str(), haveSA_only[e] && have22,
                            [&, idx](double* v) {
                                if (!haveSA_only[idx] || !have22) return;
                                luSA_only[idx].solve(v);
                                std::vector<double> corr(static_cast<size_t>(n_K), 0.0);
                                for (int j = 0; j < n_A; ++j) {
                                    const double zj = v[j];
                                    if (zj == 0.0) continue;
                                    const double* col = A21.data() + static_cast<size_t>(j) * n_K;
                                    for (int i = 0; i < n_K; ++i) corr[i] += col[i] * zj;
                                }
                                for (int i = 0; i < n_K; ++i) v[n_A + i] -= corr[i];
                                lu22.solve(v + n_A);
                            },
                            /*measure_kappa=*/false});
        }
    }

    for (size_t e = 0; e < eps_list.size(); ++e) {
        std::ostringstream nm;
        nm << "11." << (e + 1) << " DtN, S_A UND A22^-1 mit rel. Fehler " << eps_list[e];
        const size_t idx = e;
        defs.push_back({nm.str(), (idx < haveSA_both.size()) && haveSA_both[idx] && have22,
                        [&, idx](double* v) {
                            if (idx >= haveSA_both.size() || !haveSA_both[idx] || !have22) return;
                            luSA_both[idx].solve(v);
                            std::vector<double> corr(static_cast<size_t>(n_K), 0.0);
                            for (int j = 0; j < n_A; ++j) {
                                const double zj = v[j];
                                if (zj == 0.0) continue;
                                const double* col = A21.data() + static_cast<size_t>(j) * n_K;
                                for (int i = 0; i < n_K; ++i) corr[i] += col[i] * zj;
                            }
                            for (int i = 0; i < n_K; ++i) v[n_A + i] -= corr[i];
                            lu22.solve(v + n_A);
                        },
                        /*measure_kappa=*/false});
    }

    out << "[PC-Studie] GMRES je Variante (Toleranz " << opt.gmres_tol
        << ", hoechstens " << opt.gmres_max_iter << " Iterationen, Restart "
        << (opt.gmres_restart > 0 ? std::to_string(opt.gmres_restart) : std::string("aus")) << ")\n";

    for (const auto& d : defs) {
        BlockPcVariant v;
        v.name = d.name;
        if (!d.available) {
            out << "[PC-Studie]   " << d.name << ": nicht verfuegbar (Zerlegung gescheitert)\n";
            res.variants.push_back(v);
            continue;
        }

        const DenseGmresResult g = dense_gmres(A, n, b, d.apply,
                                               opt.gmres_tol, opt.gmres_max_iter,
                                               opt.gmres_restart);
        v.gmres_iters     = g.iterations;
        v.gmres_relres    = g.relative_res;
        v.gmres_converged = g.converged;

        if (opt.measure_kappa && d.measure_kappa) {
            // M^-1 A explizit aufstellen: spaltenweise M^-1 auf A anwenden.
            std::vector<double> MA(A.begin(), A.begin() + static_cast<size_t>(n) * n);
            for (int j = 0; j < n; ++j) d.apply(MA.data() + static_cast<size_t>(j) * n);
            const ConditionEstimate c = estimate_condition(MA, n, opt.power_iters, opt.power_tol);
            v.kappa = c.kappa; v.sigma_max = c.sigma_max; v.sigma_min = c.sigma_min;
        }

        out << "[PC-Studie]   " << std::left << std::setw(48) << d.name << std::right
            << "  Iter = " << std::setw(6) << v.gmres_iters
            << "  rel.Res = " << std::setw(11) << v.gmres_relres;
        if (opt.measure_kappa && d.measure_kappa) out << "  kappa_2 = " << v.kappa;
        if (!v.gmres_converged) out << "   (NICHT konvergiert)";
        out << "\n";

        res.variants.push_back(v);
    }

    // --- 3b. Kosten der inneren Loesungen -----------------------------------
    // Die Toleranz wird aus der Produktivlage (Reihe 11) abgeleitet, nicht
    // geraten: gemessen wird, was der Zweig tatsaechlich verlangt.
    {
        double need = 0.0;
        int    outer = -1;
        for (const auto& v : res.variants) {
            if (v.name.rfind("11.", 0) != 0 || !v.gmres_converged) continue;
            const size_t p = v.name.rfind(' ');
            if (p != std::string::npos) need = std::atof(v.name.c_str() + p + 1);
            outer = v.gmres_iters;
            break;
        }
        if (need > 0.0 && outer > 0 && !S_A.empty()) {
            out << "[PC-Studie] Kosten der inneren Loesungen bei Toleranz " << need << " ...\n";
            res.inner_cost = estimate_inner_cost(S_A, n_A, A22, n_K, need, outer,
                                                 std::min(4000, 2 * std::max(n_A, n_K)));
        }
    }

    // --- 4. Die Kennzahl, um die es geht ------------------------------------
    if (res.variants.size() > 2 &&
        res.variants[0].gmres_converged && res.variants[2].gmres_converged &&
        res.variants[2].gmres_iters > 0) {
        res.bem_block_speedup = static_cast<double>(res.variants[0].gmres_iters)
                              / static_cast<double>(res.variants[2].gmres_iters);
    }

    return res;
}

InnerCostEstimate estimate_inner_cost(const std::vector<double>& S_A, int n_A,
                                      const std::vector<double>& A22, int n_K,
                                      double tol, int outer_iters, int max_iter) {
    InnerCostEstimate est;
    est.tol         = tol;
    est.outer_iters = outer_iters;

    auto run = [&](const std::vector<double>& M, int n) {
        std::vector<double> b(static_cast<size_t>(n));
        fill_random(b.data(), n, 606060u);
        // Unvorkonditioniert und ohne Neustart. Das ist bewusst die
        // PESSIMISTISCHE Schaetzung fuer S_A und die OPTIMISTISCHE fuer die
        // Frage, ob sich ein Vorkonditionierer lohnt: was hier steht, ist die
        // Zahl, die ein Vorkonditionierer zu druecken haette.
        const DenseGmresResult r = dense_gmres(M, n, b, [](double*) {},
                                               tol, max_iter, 0);
        return r.converged ? r.iterations : -1;
    };

    if (!S_A.empty() && n_A > 0) est.iters_S_A = run(S_A, n_A);
    if (!A22.empty() && n_K > 0) est.iters_A22 = run(A22, n_K);

    // Eine Anwendung von S_A kostet: ein A11-Matvec plus eine A22-LOESUNG,
    // denn S_A v = A11 v - A12 (A22^-1 (A21 v)). Die A22-Loesung ist der
    // teure Anteil, sie kostet iters_A22 BEM-Matvecs.
    //
    // Je aeusserer Iteration also: iters_S_A * iters_A22 BEM-Matvecs.
    if (est.iters_S_A > 0 && est.iters_A22 > 0 && outer_iters > 0) {
        est.bem_matvecs = static_cast<double>(outer_iters)
                        * static_cast<double>(est.iters_S_A)
                        * static_cast<double>(est.iters_A22);
        // Der heutige Loeser: ein BEM-Matvec je aeusserer Iteration.
        est.baseline = 4000.0;
    }
    return est;
}


// =============================================================================
// Harmonischer Ansatz
// =============================================================================

KCSplit split_K_and_C(const std::vector<double>& A1, double dt1,
                      const std::vector<double>& A2, double dt2,
                      const std::vector<double>& A3, double dt3,
                      int n, int n_A) {
    KCSplit r;
    const size_t nn = static_cast<size_t>(n) * n;
    if (n <= 0 || A1.size() < nn || A2.size() < nn || A3.size() < nn) return r;

    const double inv1 = 1.0 / dt1, inv2 = 1.0 / dt2, inv3 = 1.0 / dt3;
    const double den  = inv1 - inv2;
    // Relativ pruefen: zwei dt, die sich kaum unterscheiden, machen die
    // Differenz zu Rundungsrauschen und C zu Unsinn -- ohne dass irgendetwas
    // schiefzugehen SCHEINT.
    if (!(std::fabs(den) > 1e-6 * std::max(std::fabs(inv1), std::fabs(inv2)))) return r;
    if (!(std::fabs(inv3 - inv1) > 1e-6 * std::fabs(inv1) &&
          std::fabs(inv3 - inv2) > 1e-6 * std::max(std::fabs(inv2), 1e-300))) return r;

    r.C.resize(nn);
    r.K.resize(nn);

    // K aus derjenigen Abtastung ziehen, die den KLEINEREN Anteil C/dt traegt,
    // also aus dem GROESSEREN dt. K = A - C/dt ist eine Differenz zweier
    // grosser Zahlen; der Stellenverlust ist ~ log10(||C/dt|| / ||K||), und
    // der ist beim groesseren dt um genau das Verhaeltnis dt_gross/dt_klein
    // kleiner. Die erste Fassung nahm stur A1 -- bei dt1 = 1e-4 gegen dt2 = 1
    // sind das vier verlorene Stellen fuer nichts.
    const bool use2 = (std::fabs(inv2) < std::fabs(inv1));
    const std::vector<double>& Abase = use2 ? A2 : A1;
    const double invbase = use2 ? inv2 : inv1;

    for (size_t i = 0; i < nn; ++i) {
        r.C[i] = (A1[i] - A2[i]) / den;
        r.K[i] = Abase[i] - r.C[i] * invbase;
    }

    // --- Kontrolle: A3 VORHERSAGEN ----------------------------------------
    // A3 geht nicht in die Aufspaltung ein. Erst dadurch ist die Pruefung
    // ueberhaupt eine: mit zwei Matrizen und zwei Unbekannten je Eintrag ist
    // der Rueckbau von A1 und A2 immer exakt und sagt nichts aus.
    {
        double num = 0.0, dsq = 0.0;
        for (size_t i = 0; i < nn; ++i) {
            const double d = (r.K[i] + r.C[i] * inv3) - A3[i];
            num  += d * d;
            dsq  += A3[i] * A3[i];
        }
        r.rel_error_prediction = (dsq > 0.0) ? std::sqrt(num / dsq) : -1.0;
    }

    // --- Kontrolle: C darf nur im FEM-Block leben -------------------------
    if (n_A > 0 && n_A <= n) {
        double outside = 0.0, inside = 0.0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const double v = r.C[static_cast<size_t>(j) * n + i];
                if (i < n_A && j < n_A) inside += v * v; else outside += v * v;
            }
        r.c_outside_fem_block = (inside > 0.0) ? std::sqrt(outside / inside) : -1.0;
    }

    // --- Ausloeschung: wieviele Stellen kostet die Wahl von dt1/dt2? ------
    {
        double nK = 0.0, nCdt = 0.0;
        for (size_t i = 0; i < nn; ++i) {
            nK   += r.K[i] * r.K[i];
            const double c = r.C[i] * invbase;
            nCdt += c * c;
        }
        r.split_conditioning = (nK > 0.0) ? std::sqrt(nCdt / nK) : -1.0;
    }

    r.valid = (r.rel_error_prediction >= 0.0 && r.rel_error_prediction < 1e-8);
    return r;
}

std::vector<double> build_harmonic_system(const std::vector<double>& K,
                                          const std::vector<double>& C,
                                          double omega, int n) {
    const int N = 2 * n;
    std::vector<double> H(static_cast<size_t>(N) * N, 0.0);
    auto set = [&](int i, int j, double v) { H[static_cast<size_t>(j) * N + i] = v; };
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            const double k = K[static_cast<size_t>(j) * n + i];
            const double c = omega * C[static_cast<size_t>(j) * n + i];
            set(i,     j,     k);    // W
            set(i,     j + n, -c);   // -T
            set(i + n, j,      c);   //  T
            set(i + n, j + n,  k);   // W
        }
    return H;
}

HarmonicStudyResult study_harmonic_system(const std::vector<double>& A_transient,
                                          const KCSplit& split,
                                          double omega, double dt_ref,
                                          int n, int n_A,
                                          const BlockPcStudyOptions& opt,
                                          std::ostream& out) {
    HarmonicStudyResult r;
    r.omega = omega;
    if (!split.valid) {
        out << "[Harmonisch] Aufspaltung ungueltig (Vorhersagefehler "
            << split.rel_error_prediction << ") -- keine Studie.\n";
        return r;
    }

    const int N = 2 * n;
    out << "[Harmonisch] Baue reelles Blocksystem " << N << " x " << N
        << " bei omega = " << omega << " (f = " << (omega / (2.0 * 3.14159265358979)) << " Hz)\n";
    const std::vector<double> H = build_harmonic_system(split.K, split.C, omega, n);

    // W + T = K + omega*C. Das ist ein TRANSIENTER Operator mit dt = 1/omega --
    // dieselbe Bauart, auf die AMS und ILU im Projekt schon zielen.
    std::vector<double> WpT(static_cast<size_t>(n) * n);
    for (size_t i = 0; i < WpT.size(); ++i) WpT[i] = split.K[i] + omega * split.C[i];

    HostDenseLU luWpT;
    bool haveWpT = false;
    try { std::vector<double> t(WpT); luWpT.factorize(t, n); haveWpT = true; }
    catch (const std::exception& ex) {
        out << "[Harmonisch] (W+T) nicht zerlegbar: " << ex.what() << "\n";
    }

    std::vector<double> b(static_cast<size_t>(N));
    fill_random(b.data(), N, 8080u);

    out << "[Harmonisch] kappa_2 ...\n";
    r.cond = estimate_condition(H, N, opt.power_iters, opt.power_tol);

    const DenseGmresResult g0 = dense_gmres(H, N, b, [](double*) {},
                                            opt.gmres_tol, opt.gmres_max_iter, opt.gmres_restart);
    r.iters_plain  = g0.iterations;
    r.relres_plain = g0.relative_res;

    if (haveWpT) {
        // C-to-R (Axelsson/Kucherov): P = [[W+T, 0], [T-W, W+T]]
        //   z_r = (W+T)^-1 b_r
        //   z_i = (W+T)^-1 (b_i - (T-W) z_r)
        auto apply_ctor = [&](double* v) {
            luWpT.solve(v);
            std::vector<double> tmp(static_cast<size_t>(n), 0.0);
            for (int j = 0; j < n; ++j) {
                const double zj = v[j];
                if (zj == 0.0) continue;
                for (int i = 0; i < n; ++i) {
                    const double TmW = omega * split.C[static_cast<size_t>(j) * n + i]
                                     -         split.K[static_cast<size_t>(j) * n + i];
                    tmp[i] += TmW * zj;
                }
            }
            for (int i = 0; i < n; ++i) v[n + i] -= tmp[i];
            luWpT.solve(v + n);
        };
        const DenseGmresResult g1 = dense_gmres(H, N, b, apply_ctor,
                                                opt.gmres_tol, opt.gmres_max_iter, opt.gmres_restart);
        r.iters_ctor  = g1.iterations;
        r.relres_ctor = g1.relative_res;
    }

    // -----------------------------------------------------------------------
    // Wie genau muss (W+T)^-1 sein?
    // -----------------------------------------------------------------------
    // Die Zahl oben gilt mit EXAKT geloestem (W+T). Im Produktivcode steht
    // dort AMS oder ILU. Ohne diese Reihe waere "C-to-R traegt" eine Aussage
    // ueber eine dichte LU, die es im Ernstfall nicht gibt -- genau der
    // Fehler, den die Reihen 7 bis 11 beim DtN-Zweig aufgedeckt haben.
    //
    // Gestoert wird die MATRIX, nicht die Loesung je Aufruf: frisches Rauschen
    // bei jeder Anwendung machte M nichtlinear, die Arnoldi-Rekursion verloere
    // ihre Grundlage, und GMRES lieferte plausibel aussehenden Unsinn.
    if (haveWpT) {
        double WpT_rms = 0.0;
        for (double v : WpT) WpT_rms += v * v;
        WpT_rms = std::sqrt(WpT_rms / static_cast<double>(WpT.size()));

        const std::vector<double> eps_list = {1e-1, 1e-2, 1e-3, 1e-4};
        // Kuerzere Obergrenze als oben: eine Stufe, die nicht traegt, soll
        // nicht minutenlang gegen die Wand laufen. 500 Iterationen sind
        // gegenueber den 38 des exakten Falls reichlich Spielraum.
        const int max_it = std::min(500, opt.gmres_max_iter);

        for (double eps : eps_list) {
            std::mt19937 rng(7070u + static_cast<unsigned>(-std::log10(eps) * 10.0));
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> M(WpT);
            for (double& v : M) v += eps * WpT_rms * nd(rng);

            HostDenseLU lue;
            bool ok = false;
            try { std::vector<double> t(M); lue.factorize(t, n); ok = true; }
            catch (const std::exception&) { ok = false; }

            int iters = -1;
            if (ok) {
                auto apply_e = [&](double* v) {
                    lue.solve(v);
                    std::vector<double> tmp(static_cast<size_t>(n), 0.0);
                    for (int j = 0; j < n; ++j) {
                        const double zj = v[j];
                        if (zj == 0.0) continue;
                        for (int i = 0; i < n; ++i) {
                            const double TmW = omega * split.C[static_cast<size_t>(j) * n + i]
                                             -         split.K[static_cast<size_t>(j) * n + i];
                            tmp[i] += TmW * zj;
                        }
                    }
                    for (int i = 0; i < n; ++i) v[n + i] -= tmp[i];
                    lue.solve(v + n);
                };
                const DenseGmresResult ge = dense_gmres(H, N, b, apply_e,
                                                        opt.gmres_tol, max_it, opt.gmres_restart);
                iters = ge.converged ? ge.iterations : -1;
            }
            r.wt_eps.push_back(eps);
            r.wt_iters.push_back(iters);
        }

        // Was kostet (W+T) selbst? Dieselbe Waehrung wie beim DtN-Zweig,
        // damit die beiden Wege vergleichbar bleiben.
        const ConditionEstimate cw = estimate_condition(WpT, n, opt.power_iters, opt.power_tol);
        r.kappa_WpT = cw.kappa;

        // -------------------------------------------------------------
        // Die Toleranz, die die ARBEIT minimiert -- nicht die groebste.
        // -------------------------------------------------------------
        // Die erste Fassung nahm die groebste noch tragende Stufe. Das ist
        // falsch: eine groebere Toleranz spart innere Schritte, kostet aber
        // aeussere, und das Produkt kann bei der FEINEREN Stufe kleiner sein.
        // Am gemessenen Fall: 1e-3 gibt 81 x 2899 = 2,3e+05 je Hoehe,
        // 1e-4 gibt 41 x ~3900 = 1,6e+05 -- ein Drittel billiger, und die
        // erste Fassung haette die teurere gewaehlt.
        double best_work = -1.0;
        for (size_t i = 0; i < r.wt_eps.size(); ++i) {
            if (r.wt_iters[i] <= 0) continue;
            std::vector<double> bw(static_cast<size_t>(n));
            fill_random(bw.data(), n, 909090u);
            const DenseGmresResult gw = dense_gmres(WpT, n, bw, [](double*) {},
                                                    r.wt_eps[i], std::min(4000, 2 * n), 0);
            if (!gw.converged) continue;
            // ~10 Hoehen fuer die Nullstellensuche F = mg,
            // je aeusserer Iteration ZWEI (W+T)-Loesungen.
            const double work = 10.0 * r.wt_iters[i] * 2.0 * gw.iterations;
            out << "[Harmonisch]     eps = " << r.wt_eps[i] << ":  "
                << r.wt_iters[i] << " aeussere x " << gw.iterations
                << " innere  ->  " << work << " Matvecs\n";
            if (best_work < 0.0 || work < best_work) {
                best_work         = work;
                r.inner_iters_WpT = gw.iterations;
                r.best_eps        = r.wt_eps[i];
                r.best_outer      = r.wt_iters[i];
            }
        }
        if (best_work > 0.0) {
            r.work_harmonic  = best_work;
            r.work_transient = 1000.0 * 4000.0;
            // Je Hoehe ist (W+T) FEST und wird 2 x aeussere Iterationen lang
            // benutzt. Das ist der Unterschied zur Transiente, wo sich die
            // Matrix in jedem der 1000 Schritte aendert (die Platte bewegt
            // sich) und deshalb NICHTS amortisiert.
            r.reuse_per_setup = 2 * r.best_outer;
        }
    }

    // Bezugsgroesse: dasselbe Mass am transienten System.    // Bezugsgroesse: dasselbe Mass am transienten System.
    {
        const ConditionEstimate ct = estimate_condition(A_transient, n, opt.power_iters, opt.power_tol);
        r.kappa_transient = ct.kappa;
        std::vector<double> bt(static_cast<size_t>(n));
        fill_random(bt.data(), n, 8080u);
        const DenseGmresResult gt = dense_gmres(A_transient, n, bt, [](double*) {},
                                                opt.gmres_tol, opt.gmres_max_iter, opt.gmres_restart);
        r.iters_transient = gt.iterations;
    }

    // -------------------------------------------------------------------
    // Wie stark koppelt omega*C ueberhaupt?
    // -------------------------------------------------------------------
    //
    // Der harmonische Lauf liefert ||A_i||/||A_r|| = 0,0285. Bei 50 Hz hat
    // Aluminium 12,2 mm Skintiefe, vergleichbar mit der Plattendicke; der
    // dimensionslose Parameter w*kappa*mu0*L^2 liegt bei L = 10 mm um 1,3.
    // Der Imaginaerteil sollte also in der Groessenordnung des Realteils
    // liegen, nicht bei drei Prozent.
    //
    // Verglichen wird ZEILENWEISE und nur dort, wo C ueberhaupt traegt. Eine
    // globale Norm waere irrefuehrend: C ist nur im Leiter von null
    // verschieden (156 von 348 Hexaedern), K dagegen ueberall --
    // ||wC||/||K|| ueber alles waere klein, ohne dass daraus etwas folgt.
    {
        int    n_active = 0;
        double sum_ratio = 0.0, max_ratio = 0.0, min_ratio = 1e300;
        for (int i = 0; i < n_A; ++i) {
            double rk = 0.0, rc = 0.0;
            for (int j = 0; j < n_A; ++j) {
                const double k = split.K[static_cast<size_t>(j) * n + i];
                const double c = omega * split.C[static_cast<size_t>(j) * n + i];
                rk += k * k;  rc += c * c;
            }
            rk = std::sqrt(rk); rc = std::sqrt(rc);
            if (rk > 0.0 && rc > 1e-12 * rk) {
                const double q = rc / rk;
                ++n_active; sum_ratio += q;
                if (q > max_ratio) max_ratio = q;
                if (q < min_ratio) min_ratio = q;
            }
        }
        out << "[Harmonisch] Kopplungsstaerke omega*C gegen K, zeilenweise:\n";
        if (n_active > 0) {
            const double mean = sum_ratio / n_active;
            out << "[Harmonisch]   " << n_active << " von " << n_A
                << " FEM-Zeilen tragen C (leitende Kanten)\n"
                << "[Harmonisch]   ||wC_i||/||K_i||:  Mittel " << mean
                << ", min " << min_ratio << ", max " << max_ratio << "\n"
                << "[Harmonisch]\n"
                   "[Harmonisch]   ZWEI KORREKTUREN sind beim Vergleichen mitzurechnen:\n"
                   "[Harmonisch]\n"
                   "[Harmonisch]   1. Dieser Wert ist D-SKALIERT, und D stammt aus dt = 1e-4\n"
                   "[Harmonisch]      statt 1/omega. Der Massenterm ist in der Aequilibrierung\n"
                   "[Harmonisch]      damit 31,8-mal staerker gewichtet als im Operator, D auf\n"
                   "[Harmonisch]      den leitenden Kanten also sqrt(31,8) = 5,6-mal zu klein.\n"
                   "[Harmonisch]      Der gemessene Wert ist entsprechend gedrueckt.\n"
                   "[Harmonisch]\n"
                   "[Harmonisch]   2. Der Vergleichswert ist NICHT w*kappa*mu0*h^2. Diese\n"
                   "[Harmonisch]      Handabschaetzung unterstellt |curlW| ~ |W|/h und ist fuer\n"
                   "[Harmonisch]      Nedelec auf einem Quader um Faktor 7 zu hoch -- die drei\n"
                   "[Harmonisch]      Richtungen mischen sich, und die Zeilennormen haengen\n"
                   "[Harmonisch]      anders von den Kantenlaengen ab. Eine EXAKTE Rechnung mit\n"
                   "[Harmonisch]      denselben Basisfunktionen und derselben Quadratur ergibt\n"
                   "[Harmonisch]        8x8x10 mm -> 0,125    4x4x5 mm -> 0,031\n"
                   "[Harmonisch]       15x15x10 mm -> 0,279\n";
            const double corrected = mean * 5.64;
            out << "[Harmonisch]   korrigiert (x5,6): " << corrected << "\n";
            if (corrected < 0.02)
                out << "[Harmonisch]   -> ZU SCHWACH, auch nach beiden Korrekturen. Dann ist die\n"
                       "[Harmonisch]      Gewinnung von C der Verdaechtige.\n";
            else
                out << "[Harmonisch]   -> PLAUSIBEL: das liegt im Bereich der exakten\n"
                       "[Harmonisch]      Elementrechnung. Ein kleiner Imaginaerteil ist dann\n"
                       "[Harmonisch]      KEIN Fehler, sondern die Physik dieses Netzes -- und\n"
                       "[Harmonisch]      eine Kraft, die linear in A_i ist, ist entsprechend\n"
                       "[Harmonisch]      klein.\n";
        } else {
            out << "[Harmonisch]   KEINE Zeile traegt C -- die Massenmatrix ist leer.\n";
        }
    }


    out << "[Harmonisch] ---------------------------------------------------------\n";
    out << "[Harmonisch]   transient (dt = " << dt_ref << "):  kappa_2 = " << r.kappa_transient
        << ", GMRES " << r.iters_transient << " Iter\n";
    out << "[Harmonisch]   harmonisch:              kappa_2 = " << r.cond.kappa
        << ", GMRES " << r.iters_plain << " Iter (rel.Res " << r.relres_plain << ")\n";
    if (r.iters_ctor > 0)
        out << "[Harmonisch]   harmonisch + C-to-R:     GMRES " << r.iters_ctor
            << " Iter (rel.Res " << r.relres_ctor << ")\n";
    if (!r.wt_eps.empty()) {
        out << "[Harmonisch]\n";
        out << "[Harmonisch]   Genauigkeit des inneren (W+T)-Loesers -- im Produktivcode\n"
               "[Harmonisch]   steht dort AMS/ILU, nicht die dichte LU von oben:\n";
        for (size_t i = 0; i < r.wt_eps.size(); ++i) {
            out << "[Harmonisch]     rel. Fehler " << r.wt_eps[i] << ":  ";
            if (r.wt_iters[i] > 0) out << r.wt_iters[i] << " aeussere Iterationen\n";
            else                   out << "traegt nicht\n";
        }
        out << "[Harmonisch]   kappa_2(W+T) = " << r.kappa_WpT
            << "   (transienter Operator mit dt = 1/omega)\n";
        if (r.inner_iters_WpT > 0)
            out << "[Harmonisch]   unvorkonditionierte Krylov-Schritte fuer (W+T): "
                << r.inner_iters_WpT << "\n";
    }

    out << "[Harmonisch]\n";
    out << "[Harmonisch]   Entscheidend ist NICHT kappa allein, sondern die Arbeit fuer\n"
           "[Harmonisch]   das GANZE Kraft-ueber-Hoehe-Problem:\n";
    if (r.iters_ctor > 0 && r.iters_transient > 0) {
        const double harm = 10.0 * r.iters_ctor * 2.0;   // ~10 Hoehen, C-to-R = 2 Loesungen je Iter
        const double tran = 1000.0 * r.iters_transient;
        out << "[Harmonisch]     harmonisch: ~10 Hoehen x " << r.iters_ctor
            << " Iter x 2 (W+T)-Loesungen = " << harm << "\n";
        out << "[Harmonisch]     transient:  1000 Schritte x " << r.iters_transient
            << " Iter = " << tran << "\n";
        out << "[Harmonisch]     Verhaeltnis: " << (tran / std::max(1.0, harm)) << "\n";
    }
    if (r.work_harmonic > 0.0) {
        out << "[Harmonisch]\n";
        out << "[Harmonisch]   Und dasselbe MIT unvorkonditioniertem innerem Loeser, also\n"
               "[Harmonisch]   ohne jede Annahme ueber AMS:\n";
        out << "[Harmonisch]     harmonisch: " << r.work_harmonic << " Matvecs\n";
        out << "[Harmonisch]     transient:  " << r.work_transient << " Matvecs\n";
        out << "[Harmonisch]     Verhaeltnis: "
            << (r.work_transient / std::max(1.0, r.work_harmonic)) << "\n";
        out << "[Harmonisch]   (bei eps = " << r.best_eps << ", also der Stufe mit der\n"
               "[Harmonisch]   KLEINSTEN Gesamtarbeit -- nicht der groebsten, die noch traegt)\n";
        out << "[Harmonisch]   Das ist eine OBERE Schranke: der innere Loeser laeuft hier ohne\n"
               "[Harmonisch]   jeden Vorkonditionierer.\n";
        out << "[Harmonisch]\n";
        out << "[Harmonisch]   ENTSCHEIDEND IST DIE WIEDERVERWENDUNG\n";
        out << "[Harmonisch]     Je Hoehe ist (W+T) FEST und wird " << r.reuse_per_setup
            << " mal benutzt.\n";
        out << "[Harmonisch]     In der Transiente aendert sich die Matrix in jedem der 1000\n"
               "[Harmonisch]     Schritte, weil die Platte sich bewegt -- dort amortisiert\n"
               "[Harmonisch]     NICHTS. Hier amortisiert eine teure Zerlegung oder ein teurer\n"
               "[Harmonisch]     Vorkonditionierer-Aufbau " << r.reuse_per_setup << "-fach.\n";
        // Break-even richtig herum: ist die harmonische Arbeit bereits
        // KLEINER, braucht es gar keinen inneren Vorkonditionierer. Die erste
        // Fassung meldete hier stumpf das Verhaeltnis und schrieb "braucht nur
        // Faktor 0.597" -- ein Faktor unter 1 waere aber eine
        // VERSCHLECHTERUNG. Die Zahl war richtig, der Satz darueber falsch.
        if (r.work_harmonic < r.work_transient) {
            out << "[Harmonisch]     Break-even ist bereits erreicht: harmonisch braucht Faktor "
                << (r.work_transient / std::max(1.0, r.work_harmonic))
                << " WENIGER Arbeit,\n"
                   "[Harmonisch]     und das OHNE jeden inneren Vorkonditionierer. Jeder Faktor,\n"
                   "[Harmonisch]     den ILU oder AMS auf (W+T) bringt, kommt oben drauf.\n";
        } else {
            out << "[Harmonisch]     Break-even gegen den heutigen Lauf verlangt am inneren\n"
                   "[Harmonisch]     Loeser den Faktor "
                << (r.work_harmonic / std::max(1.0, r.work_transient)) << ".\n";
        }
        out << "[Harmonisch]     Dabei ist der transiente Vergleichswert grosszuegig gerechnet:\n"
               "[Harmonisch]     seine 4000 Iterationen sind VORKONDITIONIERT und erreichen die\n"
               "[Harmonisch]     Toleranz trotzdem nicht, waehrend die harmonische Rechnung\n"
               "[Harmonisch]     unvorkonditioniert konvergiert.\n";
    }
    out << "[Harmonisch] ---------------------------------------------------------\n";

    r.valid = true;
    return r;
}

int compare_accuracy_series(const std::vector<BlockPcVariant>& variants,
                            const std::string& prefix_a,
                            const std::string& prefix_b) {
    struct Level { int iters; double relres; bool conv; };
    auto collect = [&](const std::string& prefix) {
        std::vector<Level> lv;
        for (const auto& v : variants) {
            if (v.name.rfind(prefix, 0) != 0) continue;
            lv.push_back({v.gmres_iters, v.gmres_relres, v.gmres_converged});
        }
        return lv;
    };
    const std::vector<Level> A = collect(prefix_a);
    const std::vector<Level> B = collect(prefix_b);

    int score = 0;
    const size_t n = std::min(A.size(), B.size());
    for (size_t i = 0; i < n; ++i) {
        const Level& a = A[i];
        const Level& b = B[i];
        if (a.conv && b.conv) {
            // Beide konvergiert: die Iterationszahl entscheidet. GENAU HIER
            // sass der Fehler -- ein Gleichstand der eps-Schwelle darf nicht
            // als Gleichstand der Reihe durchgehen.
            if (b.iters < a.iters) ++score; else if (a.iters < b.iters) --score;
        } else if (b.conv)  { ++score; }
        else if (a.conv)    { --score; }
        else {
            // Keiner konvergiert: die Iterationszahl ist fuer beide die
            // Obergrenze und sagt nichts. Das Restresiduum entscheidet.
            if (b.relres < a.relres) ++score; else if (a.relres < b.relres) --score;
        }
    }
    return score;
}

void report_block_pc_study(const BlockPcStudyResult& res, std::ostream& out) {
    out << "\n[PC-Studie] ---------------------------------------------------------------\n";
    out << "[PC-Studie] Deutung\n";
    out << "[PC-Studie] ---------------------------------------------------------------\n";

    const double kA = res.block_A.kappa;
    const double kK = res.block_K.kappa;
    const double kW = res.whole.kappa;

    out << "[PC-Studie] kappa_2:  ganzes System " << kW
        << " | FEM-Block " << kA << " | BEM-Block " << kK << "\n";

    if (kK > 0.0 && kW > 0.0) {
        if (kK < 0.05 * kW) {
            out << "[PC-Studie] Der BEM-Block ist um mehr als eine Groessenordnung BESSER\n"
                   "[PC-Studie] konditioniert als das Gesamtsystem. Die schlechte Kondition\n"
                   "[PC-Studie] des Systems entsteht damit NICHT in A22. Ein Vorkonditionierer\n"
                   "[PC-Studie] fuer A22 -- Calderon, H-Matrix, gleich welcher -- kann nur\n"
                   "[PC-Studie] wegnehmen, was in A22 steckt.\n";
        } else {
            out << "[PC-Studie] Der BEM-Block traegt einen wesentlichen Teil der Kondition des\n"
                   "[PC-Studie] Gesamtsystems. Ein Vorkonditionierer fuer A22 ist damit\n"
                   "[PC-Studie] plausibel -- die Deckenmessung unten sagt, wieviel er hoechstens\n"
                   "[PC-Studie] bringen kann.\n";
        }
    }

    if (res.bem_block_speedup > 0.0) {
        out << "[PC-Studie]\n";
        out << "[PC-Studie] DECKE FUER JEDEN BEM-BLOCK-VORKONDITIONIERER\n";
        out << "[PC-Studie]   Die EXAKTE Inverse von A22 spart gegenueber der Identitaet den\n"
               "[PC-Studie]   Faktor " << res.bem_block_speedup << " an GMRES-Iterationen.\n";
        if (res.bem_block_speedup < 1.5) {
            out << "[PC-Studie]   -> Das ist die OBERE SCHRANKE. Calderon-Vorkonditionierung und\n"
                   "[PC-Studie]      H-Matrix-Naeherungsinverse sind beide nur NAEHERUNGEN an\n"
                   "[PC-Studie]      genau diese exakte Inverse und koennen deshalb nicht mehr\n"
                   "[PC-Studie]      erreichen. Der Ansatz ist damit widerlegt, ohne dass eine\n"
                   "[PC-Studie]      Zeile davon geschrieben werden musste.\n";
        } else {
            out << "[PC-Studie]   -> Der Aufwand fuer Calderon oder eine H-Matrix-Naeherungs-\n"
                   "[PC-Studie]      inverse ist damit belegt gerechtfertigt, und der Zielwert\n"
                   "[PC-Studie]      steht fest: mehr als dieser Faktor ist nicht zu holen.\n";
        }
    }

    // Variante 4 gegen Variante 3: war die Schur-IDEE falsch oder nur die
    // Naeherung S~ = mu_0*G_BEM?
    if (res.variants.size() > 4 &&
        res.variants[3].gmres_converged && res.variants[4].gmres_converged &&
        res.variants[4].gmres_iters > 0) {        const double f = static_cast<double>(res.variants[3].gmres_iters)
                       / static_cast<double>(res.variants[4].gmres_iters);
        out << "[PC-Studie]\n";
        out << "[PC-Studie] SCHUR-KOMPLEMENT\n";
        out << "[PC-Studie]   Exaktes S gegen blockdiagonal-exakt: Faktor " << f << ".\n";
        if (f > 2.0) {
            out << "[PC-Studie]   -> Die Schur-Idee traegt. Widerlegt war dann nicht sie,\n"
                   "[PC-Studie]      sondern die Naeherung S~ = mu_0*G_BEM, die den Term\n"
                   "[PC-Studie]      A21 A11^-1 A12 vollstaendig weglaesst. Die naechste Frage\n"
                   "[PC-Studie]      waere, wie dieser Term zu naehern ist -- nicht, wie A22\n"
                   "[PC-Studie]      vorkonditioniert wird.\n";
        } else {
            out << "[PC-Studie]   -> Auch das exakte Schur-Komplement bringt wenig. Die\n"
                   "[PC-Studie]      Schur-Idee als solche ist damit widerlegt, nicht nur die\n"
                   "[PC-Studie]      Naeherung.\n";
        }
    }

    // -----------------------------------------------------------------------
    // Blockdiagonal gegen blocktriangulaer -- der Strukturbefund
    // -----------------------------------------------------------------------
    // Werden die Diagonalbloecke EXAKT invertiert und der aeussere Loeser wird
    // trotzdem nicht besser oder sogar schlechter, dann liegt die Kondition
    // nicht in den Bloecken, sondern in der KOPPLUNG. Ein blockdiagonaler
    // Vorkonditionierer kann sie bauartbedingt nicht sehen, gleich wie gut
    // seine Bloecke sind. Das erklaert zugleich, warum vier unabhaengige
    // Verbesserungen des FEM-Block-Vorkonditionierers den aeusseren Loeser
    // nicht bewegt haben (convergence-findings.tex, Befund 5).
    if (res.variants.size() > 4 && res.variants[0].gmres_iters > 0) {
        const bool diag_useless =
            (res.variants[3].gmres_iters >= res.variants[0].gmres_iters) ||
            (!res.variants[3].gmres_converged && !res.variants[0].gmres_converged &&
             res.variants[3].gmres_relres >= res.variants[0].gmres_relres);
        const bool tri_works =
            res.variants[4].gmres_converged && res.variants[4].gmres_iters > 0 &&
            res.variants[4].gmres_iters * 10 < res.variants[0].gmres_iters;

        if (diag_useless && tri_works) {
            out << "[PC-Studie]\n";
            out << "[PC-Studie] STRUKTURBEFUND\n";
            out << "[PC-Studie]   Mit EXAKTEN Diagonalbloecken wird der Loeser nicht besser,\n"
                   "[PC-Studie]   mit dem exakten Schur-Komplement loest er in "
                << res.variants[4].gmres_iters << " Iterationen.\n";
            out << "[PC-Studie]   Die Kondition sitzt damit nicht in A11 und nicht in A22,\n"
                   "[PC-Studie]   sondern in der KOPPLUNG. Ein blockDIAGONALER Vorkonditionierer\n"
                   "[PC-Studie]   kann sie bauartbedingt nicht erreichen -- gleich wie gut seine\n"
                   "[PC-Studie]   Bloecke sind. Das ist der Hebel, nicht A22.\n";
        }
    }

    if (res.schur.available) {
        out << "[PC-Studie]\n";
        out << "[PC-Studie] WARUM S~ = mu_0*G_BEM SCHEITERN MUSSTE\n";
        out << "[PC-Studie]   Der weggelassene Term A21 A11^-1 A12 hat "
            << res.schur.rel_coupling << " mal die Frobeniusnorm von S.\n";
        if (res.schur.rel_coupling > 0.5) {
            out << "[PC-Studie]   -> A22 ist damit keine grobe Naeherung an S, sondern eine\n"
                   "[PC-Studie]      andere Matrix. Dass ein inneres Verfahren darauf nicht\n"
                   "[PC-Studie]      traegt, ist kein Raetsel.\n";
        }
        if (!res.schur.positive_definite) {
            out << "[PC-Studie]   S ist NICHT positiv definit (kleinster Eigenwert des\n"
                   "[PC-Studie]   symmetrischen Anteils: " << res.schur.sym_lambda_min << ").\n"
                   "[PC-Studie]   -> CG ist auf S das falsche Verfahren. Ein inneres Verfahren\n"
                   "[PC-Studie]      fuer S muss indefinit vertragen: GMRES oder BiCGStab.\n";
        }
    }

    // -----------------------------------------------------------------------
    // Die Genauigkeitsreihe: was muss der innere FEM-Loeser leisten?
    // -----------------------------------------------------------------------
    {
        int first_good = -1;
        double first_eps = 0.0;
        for (const auto& v : res.variants) {
            if (v.name.rfind("7.", 0) != 0) continue;
            if (v.gmres_converged) {
                // Der eps-Wert steht am Ende des Namens.
                const size_t p = v.name.rfind(' ');
                if (p != std::string::npos) first_eps = std::atof(v.name.c_str() + p + 1);
                first_good = v.gmres_iters;
                break;   // Liste ist nach fallender Genauigkeit sortiert:
                         // der ERSTE Treffer ist der GROBSTE, der noch reicht.
            }
        }
        out << "[PC-Studie]\n";
        out << "[PC-Studie] GENAUIGKEITSANFORDERUNG AN DEN INNEREN FEM-LOESER\n";
        if (first_good >= 0) {
            out << "[PC-Studie]   Ein relativer Fehler von " << first_eps
                << " in A11^-1 genuegt noch (" << first_good << " Iterationen).\n";
            if (first_eps >= 1e-2) {
                out << "[PC-Studie]   -> Das ist in Reichweite von ILU(0) oder wenigen\n"
                       "[PC-Studie]      AMS-Zyklen. Der blocktriangulare Weg ist damit nicht\n"
                       "[PC-Studie]      nur theoretisch, sondern praktisch gangbar.\n";
            } else {
                out << "[PC-Studie]   -> Das verlangt einen GENAUEN inneren Loeser. ILU(0) allein\n"
                       "[PC-Studie]      reicht dafuer nicht; es braucht innere Krylov-Schritte\n"
                       "[PC-Studie]      bis zu dieser Toleranz, und deren Kosten gehen in jede\n"
                       "[PC-Studie]      aeussere Iteration ein.\n";
            }
        } else {
            out << "[PC-Studie]   KEINE der gepruefeten Genauigkeiten reicht aus.\n";
            out << "[PC-Studie]   -> Der blocktriangulare Vorkonditionierer traegt zwar mit dem\n"
                   "[PC-Studie]      EXAKTEN Schur-Komplement (Variante 4), laesst aber keine\n"
                   "[PC-Studie]      bezahlbare Naeherung zu. Das ist das entscheidende Nein --\n"
                   "[PC-Studie]      und es kommt VOR dem Produktivcode, nicht danach.\n";
        }
    }

    // -----------------------------------------------------------------------
    // Welche Eliminationsrichtung ist die praktikable?
    // -----------------------------------------------------------------------
    //
    // ACHTUNG, hier stand ein Fehler. Die erste Fassung verglich nur die
    // GROBSTE Stufe, die noch konvergiert. Landen beide Richtungen auf
    // derselben Stufe -- was am Netz team28_coarse_12 tatsaechlich passiert,
    // beide bei 1e-3 --, faellt der Vergleich unentschieden aus und der
    // else-Zweig behauptete pauschal, die Richtung ueber A11 sei die bessere.
    // Dass sie auf genau dieser Stufe 1181 statt 29 Iterationen braucht, sah
    // die Logik nicht. Die Zahlen waren richtig, die Deutung darueber falsch.
    //
    // Verglichen wird deshalb jetzt die GANZE Kurve, Stufe fuer Stufe: die
    // Iterationszahl, wo beide konvergieren, sonst das Restresiduum. Eine
    // einzelne Schwelle ist als Entscheidungsgrundlage zu grob, weil sie an
    // der willkuerlichen Iterationsobergrenze haengt.
    if (res.schur.available && res.schur_A.available) {
        struct Level { double eps; int iters; double relres; bool conv; };
        auto collect = [&](const char* prefix) {
            std::vector<Level> out_lv;
            for (const auto& v : res.variants) {
                if (v.name.rfind(prefix, 0) != 0) continue;
                const size_t p = v.name.rfind(' ');
                Level lv;
                lv.eps    = (p != std::string::npos) ? std::atof(v.name.c_str() + p + 1) : 0.0;
                lv.iters  = v.gmres_iters;
                lv.relres = v.gmres_relres;
                lv.conv   = v.gmres_converged;
                out_lv.push_back(lv);
            }
            return out_lv;
        };
        const std::vector<Level> L11 = collect("7.");
        const std::vector<Level> L22 = collect("9.");

        out << "[PC-Studie]\n";
        out << "[PC-Studie] WELCHE ELIMINATIONSRICHTUNG?\n";
        out << "[PC-Studie]   ueber A11 (kappa " << res.block_A.kappa << "):  S   hat kappa_2 = "
            << res.schur.kappa << (res.schur.positive_definite ? ", definit" : ", INDEFINIT") << "\n";
        out << "[PC-Studie]   ueber A22 (kappa " << res.block_K.kappa << "):  S_A hat kappa_2 = "
            << res.schur_A.kappa << (res.schur_A.positive_definite ? ", definit" : ", INDEFINIT") << "\n";

        const size_t n_lv = std::min(L11.size(), L22.size());
        for (size_t i = 0; i < n_lv; ++i) {
            out << "[PC-Studie]     eps = " << L11[i].eps << ":  ueber A11 "
                << L11[i].iters << " Iter / " << L11[i].relres
                << "   ueber A22 " << L22[i].iters << " Iter / " << L22[i].relres << "\n";
        }
        // Die Regel selbst steht in compare_accuracy_series() und wird dort
        // geprueft -- nicht hier dupliziert, damit sie nicht auseinanderlaufen
        // kann.
        const int score = compare_accuracy_series(res.variants, "7.", "9.");
        const int wins22 = (score > 0) ? score : 0;
        const int wins11 = (score < 0) ? -score : 0;
        const int levels = static_cast<int>(n_lv);

        out << "[PC-Studie]   Stufen gewonnen: ueber A22 " << wins22
            << ", ueber A11 " << wins11 << " von " << levels << "\n";

        if (wins22 > wins11) {
            out << "[PC-Studie]   -> Die Richtung ueber A22 ist auf der ganzen Kurve die\n"
                   "[PC-Studie]      bessere. Sie ist zugleich die billigere: A22 ist SPD und\n"
                   "[PC-Studie]      gut konditioniert, ein inneres CG darauf ist kurz, waehrend\n"
                   "[PC-Studie]      A11 mit seinem Fastnullraum-Kontinuum teuer zu invertieren\n"
                   "[PC-Studie]      ist. S_A lebt im FEM-Raum, also dort, wo AMS zu Hause ist.\n"
                   "[PC-Studie]      Und HIER waere ein besserer A22-Vorkonditionierer\n"
                   "[PC-Studie]      tatsaechlich am richtigen Ort -- nicht als Diagonalblock,\n"
                   "[PC-Studie]      wo er widerlegt ist, sondern im inneren Loeser.\n";
        } else if (wins11 > wins22) {
            out << "[PC-Studie]   -> Die Richtung ueber A11 ist auf der ganzen Kurve die bessere.\n";
        } else {
            out << "[PC-Studie]   -> Kein Unterschied ueber die Kurve. Dann entscheiden die\n"
                   "[PC-Studie]      Kosten der inneren Loesung, nicht die Iterationszahl.\n";
        }
    }

    // -----------------------------------------------------------------------
    // Reicht ein NAEHERUNGSWEISER S_A-Loeser? (Reihen 10 und 11)
    // -----------------------------------------------------------------------
    // Erst diese Reihen machen die Empfehlung belastbar. Reihe 9 nahm an, dass
    // S_A exakt geloest wird -- im Produktivcode steht dort AMS.
    {
        auto first_conv = [&](const char* prefix) {
            for (const auto& v : res.variants)
                if (v.name.rfind(prefix, 0) == 0 && v.gmres_converged) {
                    const size_t p = v.name.rfind(' ');
                    return std::make_pair(
                        (p != std::string::npos) ? std::atof(v.name.c_str() + p + 1) : 0.0,
                        v.gmres_iters);
                }
            return std::make_pair(0.0, -1);
        };
        const auto only_SA = first_conv("10.");
        const auto both    = first_conv("11.");
        const auto only_22 = first_conv("9.");

        if (only_SA.second > 0 || both.second > 0) {
            out << "[PC-Studie]\n";
            out << "[PC-Studie] REICHT EIN NAEHERUNGSWEISER S_A-LOESER?\n";
            if (only_22.second > 0)
                out << "[PC-Studie]   nur A22^-1 ungenau:        ab " << only_22.first
                    << "  (" << only_22.second << " Iterationen)\n";
            if (only_SA.second > 0)
                out << "[PC-Studie]   nur S_A-Loeser ungenau:    ab " << only_SA.first
                    << "  (" << only_SA.second << " Iterationen)\n";
            if (both.second > 0)
                out << "[PC-Studie]   BEIDE ungenau:             ab " << both.first
                    << "  (" << both.second << " Iterationen)  <- die Produktivlage\n";

            if (both.second > 0 && both.first >= 1e-2) {
                out << "[PC-Studie]   -> Beide Anteile duerfen grob sein. Der DtN-Zweig ist damit\n"
                       "[PC-Studie]      praktisch gebaut werden koennen: AMS auf S_A und ein\n"
                       "[PC-Studie]      kurzes CG auf A22 genuegen.\n";
            } else if (both.second > 0) {
                out << "[PC-Studie]   -> Die Produktivlage verlangt " << both.first
                    << " in BEIDEN Anteilen. Das ist\n"
                       "[PC-Studie]      die Zahl, an der der Umbau zu rechnen ist: Kosten der\n"
                       "[PC-Studie]      inneren Loesungen bis dorthin, mal " << both.second
                    << " aeussere Iterationen.\n";
            } else {
                out << "[PC-Studie]   -> Sind BEIDE Anteile ungenau, traegt der Zweig auf keiner\n"
                       "[PC-Studie]      der geprueften Stufen. Das ist das Nein zum DtN-Weg --\n"
                       "[PC-Studie]      und es kommt vor dem Produktivcode.\n";
            }

            // Welcher der beiden Anteile gibt den Ausschlag? Das entscheidet,
            // wo Arbeit sich lohnt -- am AMS oder am A22-Vorkonditionierer.
            // -------------------------------------------------------------
            // Was kostet diese Genauigkeit?
            // -------------------------------------------------------------
            // Bis hierher wurde mit exakten LU-Zerlegungen der inneren
            // Operatoren gerechnet. Die gibt es im Ernstfall nicht. 127
            // aeussere Iterationen sind gegenueber 4000 stagnierenden nur dann
            // ein Fortschritt, wenn jede davon nicht das Hundertfache kostet.
            if (res.inner_cost.bem_matvecs > 0.0) {
                out << "[PC-Studie]\n";
                out << "[PC-Studie] WAS KOSTET DIESE GENAUIGKEIT?\n";
                out << "[PC-Studie]   unvorkonditionierte Krylov-Schritte bis "
                    << res.inner_cost.tol << ":\n";
                out << "[PC-Studie]     S_A: " << res.inner_cost.iters_S_A
                    << "   A22: " << res.inner_cost.iters_A22 << "\n";
                out << "[PC-Studie]   Eine S_A-Anwendung kostet eine A22-Loesung.\n";
                out << "[PC-Studie]   Je Zeitschritt also rund "
                    << res.inner_cost.outer_iters << " x " << res.inner_cost.iters_S_A
                    << " x " << res.inner_cost.iters_A22 << " = "
                    << res.inner_cost.bem_matvecs << " BEM-Matvecs,\n";
                out << "[PC-Studie]   gegen " << res.inner_cost.baseline
                    << " des heutigen Loesers -- Faktor "
                    << (res.inner_cost.bem_matvecs / res.inner_cost.baseline) << ".\n";
                out << "[PC-Studie]   Das ist die Zahl, gegen die der Umbau zu rechnen ist. Sie\n"
                       "[PC-Studie]   ist eine OBERE Schranke: beide inneren Loesungen laufen hier\n"
                       "[PC-Studie]   unvorkonditioniert. Genau um diese beiden Faktoren geht es,\n"
                       "[PC-Studie]   wenn ueber innere Vorkonditionierer geredet wird.\n";
            }

            if (only_SA.second > 0 && only_22.second > 0) {
                if (only_SA.first < only_22.first)
                    out << "[PC-Studie]   Der S_A-Loeser ist der empfindlichere Anteil: dort lohnt\n"
                           "[PC-Studie]   die Arbeit (AMS), nicht am A22-Vorkonditionierer.\n";
                else if (only_22.first < only_SA.first)
                    out << "[PC-Studie]   A22^-1 ist der empfindlichere Anteil: HIER ist ein\n"
                           "[PC-Studie]   besserer A22-Vorkonditionierer (Calderon, H-Matrix) am\n"
                           "[PC-Studie]   richtigen Ort -- im inneren Loeser, nicht als\n"
                           "[PC-Studie]   Diagonalblock.\n";
            }
        }
    }

    out << "[PC-Studie] ---------------------------------------------------------------\n";
}

} // namespace bf
