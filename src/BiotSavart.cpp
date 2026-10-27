#include "BiotSavart.cuh"

#include <cmath>
#include <vector>

namespace biot_savart {
namespace {

constexpr double MU_0 = 4.0e-7 * 3.14159265358979323846;

/// Stammfunktion des inneren r-Integrals:
///   F(r,h) = ln(r + sqrt(r^2+h^2)) - r/sqrt(r^2+h^2)
double inner_antiderivative(double r, double h) {
    const double s = std::sqrt(r * r + h * h);
    if (s <= 0.0) return 0.0;              // r = h = 0, Beitrag null
    return std::log(r + s) - r / s;
}

/// Gauss-Legendre-Knoten und -Gewichte auf [-1,1], per Newton auf den
/// Legendre-Polynomen. Bewusst berechnet statt tabelliert: die Ordnung soll
/// frei waehlbar sein, damit die Konvergenz nachweisbar ist.
void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
    x.assign(n, 0.0);
    w.assign(n, 0.0);
    const double pi = 3.14159265358979323846;
    for (int i = 0; i < n; ++i) {
        // Startwert nach Chebyshev
        double t = std::cos(pi * (i + 0.75) / (n + 0.5));
        for (int it = 0; it < 100; ++it) {
            // Legendre P_n und P_n' per Rekursion
            double p0 = 1.0, p1 = t;
            for (int k = 2; k <= n; ++k) {
                const double p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k;
                p0 = p1; p1 = p2;
            }
            const double dp = n * (t * p1 - p0) / (t * t - 1.0);
            const double dt = -p1 / dp;
            t += dt;
            if (std::fabs(dt) < 1e-15) break;
        }
        double p0 = 1.0, p1 = t;
        for (int k = 2; k <= n; ++k) {
            const double p2 = ((2 * k - 1) * t * p1 - (k - 1) * p0) / k;
            p0 = p1; p1 = p2;
        }
        const double dp = n * (t * p1 - p0) / (t * t - 1.0);
        x[i] = t;
        w[i] = 2.0 / ((1.0 - t * t) * dp * dp);
    }
}

} // namespace

double coil_bz_on_axis(double r1, double r2, double z1, double z2,
                       double J, double z, int n_gauss)
{
    if (n_gauss < 2) n_gauss = 2;
    std::vector<double> gx, gw;
    gauss_legendre(n_gauss, gx, gw);

    const double half = 0.5 * (z2 - z1);
    const double mid  = 0.5 * (z2 + z1);

    double acc = 0.0;
    for (int i = 0; i < n_gauss; ++i) {
        const double zeta = mid + half * gx[i];
        const double h    = z - zeta;
        const double f    = inner_antiderivative(r2, h) - inner_antiderivative(r1, h);
        acc += gw[i] * half * f;
    }
    return 0.5 * MU_0 * J * acc;
}

double team28_bz_on_axis(double z, double current) {
    // Geometrie aus Fig. 2 der Aufgabenstellung. Die Pfeile dort zeigen auf die
    // Spulenmittellinien: w1 bei 41 mm, w2 bei 41 + 46.5 = 87.5 mm; mit den
    // Breiten 28 und 15 mm folgen 27..55 und 80..95 mm. Das deckt sich exakt
    // mit dem Netz.
    const double zi1 = -0.052, zi2 = 0.0;

    const double r1_i = 0.027, r2_i = 0.055;   // w1
    const double r1_o = 0.080, r2_o = 0.095;   // w2
    const int    w_i  = 960;
    const int    w_o  = 576;

    const double A_i = (r2_i - r1_i) * (zi2 - zi1);
    const double A_o = (r2_o - r1_o) * (zi2 - zi1);

    const double J_i =  w_i * current / A_i;
    const double J_o = -w_o * current / A_o;   // gegensinnig gewickelt

    return coil_bz_on_axis(r1_i, r2_i, zi1, zi2, J_i, z)
         + coil_bz_on_axis(r1_o, r2_o, zi1, zi2, J_o, z);
}

} // namespace biot_savart
