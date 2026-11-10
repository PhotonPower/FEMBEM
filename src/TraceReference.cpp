#include "TraceReference.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

namespace trace {
namespace {

const double MU0 = 4e-7 * 3.14159265358979323846;
const double PI = 3.14159265358979323846;

// K(m) mit p = 1 - m: K = pi / (2 AGM(1, sqrt(p)))
double ellipk_p(double p)
{
    double a = 1.0, b = std::sqrt(p);
    for (int i = 0; i < 40 && std::fabs(a - b) > 1e-16 * a; ++i) { const double an = 0.5 * (a + b), bn = std::sqrt(a * b); a = an; b = bn; }
    return PI / (2.0 * a);
}

void loop_fields(double a, double zp, double rho, double z, double& Aphi, double& Br, double& Bz)
{
    const double dz = z - zp;
    const double s2 = (a + rho) * (a + rho) + dz * dz, d2 = (a - rho) * (a - rho) + dz * dz;
    const double p = d2 / s2, m = 1.0 - p;
    const double K = ellipk_p(p), E = std::comp_ellint_2(std::sqrt(std::min(1.0, std::max(0.0, m))));
    Bz = MU0 / (2 * PI) / std::sqrt(s2) * (K + (a * a - rho * rho - dz * dz) / d2 * E);
    Br = MU0 / (2 * PI) * dz / (rho * std::sqrt(s2)) * (-K + (a * a + rho * rho + dz * dz) / d2 * E);
    const double k = std::sqrt(m);
    Aphi = MU0 / (PI * k) * std::sqrt(a / rho) * ((1 - m / 2) * K - E);
}

std::vector<double> graded(double lo, double hi, bool anchor_lo, double q, int levels)
{
    const double L = hi - lo;
    std::set<double> pts;
    for (int k = 0; k <= levels; ++k) { const double d = L * std::pow(q, k); pts.insert(anchor_lo ? lo + d : hi - d); }
    pts.insert(anchor_lo ? lo : hi);
    return std::vector<double>(pts.begin(), pts.end());
}

void gauss(int n, std::vector<double>& x, std::vector<double>& w)
{
    x.resize(n); w.resize(n);
    for (int i = 0; i < n; ++i) {
        double z = std::cos(PI * (i + 0.75) / (n + 0.5)), pp = 1;
        for (int it = 0; it < 100; ++it) {
            double p1 = 1, p2 = 0;
            for (int j = 0; j < n; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j + 1) * z * p2 - j * p3) / (j + 1); }
            pp = n * (z * p1 - p2) / (z * z - 1);
            const double z1 = z; z = z1 - p1 / pp;
            if (std::fabs(z - z1) < 1e-15) break;
        }
        x[i] = z; w[i] = 2.0 / ((1 - z * z) * pp * pp);
    }
}

} // namespace

void ring_fields(const RingCoil& c, double rho, double z, double& Aphi, double& Brho, double& Bz)
{
    static std::vector<double> gx, gw;
    if (gx.empty()) gauss(8, gx, gw);
    const double dens = c.turns * c.current / ((c.r1 - c.r0) * (c.z1 - c.z0));
    const double rc = std::min(std::max(rho, c.r0), c.r1), zc = std::min(std::max(z, c.z0), c.z1);
    std::set<double> rcuts{c.r0, c.r1, rc}, zcuts{c.z0, c.z1, zc};
    const std::vector<double> rv(rcuts.begin(), rcuts.end()), zv(zcuts.begin(), zcuts.end());
    Aphi = Brho = Bz = 0.0;
    for (size_t i = 0; i + 1 < rv.size(); ++i) for (size_t j = 0; j + 1 < zv.size(); ++j) {
        const double ra = rv[i], rb = rv[i + 1], za = zv[j], zb = zv[j + 1];
        const std::vector<double> pr = graded(ra, rb, std::fabs(rho - ra) <= std::fabs(rho - rb), 0.3, 11);
        const std::vector<double> pz = graded(za, zb, std::fabs(z - za) <= std::fabs(z - zb), 0.3, 11);
        for (size_t a = 0; a + 1 < pr.size(); ++a) for (int ia = 0; ia < 8; ++ia) {
            const double r = 0.5 * (pr[a + 1] - pr[a]) * gx[ia] + 0.5 * (pr[a + 1] + pr[a]), wr = 0.5 * (pr[a + 1] - pr[a]) * gw[ia];
            for (size_t b = 0; b + 1 < pz.size(); ++b) for (int ib = 0; ib < 8; ++ib) {
                const double zz = 0.5 * (pz[b + 1] - pz[b]) * gx[ib] + 0.5 * (pz[b + 1] + pz[b]), wz = 0.5 * (pz[b + 1] - pz[b]) * gw[ib];
                double ap, br, bz;
                loop_fields(r, zz, rho, z, ap, br, bz);
                Aphi += dens * wr * wz * ap; Brho += dens * wr * wz * br; Bz += dens * wr * wz * bz;
            }
        }
    }
}

Vec3 ring_B(const std::vector<RingCoil>& coils, const Vec3& x)
{
    const double rho = std::max(std::sqrt(x.x * x.x + x.y * x.y), 1e-9);
    double Br = 0, Bz = 0;
    for (const RingCoil& c : coils) { double ap, br, bz; ring_fields(c, rho, x.z, ap, br, bz); Br += br; Bz += bz; }
    Vec3 B;
    B.x = Br * x.x / rho; B.y = Br * x.y / rho; B.z = Bz;
    return B;
}

Vec3 ring_A(const std::vector<RingCoil>& coils, const Vec3& x)
{
    const double rho = std::max(std::sqrt(x.x * x.x + x.y * x.y), 1e-9);
    double Ap = 0;
    for (const RingCoil& c : coils) { double ap, br, bz; ring_fields(c, rho, x.z, ap, br, bz); Ap += ap; }
    Vec3 A;
    A.x = -Ap * x.y / rho; A.y = Ap * x.x / rho; A.z = 0;
    return A;
}

} // namespace trace
