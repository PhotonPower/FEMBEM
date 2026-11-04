#ifndef BEM_ROW_CHECK_H
#define BEM_ROW_CHECK_H

// =============================================================================
// BemRowCheck.h -- Gegenprobe 2: exaktes Biot-Savart-Paar (a_ex, K) in die
// beiden Blockzeilen des produktiven harmonischen BEM-FEM-Pfads einsetzen
// =============================================================================
//
// Reiner Host-Code (double, kein CUDA-Kernel), nur lesend. Nichts hier aendert
// Operatoren, Matrizen oder Loesung. Bericht: docs/design/bem-row-check.md.
//
// ENTHALTEN
//   - Vektorpotential A und Induktion B einer Kreisschleife in geschlossener
//     Form (vollstaendige elliptische Integrale), Spule als Summe ueber
//     Querschnittsfaeden wie tools/team28_field_check.py,
//   - die Normalableitung (n.grad)A analytisch (aus A_phi, B_r, B_z),
//   - Kantenintegrale (4-Punkt-Gauss) und die Galerkin-Projektion eines
//     Tangentialfelds in den Rooftop-Raum eines Quads,
//   - die Quadklassen (eben / Seite / Kante) und die Residuenbuchhaltung.
//
// KONVENTIONEN (aus dem Code, nicht angenommen; Herleitung im Bericht)
//   - Rooftop-Funktion i eines Quads: evaluate_rooftop_basis() aus
//     BemIntegration.cuh, in Richtung der lokalen Kante {i, i+1}.
//   - Koeffizient c_i gehoert zu W_i; K(x) = sum_i c_i W_i(x).
//   - n ist die AEUSSERE Normale des Leiters (von der Gegenprobe selbst aus dem
//     Elternhexaeder bestimmt, nicht die von evaluate_quad_normal()).
//   - mu0 K1 = (n.grad)A, tangentialer Anteil; mu0 K2 = -n x B.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "BemIntegration.cuh"   // evaluate_rooftop_basis, integrate_boundary_mass_matrix

namespace bem_row_check {

constexpr double kMu0 = 1.2566370614359172e-06;  // 4*pi*1e-7, wie CoupledMVOperator
constexpr double kPi  = 3.14159265358979323846;

struct V3 { double x = 0, y = 0, z = 0; };
inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(double s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }
inline double dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline double norm(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 to_v3(const double3& d) { return {d.x, d.y, d.z}; }

// ---------------------------------------------------------------------------
// Elliptische Integrale ueber das arithmetisch-geometrische Mittel
// ---------------------------------------------------------------------------
inline void ellipke(double m, double& K, double& E) {
    double a = 1.0, b = std::sqrt(1.0 - m);
    double c2_sum = 0.5 * m;
    double p = 0.5;
    for (int i = 0; i < 40; ++i) {
        const double an = 0.5 * (a + b);
        const double bn = std::sqrt(a * b);
        const double c  = 0.5 * (a - b);
        p *= 2.0;
        c2_sum += p * c * c;
        a = an; b = bn;
        if (std::fabs(c) < 1e-17 * a) break;
    }
    K = kPi / (2.0 * a);
    E = K * (1.0 - c2_sum);
}

// Kreisschleife (Radius a, Strom I) in der Ebene z = 0, Aufpunkt (r, z):
// A_phi, B_r, B_z. Formeln wie loop_field() in tools/team28_field_check.py.
struct LoopField { double Aphi, Br, Bz; };

inline LoopField loop_field(double a, double I, double r, double z) {
    if (r < 1e-12) {
        // auf der Achse: A_phi = B_r = 0, B_z wie bekannt
        const double s2 = a * a + z * z;
        return {0.0, 0.0, kMu0 * I * a * a / (2.0 * s2 * std::sqrt(s2))};
    }
    const double s2 = (a + r) * (a + r) + z * z;
    double m = 4.0 * a * r / s2;
    if (m > 1.0 - 1e-15) m = 1.0 - 1e-15;
    double K, E;
    ellipke(m, K, E);
    const double d2 = (a - r) * (a - r) + z * z;
    const double c  = kMu0 * I / (2.0 * kPi * std::sqrt(s2));
    LoopField f;
    f.Bz = c * (K + (a * a - r * r - z * z) / d2 * E);
    f.Br = c * z / r * (-K + (a * a + r * r + z * z) / d2 * E);
    const double k = std::sqrt(m);
    f.Aphi = kMu0 * I / (kPi * std::max(k, 1e-12)) * std::sqrt(a / r) * ((1.0 - 0.5 * m) * K - E);
    return f;
}

// ---------------------------------------------------------------------------
// Spulen als Querschnittsfaeden
// ---------------------------------------------------------------------------
struct Filament { double a, z0, I; };

struct CoilSpec { double r0, r1, z0, z1, turns, sign; };

// TEAM 28 wie tools/team28_axisymmetric.py: Innenspule r 27..55 mm, 960
// Windungen, Aussenspule r 80..95 mm, 576 Windungen, gegensinnig, beide
// z -52..0 mm. Scheitelstrom 20 A.
inline std::vector<CoilSpec> team28_coils() {
    return {{27e-3, 55e-3, -52e-3, 0.0, 960.0, +1.0},
            {80e-3, 95e-3, -52e-3, 0.0, 576.0, -1.0}};
}

// Fadenraster: je Spule n_r x n_z Zellmitten, Strom je Faden
// sign*turns*I/(n_r*n_z). `spacing` ist die Zielzellbreite in m.
inline std::vector<Filament> make_filaments(const std::vector<CoilSpec>& coils,
                                            double I_peak, double spacing) {
    std::vector<Filament> out;
    for (const CoilSpec& c : coils) {
        const int nr = std::max(1, static_cast<int>(std::ceil((c.r1 - c.r0) / spacing - 1e-9)));
        const int nz = std::max(1, static_cast<int>(std::ceil((c.z1 - c.z0) / spacing - 1e-9)));
        const double Ifil = c.sign * c.turns * I_peak / (static_cast<double>(nr) * nz);
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nz; ++j)
                out.push_back({c.r0 + (i + 0.5) * (c.r1 - c.r0) / nr,
                               c.z0 + (j + 0.5) * (c.z1 - c.z0) / nz, Ifil});
    }
    return out;
}

// Summe ueber alle Faeden: A_phi, B_r, B_z am Punkt (r, z).
inline LoopField sum_field(const std::vector<Filament>& f, double r, double z) {
    LoopField s{0, 0, 0};
    for (const Filament& fi : f) {
        const LoopField l = loop_field(fi.a, fi.I, r, z - fi.z0);
        s.Aphi += l.Aphi; s.Br += l.Br; s.Bz += l.Bz;
    }
    return s;
}

// Kartesische Felder am Punkt p.
struct ExactField {
    V3 A, B;
    double Aphi, Br, Bz, r;
    V3 e_r, e_phi;
};

inline ExactField exact_field(const std::vector<Filament>& f, V3 p) {
    ExactField o;
    o.r = std::sqrt(p.x * p.x + p.y * p.y);
    const LoopField s = sum_field(f, o.r, p.z);
    o.Aphi = s.Aphi; o.Br = s.Br; o.Bz = s.Bz;
    if (o.r > 1e-12) { o.e_r = {p.x / o.r, p.y / o.r, 0.0}; o.e_phi = {-p.y / o.r, p.x / o.r, 0.0}; }
    else             { o.e_r = {1, 0, 0};                    o.e_phi = {0, 1, 0}; }
    o.A = o.Aphi * o.e_phi;
    o.B = o.Br * o.e_r + V3{0, 0, o.Bz};
    return o;
}

// (n.grad)A, analytisch. A = A_phi(r,z) e_phi gibt
//   (n.grad)A = (n.grad A_phi) e_phi - (A_phi / r) (n.e_phi) e_r
// mit dA_phi/dz = -B_r und dA_phi/dr = B_z - A_phi / r.
inline V3 normal_derivative_A(const ExactField& e, V3 n) {
    const double dAr = e.Bz - (e.r > 1e-12 ? e.Aphi / e.r : 0.0);
    const double dAz = -e.Br;
    const double ngA = dot(n, e.e_r) * dAr + n.z * dAz;
    const double over_r = e.r > 1e-12 ? e.Aphi / e.r : 0.0;
    return ngA * e.e_phi - (over_r * dot(n, e.e_phi)) * e.e_r;
}

// ---------------------------------------------------------------------------
// Quadratur
// ---------------------------------------------------------------------------
inline void gauss4(double x[4], double w[4]) {
    const double a = 0.3399810435848563, b = 0.8611363115940526;
    const double wa = 0.6521451548625461, wb = 0.3478548451374538;
    x[0] = -b; x[1] = -a; x[2] = a; x[3] = b;
    w[0] = wb; w[1] = wa; w[2] = wa; w[3] = wb;
}

// Kantenintegral int A.dl von p0 nach p1 mit 4-Punkt-Gauss.
template <class AField>
double edge_integral(V3 p0, V3 p1, const AField& A_at) {
    double x[4], w[4];
    gauss4(x, w);
    const V3 d = p1 - p0;
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double t = 0.5 * (x[i] + 1.0);
        s += 0.5 * w[i] * dot(A_at(p0 + t * d), d);
    }
    return s;
}

// ---------------------------------------------------------------------------
// Quad: Geometrie, Rooftop-Raum, Galerkin-Projektion
// ---------------------------------------------------------------------------
struct QuadGeom {
    double3 n[4];            // Knoten
    double outward = 1.0;    // +1: evaluate_quad_normal() zeigt nach aussen, -1: nach innen
};

inline V3 quad_point(const double3 nodes[4], double u, double v) {
    return to_v3(evaluate_quad_position(u, v, nodes));
}

// Aeussere Einheitsnormale am Punkt (u, v) (Vorzeichen aus QuadGeom::outward).
inline V3 outward_normal(const QuadGeom& q, double u, double v) {
    return q.outward * to_v3(evaluate_quad_normal(u, v, q.n));
}

// Loest die 4x4-Gleichung M c = f (Gauss mit Spaltenpivot). Gibt false bei
// Singularitaet zurueck.
inline bool solve4(double M[4][4], const double f[4], double c[4]) {
    double a[4][5];
    for (int i = 0; i < 4; ++i) { for (int j = 0; j < 4; ++j) a[i][j] = M[i][j]; a[i][4] = f[i]; }
    for (int k = 0; k < 4; ++k) {
        int p = k;
        for (int i = k + 1; i < 4; ++i) if (std::fabs(a[i][k]) > std::fabs(a[p][k])) p = i;
        if (std::fabs(a[p][k]) < 1e-300) return false;
        if (p != k) for (int j = 0; j < 5; ++j) std::swap(a[p][j], a[k][j]);
        for (int i = k + 1; i < 4; ++i) {
            const double m = a[i][k] / a[k][k];
            for (int j = k; j < 5; ++j) a[i][j] -= m * a[k][j];
        }
    }
    for (int i = 3; i >= 0; --i) {
        double s = a[i][4];
        for (int j = i + 1; j < 4; ++j) s -= a[i][j] * c[j];
        c[i] = s / a[i][i];
    }
    return true;
}

// Galerkin-Projektion eines 3D-Vektorfelds F(p, n) in den Rooftop-Raum:
// T c = f mit T = integrate_boundary_mass_matrix (die Massenmatrix des
// Operators) und f_i = int W_i . F dS (16-Punkt-Gauss). F darf einen
// Normalanteil haben, er faellt im Skalarprodukt mit W_i heraus.
// Rueckgabe false, wenn T singulaer ist.
inline bool project_to_rooftop(const QuadGeom& q,
                               const std::function<V3(V3 p, V3 n)>& F,
                               double c[4]) {
    double x[4], w[4];
    gauss4(x, w);
    double f[4] = {0, 0, 0, 0};
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
        const double u = x[a], v = x[b];
        double detJ; double3 W[4];
        evaluate_rooftop_basis(u, v, q.n, W, detJ);
        const V3 p = quad_point(q.n, u, v);
        const V3 n = outward_normal(q, u, v);
        const V3 val = F(p, n);
        for (int i = 0; i < 4; ++i) f[i] += w[a] * w[b] * detJ * dot(to_v3(W[i]), val);
    }
    double T[4][4];
    integrate_boundary_mass_matrix(q.n, T);
    return solve4(T, f, c);
}

// Rooftop-Rekonstruktion sum c_i W_i am Punkt (u, v).
inline V3 rooftop_value(const QuadGeom& q, const double c[4], double u, double v) {
    double detJ; double3 W[4];
    evaluate_rooftop_basis(u, v, q.n, W, detJ);
    V3 s;
    for (int i = 0; i < 4; ++i) s = s + c[i] * to_v3(W[i]);
    return s;
}

// ---------------------------------------------------------------------------
// Quadklassen
// ---------------------------------------------------------------------------
enum QuadClass { kPlane = 0, kSide = 1, kEdgeQuad = 2, kNumClasses = 3 };

inline const char* class_name(int c) {
    return c == kPlane ? "eben" : (c == kSide ? "Seite" : (c == kEdgeQuad ? "Kante" : "?"));
}

// eben: |n_z| > 0.99. Kante: nicht eben, und eine der vier Kanten wird mit
// einem Quad geteilt, dessen Normale um mehr als ~26 Grad abweicht
// (|n.n'| < 0.9). Seite: Rest. Die Seitenfacetten des 48-Ecks weichen nur um
// 7.5 Grad voneinander ab und sind daher keine Kante.
inline int classify_quad(V3 n, bool touches_sharp_edge) {
    if (std::fabs(n.z) > 0.99) return kPlane;
    return touches_sharp_edge ? kEdgeQuad : kSide;
}

// ---------------------------------------------------------------------------
// Residuenbuchhaltung
// ---------------------------------------------------------------------------
struct Residual {
    double num2 = 0.0;   // ||r||^2
    double ref2 = 0.0;   // ||Bezugsterm||^2
    long   n = 0;
    void add(double r, double ref) { num2 += r * r; ref2 += ref * ref; ++n; }
    double rel() const { return ref2 > 0.0 ? std::sqrt(num2 / ref2) : 0.0; }
    double abs() const { return std::sqrt(num2); }
};

// Summiert r[i] (Bezug ref[i]) in die Klasse label[i] und in "alle".
// label[i] < 0 ueberspringt den Eintrag.
inline std::vector<Residual> accumulate_by_label(const std::vector<double>& r,
                                                 const std::vector<double>& ref,
                                                 const std::vector<int>& label,
                                                 int num_labels) {
    std::vector<Residual> out(static_cast<size_t>(num_labels) + 1);
    for (size_t i = 0; i < r.size(); ++i) {
        if (label[i] < 0) continue;
        out[static_cast<size_t>(label[i])].add(r[i], ref[i]);
        out[static_cast<size_t>(num_labels)].add(r[i], ref[i]);
    }
    return out;
}

// Euklidische Norm
inline double vnorm(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// r = base + s * add (elementweise)
inline std::vector<double> axpy(const std::vector<double>& base, double s,
                                const std::vector<double>& add) {
    std::vector<double> r(base.size());
    for (size_t i = 0; i < base.size(); ++i) r[i] = base[i] + s * add[i];
    return r;
}

// Einfaches parallel_for ueber [0, n) mit std::thread.
template <class Fn>
void parallel_for(size_t n, const Fn& fn) {
    unsigned nt = std::max(1u, std::thread::hardware_concurrency());
    nt = static_cast<unsigned>(std::min<size_t>(nt, std::max<size_t>(1, n / 8)));
    if (nt <= 1) { for (size_t i = 0; i < n; ++i) fn(i); return; }
    std::vector<std::thread> th;
    for (unsigned t = 0; t < nt; ++t)
        th.emplace_back([&, t] { for (size_t i = t; i < n; i += nt) fn(i); });
    for (auto& x : th) x.join();
}

}  // namespace bem_row_check

#endif  // BEM_ROW_CHECK_H
