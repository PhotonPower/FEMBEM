// =============================================================================
// Vergleich der beiden Lorentz-Kernel auf demselben Phasorfeld
//
//   ./build/test_lorentz_kernel_agreement
//
// Reiner Host-Code, keine GPU, keine Dateien.
//
// ANLASS (docs/design/force-estimator-check.md, A1)
// -------------------------------------------------
// harmonic_lorentz_body (Kantenelemente, BEM-FEM-Pfad) und
// nodal_lorentz_harmonic_body (Q1-Knoten, Referenzpfad) sind verschiedene
// Implementierungen derselben Formel <J x B> = (sigma w/2)(A_i x rot A_r
// - A_r x rot A_i). Jeder ist einzeln gegen exakte Felder getestet
// (test_piola_production, test_reference_harmonic); dieser Test setzt beide auf
// DASSELBE Feld und vergleicht sie direkt.
//
// FAELLE
//   1. Parallelepiped, affine Felder A_r, A_i. Beide Raeume enthalten das Feld
//      exakt; Erwartung: beide gleich dem Analytischen, bis auf Rundung.
//   2. Verzerrtes Hexaeder (wie im Referenztest), glattes nichtaffines Feld
//      (quadratisch). Kantengrade = Linienintegral (5-Punkt-Gauss), Knotenwerte
//      = Feld in den Knoten. Erwartung: die Kernel weichen um den
//      Diskretisierungsfehler voneinander und vom Exakten ab; die Abweichung
//      wird nur berichtet und gegen eine lose Grenze gehalten.
//   3. Gegenproben: A_r/A_i vertauscht und A_i mit falschem Vorzeichen muessen
//      in beiden Kerneln die Kraft umkehren, also den Vergleich mit dem
//      Original reissen.
// =============================================================================

#include "Backend.cuh"
#include "HarmonicLorentz.cuh"
#include "HarmonicMaxwell.cuh"
#include "NodalMaxwell.cuh"
#include "NedelecDispatch.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;

static double3 add3_(double3 a, double3 b) { return make_double3(a.x+b.x, a.y+b.y, a.z+b.z); }
static double3 sub3_(double3 a, double3 b) { return make_double3(a.x-b.x, a.y-b.y, a.z-b.z); }
static double3 scl3_(double s, double3 a)  { return make_double3(s*a.x, s*a.y, s*a.z); }
static double  dot3_(double3 a, double3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static double3 crs3_(double3 a, double3 b) {
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static double nrm3_(double3 a) { return std::sqrt(dot3_(a, a)); }

// Feld: A(x) = c + L x + Q(x), rot A analytisch.
//   affine:  A = c + 1/2 b x x      (rot A = b)
//   quadr.:  zusaetzlich A += (q1 y^2, q2 z^2, q3 x^2) * s
struct Field {
    double3 b, c;
    double s;            // 0 = affin
    double3 A(double3 x) const {
        double3 a = add3_(scl3_(0.5, crs3_(b, x)), c);
        a.x += s * x.y * x.y; a.y += s * x.z * x.z; a.z += s * x.x * x.x;
        return a;
    }
    double3 curl(double3 x) const {
        // rot (q1 y^2, q2 z^2, q3 x^2) = (-2 q2 z, -2 q3 x, -2 q1 y), q = s
        return add3_(b, make_double3(-2.0 * s * x.z, -2.0 * s * x.x, -2.0 * s * x.y));
    }
};

struct Hex {
    double3 nodes[8];
    Hexahedron hex;
    int edge_a[12], edge_b[12];
    Hex() {
        for (int i = 0; i < 8; ++i) hex.v[i] = i;
        for (int i = 0; i < 12; ++i) element_local_edge_nodes(ELEM_HEX, i, edge_a[i], edge_b[i]);
    }
    double3 ref(int i) const {
        double r, s, t; element_reference_node(ELEM_HEX, i, r, s, t);
        return make_double3(r, s, t);
    }
};

static const double GP5[5] = {-0.906179845938664, -0.538469310105683, 0.0,
                              0.538469310105683, 0.906179845938664};
static const double GW5[5] = {0.236926885056189, 0.478628670499366, 0.568888888888889,
                              0.478628670499366, 0.236926885056189};

static std::vector<double> edge_dofs(const Hex& h, const Field& f) {
    std::vector<double> a(12);
    for (int i = 0; i < 12; ++i) {
        const double3 xa = h.nodes[h.edge_a[i]], xb = h.nodes[h.edge_b[i]];
        double s = 0.0;
        for (int q = 0; q < 5; ++q) {
            const double u = 0.5 * (GP5[q] + 1.0);
            s += 0.5 * GW5[q] * dot3_(f.A(add3_(xa, scl3_(u, sub3_(xb, xa)))), sub3_(xb, xa));
        }
        a[i] = s;
    }
    return a;
}

static std::vector<double> node_dofs(const Hex& h, const Field& f) {
    std::vector<double> a(24);
    for (int i = 0; i < 8; ++i) {
        const double3 v = f.A(h.nodes[i]);
        a[3*i] = v.x; a[3*i+1] = v.y; a[3*i+2] = v.z;
    }
    return a;
}

// Exakte Referenz: 5x5x5-Gauss ueber das trilineare Element, exakte Felder.
static double3 exact_force(const Hex& h, const Field& fr, const Field& fi, double k) {
    double3 F = make_double3(0, 0, 0);
    for (int a = 0; a < 5; ++a) for (int b = 0; b < 5; ++b) for (int c = 0; c < 5; ++c) {
        double xi = GP5[a], eta = GP5[b], zeta = GP5[c];
        double N[8]; evaluate_nodal_shape(xi, eta, zeta, N);
        double3 gradN[8]; double detJ;
        evaluate_nodal_gradients(const_cast<double3*>(h.nodes), xi, eta, zeta, gradN, detJ);
        const double3 x = nodal_map_to_physical(const_cast<double3*>(h.nodes), xi, eta, zeta);
        const double3 d = scl3_(k, sub3_(crs3_(fi.A(x), fr.curl(x)), crs3_(fr.A(x), fi.curl(x))));
        F = add3_(F, scl3_(GW5[a] * GW5[b] * GW5[c] * std::fabs(detJ), d));
    }
    return F;
}

struct Result { double3 edge, nodal2, nodal3; };

static Result run_kernels(const Hex& h, const Field& fr, const Field& fi, double sigma, double omega) {
    MaterialProperties mat; mat.kappa = sigma;
    const std::vector<double> er = edge_dofs(h, fr), ei = edge_dofs(h, fi);
    const std::vector<double> nr = node_dofs(h, fr), ni = node_dofs(h, fi);
    int edges[12], signs[12];
    for (int i = 0; i < 12; ++i) { edges[i] = i; signs[i] = 1; }
    const int etype = ELEM_HEX;
    const double3 cm = make_double3(0, 0, 0);
    Result r;
    double f[6] = {0, 0, 0, 0, 0, 0};
    harmonic_lorentz_body(0, 1, &h.hex, h.nodes, edges, signs, &etype, &mat, nullptr,
                          er.data(), ei.data(), omega, cm, &f[0], &f[1], &f[2], &f[3], &f[4], &f[5]);
    r.edge = make_double3(f[0], f[1], f[2]);
    for (int pass = 0; pass < 2; ++pass) {
        double g[6] = {0, 0, 0, 0, 0, 0};
        nodal_lorentz_harmonic_body(0, 1, &h.hex, h.nodes, &mat, nullptr, nr.data(), ni.data(),
                                    omega, pass == 0 ? 2 : 3, cm,
                                    &g[0], &g[1], &g[2], &g[3], &g[4], &g[5]);
        (pass == 0 ? r.nodal2 : r.nodal3) = make_double3(g[0], g[1], g[2]);
    }
    return r;
}

static void check(const char* name, double err, double tol) {
    const bool ok = std::isfinite(err) && err <= tol;
    std::printf("  %-62s %10.3e  (Grenze %8.1e)  %s\n", name, err, tol, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}
static void check_control(const char* name, double err, double tol) {
    const bool ok = std::isfinite(err) && err > tol;
    std::printf("  %-62s %10.3e  (Grenze %8.1e)  %s\n", name, err, tol,
                ok ? "DURCHGEFALLEN (erwartet)" : "BESTANDEN -- Kontrollfall wirkungslos");
    if (!ok) ++g_failures;
}
static double rel(double3 a, double3 b) { return nrm3_(sub3_(a, b)) / nrm3_(b); }

int main() {
    const double sigma = 3.5e7, omega = 2.0 * 3.14159265358979323846 * 50.0;
    const double k = 0.5 * sigma * omega;
    // Feldstaerken ~ Aluminiumplatte: A ~ 1e-3 Vs/m, B ~ 0.1 T
    const Field fr_aff = {make_double3(0.30, -1.10, 0.70), make_double3(2.0e-3, 5.0e-3, -4.0e-3), 0.0};
    const Field fi_aff = {make_double3(-0.45, 0.25, 0.90), make_double3(-3.0e-3, 1.0e-3, 6.0e-3), 0.0};

    // --- Fall 1: Parallelepiped (gescherte Einheitszelle, Kante ~ 1 cm)
    std::puts("\n[1] Parallelepiped, affine Felder: beide Kernel exakt");
    Hex par;
    const double M[3][3] = {{1.0, 0.3, 0.1}, {0.0, 1.2, 0.2}, {0.1, 0.0, 0.9}};
    for (int i = 0; i < 8; ++i) {
        const double3 r = par.ref(i);   // Referenzknoten in [-1,1]
        par.nodes[i] = make_double3(
            0.005 * (M[0][0]*r.x + M[0][1]*r.y + M[0][2]*r.z) + 0.02,
            0.005 * (M[1][0]*r.x + M[1][1]*r.y + M[1][2]*r.z) - 0.01,
            0.005 * (M[2][0]*r.x + M[2][1]*r.y + M[2][2]*r.z) + 0.03);
    }
    {
        const double3 Fex = exact_force(par, fr_aff, fi_aff, k);
        const Result r = run_kernels(par, fr_aff, fi_aff, sigma, omega);
        std::printf("  F_exakt = (%.6e, %.6e, %.6e) N\n", Fex.x, Fex.y, Fex.z);
        check("Kantenkernel gegen exakt", rel(r.edge, Fex), 1e-12);
        check("Knotenkernel (2x2x2) gegen exakt", rel(r.nodal2, Fex), 1e-12);
        check("Knotenkernel (3x3x3) gegen exakt", rel(r.nodal3, Fex), 1e-12);
        check("Kantenkernel gegen Knotenkernel", rel(r.edge, r.nodal3), 1e-12);
        // Gegenproben
        const Result sw = run_kernels(par, fi_aff, fr_aff, sigma, omega);   // A_r <-> A_i
        check_control("Kontrolle A_r/A_i vertauscht: Kanten gegen Original", rel(sw.edge, r.edge), 0.1);
        check_control("Kontrolle A_r/A_i vertauscht: Knoten gegen Original", rel(sw.nodal3, r.nodal3), 0.1);
        check("vertauscht: Kanten und Knoten untereinander gleich", rel(sw.edge, sw.nodal3), 1e-12);
        check("vertauscht: Kraft kehrt das Vorzeichen um (Kante)", rel(sw.edge, scl3_(-1.0, r.edge)), 1e-12);
        Field neg = fi_aff; neg.b = scl3_(-1.0, neg.b); neg.c = scl3_(-1.0, neg.c);
        const Result ng = run_kernels(par, fr_aff, neg, sigma, omega);      // A_i -> -A_i
        check_control("Kontrolle falsches Vorzeichen A_i: Kanten gegen Original", rel(ng.edge, r.edge), 0.1);
        check_control("Kontrolle falsches Vorzeichen A_i: Knoten gegen Original", rel(ng.nodal3, r.nodal3), 0.1);
    }

    // --- Fall 1b: Wirbelstromdiagnose (m_z, P) beider Pfade auf demselben Feld
    std::puts("\n[1b] Wirbelstromdiagnose: harmonic_eddy_diag_body gegen nodal_eddy_diag_body (affin)");
    {
        MaterialProperties mat; mat.kappa = sigma;
        int edges[12], signs[12];
        for (int i = 0; i < 12; ++i) { edges[i] = i; signs[i] = 1; }
        const int etype = ELEM_HEX;
        for (int part = 0; part < 2; ++part) {
            const Field& f = part == 0 ? fr_aff : fi_aff;
            const std::vector<double> ed = edge_dofs(par, f), nd = node_dofs(par, f);
            double e[3] = {0, 0, 0};
            harmonic_eddy_diag_body(0, 1, &par.hex, par.nodes, edges, signs, &etype, &mat,
                                    ed.data(), omega, &e[0], &e[1], &e[2]);
            double n[3] = {0, 0, 0};
            const std::vector<double> zero(24, 0.0);
            nodal_eddy_diag_body(0, 1, &par.hex, par.nodes, &mat, nd.data(), zero.data(),
                                 1.0 / omega, &n[0], &n[1], &n[2]);
            char buf[96];
            std::snprintf(buf, sizeof buf, "m_z Kante gegen Knoten (%s)", part == 0 ? "A_r" : "A_i");
            check(buf, std::fabs(e[0] - n[0]) / std::fabs(n[0]), 1e-12);
            std::snprintf(buf, sizeof buf, "P Kante gegen Knoten (%s)", part == 0 ? "A_r" : "A_i");
            check(buf, std::fabs(e[1] - n[1]) / std::fabs(n[1]), 1e-12);
            // Int|J| wird nicht verglichen: sqrt ist nicht polynomial, die Quadraturordnungen (2^3 / 3^3)
            // unterscheiden sich dort, ohne dass die Kernel verschieden waeren; die Groesse wird nicht verwendet.
        }
    }

    // --- Fall 2: verzerrtes Hexaeder, quadratisches Feld
    std::puts("\n[2] verzerrtes Hexaeder, quadratische Felder: Diskretisierungsabweichung");
    Hex dis;
    {
        const double d[8][3] = {{0.0, 0.0, 0.0}, {1.1, 0.0, 0.0}, {1.0, 0.9, 0.1}, {0.1, 1.0, 0.0},
                                {0.0, 0.1, 1.0}, {1.0, 0.0, 1.1}, {1.2, 1.1, 1.0}, {0.0, 1.0, 1.0}};
        // Reihenfolge wie im Referenztest; Knotennummerierung ist die des Loesers
        for (int i = 0; i < 8; ++i)
            dis.nodes[i] = make_double3(0.01 * d[i][0], 0.01 * d[i][1], 0.01 * d[i][2]);
    }
    Field fr = fr_aff, fi = fi_aff;
    fr.s = 40.0; fi.s = -25.0;   // q*x^2 ~ 40 * 1e-4 = 4e-3: gleiche Groessenordnung wie c
    {
        const double3 Fex = exact_force(dis, fr, fi, k);
        const Result r = run_kernels(dis, fr, fi, sigma, omega);
        std::printf("  F_exakt = (%.6e, %.6e, %.6e) N\n", Fex.x, Fex.y, Fex.z);
        std::printf("  Kantenkernel     F_z = %.6e N, Abweichung vom Exakten %.3e\n", r.edge.z, rel(r.edge, Fex));
        std::printf("  Knotenkernel 2^3 F_z = %.6e N, Abweichung vom Exakten %.3e\n", r.nodal2.z, rel(r.nodal2, Fex));
        std::printf("  Knotenkernel 3^3 F_z = %.6e N, Abweichung vom Exakten %.3e\n", r.nodal3.z, rel(r.nodal3, Fex));
        std::printf("  Kanten- gegen Knotenkernel (3^3)               %.3e\n", rel(r.edge, r.nodal3));
        check("Kanten- gegen Knotenkernel (lose Grenze, Elementfehler)", rel(r.edge, r.nodal3), 0.25);
        check("Knotenkernel 2^3 gegen 3^3", rel(r.nodal2, r.nodal3), 0.25);
    }

    std::printf("\n%s (%d Fehlschlaege)\n", g_failures ? "FEHLGESCHLAGEN" : "BESTANDEN", g_failures);
    return g_failures ? 1 : 0;
}
