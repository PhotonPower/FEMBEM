// =============================================================================
// Host-Test fuer include/BemNormalPart.h (bem-normal-components)
//
// Geschlossene Flaeche: Wuerfel (Kantenlaenge 1) aus ebenen Quads, n x n je Seite. Innen liegen
// Punktquellen; das Feld A ist komponentenweise harmonisch und klingt ab:
//   A = R d1/|d1|^3 + c_b d3/|d3|^3  (R dreht um z), d_k = p - p_k,
// Eichverschiebung A' = A + grad psi mit dem Dipol psi = c d2_z/|d2|^3 (harmonisch ausser in p_2, klingt ab).
//
//   [1] Zeile 2 mit exakten Daten: tangential (nur A_tan, (d_n A)_tan) gegen vollstaendig. Das Residuum
//       1/2 A - D A + G d_n A muss im vollstaendigen Fall gegen null konvergieren, im tangentialen nicht.
//   [2] Eichinvarianz: Aenderung des Operanden 1/2 T - D unter A -> A + grad psi, tangential gegen vollstaendig.
//   [3] Gegenproben: Normalanteil mit falschem Vorzeichen, nur D-Anteil, nur G-Anteil.
//   [4] Identitaetsterm: int W_i . (A.n) n = 0 auf ebenen Quads.
// =============================================================================
#include <cmath>
#include <cstdio>
#include <vector>

#include "BemNormalPart.h"

using namespace bem_normal;

static int g_fail = 0;
static void check(bool ok, const char* what, double value, double limit) {
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? "<" : ">=", limit);
    if (!ok) ++g_fail;
}

static std::vector<QuadGeom> cube_quads(int n) {
    std::vector<QuadGeom> qs;
    for (int axis = 0; axis < 3; ++axis) for (int sgn = -1; sgn <= 1; sgn += 2)
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            auto pt = [&](int a, int b) {
                const double s = -0.5 + static_cast<double>(a) / n, t = -0.5 + static_cast<double>(b) / n;
                double c[3];
                c[axis] = 0.5 * sgn; c[(axis + 1) % 3] = s; c[(axis + 2) % 3] = t;
                return make_double3(c[0], c[1], c[2]);
            };
            QuadGeom q;
            q.n[0] = pt(i, j); q.n[1] = pt(i + 1, j); q.n[2] = pt(i + 1, j + 1); q.n[3] = pt(i, j + 1);
            const V3 n0 = to_v3(evaluate_quad_normal(0.0, 0.0, q.n));
            const V3 p0 = quad_point(q.n, 0.0, 0.0);
            q.outward = dot(n0, p0) >= 0.0 ? 1.0 : -1.0;
            qs.push_back(q);
        }
    return qs;
}

static const V3 P1{0.12, -0.08, 0.05}, P2{-0.10, 0.15, -0.07}, P3{0.05, 0.10, 0.12};
static const double CB = 0.5;

// d/rho^3 und seine Richtungsableitung
static V3 unit_field(V3 d) { const double r = norm(d); return (1.0 / (r * r * r)) * d; }
static V3 unit_field_dir(V3 d, V3 v) {
    const double r = norm(d), r3 = r * r * r, r5 = r3 * r * r;
    return (1.0 / r3) * v - (3.0 * dot(d, v) / r5) * d;
}
static V3 rotz(V3 a) { return {-a.y, a.x, 0.0}; }

struct Case { bool shifted; };

static FieldFn make_A(bool shifted, bool with_normal, Gauge gg) {
    return [=](V3 p) {
        V3 a = rotz(unit_field(p - P1));
        if (with_normal) a = a + CB * unit_field(p - P3);
        if (shifted) a = a + gg.grad(p);
        return a;
    };
}
static NormalDerivFn make_dnA(bool shifted, bool with_normal, Gauge gg) {
    return [=](V3 p, V3 n) {
        V3 a = rotz(unit_field_dir(p - P1, n));
        if (with_normal) a = a + CB * unit_field_dir(p - P3, n);
        if (shifted) a = a + gg.hess_times(p, n);
        return a;
    };
}

struct Res { double res_t, res_f, op_t, op_f; std::vector<double> operand_t, operand_f; };

static Res residuals(const Geometry& g, bool shifted, bool with_normal, const Gauge& gg,
                     double sgn_n = 1.0, double wD = 1.0, double wG = 1.0) {
    const NormalDerivFn dn = make_dnA(shifted, with_normal, gg);
    const auto dens = evaluate_densities(g, make_A(shifted, with_normal, gg), dn);
    const NormalPartResult r = apply_kernels(g, dens, &dn);
    const size_t N = r.Tt.size();
    Res o;
    o.operand_t.resize(N); o.operand_f.resize(N);
    double nt = 0, nf = 0, rt = 0, rf = 0;
    for (size_t i = 0; i < N; ++i) {
        const double opt = 0.5 * r.Tt[i] - r.Dt[i];
        const double opf = opt + sgn_n * (0.5 * r.Tn[i] - wD * r.Dn[i]);
        const double rest = opt + r.Gt[i];
        const double resf = opf + r.Gt[i] + sgn_n * wG * r.Gn[i];
        o.operand_t[i] = opt; o.operand_f[i] = opf;
        nt += opt * opt; nf += opf * opf; rt += rest * rest; rf += resf * resf;
    }
    o.op_t = std::sqrt(nt); o.op_f = std::sqrt(nf);
    o.res_t = std::sqrt(rt) / o.op_t; o.res_f = std::sqrt(rf) / o.op_f;
    return o;
}

static double rel_change(const std::vector<double>& a, const std::vector<double>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += (a[i] - b[i]) * (a[i] - b[i]); d += a[i] * a[i]; }
    return std::sqrt(n / d);
}

int main() {
    Gauge gg; gg.kind = 2; gg.c = 1.0; gg.p0 = P2;
    {   // Amplitude: RMS |grad psi| = RMS |A| auf der Flaeche (wie in --bem-row-check: grad psi von der Groessenordnung A)
        const Geometry g0 = build_geometry(cube_quads(4));
        double sa = 0, sg = 0; long n = 0;
        for (const QuadSets& qs : g0.sets) for (const V3& y : qs.far.y) {
            const V3 a = rotz(unit_field(y - P1)) + CB * unit_field(y - P3);
            sa += dot(a, a); const V3 gr = gg.grad(y); sg += dot(gr, gr); ++n;
        }
        gg.c = std::sqrt(sa / sg);
        std::printf("Eichamplitude c = %.4e (RMS |A| = %.3e = RMS |grad psi|)\n", gg.c, std::sqrt(sa / n));
    }

    std::printf("[4] Identitaetsterm: int W_i . (A.n) n auf ebenen Quads\n");
    {
        const Geometry g = build_geometry(cube_quads(4));
        const NormalDerivFn dn = make_dnA(false, true, gg);
        const auto dens = evaluate_densities(g, make_A(false, true, gg), dn);
        const NormalPartResult r = apply_kernels(g, dens, &dn);
        double mn = 0, mt = 0;
        for (size_t i = 0; i < r.Tn.size(); ++i) { mn = std::max(mn, std::fabs(r.Tn[i])); mt = std::max(mt, std::fabs(r.Tt[i])); }
        check(mn < 1e-14 * mt, "max |int W.(A.n)n| gegen max |int W.A_tan|", mn / mt, 1e-14);
    }

    std::printf("[1] Zeile 2 mit exakten Daten am Wuerfel (A mit A.n != 0), relatives Residuum\n");
    double prev_f = 1e9;
    for (int n : {2, 4, 8}) {
        const Geometry g = build_geometry(cube_quads(n));
        const Res r = residuals(g, false, true, gg);
        std::printf("      n = %d (%d Quads): tangential %.3e, vollstaendig %.3e\n", n, 6 * n * n, r.res_t, r.res_f);
        check(r.res_f < prev_f, "vollstaendig: Residuum nimmt mit Verfeinerung ab", r.res_f, prev_f);
        prev_f = r.res_f;
        if (n == 8) {
            check(r.res_f < 0.02, "n = 8: vollstaendiges Residuum < 2 %", r.res_f, 0.02);
            check(r.res_t > 0.1, "n = 8: tangentiales Residuum bleibt gross", r.res_t, 0.1);
            check(r.res_t > 20.0 * r.res_f, "n = 8: tangential > 20 x vollstaendig", r.res_t / r.res_f, 20.0);
        }
    }

    std::printf("[2] Eichverschiebung A -> A + grad psi (Dipol): Residuum mit mitverschobenem K, Operand-Aenderung\n");
    {
        const Geometry g = build_geometry(cube_quads(8));
        for (int wn = 0; wn < 2; ++wn) {
            const Res r0 = residuals(g, false, wn == 1, gg), r1 = residuals(g, true, wn == 1, gg);
            const double dt = rel_change(r1.operand_t, r0.operand_t), df = rel_change(r1.operand_f, r0.operand_f);
            std::printf("      A %s A.n: Residuum tangential %.3e -> %.3e, vollstaendig %.3e -> %.3e; "
                        "Operand-Aenderung tangential %.3e, vollstaendig %.3e\n", wn ? "mit" : "ohne", r0.res_t, r1.res_t,
                        r0.res_f, r1.res_f, dt, df);
            check(r1.res_f < 0.01, "vollstaendig, verschoben: Residuum < 1 %", r1.res_f, 0.01);
            check(r1.res_f < 3.0 * r0.res_f, "vollstaendig: Residuum aendert sich durch die Verschiebung nicht (< 3 x)", r1.res_f / r0.res_f, 3.0);
            check(r1.res_t > 0.2, "tangential, verschoben: Residuum > 20 %", r1.res_t, 0.2);
            check(r1.res_t > 2.0 * r0.res_t, "tangential: Residuum waechst durch die Verschiebung (> 2 x)", r1.res_t / r0.res_t, 2.0);
            // Der Operand (1/2 T - D) A aendert sich auch im vollstaendigen Fall: 1/2 gradpsi - D gradpsi = -G d_n gradpsi,
            // kompensiert durch das mitverschobene K. Erwartung: Aenderung ungleich null (kein Fehler).
            check(df > 0.05, "vollstaendiger Operand aendert sich (kompensiert durch G d_n grad psi)", df, 0.05);
        }
    }

    std::printf("[3] Gegenproben (muessen durchfallen), n = 8, verschoben\n");
    {
        const Geometry g = build_geometry(cube_quads(8));
        const Res ok = residuals(g, true, true, gg);
        const Res neg = residuals(g, true, true, gg, -1.0, 1.0, 1.0);
        const Res onlyG = residuals(g, true, true, gg, 1.0, 0.0, 1.0);
        const Res onlyD = residuals(g, true, true, gg, 1.0, 1.0, 0.0);
        std::printf("      richtig %.3e, Vorzeichen falsch %.3e, nur G-Anteil %.3e, nur D-Anteil %.3e\n", ok.res_f,
                    neg.res_f, onlyG.res_f, onlyD.res_f);
        check(neg.res_f > 10.0 * ok.res_f, "Normalanteil mit falschem Vorzeichen faellt durch", neg.res_f / ok.res_f, 10.0);
        check(onlyG.res_f > 10.0 * ok.res_f, "nur G-Anteil (kein D) faellt durch", onlyG.res_f / ok.res_f, 10.0);
        check(onlyD.res_f > 10.0 * ok.res_f, "nur D-Anteil (kein G) faellt durch", onlyD.res_f / ok.res_f, 10.0);
    }

    std::printf("\n%s (%d Fehler)\n", g_fail ? "FEHLGESCHLAGEN" : "BESTANDEN", g_fail);
    return g_fail ? 1 : 0;
}
