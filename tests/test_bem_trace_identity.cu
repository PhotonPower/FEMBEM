// =============================================================================
// Host-Test: Identitaet d = mu0 (K2 - K1) = - grad_G(A.n) + C am exakten
// Biot-Savart-Feld (bem-trace-consistency, Schritt 1)
//
//   [1] glatte Flaechen mit analytischer Normale: Zylinder, Kugel, Ebene z = const,
//       gekippte Ebene. d (direkt aus K1, K2) gegen grad- plus Kruemmungsterm.
//       Zylinder und Kugel: d gegen geschlossene Formel A_tan / R.
//       Ebene z = const: d = 0.
//   [2] Gegenproben: falsches Vorzeichen des Kruemmungsterms bzw. fehlender
//       grad_G(A.n)-Term muessen durchfallen.
//   [3] facettierter Zylinder (ebene Quads, W = 0): Identitaet exakt, und die
//       Flaechen-L2-Abweichung von d gegen die Formel der glatten Flaeche nimmt
//       unter Verfeinerung etwa linear ab. (Im Koeffizientenraum stagniert sie:
//       die langen z-Kanten tragen d_z ~ B_r*delta mit Kantenlaenge als Gewicht,
//       das skaliert wie die kurzen phi-Kanten; deshalb nur als Info.)
//   [4] Rooftop-Projektion der Terme (quad_trace_projection) gegen [3].
//
// Reiner Host-Code, keine GPU, kein Netz.
// =============================================================================
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "BemRowCheck.h"

using namespace bem_row_check;

static int g_fail = 0;
static void check(bool ok, const char* what, double value, double limit) {
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? "<" : ">=", limit);
    if (!ok) ++g_fail;
}

struct Surf { V3 p, xu, xv, n, nu, nv; };

static Surf cylinder(double R, double s, double ph, double z) {
    const V3 er{std::cos(ph), std::sin(ph), 0}, ep{-std::sin(ph), std::cos(ph), 0};
    return {R * er + V3{0, 0, z}, R * ep, {0, 0, 1}, s * er, s * ep, {0, 0, 0}};
}
static Surf sphere(double R, double th, double ph) {   // n nach aussen
    const V3 er{std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph), std::cos(th)};
    const V3 et{std::cos(th) * std::cos(ph), std::cos(th) * std::sin(ph), -std::sin(th)};
    const V3 ep{-std::sin(ph), std::cos(ph), 0};
    // x_u = R e_theta, x_v = R sin(th) e_phi, n = e_r, n_u = e_theta, n_v = sin(th) e_phi
    return {R * er, R * et, R * std::sin(th) * ep, er, et, std::sin(th) * ep};
}
static Surf plane_z(double z0, double s, double x, double y) {
    return {{x, y, z0}, {1, 0, 0}, {0, 1, 0}, {0, 0, s}, {0, 0, 0}, {0, 0, 0}};
}
static Surf tilted_plane(double x, double y) {          // Ebene durch (0,0,0.02), Normale (1,0,2)/sqrt5
    const double c = 1.0 / std::sqrt(5.0);
    const V3 n{c, 0, 2 * c};
    const V3 t1{2 * c, 0, -c};
    return {V3{x * t1.x, y, 0.02 + x * t1.z}, t1, {0, 1, 0}, n, {0, 0, 0}, {0, 0, 0}};
}

struct Dev { double rel = 0; double dn = 0; double kn = 0; };

// relative Abweichung |d - (wg grad + wc curv)| / |d|, gemittelt ueber Punkte (RMS)
template <class Gen>
static Dev deviation(const std::vector<Filament>& fil, const Gen& gen, int N, double wg, double wc) {
    double num = 0, den = 0, k2 = 0;
    for (int i = 0; i < N; ++i) {
        const Surf sf = gen(i);
        const ExactField ef = exact_field(fil, sf.p);
        const TraceTerms t = trace_terms(ef, sf.xu, sf.xv, sf.n, sf.nu, sf.nv);
        const V3 pred = wg * t.grad + wc * t.curv;
        num += dot(t.d - pred, t.d - pred);
        den += dot(t.d, t.d);
        k2 += dot(t.K2, t.K2);
    }
    return {std::sqrt(num / (den > 0 ? den : 1.0)), std::sqrt(den), std::sqrt(k2)};
}

int main() {
    const std::vector<Filament> fil = make_filaments(team28_coils(), 20.0, 2e-3);
    std::printf("Quelle: %zu Faeden\n", fil.size());
    const int N = 200;

    std::printf("[1] glatte Flaechen: d gegen grad- plus Kruemmungsterm\n");
    auto cyl_o = [&](int i) { return cylinder(0.060, +1, 0.37 + 0.11 * i, -0.040 + 0.00015 * i); };
    auto cyl_i = [&](int i) { return cylinder(0.060, -1, 0.37 + 0.11 * i, -0.040 + 0.00015 * i); };
    auto sph   = [&](int i) { return sphere(0.020, 0.4 + 0.01 * i, 0.2 + 0.07 * i); };
    auto pln   = [&](int i) { return plane_z(0.02, +1, 0.003 * (i % 20) - 0.03, 0.002 * (i / 20)); };
    auto tpl   = [&](int i) { return tilted_plane(0.001 * (i % 20) - 0.01, 0.002 * (i / 20) - 0.01); };

    const Dev dc = deviation(fil, cyl_o, N, 1, 1), di = deviation(fil, cyl_i, N, 1, 1);
    const Dev ds = deviation(fil, sph, N, 1, 1), dp = deviation(fil, pln, N, 1, 1),
              dt = deviation(fil, tpl, N, 1, 1);
    check(dc.rel < 1e-10, "Zylinder n=+e_r:   |d - Vorhersage|/|d|", dc.rel, 1e-10);
    check(di.rel < 1e-10, "Zylinder n=-e_r:   |d - Vorhersage|/|d|", di.rel, 1e-10);
    check(ds.rel < 1e-10, "Kugel:             |d - Vorhersage|/|d|", ds.rel, 1e-10);
    check(dt.rel < 1e-10, "gekippte Ebene:    |d - Vorhersage|/|d|", dt.rel, 1e-10);
    check(dp.dn < 1e-12 * dp.kn, "Ebene z=const:     |d| gegen |K2|", dp.dn / dp.kn, 1e-12);

    // geschlossene Formeln: Zylinder d = +-A_phi/R e_phi, Kugel d = A_tan / R
    double ez = 0, es = 0, nz = 0, ns = 0;
    for (int i = 0; i < N; ++i) {
        { const Surf sf = cyl_o(i); const ExactField ef = exact_field(fil, sf.p);
          const TraceTerms t = trace_terms(ef, sf.xu, sf.xv, sf.n, sf.nu, sf.nv);
          const V3 ex = (ef.Aphi / 0.060) * ef.e_phi; ez += dot(t.d - ex, t.d - ex); nz += dot(ex, ex); }
        { const Surf sf = cyl_i(i); const ExactField ef = exact_field(fil, sf.p);
          const TraceTerms t = trace_terms(ef, sf.xu, sf.xv, sf.n, sf.nu, sf.nv);
          const V3 ex = (-ef.Aphi / 0.060) * ef.e_phi; ez += dot(t.d - ex, t.d - ex); nz += dot(ex, ex); }
        { const Surf sf = sph(i); const ExactField ef = exact_field(fil, sf.p);
          const TraceTerms t = trace_terms(ef, sf.xu, sf.xv, sf.n, sf.nu, sf.nv);
          const V3 ex = (1.0 / 0.020) * ef.A; es += dot(t.d - ex, t.d - ex); ns += dot(ex, ex); }
    }
    check(std::sqrt(ez / nz) < 1e-10, "Zylinder: d = +-A_phi/R e_phi (geschlossen)", std::sqrt(ez / nz), 1e-10);
    check(std::sqrt(es / ns) < 1e-10, "Kugel:    d = A_tan/R (geschlossen)", std::sqrt(es / ns), 1e-10);
    std::printf("      |d|/|K2|: Zylinder %.3f, Kugel %.3f, gekippte Ebene %.3f\n", dc.dn / dc.kn, ds.dn / ds.kn,
                dt.dn / dt.kn);

    std::printf("[2] Gegenproben (muessen durchfallen)\n");
    const Dev c1 = deviation(fil, cyl_o, N, 1, -1), c2 = deviation(fil, sph, N, 1, -1);
    const Dev c3 = deviation(fil, tpl, N, 0, 1), c4 = deviation(fil, tpl, N, -1, 1);
    check(c1.rel > 0.5, "Zylinder, Vorzeichen des Kruemmungsterms falsch: Abweichung gross", c1.rel, 0.5);
    check(c2.rel > 0.5, "Kugel,    Vorzeichen des Kruemmungsterms falsch: Abweichung gross", c2.rel, 0.5);
    check(c3.rel > 0.5, "gekippte Ebene, grad_G(A.n) vergessen: Abweichung gross", c3.rel, 0.5);
    check(c4.rel > 0.5, "gekippte Ebene, Vorzeichen von grad_G(A.n) falsch: Abweichung gross", c4.rel, 0.5);

    std::printf("[3] facettierter Zylinder R = 60 mm, z -40..-10 mm\n");
    const double R = 0.060, z0 = -0.040, z1 = -0.010;
    double prev = 0;
    for (int nf = 12; nf <= 192; nf *= 2) {
        double num = 0, den = 0, k2n = 0, dn2 = 0, idn = 0, idd = 0, l2n = 0, l2d = 0;
        for (int k = 0; k < nf; ++k) {
            const double p0 = 2 * kPi * k / nf, p1 = 2 * kPi * (k + 1) / nf;
            QuadGeom q;
            q.n[0] = {R * std::cos(p0), R * std::sin(p0), z0};
            q.n[1] = {R * std::cos(p1), R * std::sin(p1), z0};
            q.n[2] = {R * std::cos(p1), R * std::sin(p1), z1};
            q.n[3] = {R * std::cos(p0), R * std::sin(p0), z1};
            q.outward = 1.0;
            QuadTrace qt;
            if (!quad_trace_projection(q, fil, 1.0, 1.0, qt)) { check(false, "Projektion", 0, 0); continue; }
            // glatte Referenz: d_s = A_phi(p)/R e_phi, auf die Rooftops projiziert
            double cs[4];
            project_to_rooftop(q, [&](V3 p, V3) {
                const ExactField ef = exact_field(fil, p);
                return (ef.Aphi / R) * ef.e_phi; }, cs);
            {   // Flaechen-L2-Mass: punktweise d_facet - d_glatt, 16-Punkt-Gauss
                double xg[4], wg[4];
                gauss4(xg, wg);
                for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
                    const QuadFrame fr = quad_frame(q, xg[a], xg[b]);
                    const ExactField ef = exact_field(fil, fr.p);
                    const TraceTerms t = trace_terms(ef, fr.xu, fr.xv, fr.n, fr.nu, fr.nv);
                    const V3 ds = (ef.Aphi / R) * ef.e_phi;
                    const double dS = wg[a] * wg[b] * norm(cross(fr.xu, fr.xv));
                    l2n += dS * dot(t.d - ds, t.d - ds);
                    l2d += dS * dot(ds, ds);
                }
            }
            for (int i = 0; i < 4; ++i) {
                num += (qt.d[i] - cs[i]) * (qt.d[i] - cs[i]);
                den += cs[i] * cs[i];
                k2n += qt.k2[i] * qt.k2[i];
                dn2 += qt.d[i] * qt.d[i];
                idn += (qt.d[i] - qt.pred[i]) * (qt.d[i] - qt.pred[i]);
                idd += qt.d[i] * qt.d[i];
            }
        }
        const double err = std::sqrt(l2n / l2d);
        std::printf("      %3d Facetten: L2-Abweichung d_facet gegen d_glatt = %.3e%s, "
                    "Koeffizientenraum %.3e (Info), ||d - Vorhersage||/||d|| = %.2e, ||d||/||K2|| = %.3f\n", nf, err,
                    prev > 0 ? (std::string("  (Faktor ") + std::to_string(prev / err).substr(0, 4) + ")").c_str() : "",
                    std::sqrt(num / den), std::sqrt(idn / idd), std::sqrt(dn2 / k2n));
        check(std::sqrt(idn / idd) < 1e-10, "Facetten: Identitaet auf den Rooftop-Koeffizienten", std::sqrt(idn / idd), 1e-10);
        if (prev > 0) check(err < prev, "Facetten: Abweichung von der glatten Formel nimmt ab", err, prev);
        prev = err;
    }
    check(prev < 0.02, "192 Facetten: L2-Abweichung < 2 %", prev, 0.02);

    std::printf("[5] Eichverschiebung A -> A + grad psi (Schritt 2): Identitaet, K2 invariant, K1 verschoben\n");
    {
        // Gauge selbst: grad psi gegen zentrale Differenz von psi, H v gegen zentrale Differenz von grad psi
        double eg = 0, eh = 0, ng = 0, nh = 0;
        for (int kind = 0; kind < 4; ++kind) {
            Gauge gg; gg.kind = kind; gg.c = kind == 2 ? 3.0e-6 : 3.0;
            for (int i = 0; i < 20; ++i) {
                const V3 p{0.01 * i - 0.05, 0.02 - 0.003 * i, 0.03 + 0.004 * i}, v{0.3, -0.7, 0.5};
                const double h = 1e-6;
                const V3 gx = gg.grad(p);
                V3 fd{(gg.psi(p + V3{h, 0, 0}) - gg.psi(p - V3{h, 0, 0})) / (2 * h),
                      (gg.psi(p + V3{0, h, 0}) - gg.psi(p - V3{0, h, 0})) / (2 * h),
                      (gg.psi(p + V3{0, 0, h}) - gg.psi(p - V3{0, 0, h})) / (2 * h)};
                eg += dot(gx - fd, gx - fd); ng += dot(gx, gx);
                const V3 hv = gg.hess_times(p, v);
                const V3 fh = (1.0 / (2 * h)) * (gg.grad(p + h * v) - gg.grad(p - h * v));
                eh += dot(hv - fh, hv - fh); nh += dot(hv, hv);
            }
        }
        check(std::sqrt(eg / ng) < 1e-8, "psi: grad gegen zentrale Differenz", std::sqrt(eg / ng), 1e-8);
        check(std::sqrt(eh / nh) < 1e-6, "psi: H v gegen zentrale Differenz von grad psi", std::sqrt(eh / nh), 1e-6);
    }
    for (int kind = 0; kind < 4; ++kind) {
        Gauge gg; gg.kind = kind; gg.c = kind == 2 ? 1.0e-6 : 5.0e-3;
        double worst_id = 0, k2max = 0, k2dev = 0, k1shift = 0, wrong_old = 0, den = 0;
        auto run = [&](auto gen, int n) {
            for (int i = 0; i < n; ++i) {
                const Surf sf = gen(i);
                const ExactField ef = exact_field(fil, sf.p);
                const TraceTerms t0 = trace_terms(ef, sf.xu, sf.xv, sf.n, sf.nu, sf.nv);
                const TraceTerms t1 = trace_terms_g(ef, sf.p, gg, sf.xu, sf.xv, sf.n, sf.nu, sf.nv);
                worst_id = std::max(worst_id, norm(t1.d - (t1.grad + t1.curv)));
                k2max = std::max(k2max, norm(t1.K2));
                k2dev = std::max(k2dev, norm(t1.K2 - t0.K2));
                const V3 hn = gg.hess_times(sf.p, sf.n);
                const V3 exp_shift = hn - dot(hn, sf.n) * sf.n;               // tang (H n)
                k1shift += dot(t1.K1 - t0.K1 - exp_shift, t1.K1 - t0.K1 - exp_shift);
                den += dot(exp_shift, exp_shift);
                // Gegenprobe: Vorhersage mit dem UNverschobenen A.n
                wrong_old += dot(t1.d - (t0.grad + t0.curv), t1.d - (t0.grad + t0.curv));
            }
        };
        run(cyl_o, N); run(sph, N); run(pln, N); run(tpl, N);
        worst_id /= k2max; k2dev /= k2max;     // absolute Abweichungen gegen max |K2| (auf der Achse ist K2 = 0)
        std::printf("  psi kind %d (c = %.1e):\n", kind, gg.c);
        check(worst_id < 1e-10, "  |d' - (grad + curv)'| / |K2| (alle Flaechen)", worst_id, 1e-10);
        check(k2dev < 1e-12, "  K2 invariant", k2dev, 1e-12);
        check(std::sqrt(k1shift / den) < 1e-10, "  K1' - K1 = tang(H n)", std::sqrt(k1shift / den), 1e-10);
        check(std::sqrt(wrong_old / (den > 0 ? den : 1)) > 0.5,
              "  Gegenprobe: Vorhersage mit unverschobenem A.n faellt durch (Abweichung gegen Verschiebung)",
              std::sqrt(wrong_old / den), 0.5);
    }

    std::printf("\n%s (%d Fehler)\n", g_fail ? "FEHLGESCHLAGEN" : "BESTANDEN", g_fail);
    return g_fail ? 1 : 0;
}
