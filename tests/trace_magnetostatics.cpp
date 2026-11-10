// =============================================================================
// Phase 2: Magnetostatik der TEAM-28-Spulen mit der Maxwell-Spur-Kopplung (dicht, Host). Messtreiber, kein Test.
//   trace_magnetostatics <vtk> [--plate-z=<mm>] [--seed=<n>] [--dump=<csv>] [--flip-lambda] [--no-harmonic] [--cm0]
//                        [--no-project] [--no-renorm] [--no-air]
// sigma = 0, mu = mu0 ueberall, Quelle: Innenspule (Komp. 2, N = 960, +) und Aussenspule (Komp. 1, N = 576, -), I = 20 A (Gleichstrom, Betrag
// der Amplitude des Produktionsfalls). Platte (Komp. 3): FEM-Koerper ohne Wirkung. Referenz: Volumenstrom-Biot-Savart (TraceReference.h).
// =============================================================================
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "TraceBemOps.h"
#include "TraceFem.h"
#include "TraceMeshes.h"
#include "TraceReference.h"
#include "TraceSystem.h"

using namespace trace;
static const double PI = 3.14159265358979323846;
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static bool has(int argc, char** argv, const char* f) { for (int i = 2; i < argc; ++i) if (!std::strcmp(argv[i], f)) return true; return false; }
static double opt_val(int argc, char** argv, const char* key, double def)
{
    const size_t n = std::strlen(key);
    for (int i = 2; i < argc; ++i) if (!std::strncmp(argv[i], key, n)) return std::atof(argv[i] + n);
    return def;
}
static std::string opt_str(int argc, char** argv, const char* key)
{
    const size_t n = std::strlen(key);
    for (int i = 2; i < argc; ++i) if (!std::strncmp(argv[i], key, n)) return std::string(argv[i] + n);
    return "";
}

// Meridian- und Laengszyklus einer Spule (rechteckiger Ringquerschnitt) auf der Randflaeche
static void coil_cycles(const FemProblem& F, int comp, std::vector<std::pair<int, int>>& mer, std::vector<std::pair<int, int>>& lon, int* surf_comp)
{
    const TraceSurface& S = F.surface;
    std::vector<char> in_node(F.num_nodes, 0);
    for (int h = 0; h < F.num_hexes; ++h) if (F.hex_comp[h] == comp) for (int k = 0; k < 8; ++k) in_node[F.hex[h][k]] = 1;
    std::vector<int> sn;                                   // Flaechenknoten der Spule
    for (int v = 0; v < S.num_nodes(); ++v) if (in_node[F.surf_node_to_fem[v]]) sn.push_back(v);
    *surf_comp = S.node_comp[sn[0]];
    auto pos = [&](int v) { return F.X[F.surf_node_to_fem[v]]; };
    double th0 = 1e300, rmin = 1e300, zmin = 1e300, rmax = -1e300, zmax = -1e300;
    for (int v : sn) { const Vec3 p = pos(v); th0 = std::min(th0, std::atan2(p.y, p.x)); const double r = std::hypot(p.x, p.y); rmin = std::min(rmin, r); rmax = std::max(rmax, r); zmin = std::min(zmin, p.z); zmax = std::max(zmax, p.z); }
    // Meridian bei kleinstem Azimut: Knoten auf dem Rechteckumfang, nach Polarwinkel um die Mitte im Uhrzeigersinn
    std::vector<std::pair<double, int>> ring;
    const double rc = 0.5 * (rmin + rmax), zc = 0.5 * (zmin + zmax);
    for (int v : sn) { const Vec3 p = pos(v); if (std::fabs(std::atan2(p.y, p.x) - th0) < 1e-7) ring.push_back({-std::atan2(p.z - zc, std::hypot(p.x, p.y) - rc), v}); }
    std::sort(ring.begin(), ring.end());
    std::vector<int> m;
    for (auto& r : ring) m.push_back(r.second);
    mer = cycle_from_nodes(S, m);
    // Laengskreis an der inneren unteren Kante
    std::vector<std::pair<double, int>> corner;
    for (int v : sn) { const Vec3 p = pos(v); if (std::fabs(std::hypot(p.x, p.y) - rmin) < 1e-9 && std::fabs(p.z - zmin) < 1e-9) corner.push_back({std::atan2(p.y, p.x), v}); }
    std::sort(corner.begin(), corner.end());
    std::vector<int> l;
    for (auto& c : corner) l.push_back(c.second);
    lon = cycle_from_nodes(S, l);
    for (const auto& e : mer) if (e.first < 0) throw std::runtime_error("Meridian: Kante fehlt");
    for (const auto& e : lon) if (e.first < 0) throw std::runtime_error("Laengskreis: Kante fehlt");
}

struct Metric { double proj, l2, ratio; };
static Metric metric(const std::vector<Vec3>& t, const std::vector<Vec3>& r, const std::vector<int>& idx)
{
    double tr = 0, rr = 0, dd = 0, tt = 0;
    for (int i : idx) { tr += dot(t[i], r[i]); rr += dot(r[i], r[i]); tt += dot(t[i], t[i]); dd += dot(t[i] - r[i], t[i] - r[i]); }
    return {tr / rr, std::sqrt(dd / rr), std::sqrt(tt / rr)};
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "Aufruf: trace_magnetostatics <vtk> [Optionen]\n"); return 2; }
    const std::string vtk = argv[1];
    const double plate_z = opt_val(argc, argv, "--plate-z=", 11.3);
    const unsigned seed = static_cast<unsigned>(opt_val(argc, argv, "--seed=", 0));
    const double MU0 = 4e-7 * PI, I = 20.0, N_in = 960, N_out = 576, A_in = 1.456e-3, A_out = 7.8e-4;
    double t_all = now();

    // ---- Aufbau ----
    // Plattenhoehe: Datei hat die Platte bei z = 3.8 mm (Kopfzeile z0=3.8)
    std::vector<Vec3> shift(4);
    shift[3] = v3(0, 0, (plate_z - 3.8) * 1e-3);
    FemMaterials mats; mats.mu_r = {1, 1, 1, 1}; mats.kappa = {0, 0, 0, 0};
    FemProblem F = build_fem_problem(vtk, mats, MU0, [&](const Vec3& x, int comp) {
        const double rho = std::hypot(x.x, x.y);
        double j = 0;
        if (comp == 2) j = +N_in * I / A_in; else if (comp == 1) j = -N_out * I / A_out;
        return v3(-j * x.y / rho, j * x.x / rho, 0.0);
    }, 1e-3, &shift);
    TraceSurface& S = F.surface;
    compute_harmonic_fields(S);
    std::printf("Netz %s: %d Hexaeder, %d FEM-Kanten, %d Randquads, %d Flaechenknoten, %d Komponenten (g = %d, %d, %d), nH = %d, Platte bei z = %.1f mm  [Aufbau %.1f s]\n",
                vtk.c_str(), F.num_hexes, F.num_edges, S.num_quads(), S.num_nodes(), S.num_components, S.comp_genus[0], S.comp_genus[1 % S.num_components], S.comp_genus[2 % S.num_components], S.nH, plate_z, now() - t_all);
    int sc_out = -1, sc_in = -1;
    {
        std::vector<std::pair<int, int>> m1, l1, m2, l2;
        coil_cycles(F, 1, m1, l1, &sc_out);
        coil_cycles(F, 2, m2, l2, &sc_in);
        const double s1 = normalize_harmonic_to_cycles(S, sc_out, {m1, l1}), s2 = normalize_harmonic_to_cycles(S, sc_in, {m2, l2});
        std::printf("Zyklen: Aussenspule Meridian %zu Kanten, Laengs %zu Kanten (kleinster Singulaerwert der Zirkulationsmatrix %.3f); Innenspule %zu / %zu (%.3f)\n", m1.size(), l1.size(), s1, m2.size(), l2.size(), s2);
    }
    double t0 = now();
    const BoundaryOps O = assemble_boundary_ops(S);
    const DerivedOps D = derive_ops(S, O);
    const double t_bem = now() - t0;

    // ---- Quelle ----
    const bool do_proj = !has(argc, argv, "--no-project"), do_ren = !has(argc, argv, "--no-renorm");
    std::vector<SourceDiagnostics> sd;
    const std::vector<double> ff = prepare_source(F, vtk, {{1, -N_out * I}, {2, +N_in * I}}, do_proj, do_ren, &sd);
    for (const auto& d : sd)
        std::printf("Quelle Spule %d (%s): %d Schnittschichten, ||G^T f||/||f|| %.2e -> %.2e, Nettostrom vorher %.2f ... %.2f A, nach Projektion %.4f A (Streuung %.1e), Faktor %.6f, final %.6f A (Streuung %.1e; Soll %+.0f A)\n",
                    d.comp, d.comp == 1 ? "aussen" : "innen", d.K, d.rel_gt_before, d.rel_gt_after, d.net_min_before, d.net_max_before, d.net_proj, d.net_proj_spread, d.scale, d.net_final, d.net_final_spread, d.comp == 1 ? -N_out * I : N_in * I);
    const Vec f = Eigen::Map<const Vec>(ff.data(), ff.size());

    // ---- Loesung ----
    SchurOptions so;
    if (seed != 0) so.gauged = make_tree_gauge(F, seed);
    else so.gauged = make_tree_gauge(F, 0);
    so.flip_lambda = has(argc, argv, "--flip-lambda");
    if (has(argc, argv, "--no-harmonic")) so.drop_c.assign(S.nH, 1);
    if (has(argc, argv, "--cm0")) {   // c_Meridian beider Spulen := 0 (Spalte 0 jeder Spulenkomponente)
        so.drop_c.assign(S.nH, 0);
        std::set<int> seen;
        for (int k = 0; k < S.nH; ++k) if ((S.H_comp[k] == sc_out || S.H_comp[k] == sc_in) && !seen.count(S.H_comp[k])) { so.drop_c[k] = 1; seen.insert(S.H_comp[k]); }
    }
    const SchurResult R = solve_schur(F, O, D, f, so);
    std::printf("Loesung: n_innen %d, n_Gamma = %d (a) + %d (psi) + %d (c) = %d, Schurkomplement %.2f GB, Residuum %.2e (skaliertes System), Baum seed %u (%d Baumkanten)\n",
                R.nI, R.nGa, R.nPsi, R.nC, R.nGa + R.nPsi + R.nC, R.schur_gb, R.residual_rel, seed, static_cast<int>(std::count(so.gauged.begin(), so.gauged.end(), 1)));
    std::printf("Laufzeit: BEM-Operatoren %.1f s, Elimination %.1f s, Schurkomplement %.1f s, Loesen %.1f s\n", t_bem, R.t_eliminate, R.t_schur, R.t_solve);
    for (int sc : {sc_out, sc_in}) {
        int first = -1, second = -1;
        for (int k = 0; k < S.nH; ++k) if (S.H_comp[k] == sc) { if (first < 0) first = k; else second = k; }
        const double target = (sc == sc_out) ? -N_out * I : N_in * I;
        std::printf("Spule %s: c_Meridian = %+.4f A (Soll %+.0f A, rel. Abweichung %.2e), c_Laengs = %+.3e A (%.2e von N I)\n", sc == sc_out ? "aussen" : "innen", R.c[first], target, std::fabs(R.c[first] / target - 1), R.c[second], std::fabs(R.c[second] / target));
    }

    // ---- Feld in den Hexaedern gegen Volumenstrom-Biot-Savart ----
    const std::vector<RingCoil> coils = {{27e-3, 55e-3, -52e-3, 0.0, N_in, +I}, {80e-3, 95e-3, -52e-3, 0.0, N_out, -I}};
    const std::vector<double> a(R.a.data(), R.a.data() + R.a.size());
    std::vector<Vec3> Bh(F.num_hexes), Br(F.num_hexes);
    t0 = now();
    for (int h = 0; h < F.num_hexes; ++h) { Bh[h] = F.curl_at_center(h, a); Br[h] = ring_B(coils, F.center(h)); }
    std::printf("Referenz an %d Elementmitten: %.1f s\n", F.num_hexes, now() - t0);
    std::map<int, std::vector<int>> by;
    for (int h = 0; h < F.num_hexes; ++h) by[F.hex_comp[h]].push_back(h);
    const char* names[4] = {"", "Aussenspule", "Innenspule", "Platte"};
    std::printf("Feld (curl a) gegen Volumenstrom-Biot-Savart, Elementmitten (proj = <B,Bref>/<Bref,Bref>, L2 = |B-Bref|/|Bref|, |B|/|Bref|):\n");
    for (int c : {3, 2, 1}) { const Metric m = metric(Bh, Br, by[c]); std::printf("  %-12s n = %5zu  proj %.4f  L2 %.4f  |B|/|Bref| %.4f\n", names[c], by[c].size(), m.proj, m.l2, m.ratio); }
    {   // Platte: B_z je Radiusbin (wie force-excess-localization.md)
        const double edges[5] = {0.0, 0.02, 0.04, 0.06, 0.10};
        std::printf("  Platte, B_z je Radiusbin [m]: n, B_z Biot-Savart, B_z gerechnet, Verhaeltnis\n");
        for (int b = 0; b < 4; ++b) {
            double sh = 0, sr = 0; int n = 0;
            for (int h : by[3]) { const Vec3 c = F.center(h); const double r = std::hypot(c.x, c.y); if (r >= edges[b] && r < edges[b + 1]) { sh += Bh[h].z; sr += Br[h].z; ++n; } }
            if (n) std::printf("    %.2f bis %.2f  n = %4d  %+.5f  %+.5f  %.4f\n", edges[b], edges[b + 1], n, sr / n, sh / n, sh / sr);
        }
        // Zentrum / Rueckfluss / Rand: Mittel ueber Bins
    }

    // ---- Luftraum ueber die Darstellungsformel (R-B) ----
    if (!has(argc, argv, "--no-air")) {
        Vec psi = R.psi;
        Vec lam = -(D.G * psi) + D.H * R.c;
        std::vector<double> lamv(lam.data(), lam.data() + lam.size()), at(S.num_edges()), g(S.num_quads()), nA(S.num_quads(), 0.0);
        for (int e = 0; e < S.num_edges(); ++e) at[e] = a[F.surf_edge_to_fem[e]];
        const Vec as = Eigen::Map<const Vec>(at.data(), at.size());
        const Vec flux = D.R * as;
        for (int q = 0; q < S.num_quads(); ++q) g[q] = flux[q] / O.area[q];
        const double zp = plate_z * 1e-3, tp = 3e-3;
        struct Pt { const char* label; Vec3 x; };
        std::vector<Pt> pts;
        for (double d : {1e-3, 5e-3, 20e-3}) {
            char* l;
            l = new char[64]; std::snprintf(l, 64, "Platte oben, Achse, %2.0f mm", d * 1e3); pts.push_back({l, v3(0, 0, zp + tp + d)});
            l = new char[64]; std::snprintf(l, 64, "Platte Seite (rho=65+d), %2.0f mm", d * 1e3); pts.push_back({l, v3(65e-3 + d, 0, zp + 1.5e-3)});
            l = new char[64]; std::snprintf(l, 64, "Platte unten (rho=70), %2.0f mm", d * 1e3); pts.push_back({l, v3(70e-3, 0, zp - d)});
            l = new char[64]; std::snprintf(l, 64, "Innenspule aussen, %2.0f mm", d * 1e3); pts.push_back({l, v3(55e-3 + d, 0, -26e-3)});
            l = new char[64]; std::snprintf(l, 64, "Innenspule innen (Loch), %2.0f mm", d * 1e3); pts.push_back({l, v3(27e-3 - d, 0, -26e-3)});
            l = new char[64]; std::snprintf(l, 64, "Aussenspule aussen, %2.0f mm", d * 1e3); pts.push_back({l, v3(95e-3 + d, 0, -26e-3)});
            l = new char[64]; std::snprintf(l, 64, "Spulen unten (rho=41), %2.0f mm", d * 1e3); pts.push_back({l, v3(41e-3, 0, -52e-3 - d)});
        }
        pts.push_back({"Achse z = -26 mm", v3(0, 0, -26e-3)});
        pts.push_back({"Achse z = +30 mm", v3(0, 0, 30e-3)});
        pts.push_back({"Achse z = +60 mm", v3(0, 0, 60e-3)});
        std::printf("Luftraum (R-B) aus lambda = -G psi + H c und g = R a / |q|: Punkt, |B_ref| [T], relativer Fehler |B - B_ref|/|B_ref|\n");
        t0 = now();
        double worst = 0;
        for (const Pt& p : pts) {
            const FieldPoint fp = evaluate_exterior(S, lamv, at, g, nA, p.x, MU0, ExteriorTerms(), 6, 6);
            const Vec3 br = ring_B(coils, p.x);
            const double err = norm(fp.B - br) / norm(br);
            std::printf("  %-36s %.5e  %.3e\n", p.label, norm(br), err);
            worst = std::max(worst, err);
        }
        {   // Quadraturkontrolle der Darstellungsformel am naechsten Punkt
            const Vec3 x = v3(0, 0, zp + tp + 1e-3);
            const FieldPoint a6 = evaluate_exterior(S, lamv, at, g, nA, x, MU0, ExteriorTerms(), 6, 6), a8 = evaluate_exterior(S, lamv, at, g, nA, x, MU0, ExteriorTerms(), 8, 10);
            std::printf("  Quadraturkontrolle Auswertung (1 mm ueber der Platte): |B(6x6) - B(8x10)|/|B| = %.2e  [%.1f s fuer %zu Punkte]\n", norm(a6.B - a8.B) / norm(a8.B), now() - t0, pts.size());
        }
        std::printf("  groesster Fehler im Luftraum: %.3e\n", worst);
    }

    const std::string dump = opt_str(argc, argv, "--dump=");
    if (!dump.empty()) {   // Dump im Format von tools/compare_fields.py (SI, Re/Im, body 1 = Platte, 2 = Innenspule, 3 = Aussenspule)
        FILE* fp = std::fopen(dump.c_str(), "w");
        std::fprintf(fp, "elem,body,x,y,z,Arx,Ary,Arz,Aix,Aiy,Aiz,Brx,Bry,Brz,Bix,Biy,Biz\n");
        for (int h = 0; h < F.num_hexes; ++h) {
            const Vec3 c = F.center(h), A = F.a_at_center(h, a);
            const int body = F.hex_comp[h] == 3 ? 1 : (F.hex_comp[h] == 2 ? 2 : 3);
            std::fprintf(fp, "%d,%d,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,0,0,0,%.9g,%.9g,%.9g,0,0,0\n", h, body, c.x, c.y, c.z, A.x, A.y, A.z, Bh[h].x, Bh[h].y, Bh[h].z);
        }
        std::fclose(fp);
        std::printf("Dump: %s\n", dump.c_str());
    }
    std::printf("Gesamtlaufzeit %.1f s\n", now() - t_all);
    return 0;
}
