// =============================================================================
// Phase 1b, Schritt 4: gekoppeltes System (S) und (J) auf kleinen Netzen, dicht.
//   [T8]  Kugel mit Azimutalstrom (FEM-Kugel, BEM-Aussenraum) gegen die exakte Loesung, Konvergenzordnung
//   [T6]  Symmetrie von (S), Unsymmetrie von (J)
//   [T7]  Nullraum per SVD (Ring mit sigma = 0 + Platte mit sigma > 0, K = 2, g = 1)
//   [T4]  Eichtest a -> a + G psi_h (nichtharmonisch)
//   [R3]  diskret divergenzfreie Quelle, c_Meridian gegen Schnittscheibe
// Reiner Host-Test (Netze werden als VTK geschrieben und ueber den Bestand gelesen), mu0 = 1.
// =============================================================================
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "TraceHexMeshes.h"
#include "TraceBemOps.h"
#include "TraceFem.h"
#include "TraceSystem.h"

using namespace trace;

static int g_fail = 0;
static void check(bool ok, const char* what, double value, double limit)
{
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? "<" : ">=", limit);
    if (!ok) ++g_fail;
}
static void check_ge(bool ok, const char* what, double value, double limit)
{
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? ">=" : "<", limit);
    if (!ok) ++g_fail;
}
static const double PI = 3.14159265358979323846;
static const double MU0 = 1.0;
static double order(double a, double b) { return std::log(a / b) / std::log(2.0); }

// ---------------- T8: Kugel mit Azimutalstrom ----------------
// J_s = j0 (z x r) = j0 (-y, x, 0). Exakt (mu0 = 1, R = 1): A = a(r) sin(th) e_phi, a_in = alpha r - j0 r^3/10, a_out = beta r^-2,
// alpha = j0/6, beta = j0/15. Spuren: psi = beta cos(th), g = 2 beta cos(th).
struct BallResult { double e_psi, e_g, e_B, e_B_fem, d_B; };

// Loest die Kugelaufgabe mit den Optionen und liefert die Fehler gegen die exakte Loesung
static BallResult solve_ball(int n, CoupledOptions opt, bool project, double* gradf_before = nullptr, double* gradf_after = nullptr, int* ngauged = nullptr, int* nquads = nullptr)
{
    const double j0 = 1.0, beta = j0 / 15, alpha = j0 / 6;
    const std::string path = "trace_ball_" + std::to_string(n) + ".vtk";
    write_vtk(path, ball_hex(n, 1.0), "ball");
    FemMaterials mats;
    mats.mu_r = {1.0, 1.0}; mats.kappa = {0.0, 0.0};
    FemProblem F = build_fem_problem(path, mats, MU0, [&](const Vec3& x, int) { return v3(-j0 * x.y, j0 * x.x, 0.0); });
    if (gradf_before) {
        std::vector<double> gt(F.num_nodes, 0.0); double fn = 0;
        for (int e = 0; e < F.num_edges; ++e) { gt[F.edges[e][1]] += F.f[e]; gt[F.edges[e][0]] -= F.f[e]; fn += F.f[e] * F.f[e]; }
        double s = 0; for (double v : gt) s += v * v;
        *gradf_before = std::sqrt(s / fn);
    }
    if (project) { const ProjectionResult pr = project_source(path, F.f); if (gradf_after) *gradf_after = pr.rel_after; }
    if (ngauged) *ngauged = F.num_gauged;
    TraceSurface& S = F.surface;
    if (nquads) *nquads = S.num_quads();
    compute_harmonic_fields(S);
    const BoundaryOps O = assemble_boundary_ops(S);
    const DerivedOps D = derive_ops(S, O);
    const Vec f = Eigen::Map<const Vec>(F.f.data(), F.f.size());
    opt.sigma_factor = 0.0;
    const Coupled C = build_coupled(F, O, D, f, opt);
    const Vec x = C.solve();
    const Vec a = x.head(C.nE), psi = x.segment(C.nE, C.nV);
    Vec psi_ex(C.nV);
    for (int i = 0; i < C.nV; ++i) psi_ex[i] = beta * S.X[i].z / norm(S.X[i]);
    const Vec ps = psi - Vec::Constant(C.nV, psi.mean()), pe = psi_ex - Vec::Constant(C.nV, psi_ex.mean());
    BallResult r;
    r.e_psi = (ps - pe).norm() / pe.norm();
    std::vector<double> asurf(S.num_edges());
    for (int e = 0; e < S.num_edges(); ++e) asurf[e] = a[F.surf_edge_to_fem[e]];
    const Vec flux = D.R * Eigen::Map<const Vec>(asurf.data(), asurf.size());
    const auto gex = quad_average(S, [&](const Vec3& x, const Vec3&) { return 2 * beta * x.z / norm(x); });
    double dg = 0, ng = 0;
    for (int q = 0; q < S.num_quads(); ++q) { const double g = flux[q] / O.area[q]; dg += O.area[q] * (g - gex[q]) * (g - gex[q]); ng += O.area[q] * gex[q] * gex[q]; }
    r.e_g = std::sqrt(dg / ng);
    double dB = 0, nB = 0;
    const std::vector<double> av(a.data(), a.data() + a.size());
    for (int h = 0; h < F.num_hexes; ++h) {
        const Vec3 c = F.center(h);
        const double rr = norm(c);
        if (rr < 1e-12) continue;
        const double ct = c.z / rr, st = std::sqrt(std::max(0.0, 1 - ct * ct));
        const double ar = alpha * rr - j0 * rr * rr * rr / 10, rap = 2 * alpha * rr - 0.4 * j0 * rr * rr * rr;
        const double Br = 2 * ar * ct / rr, Bth = -rap * st / rr;
        const double rho = std::sqrt(c.x * c.x + c.y * c.y);
        const double cp = rho > 0 ? c.x / rho : 1, sp = rho > 0 ? c.y / rho : 0;
        const Vec3 Bex = Br * ((1.0 / rr) * c) + Bth * v3(ct * cp, ct * sp, -st);
        const Vec3 Bh = F.curl_at_center(h, av);
        dB += dot(Bh - Bex, Bh - Bex); nB += dot(Bex, Bex);
    }
    r.e_B = std::sqrt(dB / nB);

    // Referenz: reines FEM-Problem mit EXAKTEN Dirichlet-Daten auf dem Rand (gleiche Quelle, gleiches Netz); dort steckt der reine
    // FEM-Diskretisierungsfehler, die Kopplung ist nicht beteiligt. Gauge durch kleine Regularisierung (konsistentes, singulaeres System).
    {
        const int E = F.num_edges;
        auto Aex = [&](const Vec3& x) { const double r2 = dot(x, x), c = alpha - j0 * r2 / 10; return v3(-c * x.y, c * x.x, 0.0); };
        static const double gx[4] = {0.0694318442029737, 0.3300094782075719, 0.6699905217924281, 0.9305681557970263},
                            gw[4] = {0.1739274225687269, 0.3260725774312731, 0.3260725774312731, 0.1739274225687269};
        auto line = [&](int e) {
            const Vec3 p = F.X[F.edges[e][0]], q = F.X[F.edges[e][1]];
            double sum = 0;
            for (int i = 0; i < 4; ++i) sum += gw[i] * dot(Aex(p + gx[i] * (q - p)), q - p);
            return sum;
        };
        std::vector<char> bnd(E, 0);
        for (int e : F.surf_edge_to_fem) bnd[e] = 1;
        const Mat K = dense_K(F, 0.0);
        Vec afem = Vec::Zero(E);
        std::vector<int> in;
        for (int e = 0; e < E; ++e) { if (bnd[e]) afem[e] = line(e); else in.push_back(e); }
        const int ni = static_cast<int>(in.size());
        Mat Kii(ni, ni);
        Vec rhs(ni);
        for (int i = 0; i < ni; ++i) {
            double sum = f[in[i]];
            for (int e = 0; e < E; ++e) if (bnd[e]) sum -= K(in[i], e) * afem[e];
            rhs[i] = sum;
            for (int j = 0; j < ni; ++j) Kii(i, j) = K(in[i], in[j]);
        }
        const double eps = 1e-9 * Kii.diagonal().mean();
        for (int i = 0; i < ni; ++i) Kii(i, i) += eps;
        const Vec ai = Kii.ldlt().solve(rhs);
        for (int i = 0; i < ni; ++i) afem[in[i]] = ai[i];
        const std::vector<double> avf(afem.data(), afem.data() + E);
        double dBf = 0, dBd = 0;
        for (int h = 0; h < F.num_hexes; ++h) {
            const Vec3 c = F.center(h);
            const double rr = norm(c);
            if (rr < 1e-12) continue;
            const double ct = c.z / rr, st = std::sqrt(std::max(0.0, 1 - ct * ct));
            const double ar = alpha * rr - j0 * rr * rr * rr / 10, rap = 2 * alpha * rr - 0.4 * j0 * rr * rr * rr;
            const double rho = std::sqrt(c.x * c.x + c.y * c.y);
            const double cp = rho > 0 ? c.x / rho : 1, sp = rho > 0 ? c.y / rho : 0;
            const Vec3 Bex = (2 * ar * ct / rr) * ((1.0 / rr) * c) + (-rap * st / rr) * v3(ct * cp, ct * sp, -st);
            const Vec3 Bf = F.curl_at_center(h, avf), Bh = F.curl_at_center(h, av);
            dBf += dot(Bf - Bex, Bf - Bex); dBd += dot(Bh - Bf, Bh - Bf);
        }
        r.e_B_fem = std::sqrt(dBf / nB);
        r.d_B = std::sqrt(dBd / nB);
    }
    return r;
}

static void test_T8()
{
    std::printf("[T8] Kugel R=1, J_s = j0 (-y,x,0), j0 = 1, FEM innen (n^3 Hexaeder, Cube-to-Ball), BEM aussen; (S) mit Projektion der Quelle\n");
    BallResult prev{}, last{}, prev2{};
    for (int n : {4, 6, 8, 10}) {
        double gb = 0, ga = 0; int ngd = 0, nq = 0;
        const BallResult r = solve_ball(n, CoupledOptions(), true, &gb, &ga, &ngd, &nq);
        std::printf("    n=%2d (%4d Hex, %4d Randquads, %4d Eichkanten): |G^T f|/|f| vor/nach Projektion %.2e / %.2e | Fehler psi %.3e, g %.3e, B_innen %.3e (reines FEM mit exakten Randdaten: %.3e; |B_gekoppelt - B_FEM|/|B| = %.3e)", n, n * n * n, nq, ngd, gb, ga, r.e_psi, r.e_g, r.e_B, r.e_B_fem, r.d_B);
        if (prev.e_psi > 0) std::printf("  Ordnung %.2f / %.2f / %.2f", order(prev.e_psi, r.e_psi), order(prev.e_g, r.e_g), order(prev.e_B, r.e_B));
        std::printf("\n");
        prev2 = prev; prev = r; last = r;
    }
    // Varianten auf n = 8
    const BallResult rS = solve_ball(8, CoupledOptions(), true);
    CoupledOptions oj; oj.unsymmetric = true;
    const BallResult rJ = solve_ball(8, oj, true);
    CoupledOptions of; of.flip_lambda = true;
    const BallResult rF = solve_ball(8, of, true);
    const BallResult rN = solve_ball(8, CoupledOptions(), false);
    std::printf("    n=8 Varianten: (S) psi %.3e g %.3e B %.3e | (J) psi %.3e g %.3e B %.3e | lambda-Vorzeichen umgekehrt psi %.3e g %.3e B %.3e | ohne Projektion psi %.3e g %.3e B %.3e\n",
                rS.e_psi, rS.e_g, rS.e_B, rJ.e_psi, rJ.e_g, rJ.e_B, rF.e_psi, rF.e_g, rF.e_B, rN.e_psi, rN.e_g, rN.e_B);
    check(last.d_B < 0.5 * last.e_B_fem, "Kopplung: |B_gekoppelt - B_FEM(exakte Randdaten)| ist klein gegen den FEM-Fehler (n = 10)", last.d_B / last.e_B_fem, 0.5);
    check(last.e_B < 1.15 * last.e_B_fem, "der gekoppelte B-Fehler liegt nicht mehr als 15 % ueber dem reinen FEM-Fehler (n = 10)", last.e_B / last.e_B_fem, 1.15);
    // Erwartung (Theorie T8 "~h^2") fuer niedrigste Ordnung Nedelec nicht haltbar: B = rot a ist nur O(h). Gepruefte Aussagen:
    check(last.e_psi < 0.1 && last.e_g < 0.15 && last.e_B < 0.1, "feinstes Netz (n = 10): Fehler psi, g, B < 10 %", std::max({last.e_psi, last.e_g, last.e_B}), 0.15);
    check_ge(order(prev2.e_psi, last.e_psi) >= 0.5 * std::log(10.0 / 8.0) / std::log(2.0) * 2.0 || last.e_psi < prev2.e_psi * 0.8, "psi konvergiert (n = 8 -> 10: Fehler faellt um >= 20 %)", 1.0 - last.e_psi / prev2.e_psi, 0.2);
    check(std::fabs(rJ.e_psi - rS.e_psi) < 0.5 * rS.e_psi + 0.02 && rJ.e_B < 2 * rS.e_B + 0.02, "(J) liefert dieselbe Genauigkeit wie (S)", std::fabs(rJ.e_psi - rS.e_psi), 0.5 * rS.e_psi + 0.02);
    check(rF.e_psi > 5 * rS.e_psi && rF.e_B > 2 * rS.e_B, "Gegenprobe: Vorzeichen von lambda umgekehrt (Abweichung >= Faktor 5 bzw. 2)", std::min(rF.e_psi / rS.e_psi, rF.e_B / rS.e_B), 2.0);
}

// ---------------- Ring (sigma = 0) + Platte (sigma > 0) ----------------
struct RingPlate {
    std::string path;
    FemProblem F;
    BoundaryOps O;
    DerivedOps D;
    int nu = 8, nr = 2, nz = 2;
    int ring_comp = -1;                                 // Flaechenkomponente des Rings
    std::vector<std::pair<int, int>> meridian, longitudinal;
    double J0 = 1.0;
};

static RingPlate make_ring_plate(bool with_cycles)
{
    RingPlate R;
    R.path = "trace_ringplate.vtk";
    const HexMesh ring = ring_hex(R.nu, R.nr, R.nz, 1.5, 2.5, -0.5, 0.5, 1, 0.7);   // ungleichmaessige Winkelteilung (R3)
    const HexMesh plate = box_hex(3, 3, 1, -1.0, 1.0, -1.0, 1.0, -2.0, -1.2, 2);
    write_vtk(R.path, merge_hex(ring, plate), "ring+plate");
    FemMaterials mats;
    mats.mu_r = {1.0, 1.0, 1.0}; mats.kappa = {0.0, 0.0, 1.0};
    const double J0 = R.J0;
    R.F = build_fem_problem(R.path, mats, MU0, [J0](const Vec3& x, int comp) {
        if (comp != 1) return v3(0, 0, 0);
        const double rho = std::sqrt(x.x * x.x + x.y * x.y);
        return v3(-J0 * x.y / rho, J0 * x.x / rho, 0.0);      // azimutale Stromdichte, |J| = J0, Querschnitt 1 x 1: Strom I = J0
    });
    TraceSurface& S = R.F.surface;
    compute_harmonic_fields(S);
    // Komponente des Rings: Flaechenknoten zum FEM-Knoten 0 (Ringknoten (i=0,j=0,k=0))
    int snode0 = -1;
    for (int v = 0; v < S.num_nodes(); ++v) if (R.F.surf_node_to_fem[v] == 0) snode0 = v;
    R.ring_comp = S.node_comp[snode0];
    if (with_cycles) {
        std::vector<int> fem2surf(R.F.num_nodes, -1);
        for (int v = 0; v < S.num_nodes(); ++v) fem2surf[R.F.surf_node_to_fem[v]] = v;
        auto rn = [&](int i, int j, int k) { return fem2surf[((i % R.nu) * (R.nr + 1) + j) * (R.nz + 1) + k]; };
        // Meridian im Uhrzeigersinn in der (rho, z)-Ebene bei i = 0: (j=0,k=0) -> k hoch -> j hoch -> k runter -> j runter
        std::vector<int> mer;
        for (int k = 0; k < R.nz; ++k) mer.push_back(rn(0, 0, k));
        for (int j = 0; j < R.nr; ++j) mer.push_back(rn(0, j, R.nz));
        for (int k = R.nz; k > 0; --k) mer.push_back(rn(0, R.nr, k));
        for (int j = R.nr; j > 0; --j) mer.push_back(rn(0, j, 0));
        std::vector<int> lon;
        for (int i = 0; i < R.nu; ++i) lon.push_back(rn(i, 0, 0));      // Innenkante unten, um die Achse
        R.meridian = cycle_from_nodes(S, mer);
        R.longitudinal = cycle_from_nodes(S, lon);
        normalize_harmonic_to_cycles(S, R.ring_comp, {R.meridian, R.longitudinal});
    }
    R.O = assemble_boundary_ops(S);
    R.D = derive_ops(S, R.O);
    return R;
}

static void test_T6_T7_T4_R3()
{
    std::printf("[Ring + Platte] Ring: 8x2x2 Hexaeder (Rechteckquerschnitt, sigma = 0, g = 1), Platte 3x3x1 (sigma = 1, g = 0), K = 2 Flaechenkomponenten\n");
    RingPlate R = make_ring_plate(true);
    const FemProblem& F = R.F;
    const TraceSurface& S = F.surface;
    std::printf("    %d Hexaeder, %d FEM-Kanten, %d Randquads, %d Flaechenknoten, %d Komponenten, nH = %d, %d Eichkanten (Baum-Kobaum, sigma = 0)\n",
                F.num_hexes, F.num_edges, S.num_quads(), S.num_nodes(), S.num_components, S.nH, F.num_gauged);
    const Vec f = Eigen::Map<const Vec>(F.f.data(), F.f.size());

    // ---- T6: Symmetrie ----
    CoupledOptions os; os.sigma_factor = 1.0;
    CoupledOptions oj = os; oj.unsymmetric = true;
    const Coupled CS = build_coupled(F, R.O, R.D, f, os), CJ = build_coupled(F, R.O, R.D, f, oj);
    const double asS = (CS.A - CS.A.transpose()).norm() / CS.A.norm(), asJ = (CJ.A - CJ.A.transpose()).norm() / CJ.A.norm();
    std::printf("  [T6] |A - A^T|/|A|: (S) %.3e, (J) %.3e\n", asS, asJ);
    check(asS <= 1e-12, "T6: (S) symmetrisch", asS, 1e-12);
    check_ge(asJ >= 1e-2, "T6: (J) unsymmetrisch", asJ, 1e-2);
    {   // Gegenprobe: Q-Block verfaelscht (Cm-Anteil falsches Vorzeichen) -> (S) nicht mehr symmetrisch? Nein: Symmetrie ist strukturell.
        // Gegenprobe stattdessen: (3,1)-Block ohne Transposition (Q statt Q^T-Zuordnung) macht (S) unsymmetrisch.
        Coupled Cb = CS;
        Mat B = Cb.A.block(S.num_edges() * 0 + F.num_edges + S.num_nodes(), 0, S.nH, F.num_edges);
        Cb.A.block(F.num_edges + S.num_nodes(), 0, S.nH, F.num_edges) = 0.9 * B;
        const double as = (Cb.A - Cb.A.transpose()).norm() / Cb.A.norm();
        check_ge(as > 1e-6, "T6-Gegenprobe: gestoerter (3,1)-Block verletzt die Symmetrie messbar", as, 1e-6);
    }

    // ---- T7: Nullraum per SVD ----
    // Zaehlung mit zwei Schwellen: 1e-10 (exakte Nullvektoren) und 1e-5 (Nullvektoren bis auf Quadraturniveau: die psi-Konstante ist wegen
    // (1/2 + D) 1 = 0 nur bis auf den Quadraturfehler exakt, ein Gradient mit Randanteil wegen des Q-Blocks ebenso).
    struct Null { int n10, n5; double smin_rel, gap_lo, gap_hi; };
    auto nullity = [&](const Coupled& C) {
        const Mat A = C.reduced();
        Eigen::BDCSVD<Mat> svd(A);
        const Vec sv = svd.singularValues();
        Null r{0, 0, sv[sv.size() - 1] / sv[0], 0, 0};
        r.gap_lo = 0; r.gap_hi = 1e300;     // groesster Wert unter, kleinster Wert ueber der 1e-5-Schwelle (relativ)
        for (int i = 0; i < sv.size(); ++i) {
            const double rel = sv[i] / sv[0];
            if (rel < 1e-10) ++r.n10;
            if (rel < 1e-5) { ++r.n5; r.gap_lo = std::max(r.gap_lo, rel); } else r.gap_hi = std::min(r.gap_hi, rel);
        }
        return r;
    };
    const Null n_full = nullity(CS);
    CoupledOptions o_nopin = os; o_nopin.pin_psi = false;
    CoupledOptions o_notree = os; o_notree.gauge_tree = false;
    CoupledOptions o_nobem = os; o_nobem.include_bem = false;
    const Null n_nopin = nullity(build_coupled(F, R.O, R.D, f, o_nopin));
    const Null n_notree = nullity(build_coupled(F, R.O, R.D, f, o_notree));
    const Null n_nobem = nullity(build_coupled(F, R.O, R.D, f, o_nobem));
    const Null n_J = nullity(CJ);
    std::printf("  [T7] Nullraumdimension, Schwellen 1e-10 / 1e-5 (relativ zu sigma_max); Luecke = groesster Wert unter / kleinster Wert ueber 1e-5:\n");
    std::printf("       (S) vollstaendig:           %d / %d   (kleinster Singulaerwert %.2e)\n", n_full.n10, n_full.n5, n_full.smin_rel);
    std::printf("       (J) vollstaendig:           %d / %d   (kleinster Singulaerwert %.2e)\n", n_J.n10, n_J.n5, n_J.smin_rel);
    std::printf("       ohne psi-Konstanten:        %d / %d   (Luecke %.2e | %.2e; erwartet K = %d)\n", n_nopin.n10, n_nopin.n5, n_nopin.gap_lo, n_nopin.gap_hi, S.num_components);
    std::printf("       ohne Baum-Kobaum:           %d / %d   (Luecke %.2e | %.2e; Eichkanten %d)\n", n_notree.n10, n_notree.n5, n_notree.gap_lo, n_notree.gap_hi, F.num_gauged);
    std::printf("       ohne BEM-Zeilen (nur K):    %d / %d   (alte Aufgabe: harmonischer Nullvektor der Spule)\n", n_nobem.n10, n_nobem.n5);
    check(n_full.n10 == 0 && n_full.n5 == 0, "T7: kein Nullvektor nach Baum-Kobaum + je einer psi-Konstante, OHNE Kohomologie-Eichung der Spule", n_full.n5, 1);
    check(n_J.n10 == 0 && n_J.n5 == 0, "T7: (J) ebenfalls ohne Nullvektor", n_J.n5, 1);
    check_ge(n_nopin.n5 == S.num_components, "T7-Gegenprobe: ohne Festlegung der psi-Konstanten Nullraum = K (Schwelle 1e-5)", n_nopin.n5, S.num_components);
    check_ge(n_notree.n5 == F.num_gauged, "T7-Gegenprobe: ohne Baum-Kobaum waechst der Nullraum genau um die Zahl der Eichkanten (Schwelle 1e-5)", n_notree.n5, F.num_gauged);
    check_ge(n_nobem.n10 == 1, "T7-Gegenprobe: ohne BEM-Zeilen bleibt genau ein harmonischer Nullvektor (Spule; alte Kohomologie-Eichung noetig)", n_nobem.n10, 1);

    // ---- T4: Eichtest, Ring (sigma = 0): a -> a + G psi_h, psi_h nichtharmonisch ----
    {
        const int E = F.num_edges;
        auto psi_h = [&](const Vec3& x) { return x.z * std::exp(-(x.x * x.x + x.y * x.y + x.z * x.z) / 4.0); };
        Vec a_shift = Vec::Zero(E);
        for (int e = 0; e < E; ++e) {
            const int v0 = F.edges[e][0], v1 = F.edges[e][1];
            if (F.hex_comp.empty()) break;
            // nur Ringknoten (id < Ringknotenzahl): Platte unveraendert (sigma > 0: Eichfreiheit nur durch V = 0-Konvention)
            const int nring = R.nu * (R.nr + 1) * (R.nz + 1);
            if (v0 < nring && v1 < nring) a_shift[e] = psi_h(F.X[v1]) - psi_h(F.X[v0]);
        }
        Vec x = Vec::Zero(CS.A.rows());
        x.head(E) = a_shift;
        const Vec r = CS.A * x;
        const int nE = E, nV = S.num_nodes();
        const Vec r1 = r.head(nE), r2 = r.segment(nE, nV), r3 = r.tail(S.nH);
        // Vergleichsgroessen: Normen der Bloecke, die auf a wirken, mal ||a_shift||
        const double an = a_shift.norm();
        const double n1 = CS.A.topLeftCorner(nE, nE).norm() * an, n2 = CS.A.block(nE, 0, nV, nE).norm() * an, n3 = CS.A.block(nE + nV, 0, S.nH, nE).norm() * an;
        std::printf("  [T4] a -> a + G psi_h (psi_h = z exp(-r^2/4), nichtharmonisch, nur Ring): relative Aenderung der Zeilen: (S1) %.2e, (S2) %.2e, (S3) %.2e\n", r1.norm() / n1, r2.norm() / n2, r3.norm() / n3);
        // Gegenprobe: ein Nicht-Gradient (Rotationsanteil) auf Ringkanten: a_rot = zufaellig auf Ringkanten
        Vec a_rnd = Vec::Zero(E);
        unsigned st = 99u;
        for (int e = 0; e < E; ++e) { st = st * 1664525u + 1013904223u; const int nring = R.nu * (R.nr + 1) * (R.nz + 1); if (F.edges[e][0] < nring && F.edges[e][1] < nring) a_rnd[e] = (static_cast<double>(st >> 8) / 16777216.0) - 0.5; }
        Vec xr = Vec::Zero(CS.A.rows());
        xr.head(E) = a_rnd;
        const Vec rr = CS.A * xr;
        const double ar = a_rnd.norm();
        const double m1 = rr.head(nE).norm() / (CS.A.topLeftCorner(nE, nE).norm() * ar), m2 = rr.segment(nE, nV).norm() / (CS.A.block(nE, 0, nV, nE).norm() * ar), m3 = rr.tail(S.nH).norm() / (CS.A.block(nE + nV, 0, S.nH, nE).norm() * ar);
        std::printf("       Gegenprobe a + (zufaelliger Nicht-Gradient): (S1) %.2e, (S2) %.2e, (S3) %.2e\n", m1, m2, m3);
        check(r1.norm() / n1 <= 1e-12 && r2.norm() / n2 <= 1e-12, "T4: Flussbloecke (S1, S2) exakt invariant (<= 1e-12)", std::max(r1.norm() / n1, r2.norm() / n2), 1e-12);
        const double q_thr = 1e-5;   // aus Selbstkonvergenz (test_trace_quadrature, [Q7]: Q auf dem Torus 2e-7 relativ) mit Faktor 50 Reserve
        check(r3.norm() / n3 <= q_thr, "T4: Q-Block (S3) auf Quadraturniveau (<= 1e-5, Q7: 2e-7)", r3.norm() / n3, q_thr);
        check_ge(m1 > 1e-3 && m2 > 1e-3 && m3 > 1e-3, "T4-Gegenprobe: Nicht-Gradient aendert alle Zeilen um O(1) (>= 1e-3)", std::min({m1, m2, m3}), 1e-3);
    }

    // ---- R3 + c_Meridian: Quelle mit/ohne Projektion ----
    {
        const int nring = R.nu * (R.nr + 1) * (R.nz + 1);
        auto node = [&](int i, int j, int k) { return ((i % R.nu) * (R.nr + 1) + j) * (R.nz + 1) + k; };
        // Schnittschicht i: alle phi-Kanten zwischen Scheibe i und i+1; Vorzeichen +1, wenn die globale Richtung (vmin -> vmax) in wachsendes phi zeigt
        std::map<std::pair<int, int>, int> eid;
        for (int e = 0; e < F.num_edges; ++e) eid[{F.edges[e][0], F.edges[e][1]}] = e;
        std::vector<std::vector<std::pair<int, double>>> layers(R.nu);
        for (int i = 0; i < R.nu; ++i) for (int j = 0; j <= R.nr; ++j) for (int k = 0; k <= R.nz; ++k) {
            const int a = node(i, j, k), b = node(i + 1, j, k);
            const auto key = std::make_pair(std::min(a, b), std::max(a, b));
            layers[i].push_back({eid.at(key), (a < b) ? +1.0 : -1.0});
        }
        std::vector<int> fem2surf(F.num_nodes, -1);
        for (int v = 0; v < S.num_nodes(); ++v) fem2surf[F.surf_node_to_fem[v]] = v;
        std::vector<int> surf_of_fem_edge(F.num_edges, -1);
        for (int e = 0; e < S.num_edges(); ++e) surf_of_fem_edge[F.surf_edge_to_fem[e]] = e;
        // Spalten von Q: Meridian = Spalte 0 der Ringkomponente, Laengs = Spalte 1 (Reihenfolge in S.H_comp)
        int colM = -1, colL = -1;
        for (int k = 0; k < S.nH; ++k) if (S.H_comp[k] == R.ring_comp) { if (colM < 0) colM = k; else colL = k; }
        const Mat& Q = R.D.Q;

        // Maske: Platte (Knoten >= nring) nimmt an der Projektion nicht teil
        std::vector<char> blocked(F.num_edges, 0);
        for (int e = 0; e < F.num_edges; ++e) if (F.edges[e][0] >= nring || F.edges[e][1] >= nring) blocked[e] = 1;
        auto analyse = [&](bool project, double* spread, double* cmer_solved, double* clon_solved, double* spread_c) {
            Vec fv = f;
            double rel_after = 0;
            if (project) { std::vector<double> ff(fv.data(), fv.data() + fv.size()); const ProjectionResult pr = project_source(R.path, ff, &blocked); rel_after = pr.rel_after; fv = Eigen::Map<const Vec>(ff.data(), ff.size()); }
            (void)rel_after;
            CoupledOptions oo; oo.sigma_factor = 1.0;
            const Coupled C = build_coupled(F, R.O, R.D, fv, oo);
            const Vec x = C.solve();
            const double cM = x[C.nE + C.nV + colM - 0 - 0 + 0] * 0 + x[C.nE + C.nV + colM], cL = x[C.nE + C.nV + colL];
            *cmer_solved = cM; *clon_solved = cL;
            std::vector<double> Fi(R.nu), ci(R.nu);
            double mn = 1e300, mx = -1e300, cmn = 1e300, cmx = -1e300;
            for (int i = 0; i < R.nu; ++i) {
                double sum = 0, qM = 0, qL = 0;
                for (const auto& le : layers[i]) {
                    sum += le.second * fv[le.first];
                    const int se = surf_of_fem_edge[le.first];
                    if (se >= 0) { qM += le.second * Q(se, colM); qL += le.second * Q(se, colL); }
                }
                Fi[i] = sum;
                ci[i] = (sum - qL * cL) / qM;     // c_Meridian aus der Zeile (S1) getestet mit der Schnittschicht i
                mn = std::min(mn, sum); mx = std::max(mx, sum); cmn = std::min(cmn, ci[i]); cmx = std::max(cmx, ci[i]);
            }
            *spread = (mx - mn) / std::fabs(0.5 * (mx + mn));
            *spread_c = (cmx - cmn) / std::fabs(0.5 * (cmx + cmn));
            (void)Fi; (void)nring;
        };
        double sp0, cm0, cl0, sc0, sp1, cm1, cl1, sc1;
        analyse(false, &sp0, &cm0, &cl0, &sc0);
        analyse(true, &sp1, &cm1, &cl1, &sc1);
        // ||G^T f||/||f|| im sigma = 0-Gebiet
        auto gt = [&](const Vec& fv) {
            std::vector<double> g(F.num_nodes, 0.0);
            for (int e = 0; e < F.num_edges; ++e) { g[F.edges[e][1]] += fv[e]; g[F.edges[e][0]] -= fv[e]; }
            double s = 0; for (int v = 0; v < nring; ++v) s += g[v] * g[v];
            return std::sqrt(s) / fv.norm();
        };
        std::vector<double> ff(f.data(), f.data() + f.size());
        const ProjectionResult prm = project_source(R.path, ff, &blocked);
        const Vec fp = Eigen::Map<const Vec>(ff.data(), ff.size());
        {   // Beobachtung: dieselbe Projektion ohne Maske auf dem Netz aus zwei getrennten Koerpern
            std::vector<double> fu(f.data(), f.data() + f.size());
            const ProjectionResult pru = project_source(R.path, fu);
            double nf = 0, nu2 = 0; for (size_t i = 0; i < fu.size(); ++i) { nf += f[i] * f[i]; nu2 += fu[i] * fu[i]; }
            std::printf("       Beobachtung: project_out_gradient OHNE Maske auf Ring + Platte (getrennte Koerper): ||f|| %.3f -> %.3f, ||G^T f||/||f|| nachher %.2e; MIT Maske (Platte gesperrt): %.2e (||f|| %.3f)\n",
                        std::sqrt(nf), std::sqrt(nu2), pru.rel_after, prm.rel_after, fp.norm());
        }
        std::printf("  [R3] ||G^T f||/||f|| auf den Ringknoten: unprojiziert %.2e, projiziert %.2e\n", gt(f), gt(fp));
        std::printf("       Strom durch die Schnittschichten, relative Streuung ueber die %d Schichten: unprojiziert %.2e, projiziert %.2e\n", R.nu, sp0, sp1);
        std::printf("       c_Meridian aus (S1) mit Schnittschicht i: Streuung ueber i: unprojiziert %.2e, projiziert %.2e\n", sc0, sc1);
        std::printf("       Loesung des Gesamtsystems: c_Meridian = %.6f (unprojiziert), %.6f (projiziert); c_Laengs = %.2e / %.2e; erwartet |c_Meridian| = I = %.1f\n", cm0, cm1, cl0, cl1, R.J0);
        check(gt(fp) <= 1e-9, "R3: projizierte Quelle ist diskret divergenzfrei (<= 1e-9)", gt(fp), 1e-9);
        check_ge(sp0 > 100 * sp1 && sc0 > 100 * sc1, "R3: ohne Projektion haengt der Strom (bzw. c_Meridian) von der Schnittschicht ab, mit Projektion nicht (Faktor >= 100)", std::min(sp0 / sp1, sc0 / sc1), 100.0);
    }
}

// ---------------- Phase 2, Schritt 0a: Skalierung ----------------
static void test_scaling()
{
    std::printf("[0a] Skalierung (psi~ = mu0 psi, c~ = mu0 c, Matrix mal mu0): Kugel n = 4 und Ring + Platte mit mu0 = 4 pi 1e-7\n");
    const double MU = 4e-7 * PI;
    write_vtk("trace_ball_scal.vtk", ball_hex(4, 1.0), "ball");
    FemMaterials mats;
    mats.mu_r = {1.0, 1.0}; mats.kappa = {0.0, 0.0};
    FemProblem F = build_fem_problem("trace_ball_scal.vtk", mats, MU, [&](const Vec3& x, int) { return v3(-x.y, x.x, 0.0); });
    project_source("trace_ball_scal.vtk", F.f);
    TraceSurface& S = F.surface;
    compute_harmonic_fields(S);
    const BoundaryOps O = assemble_boundary_ops(S);
    const DerivedOps D = derive_ops(S, O);
    const Vec f = Eigen::Map<const Vec>(F.f.data(), F.f.size());
    CoupledOptions o1; o1.sigma_factor = 0.0;
    CoupledOptions o0 = o1; o0.scaled = false;
    const Coupled C1 = build_coupled(F, O, D, f, o1), C0 = build_coupled(F, O, D, f, o0);
    const Vec x1 = C1.solve(), x0 = C0.solve();
    const double dA = (x1.head(C1.nE) - x0.head(C1.nE)).norm() / x1.head(C1.nE).norm();
    const double dP = (x1.segment(C1.nE, C1.nV) - x0.segment(C1.nE, C1.nV)).norm() / x1.segment(C1.nE, C1.nV).norm();
    // Rueckwaertsfehler beider Loesungen im jeweiligen System (relativ zu |A||x|)
    auto resid = [&](const Coupled& C, const Vec& x) {
        Vec xs = x;
        if (C.scaled) xs.tail(C.nV + C.nH) *= C.mu0;
        const Mat A = C.reduced();
        Vec xr(C.keep.size()), br(C.keep.size());
        for (size_t i = 0; i < C.keep.size(); ++i) { xr[i] = xs[C.keep[i]]; br[i] = C.b[C.keep[i]]; }
        return (A * xr - br).norm() / (A.norm() * xr.norm());
    };
    const double asym1 = (C1.A - C1.A.transpose()).norm() / C1.A.norm();
    Eigen::BDCSVD<Mat> s1(C1.reduced()), s0(C0.reduced());
    const double cond1 = s1.singularValues()(0) / s1.singularValues().tail(1)(0), cond0 = s0.singularValues()(0) / s0.singularValues().tail(1)(0);
    std::printf("     Kugel n=4: |a_skal - a_unskal|/|a| = %.2e, |psi_skal - psi_unskal|/|psi| = %.2e; Rueckwaertsfehler skaliert %.2e, unskaliert %.2e; Kond %.2e (skaliert) gegen %.2e (unskaliert); Asymmetrie skaliert %.2e\n",
                dA, dP, resid(C1, x1), resid(C0, x0), cond1, cond0, asym1);
    check(asym1 <= 1e-15, "T6 (skaliert): Symmetrie 1e-15", asym1, 1e-15);
    check(cond1 < 1e-3 * cond0, "skalierte Matrix ist um >= 3 Dekaden besser konditioniert als die unskalierte (mu0 = 4 pi 1e-7)", cond1 / cond0, 1e-3);
    check(dA <= 1e-6 && dP <= 1e-6, "skaliert und unskaliert liefern dieselbe Loesung (<= 1e-6; die unskalierte ist die ungenauere)", std::max(dA, dP), 1e-6);
    // mu0 = 1: Skalierung ist die Identitaet
    {
        FemProblem F1 = build_fem_problem("trace_ball_scal.vtk", mats, 1.0, [&](const Vec3& x, int) { return v3(-x.y, x.x, 0.0); });
        project_source("trace_ball_scal.vtk", F1.f);
        TraceSurface& S1 = F1.surface;
        compute_harmonic_fields(S1);
        const BoundaryOps O1 = assemble_boundary_ops(S1);
        const DerivedOps D1 = derive_ops(S1, O1);
        const Vec f1 = Eigen::Map<const Vec>(F1.f.data(), F1.f.size());
        const Vec xa = build_coupled(F1, O1, D1, f1, o1).solve(), xb = build_coupled(F1, O1, D1, f1, o0).solve();
        check((xa - xb).norm() / xa.norm() <= 1e-12, "mu0 = 1: skalierte und unskalierte Loesung stimmen auf 1e-12 ueberein (T8-Aufbau)", (xa - xb).norm() / xa.norm(), 1e-12);
    }
}

// ---------------- Phase 2, Schritt 1: duenne Elimination + dichtes Schurkomplement gegen die dichte Loesung ----------------
static void test_schur()
{
    std::printf("[S] solve_schur (duenne Elimination der inneren Kanten, dichtes Schurkomplement) gegen die dichte Gesamtloesung: Kugel n = 6, mu0 = 4 pi 1e-7\n");
    const double MU = 4e-7 * PI;
    write_vtk("trace_ball_schur.vtk", ball_hex(6, 1.0), "ball");
    FemMaterials mats;
    mats.mu_r = {1.0, 1.0}; mats.kappa = {0.0, 0.0};
    FemProblem F = build_fem_problem("trace_ball_schur.vtk", mats, MU, [&](const Vec3& x, int) { return v3(-x.y, x.x, 0.0); });
    project_source("trace_ball_schur.vtk", F.f);
    TraceSurface& S = F.surface;
    compute_harmonic_fields(S);
    const BoundaryOps O = assemble_boundary_ops(S);
    const DerivedOps D = derive_ops(S, O);
    const Vec f = Eigen::Map<const Vec>(F.f.data(), F.f.size());
    CoupledOptions o; o.sigma_factor = 0.0;
    const Vec xd = build_coupled(F, O, D, f, o).solve();
    const SchurResult r = solve_schur(F, O, D, f);
    const double da = (r.a - xd.head(F.num_edges)).norm() / xd.head(F.num_edges).norm();
    const double dp = (r.psi - xd.segment(F.num_edges, S.num_nodes())).norm() / xd.segment(F.num_edges, S.num_nodes()).norm();
    std::printf("     n_innen %d, n_Gamma(a) %d, psi %d, c %d; Schurkomplement %.4f GB; Zeiten: eliminieren %.2f s, Schur %.2f s, loesen %.2f s; LDL^T: %s; Residuum %.2e\n",
                r.nI, r.nGa, r.nPsi, r.nC, r.schur_gb, r.t_eliminate, r.t_schur, r.t_solve, r.used_ldlt ? "ja" : "nein (SparseLU)", r.residual_rel);
    check(da <= 1e-8 && dp <= 1e-8, "solve_schur = dichte Loesung (<= 1e-8)", std::max(da, dp), 1e-8);
    check(r.residual_rel <= 1e-10, "Residuum des skalierten Gesamtsystems (<= 1e-10)", r.residual_rel, 1e-10);
    // Gegenprobe: anderer Baum (Seed): a anders, B gleich
    SchurOptions so; so.gauged = make_tree_gauge(F, 7u);
    const SchurResult r2 = solve_schur(F, O, D, f, so);
    double dB = 0, nB = 0;
    const std::vector<double> a1(r.a.data(), r.a.data() + r.a.size()), a2(r2.a.data(), r2.a.data() + r2.a.size());
    for (int h = 0; h < F.num_hexes; ++h) { const Vec3 b1 = F.curl_at_center(h, a1), b2 = F.curl_at_center(h, a2); dB += dot(b1 - b2, b1 - b2); nB += dot(b1, b1); }
    const double dAa = (r.a - r2.a).norm() / r.a.norm();
    std::printf("     anderer Baum (Seed 7): |a1 - a2|/|a| = %.2e, |B1 - B2|/|B| = %.2e, Residuum %.2e\n", dAa, std::sqrt(dB / nB), r2.residual_rel);
    check(std::sqrt(dB / nB) <= 1e-8, "anderer Baum: B gleich (<= 1e-8)", std::sqrt(dB / nB), 1e-8);
    check_ge(dAa >= 1e-2, "anderer Baum: a verschieden (>= 1e-2)", dAa, 1e-2);
}

int main()
{
    test_scaling();
    test_schur();
    test_T8();
    test_T6_T7_T4_R3();
    std::printf(g_fail == 0 ? "ALLE TESTS BESTANDEN\n" : "FEHLGESCHLAGEN: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
