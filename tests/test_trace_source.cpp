// =============================================================================
// Phase 2, Schritt 0b: Nettostrom der Quelle und c_Meridian (Ring mit ungleichmaessiger Winkelteilung).
// Gemessen: <f, v_cut(i)> je Schnittschicht vor/nach der Projektion, gelostes c_Meridian, die Spalte qM = <v_cut, (1/2 + n x C) h_Meridian>.
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
static const double PI = 3.14159265358979323846;

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

struct RingRun { double cm, cl, net_spread_before, net_proj, gt_after, net_final_err, net_final_spread; };

// Ring nu x 2 x 2, ungleichmaessige Winkelteilung; Quelle je nach Optionen aufbereitet; Ergebnis: gelostes c_Meridian
static RingRun run_ring(int nu, bool project, bool renormalize)
{
    const int nr = 2, nz = 2;
    const std::string path = "trace_src_ring_" + std::to_string(nu) + ".vtk";
    write_vtk(path, ring_hex(nu, nr, nz, 1.5, 2.5, -0.5, 0.5, 1, 0.7), "ring");
    FemMaterials mats; mats.mu_r = {1, 1}; mats.kappa = {0, 0};
    FemProblem F = build_fem_problem(path, mats, 1.0, [](const Vec3& x, int) {
        const double rho = std::sqrt(x.x * x.x + x.y * x.y); return v3(-x.y / rho, x.x / rho, 0.0); });   // |J| = 1 A/m^2, Querschnitt 1 x 1 m^2: N I = 1 A
    TraceSurface& S = F.surface;
    compute_harmonic_fields(S);
    std::vector<int> fem2surf(F.num_nodes, -1);
    for (int v = 0; v < S.num_nodes(); ++v) fem2surf[F.surf_node_to_fem[v]] = v;
    auto rn = [&](int i, int j, int k) { return fem2surf[((i % nu) * (nr + 1) + j) * (nz + 1) + k]; };
    std::vector<int> mer, lon;
    for (int k = 0; k < nz; ++k) mer.push_back(rn(0, 0, k));
    for (int j = 0; j < nr; ++j) mer.push_back(rn(0, j, nz));
    for (int k = nz; k > 0; --k) mer.push_back(rn(0, nr, k));
    for (int j = nr; j > 0; --j) mer.push_back(rn(0, j, 0));
    for (int i = 0; i < nu; ++i) lon.push_back(rn(i, 0, 0));
    normalize_harmonic_to_cycles(S, 0, {cycle_from_nodes(S, mer), cycle_from_nodes(S, lon)});
    const BoundaryOps O = assemble_boundary_ops(S);
    const DerivedOps D = derive_ops(S, O);
    std::vector<SourceDiagnostics> diag;
    const std::vector<double> ff = prepare_source(F, path, {{1, 1.0}}, project, renormalize, &diag);
    const Vec f = Eigen::Map<const Vec>(ff.data(), ff.size());
    CoupledOptions o; o.sigma_factor = 0;
    const Vec x = build_coupled(F, O, D, f, o).solve();
    RingRun r{};
    r.cm = x[F.num_edges + S.num_nodes() + 0];
    r.cl = x[F.num_edges + S.num_nodes() + 1];
    r.net_spread_before = (diag[0].net_max_before - diag[0].net_min_before) / std::fabs(0.5 * (diag[0].net_max_before + diag[0].net_min_before));
    r.net_proj = diag[0].net_proj;
    r.gt_after = diag[0].rel_gt_after;
    r.net_final_err = std::fabs(diag[0].net_final - 1.0);
    r.net_final_spread = diag[0].net_final_spread;
    if (nu == 8 && project && renormalize) std::printf("     (nu = 8: %d Schnittschichten erkannt, |G^T f|/|f| vorher %.2e, Nettostrom vorher %.4f ... %.4f)\n", diag[0].K, diag[0].rel_gt_before, diag[0].net_min_before, diag[0].net_max_before);
    return r;
}

int main()
{
    std::printf("[0b] Ring mit ungleichmaessiger Winkelteilung, N I = 1: Nettostrom je Schnittschicht und gelostes c_Meridian\n");
    double prev_dev = 0;
    for (int nu : {8, 16, 32}) {
        const RingRun full = run_ring(nu, true, true), projonly = run_ring(nu, true, false), none = run_ring(nu, false, false);
        std::printf("  nu = %2d: Nettostrom projiziert %.8f (Defizit %.3e), c_Meridian: ohne Korrektur %.6f, nur Projektion %.6f, mit Projektion + Renormierung %.8f (|c-1| = %.2e), c_Laengs %.1e\n",
                    nu, projonly.net_proj, 1 - projonly.net_proj, none.cm, projonly.cm, full.cm, std::fabs(full.cm - 1), full.cl);
        check(full.gt_after <= 1e-12, "G^T f = 0 (<= 1e-12)", full.gt_after, 1e-12);
        check(full.net_final_err <= 1e-12 && full.net_final_spread <= 1e-12, "Nettostrom = N I in jeder Schnittschicht (<= 1e-12)", std::max(full.net_final_err, full.net_final_spread), 1e-12);
        check(std::fabs(full.cm - 1) <= 3e-4, "gelostes c_Meridian = N I (<= 3e-4; Rest: qM = 1 + O(1e-4), BEM-Diskretisierung)", std::fabs(full.cm - 1), 3e-4);
        // Gegenproben: ohne Renormierung weicht c_Meridian um das Defizit ab, ohne Projektion streut der Nettostrom
        check_ge(std::fabs(projonly.cm - 1) >= 10 * std::fabs(full.cm - 1), "Gegenprobe ohne Renormierung: |c_Meridian - N I| >= Faktor 10 groesser", std::fabs(projonly.cm - 1) / std::fabs(full.cm - 1), 10.0);
        check_ge(none.net_spread_before >= 1e-2 || nu > 8, "Gegenprobe ohne Projektion: Nettostrom streut ueber die Schichten (nu = 8: >= 1 %)", none.net_spread_before, 1e-2);
        if (nu > 8) check_ge(std::fabs(1 - projonly.net_proj) <= 0.4 * prev_dev, "Defizit des projizierten Nettostroms faellt mit Verfeinerung (>= Faktor 2.5 je Halbierung)", prev_dev / std::fabs(1 - projonly.net_proj), 2.5);
        prev_dev = std::fabs(1 - projonly.net_proj);
    }
    std::printf(g_fail == 0 ? "ALLE TESTS BESTANDEN\n" : "FEHLGESCHLAGEN: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
