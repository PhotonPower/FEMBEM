// =============================================================================
// T9 (Phase 1b, Schritt 1): Hodge-Strukturen der Randflaechen. Reiner Host-Test, ohne CUDA.
//   R G = 0 exakt; dim ker R - dim range G = 2 sum g; Euler-Charakteristik; harmonische Felder in ker R und
//   orthogonal zu range G; Zirkulationsmatrix regulaer; Gegenproben.
// =============================================================================
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <random>

#include "TraceMeshes.h"
#include "TraceSurface.h"

using namespace trace;

static int g_fail = 0;
static void check(bool ok, const char* what, double value, double limit)
{
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? "<" : ">=", limit);
    if (!ok) ++g_fail;
}
static void check_eq(bool ok, const char* what, long a, long b)
{
    std::printf("  [%s] %s  (%ld %s %ld)\n", ok ? "OK " : "FEHLER", what, a, ok ? "==" : "!=", b);
    if (!ok) ++g_fail;
}

static int numerical_rank(const Eigen::MatrixXd& A)
{
    Eigen::BDCSVD<Eigen::MatrixXd> svd(A);
    const auto s = svd.singularValues();
    const double tol = 1e-10 * std::max(1.0, s.size() ? s[0] : 1.0);
    int r = 0;
    for (int i = 0; i < s.size(); ++i) if (s[i] > tol) ++r;
    return r;
}

static Eigen::MatrixXd to_eigen(const std::vector<double>& d, int rows, int cols)
{
    Eigen::MatrixXd M(rows, cols);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) M(i, j) = d[static_cast<size_t>(i) * cols + j];
    return M;
}

// Allgemeine Pruefung einer Flaeche; erwartete Geschlechter je Komponente (in Komponentenreihenfolge, summiert verglichen)
static TraceSurface run_surface(const char* name, const QuadMesh& m, int expect_K, int expect_sum_g, bool with_harmonics = true)
{
    std::printf("[%s]\n", name);
    TraceSurface S = build_trace_surface(m.X, m.quads);
    check_eq(S.manifold_oriented, "geschlossen und konsistent orientiert", S.manifold_oriented, 1);
    check_eq(S.num_components == expect_K, "Zusammenhangskomponenten", S.num_components, expect_K);
    check_eq(S.total_genus() == expect_sum_g, "Summe der Geschlechter (aus Euler je Komponente)", S.total_genus(), expect_sum_g);
    const int V = S.num_nodes(), E = S.num_edges(), F = S.num_quads();
    check_eq(V - E + F == 2 * S.num_components - 2 * S.total_genus(), "Euler V-E+F = 2K-2 sum g", V - E + F, 2 * S.num_components - 2 * S.total_genus());
    check_eq(E == 2 * F, "E = 2F (nur Quads)", E, 2 * F);

    const Eigen::MatrixXd G = to_eigen(S.G_dense(), E, V), R = to_eigen(S.R_dense(), F, E);
    const double rg = (R * G).cwiseAbs().maxCoeff();
    check(rg == 0.0, "max|R G| exakt 0", rg, 1e-300);
    std::mt19937 rng(7);
    std::normal_distribution<double> nd;
    Eigen::VectorXd psi(V);
    for (int i = 0; i < V; ++i) psi[i] = nd(rng) * std::exp(-0.5 * i / V);
    check((R * (G * psi)).cwiseAbs().maxCoeff() <= 1e-13, "max|R G psi| (nichtharmonisch)", (R * (G * psi)).cwiseAbs().maxCoeff(), 1e-13);

    // Dimensionen unabhaengig ueber numerischen Rang
    const int rR = numerical_rank(R), rG = numerical_rank(G);
    check_eq(rR == F - S.num_components, "rank R = F-K", rR, F - S.num_components);
    check_eq(rG == V - S.num_components, "rank G = V-K", rG, V - S.num_components);
    check_eq((E - rR) - rG == 2 * S.total_genus(), "dim ker R - dim range G = 2 sum g", (E - rR) - rG, 2 * S.total_genus());

    // Gegenprobe: ein Quad umdrehen macht die Orientierung inkonsistent
    {
        QuadMesh bad = m;
        std::swap(bad.quads[0][1], bad.quads[0][3]);
        TraceSurface Sb = build_trace_surface(bad.X, bad.quads);
        check_eq(!Sb.manifold_oriented, "Gegenprobe: umgedrehtes Quad wird erkannt", !Sb.manifold_oriented, 1);
    }

    if (with_harmonics) {
        compute_harmonic_fields(S);
        check_eq(S.nH == 2 * S.total_genus(), "Anzahl harmonischer Felder = 2 sum g", S.nH, 2 * S.total_genus());
        if (S.nH > 0) {
            const Eigen::MatrixXd H = to_eigen(S.H, E, S.nH);
            check(S.harmonic_residual_R <= 1e-9, "max |R H_k| (H in ker R)", S.harmonic_residual_R, 1e-9);
            check(S.harmonic_residual_G <= 1e-9, "max |G^T H_k| (H senkrecht range G)", S.harmonic_residual_G, 1e-9);
            const Eigen::MatrixXd gram = H.transpose() * H;
            check((gram - Eigen::MatrixXd::Identity(S.nH, S.nH)).cwiseAbs().maxCoeff() <= 1e-9, "H orthonormal", (gram - Eigen::MatrixXd::Identity(S.nH, S.nH)).cwiseAbs().maxCoeff(), 1e-9);
            // Gegenprobe: eine Zufallsspalte ist keines von beiden
            Eigen::VectorXd c(E);
            for (int e = 0; e < E; ++e) c[e] = nd(rng);
            check((R * c).norm() > 1.0, "Gegenprobe: Zufallsvektor liegt nicht in ker R", (R * c).norm(), 1.0);
            check((G.transpose() * c).norm() > 1.0, "Gegenprobe: Zufallsvektor nicht senkrecht zu range G", (G.transpose() * c).norm(), 1.0);
            // Gegenprobe: Gradient wird von H nicht erfasst
            check((H.transpose() * (G * psi)).cwiseAbs().maxCoeff() <= 1e-9 * (G * psi).norm(), "H^T (G psi) = 0", (H.transpose() * (G * psi)).cwiseAbs().maxCoeff(), 1e-9 * (G * psi).norm());
        }
    }
    return S;
}

// Zirkulationsmatrix auf Meridian und Laengskreis des Torus
static void torus_cycles(const char* name, const TorusMesh& T, TraceSurface S, int comp)
{
    std::printf("[%s: Zirkulationen auf den Erzeugern]\n", name);
    std::vector<int> mer, lon;
    for (int j = 0; j < T.nv; ++j) mer.push_back(T.node(0, j));
    for (int i = 0; i < T.nu; ++i) lon.push_back(T.node(i, 0));
    const auto cm = cycle_from_nodes(S, mer), cl = cycle_from_nodes(S, lon);
    // Gradient hat auf jedem Zyklus Zirkulation exakt null
    std::mt19937 rng(3);
    std::normal_distribution<double> nd;
    std::vector<double> psi(S.num_nodes());
    for (double& p : psi) p = nd(rng);
    std::vector<double> gpsi(S.num_edges());
    for (int e = 0; e < S.num_edges(); ++e) gpsi[e] = psi[S.edges[e][1]] - psi[S.edges[e][0]];
    check(std::fabs(cycle_circulation(cm, gpsi.data())) < 1e-12 && std::fabs(cycle_circulation(cl, gpsi.data())) < 1e-12,
          "Gradient: Zirkulation auf beiden Zyklen exakt 0", std::fabs(cycle_circulation(cm, gpsi.data())) + std::fabs(cycle_circulation(cl, gpsi.data())), 1e-12);
    const double smin = normalize_harmonic_to_cycles(S, comp, {cm, cl});
    check(smin > 0.1, "Zirkulationsmatrix regulaer (kleinster Singulaerwert)", -smin, -0.1);
    // nach Normierung ist sie die Einheitsmatrix
    std::vector<double> col(S.num_edges());
    int k0 = -1;
    for (int k = 0; k < S.nH; ++k) if (S.H_comp[k] == comp) { k0 = k; break; }
    for (int k = 0; k < 2; ++k) {
        for (int e = 0; e < S.num_edges(); ++e) col[e] = S.H[static_cast<size_t>(e) * S.nH + k0 + k];
        const double a = cycle_circulation(cm, col.data()), b = cycle_circulation(cl, col.data());
        check(std::fabs(a - (k == 0 ? 1.0 : 0.0)) < 1e-10 && std::fabs(b - (k == 1 ? 1.0 : 0.0)) < 1e-10, "normierte Spalte hat Zirkulation e_k", std::fabs(a - (k == 0)) + std::fabs(b - (k == 1)), 1e-10);
    }
}

int main()
{
    run_surface("Wuerfel 4x4x4", cube_mesh(1.0, 4), 1, 0);
    run_surface("Kugel (Cube-Sphere) 6x6x6", sphere_mesh(1.0, 6), 1, 0);
    run_surface("Quader 3x2x4", cuboid_mesh(1.0, 0.5, 2.0, 3, 2, 4), 1, 0);
    {
        TorusMesh T = torus_mesh(3.0, 1.0, 16, 8);
        TraceSurface S = run_surface("Torus 16x8", T.mesh, 1, 1);
        torus_cycles("Torus 16x8", T, S, 0);
    }
    {
        TorusMesh T1 = torus_mesh(3.0, 1.0, 12, 6), T2 = torus_mesh(3.0, 1.0, 12, 6, v3(0, 0, 10));
        run_surface("zwei Tori", merge_meshes(T1.mesh, T2.mesh), 2, 2);
    }
    {
        TorusMesh T = torus_mesh(3.0, 1.0, 12, 6);
        QuadMesh box = cuboid_mesh(1.0, 1.0, 0.5, 3, 3, 2);
        run_surface("Quader + Torus", merge_meshes(translate_mesh(box, v3(0, 0, 8)), T.mesh), 2, 1);
    }
    std::printf(g_fail == 0 ? "ALLE TESTS BESTANDEN\n" : "FEHLGESCHLAGEN: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
