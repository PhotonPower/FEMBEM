// =============================================================================
// Abnahmepruefung: CoupledMVOperator::apply_many() gegen apply()
//
//   cmake --build build --config Release --target test_apply_many
//   ./build/Release/test_apply_many
//
// Reiner Host-Code, keine GPU, kein Netz.
//
// WORUM ES GEHT
// -------------
// apply_many() rechnet die G-/H-Bloecke fuer alle Spalten gemeinsam
// (dgemm) und den Rest spaltenweise. Das muss bis auf Rundung dasselbe
// ergeben wie apply() je Spalte. Der Aufbau deckt alles ab, was dabei
// schiefgehen kann:
//   * nicht quadratische Bloecke (vertauschte m/n oder Leading Dimensions
//     fielen bei quadratischen nicht auf),
//   * dichte, transponiert gespeicherte, ACA- und gespiegelte ACA-Bloecke
//     in G, dichte und ACA-Bloecke in H,
//   * Eichung mit Kohomologie-Projektoren, mit und ohne Aequilibrierung,
//   * eine Leading Dimension groesser als die Vektorlaenge.
// Ausserdem tastet sample_operator_dense() (jetzt ueber apply_many) diesen
// Operator ab, verglichen mit apply(e_j).
//
// GEGENPROBE
// ----------
// Ein Operator mit denselben Diagonalbloecken (also derselben
// Aequilibrierung), aber ohne die uebrigen G-/H-Bloecke, muss deutlich
// abweichen -- sonst saehe der Vergleich oben einen fehlenden Block nicht.
// =============================================================================

#include "Backend.cuh"
#include "CoupledSolver.cuh"
#include "HarmonicSolver.cuh"
#include "Mesh.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace bf { extern Backend g_backend; }

static int g_failures = 0;

static void check(bool ok, const char* what) {
    std::printf("  %-14s %s\n", ok ? "ok" : "FEHLGESCHLAGEN", what);
    if (!ok) ++g_failures;
}

static const int NF = 7;
static const int NFACES = 2;
static const int NB = NFACES * 4;
static const int N = NF + NB;

// Gruppen der BEM-Freiheitsgrade: [0,3), [3,5), [5,8) -- ungleich gross,
// damit die Nebendiagonalbloecke nicht quadratisch sind.
static const int g0 = 0, n0 = 3, g1 = 3, n1 = 2, g2 = 5, n2 = 3;

// Daten muessen leben, solange die Operatoren leben (nicht besitzende Zeiger).
struct Data {
    std::vector<int> row, col, map, sgn;
    std::vector<double> val, c0, c1;
    std::vector<char> gauged;
    std::vector<double3> nodes;
    std::vector<Mesh::QuadFace> faces;
    std::vector<double> G00, G11, G22, S10, U20, V20, U21, V21, U12, V12, H0, UH, VH;
};

static void fill(std::vector<double>& v, size_t n, std::mt19937& rng, double scale) {
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    v.resize(n);
    for (double& x : v) x = scale * uni(rng);
}

static void diag_block(std::vector<double>& v, int m, std::mt19937& rng) {
    fill(v, static_cast<size_t>(m) * m, rng, 1.0e7);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < i; ++j) v[static_cast<size_t>(j) * m + i] = v[static_cast<size_t>(i) * m + j];
    for (int i = 0; i < m; ++i) v[static_cast<size_t>(i) * m + i] = 8.0e8 + 1.0e7 * i;
}

static Data make_data() {
    Data d;
    std::mt19937 rng(11u);
    d.row.assign(NF + 1, 0);
    for (int i = 0; i < NF; ++i) {
        if (i > 0)      { d.col.push_back(i - 1); d.val.push_back(-1.0); }
        d.col.push_back(i); d.val.push_back(4.0 + 0.1 * i);
        if (i < NF - 1) { d.col.push_back(i + 1); d.val.push_back(-1.3); }
        d.row[i + 1] = static_cast<int>(d.col.size());
    }
    d.map = {0, 1, 2, 3, 3, 4, 5, 6};
    d.sgn = {1, -1, 1, 1, -1, 1, 1, -1};
    d.nodes = {{0.00, 0.00, 0.00}, {1.00, 0.05, 0.00}, {1.10, 1.00, 0.10},
               {0.05, 0.95, 0.00}, {2.10, 0.10, 0.05}, {2.00, 1.05, 0.00}};
    d.faces.resize(NFACES);
    d.faces[0].v[0] = 0; d.faces[0].v[1] = 1; d.faces[0].v[2] = 2; d.faces[0].v[3] = 3;
    d.faces[1].v[0] = 1; d.faces[1].v[1] = 4; d.faces[1].v[2] = 5; d.faces[1].v[3] = 2;

    diag_block(d.G00, n0, rng);
    diag_block(d.G11, n1, rng);
    diag_block(d.G22, n2, rng);
    fill(d.S10, static_cast<size_t>(n1) * n0, rng, 3.0e7);   // (g1,g0) 2x3, gespeichert
    fill(d.U20, static_cast<size_t>(n2) * 2, rng, 5.0e3);    // (g2,g0) 3x3, Rang 2
    fill(d.V20, static_cast<size_t>(n0) * 2, rng, 5.0e3);
    fill(d.U21, static_cast<size_t>(n2) * 1, rng, 5.0e3);    // (g2,g1) 3x2, Rang 1
    fill(d.V21, static_cast<size_t>(n1) * 1, rng, 5.0e3);
    fill(d.U12, static_cast<size_t>(n1) * 2, rng, 5.0e3);    // (g1,g2) 2x3, Rang 2
    fill(d.V12, static_cast<size_t>(n2) * 2, rng, 5.0e3);
    fill(d.H0, static_cast<size_t>(n0) * NB, rng, 0.3);      // H (g0, alle) 3x8 dicht
    fill(d.UH, static_cast<size_t>(NB - n0) * 2, rng, 0.4);  // H (g1.., alle) 5x8, Rang 2
    fill(d.VH, static_cast<size_t>(NB) * 2, rng, 0.4);

    d.gauged.assign(NF, 0);
    d.gauged[2] = 1; d.gauged[5] = 1;
    d.c0.assign(NF, 0.0); d.c1.assign(NF, 0.0);
    d.c0[0] = 1.0; d.c0[1] = -0.5; d.c0[3] = 0.7;
    d.c1[4] = 1.0; d.c1[6] = 0.3;
    return d;
}

static void build(CoupledMVOperator& op, Data& d, bool offdiag, bool equilibrate) {
    op.setup_fem_matrix(NF, static_cast<int>(d.val.size()), d.row.data(), d.col.data(), d.val.data());
    op.setup_bem_coupling(NB, d.map.data(), d.sgn.data());
    op.setup_bem_mesh(NFACES, d.faces.data(), d.nodes.data());
    op.add_dense_block(n0, n0, g0, g0, d.G00.data());
    op.add_dense_block(n1, n1, g1, g1, d.G11.data());
    op.add_dense_block(n2, n2, g2, g2, d.G22.data());
    if (offdiag) {
        op.add_dense_block(n1, n0, g1, g0, d.S10.data());
        op.add_dense_block_transposed(n0, n1, g0, g1, d.S10.data());
        op.add_aca_block(n2, n0, 2, g2, g0, d.U20.data(), d.V20.data());
        op.add_aca_block_mirrored(n0, n2, 2, g0, g2, d.U20.data(), d.V20.data());
        op.add_aca_block(n2, n1, 1, g2, g1, d.U21.data(), d.V21.data());
        op.add_aca_block(n1, n2, 2, g1, g2, d.U12.data(), d.V12.data());
        op.add_dense_block_H(n0, NB, g0, 0, d.H0.data());
        op.add_aca_block_H(NB - n0, NB, 2, g1, 0, d.UH.data(), d.VH.data());
    }
    op.setup_gauge(d.gauged.data());
    double s0 = 0, s1 = 0;
    for (int i = 0; i < NF; ++i) { s0 += d.c0[i] * d.c0[i]; s1 += d.c1[i] * d.c1[i]; }
    op.add_cohomology_projector(d.c0.data(), s0);
    op.add_cohomology_projector(d.c1.data(), s1);
    if (equilibrate) op.setup_equilibration();
}

// max |apply_many - apply| / max |apply| ueber nrhs Spalten mit ld > N
static double compare(CoupledMVOperator& op, int nrhs, std::vector<double>* ref_out = nullptr) {
    const int ld = N + 3;
    std::mt19937 rng(5u);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<double> X(static_cast<size_t>(ld) * nrhs), Y(X.size(), 0.0), R(X.size(), 0.0);
    for (double& x : X) x = uni(rng);
    op.apply_many(X.data(), Y.data(), ld, nrhs);
    for (int j = 0; j < nrhs; ++j)
        op.apply(X.data() + static_cast<size_t>(j) * ld, R.data() + static_cast<size_t>(j) * ld);
    double diff = 0.0, scale = 0.0;
    for (int j = 0; j < nrhs; ++j)
        for (int i = 0; i < N; ++i) {
            const size_t k = static_cast<size_t>(j) * ld + i;
            diff = std::max(diff, std::fabs(Y[k] - R[k]));
            scale = std::max(scale, std::fabs(R[k]));
        }
    if (ref_out) *ref_out = R;
    return diff / scale;
}

int main() {
    bf::g_backend = bf::Backend::CPU;
    Data d = make_data();
    char buf[200];

    std::printf("apply_many gegen apply, aequilibriert\n");
    CoupledMVOperator op(NF);
    build(op, d, true, true);
    std::vector<double> ref_full;
    const double e_eq = compare(op, 5, &ref_full);
    std::snprintf(buf, sizeof buf, "5 Spalten, ld = N+3: relative Abweichung %.1e < 1e-13", e_eq);
    check(e_eq < 1e-13, buf);
    const double e_one = compare(op, 1);
    std::snprintf(buf, sizeof buf, "1 Spalte (Rueckfall auf apply): %.1e", e_one);
    check(e_one == 0.0, buf);

    std::printf("apply_many gegen apply, ohne Aequilibrierung\n");
    CoupledMVOperator op_raw(NF);
    build(op_raw, d, true, false);
    const double e_raw = compare(op_raw, 4);
    std::snprintf(buf, sizeof buf, "4 Spalten: relative Abweichung %.1e < 1e-13", e_raw);
    check(e_raw < 1e-13, buf);

    std::printf("sample_operator_dense ueber apply_many\n");
    std::vector<double> A(static_cast<size_t>(N) * N, 0.0);
    bf::sample_operator_dense(op, N, A.data());
    double worst = 0.0, amax = 0.0;
    std::vector<double> e(N, 0.0), col(N, 0.0);
    for (int j = 0; j < N; ++j) {
        e[j] = 1.0;
        op.apply(e.data(), col.data());
        e[j] = 0.0;
        for (int i = 0; i < N; ++i) {
            worst = std::max(worst, std::fabs(A[static_cast<size_t>(j) * N + i] - col[i]));
            amax = std::max(amax, std::fabs(col[i]));
        }
    }
    std::snprintf(buf, sizeof buf, "Spalte j = apply(e_j), relative Abweichung %.1e < 1e-13",
                  worst / amax);
    check(worst / amax < 1e-13, buf);

    std::printf("Gegenprobe\n");
    CoupledMVOperator op_diag(NF);
    build(op_diag, d, false, true);
    std::vector<double> ref_diag;
    compare(op_diag, 5, &ref_diag);
    double dmax = 0.0, smax = 0.0;
    for (size_t k = 0; k < ref_full.size(); ++k) {
        dmax = std::max(dmax, std::fabs(ref_full[k] - ref_diag[k]));
        smax = std::max(smax, std::fabs(ref_full[k]));
    }
    std::snprintf(buf, sizeof buf,
                  "ohne Nebendiagonal- und H-Bloecke weicht y um %.2f (relativ) ab", dmax / smax);
    check(dmax / smax > 1e-2, buf);

    if (g_failures) {
        std::printf("\n%d Pruefung(en) fehlgeschlagen\n", g_failures);
        return 1;
    }
    std::printf("\nalle Pruefungen bestanden\n");
    return 0;
}
