// =============================================================================
// Host-Test: Calderon-Aussenraumtest der K-Zeile, ohne FEM
// (docs/design/bem-coupling-fixes.md, Schritt 4)
//
// Geschlossenes Quadnetz (Wuerfel, aussen orientiert), Punktquelle im Inneren:
// u(x) = e / |x - x0| (e fester Vektor, harmonisch im Aussenraum, abklingend).
// Dichte G, D, T mit denselben Integratoren wie die BEM-Bloecke (fem.cu:
// Sauter-Schwab fuer Selbst- und Nachbarpaare, Gauss sonst). a = Galerkin-
// Projektion der Tangentialspur P_t u, k = Projektion von (du/dn)_t.
//
// Aussenraum (Normale nach aussen): (1/2 - D) u = -G du/dn, also in den Zeilen
// des Operators  (+1/2 T - D) a + G k = 0. Der Code hatte (-1/2 T - D) a + G k
// (Innenraumform). Der Test misst beide und verlangt, dass die Aussenform
// deutlich kleiner ist.
//
// Grenze der Darstellung: D wirkt nur auf die TANGENTIALspur (Rooftop). Der
// Normalanteil u.n der Quelle faellt weg und fehlt an den Wuerfelkanten; das
// Restresiduum der Aussenform kommt daher (gemessen, nicht zu null zu bringen).
// =============================================================================

#include <cmath>
#include <cstdio>
#include <map>
#include <tuple>
#include <vector>

#include "BemIntegration.cuh"
#include "BemRowCheck.h"
#include "Mesh.cuh"
#include "SauterSchwab.cuh"
#include "SingularQuadPair.cuh"

using namespace bem_row_check;

static int g_fail = 0;
static void check(bool ok, const char* what, double v, double lim, bool greater = false) {
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, v,
                greater ? (ok ? ">" : "<=") : (ok ? "<" : ">="), lim);
    if (!ok) ++g_fail;
}

struct CubeMesh {
    std::vector<double3> nodes;
    std::vector<Mesh::QuadFace> faces;
};

// Wuerfel [-h, h]^3, n x n Quads je Seite, Knotenreihenfolge so, dass
// cross(dr_du, dr_dv) nach aussen zeigt.
static CubeMesh make_cube(int n, double h) {
    CubeMesh m;
    std::map<std::tuple<int, int, int>, int> id;
    auto node = [&](int i, int j, int k) {
        auto key = std::make_tuple(i, j, k);
        auto it = id.find(key);
        if (it != id.end()) return it->second;
        const int v = static_cast<int>(m.nodes.size());
        m.nodes.push_back(make_double3(-h + 2.0 * h * i / n, -h + 2.0 * h * j / n, -h + 2.0 * h * k / n));
        id[key] = v;
        return v;
    };
    // 6 Seiten: Achse a (fest), u-Achse b, v-Achse c, mit (b x c) = +a-Richtung fuer sign = +1
    const int axes[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};   // (fest, u, v) zyklisch: u x v = fest
    for (int ax = 0; ax < 3; ++ax) {
        for (int side = 0; side < 2; ++side) {
            const int fa = axes[ax][0], ua = axes[ax][1], va = axes[ax][2];
            for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) {
                int c[4][3];
                const int du[4] = {0, 1, 1, 0}, dv[4] = {0, 0, 1, 1};
                int order[4] = {0, 1, 2, 3};
                if (side == 0) { order[1] = 3; order[3] = 1; }   // innere Seite: Umlaufsinn umkehren
                int vid[4];
                for (int q = 0; q < 4; ++q) {
                    int idx[3];
                    idx[fa] = side ? n : 0;
                    idx[ua] = a + du[order[q]];
                    idx[va] = b + dv[order[q]];
                    vid[q] = node(idx[0], idx[1], idx[2]);
                    (void)c;
                }
                Mesh::QuadFace f;
                for (int q = 0; q < 4; ++q) f.v[q] = vid[q];
                f.parent_hex_id = 0; f.local_face_id = 0;
                m.faces.push_back(f);
            }
        }
    }
    return m;
}

int main() {
    std::printf("=== test_bem_calderon ===\n");
    const int n = 3;
    const double h = 0.05;
    CubeMesh cm = make_cube(n, h);
    const int nq = static_cast<int>(cm.faces.size());
    const int nb = 4 * nq;
    std::printf("  Wuerfel %d x %d je Seite: %d Quads, %zu Knoten, %d Freiheitsgrade\n", n, n, nq, cm.nodes.size(), nb);

    // Orientierung pruefen: alle Normalen nach aussen
    int inward = 0;
    std::vector<QuadGeom> geom(static_cast<size_t>(nq));
    for (int q = 0; q < nq; ++q) {
        for (int i = 0; i < 4; ++i) geom[q].n[i] = cm.nodes[cm.faces[q].v[i]];
        const V3 nrm = to_v3(evaluate_quad_normal(0, 0, geom[q].n));
        const V3 p = quad_point(geom[q].n, 0, 0);
        if (dot(nrm, p) < 0) { ++inward; geom[q].outward = -1.0; }
    }
    check(inward == 0, "alle Quadnormalen zeigen nach aussen", inward, 0.5);

    // Dichte Matrizen: wie compute_dense_block_kernel(_H)_body
    std::vector<double> G(static_cast<size_t>(nb) * nb), D(static_cast<size_t>(nb) * nb);
    for (int p = 0; p < nq; ++p) for (int q = 0; q < nq; ++q) {
        const Mesh::QuadFace &fp = cm.faces[p], &fq = cm.faces[q];
        double3 np_[4], nq_[4];
        int gp[4], gq[4];
        for (int i = 0; i < 4; ++i) { np_[i] = cm.nodes[fp.v[i]]; nq_[i] = cm.nodes[fq.v[i]]; gp[i] = fp.v[i]; gq[i] = fq.v[i]; }
        int shared = 0;
        for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) if (fp.v[i] == fq.v[j]) ++shared;
        double bg[4][4], bd[4][4];
        if (shared == 4) { integrate_singular_self_patch(np_, bg); for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) bd[r][c] = 0.0; }
        else if (shared == 1 || shared == 2) { integrate_singular_quad_pair(np_, gp, nq_, gq, 0, bg); integrate_singular_quad_pair(np_, gp, nq_, gq, 1, bd); }
        else { integrate_regular_quad_pair(np_, nq_, bg); integrate_regular_quad_pair_dlp(np_, nq_, bd); }
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
            G[static_cast<size_t>(4 * p + r) * nb + 4 * q + c] = bg[r][c];
            D[static_cast<size_t>(4 * p + r) * nb + 4 * q + c] = bd[r][c];
        }
    }
    std::vector<double> T(static_cast<size_t>(nb) * 4);   // blockdiagonal, 4 x 4 je Quad
    for (int q = 0; q < nq; ++q) {
        double Tq[4][4];
        integrate_boundary_mass_matrix(geom[q].n, Tq);
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) T[static_cast<size_t>(q) * 16 + r * 4 + c] = Tq[r][c];
    }

    // Quelle
    const V3 e{0.7, -0.4, 0.5}, x0{0.011, -0.017, 0.013};
    auto Fu = [&](V3 p, V3 nrm) {
        const V3 d = p - x0; const double r = norm(d);
        const V3 u = (1.0 / r) * e;
        return u - dot(u, nrm) * nrm;                    // P_t u
    };
    auto Fdn = [&](V3 p, V3 nrm) {
        const V3 d = p - x0; const double r = norm(d);
        const V3 dn = (-dot(d, nrm) / (r * r * r)) * e;  // (n.grad) u
        return dn - dot(dn, nrm) * nrm;                  // tangential
    };
    std::vector<double> a(static_cast<size_t>(nb)), k(static_cast<size_t>(nb));
    for (int q = 0; q < nq; ++q) {
        double ca[4], ck[4];
        project_to_rooftop(geom[q], Fu, ca);
        project_to_rooftop(geom[q], Fdn, ck);
        for (int i = 0; i < 4; ++i) { a[4 * q + i] = ca[i]; k[4 * q + i] = ck[i]; }
    }
    auto matvec = [&](const std::vector<double>& M, const std::vector<double>& x) {
        std::vector<double> y(static_cast<size_t>(nb), 0.0);
        for (int i = 0; i < nb; ++i) { double s = 0; for (int j = 0; j < nb; ++j) s += M[static_cast<size_t>(i) * nb + j] * x[j]; y[i] = s; }
        return y;
    };
    std::vector<double> Da = matvec(D, a), Gk = matvec(G, k), Ta(static_cast<size_t>(nb));
    for (int q = 0; q < nq; ++q) for (int r = 0; r < 4; ++r) {
        double s = 0; for (int c = 0; c < 4; ++c) s += T[static_cast<size_t>(q) * 16 + r * 4 + c] * a[4 * q + c];
        Ta[4 * q + r] = s;
    }
    auto resid = [&](double half, double dsign, double ksign, double& nrm_ref) {
        double nr = 0, nn = 0;
        for (int i = 0; i < nb; ++i) {
            const double lin = half * 0.5 * Ta[i] + dsign * Da[i];
            const double r = lin + ksign * Gk[i];
            nr += r * r; nn += lin * lin;
        }
        nrm_ref = std::sqrt(nn);
        return std::sqrt(nr / nn);
    };
    double ref, ext_p, ext_m, int_p, int_m;
    ext_p = resid(+1.0, -1.0, +1.0, ref);   // (+1/2 T - D) a + G k   (Aussenraum)
    ext_m = resid(+1.0, -1.0, -1.0, ref);
    int_p = resid(-1.0, -1.0, +1.0, ref);   // (-1/2 T - D) a + G k   (Innenraumform, wie der Code bisher)
    int_m = resid(-1.0, -1.0, -1.0, ref);
    std::printf("  ||G k|| = %.4e, ||D a|| = %.4e, ||1/2 T a|| = %.4e\n", std::sqrt([&]{double s=0;for(double x:Gk)s+=x*x;return s;}()),
                std::sqrt([&]{double s=0;for(double x:Da)s+=x*x;return s;}()),
                std::sqrt([&]{double s=0;for(double x:Ta)s+=0.25*x*x;return s;}()));
    std::printf("  Aussenform (+1/2 T - D) a + G k:  k = +du/dn: %.2f %%,  k = -du/dn: %.2f %%\n", 100 * ext_p, 100 * ext_m);
    std::printf("  Innenform  (-1/2 T - D) a + G k:  k = +du/dn: %.2f %%,  k = -du/dn: %.2f %%\n", 100 * int_p, 100 * int_m);
    check(ext_p < 0.3, "Aussenform verschwindet mit k = +du/dn (gemessene Schwelle, Wuerfelkanten begrenzen)", ext_p, 0.3);
    check(ext_p < 0.5 * std::fmin(int_p, int_m), "Gegenprobe: Innenraumform wird erkannt (mind. doppelt so gross)", int_p / ext_p, 2.0, true);
    check(ext_m > 0.5, "Gegenprobe: falsches Vorzeichen von k wird erkannt", ext_m, 0.5, true);

    std::printf("\n%s (%d Fehler)\n", g_fail ? "FEHLER" : "ALLE PRUEFUNGEN BESTANDEN", g_fail);
    return g_fail ? 1 : 0;
}
