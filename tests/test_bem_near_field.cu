// =============================================================================
// Host-Test: Nahfeldquadratur disjunkter Paare und duenner Quader als Calderon-Test
// (docs/design/bem-near-field.md)
//
//   B1  Paartest ohne FEM: zwei parallele, gegenueberliegende Quads (Kantenlaenge s,
//       Abstand d), quadratisch und im Plattenaspekt, auch seitlich versetzt:
//       relativer Fehler der 4x4-Bloecke von integrate_regular_quad_pair (G) und
//       integrate_regular_quad_pair_dlp (D) gegen eine gleichmaessige Unterteilung
//       beider Parametergebiete (integrate_adjacent_quad_pair(_dlp), Konvergenz der
//       Referenz wird belegt).
//   B2  Duenner Quader (Plattenaspekt) und Wuerfel gleicher Panelkante als
//       Calderon-Aussenraumtest wie test_bem_calderon.cu, Residuum der K-Zeile gesamt
//       und getrennt nach Deckel/Boden und Seiten.
//
// Aufruf:  test_bem_near_field            schnelle Pruefungen (ctest, Marke host)
//          test_bem_near_field --table    Tabellen fuer den Bericht (dauert Minuten)
// Reiner Host-Code, keine GPU.
// =============================================================================

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// ---------------------------------------------------------------------------
// B1: Paartest
// ---------------------------------------------------------------------------
static double block_rel(const double a[4][4], const double b[4][4]) {
    double n = 0, d = 0;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { n += (a[i][j] - b[i][j]) * (a[i][j] - b[i][j]); d += b[i][j] * b[i][j]; }
    return std::sqrt(n / d);
}

struct PairGeom { double3 p[4], q[4]; };

// p unten (z = 0, Normale -z), q oben (z = d, Normale +z), Rechtecke sx x sy, q um (ox, oy) versetzt
static PairGeom make_pair(double sx, double sy, double d, double ox, double oy) {
    PairGeom g;
    g.p[0] = make_double3(0, 0, 0);   g.p[1] = make_double3(0, sy, 0);
    g.p[2] = make_double3(sx, sy, 0); g.p[3] = make_double3(sx, 0, 0);
    g.q[0] = make_double3(ox, oy, d);       g.q[1] = make_double3(ox + sx, oy, d);
    g.q[2] = make_double3(ox + sx, oy + sy, d); g.q[3] = make_double3(ox, oy + sy, d);
    return g;
}

struct PairErr { double g, d, gconv, dconv, gn, dn; int lvl; };

// Fehler der Produktionsregel und Konvergenz der Referenz (Unterteilung n_ref gegen 2 n_ref)
static PairErr pair_errors(const PairGeom& pg, int n_ref, bool swap_roles = false) {
    const double3* P = swap_roles ? pg.q : pg.p;
    const double3* Q = swap_roles ? pg.p : pg.q;
    double g[4][4], d[4][4], gr[4][4], dr[4][4], gr2[4][4], dr2[4][4];
    integrate_regular_quad_pair(P, Q, g);
    integrate_regular_quad_pair_dlp(P, Q, d);
    integrate_adjacent_quad_pair(P, Q, gr, n_ref);
    integrate_adjacent_quad_pair_dlp(P, Q, dr, n_ref);
    integrate_adjacent_quad_pair(P, Q, gr2, 2 * n_ref);
    integrate_adjacent_quad_pair_dlp(P, Q, dr2, 2 * n_ref);
    PairErr e;
    // Prototyp --bem-near-subdiv (N = 8): abstandsabhaengige Unterteilung
    e.lvl = near_subdivision_level(P, Q, 8);
    double gn[4][4], dn[4][4];
    if (e.lvl > 0) { integrate_adjacent_quad_pair(P, Q, gn, e.lvl); integrate_adjacent_quad_pair_dlp(P, Q, dn, e.lvl); }
    else { std::memcpy(gn, g, sizeof gn); std::memcpy(dn, d, sizeof dn); }
    e.gn = block_rel(gn, gr2); e.dn = block_rel(dn, dr2);
    e.g = block_rel(g, gr2); e.d = block_rel(d, dr2);
    e.gconv = block_rel(gr, gr2); e.dconv = block_rel(dr, dr2);
    return e;
}

static const double kDs[7] = {0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0};

static void run_pairs(bool table, int n_ref) {
    std::printf("[B1] Paartest (Referenz: gleichmaessige Unterteilung %d x %d bzw. %d x %d je Quad)\n", n_ref, n_ref, 2 * n_ref, 2 * n_ref);
    struct Case { const char* name; double sx, sy, ox, oy; };
    const Case cases[4] = {
        {"quadratisch, ausgerichtet", 1.0, 1.0, 0.0, 0.0},
        {"quadratisch, um s/2 versetzt", 1.0, 1.0, 0.5, 0.0},
        {"Rechteck 1:2, ausgerichtet", 1.0, 2.0, 0.0, 0.0},
        {"Rechteck 1:2, um s/2 in x und y versetzt", 1.0, 2.0, 0.5, 0.5}};
    struct Res { double eg, ed, cg, cd, gn, dn; int lvl; };
    std::vector<Res> res(4 * 7);
    parallel_for(4 * 7, [&](size_t it) {
        const Case& c = cases[it / 7];
        const int k = static_cast<int>(it % 7);
        const PairGeom pg = make_pair(c.sx, c.sy, kDs[k] * c.sx, c.ox, c.oy);   // d/s mit s = sx
        const PairErr e = pair_errors(pg, n_ref);
        const PairErr e2 = pair_errors(pg, n_ref, true);   // q als Beobachter: anderer Normalenfaktor bei D
        res[it] = {std::max(e.g, e2.g), std::max(e.d, e2.d), std::max(e.gconv, e2.gconv), std::max(e.dconv, e2.dconv),
                   std::max(e.gn, e2.gn), std::max(e.dn, e2.dn), std::max(e.lvl, e2.lvl)};
    });
    if (table) std::printf("\n| Fall | d/s | Fehler G | Fehler D | Konvergenz Ref. G | Konvergenz Ref. D | Level | Fehler G mit Unterteilung | Fehler D mit Unterteilung |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    double worst_conv = 0;
    double thr_g = 0, thr_d = 0;
    double err_at[7][2] = {}, err_near[7][2] = {};
    for (size_t it = 0; it < res.size(); ++it) {
        const int k = static_cast<int>(it % 7);
        err_at[k][0] = std::max(err_at[k][0], res[it].eg);
        err_at[k][1] = std::max(err_at[k][1], res[it].ed);
        if (kDs[k] >= 0.2) worst_conv = std::max({worst_conv, res[it].cg, res[it].cd});
        err_near[k][0] = std::max(err_near[k][0], res[it].gn);
        err_near[k][1] = std::max(err_near[k][1], res[it].dn);
        if (table) std::printf("| %s | %.2f | %.2e | %.2e | %.1e | %.1e | %d | %.2e | %.2e |\n", cases[it / 7].name, kDs[k], res[it].eg, res[it].ed, res[it].cg, res[it].cd, res[it].lvl, res[it].gn, res[it].dn);
    }
    for (int type = 0; type < 2; ++type) {
        double thr = 0;
        for (int k = 6; k >= 0; --k) { if (err_at[k][type] >= 0.01) { thr = kDs[std::min(k + 1, 6)]; break; } if (k == 0) thr = kDs[0]; }
        (type == 0 ? thr_g : thr_d) = thr;
    }
    if (table) {
        std::printf("\nMaximum ueber die Faelle, je d/s (Produktionsregel | mit --bem-near-subdiv=8):\n\n| d/s | Fehler G | Fehler D | Fehler G (Unterteilung) | Fehler D (Unterteilung) |\n|---:|---:|---:|---:|---:|\n");
        for (int k = 0; k < 7; ++k) std::printf("| %.2f | %.2e | %.2e | %.2e | %.2e |\n", kDs[k], err_at[k][0], err_at[k][1], err_near[k][0], err_near[k][1]);
    }
    std::printf("      Schwelle d/s, ab der alle groesseren Abstaende unter 1 %% liegen: G %.2f, D %.2f\n", thr_g, thr_d);
    check(worst_conv < (table ? 1e-3 : 3e-3), "Referenz konvergiert (n gegen 2n, d/s >= 0.2)", worst_conv, table ? 1e-3 : 3e-3);
    check(err_at[0][0] > 0.05, "Gegenprobe: bei d/s = 0.05 ist die Produktionsregel fuer G nicht genau (> 5 %)", err_at[0][0], 0.05, true);
    check(err_at[6][0] < 0.01, "bei d/s = 4 ist G unter 1 %", err_at[6][0], 0.01);
    check(err_near[3][0] < 0.01 && err_near[3][1] < 0.02, "Unterteilung N = 8: bei d/s = 0.5 G unter 1 %, D unter 2 %", std::max(err_near[3][0], err_near[3][1] / 2), 0.01);
    check(err_near[4][1] < err_at[4][1] && err_near[3][1] < err_at[3][1], "Unterteilung verbessert D bei d/s = 0.5 und 1", err_near[3][1], err_at[3][1]);
}

// ---------------------------------------------------------------------------
// B2: Calderon-Aussenraumtest, Quader nx x ny x nz Zellen
// ---------------------------------------------------------------------------
struct BoxMesh { std::vector<double3> nodes; std::vector<Mesh::QuadFace> faces; };

// Quader [0,Lx] x [0,Ly] x [0,Lz], nx x ny x nz Zellen, Normalen nach aussen
static BoxMesh make_box(double Lx, double Ly, double Lz, int nx, int ny, int nz) {
    BoxMesh m;
    std::map<std::tuple<int, int, int>, int> id;
    auto node = [&](int i, int j, int k) {
        auto key = std::make_tuple(i, j, k);
        auto it = id.find(key);
        if (it != id.end()) return it->second;
        const int v = static_cast<int>(m.nodes.size());
        m.nodes.push_back(make_double3(Lx * i / nx, Ly * j / ny, Lz * k / nz));
        id[key] = v;
        return v;
    };
    const int N[3] = {nx, ny, nz};
    const int axes[3][3] = {{0, 1, 2}, {1, 2, 0}, {2, 0, 1}};
    for (int ax = 0; ax < 3; ++ax) for (int side = 0; side < 2; ++side) {
        const int fa = axes[ax][0], ua = axes[ax][1], va = axes[ax][2];
        for (int a = 0; a < N[ua]; ++a) for (int b = 0; b < N[va]; ++b) {
            const int du[4] = {0, 1, 1, 0}, dv[4] = {0, 0, 1, 1};
            int order[4] = {0, 1, 2, 3};
            if (side == 0) { order[1] = 3; order[3] = 1; }
            Mesh::QuadFace f;
            for (int q = 0; q < 4; ++q) {
                int idx[3];
                idx[fa] = side ? N[fa] : 0;
                idx[ua] = a + du[order[q]];
                idx[va] = b + dv[order[q]];
                f.v[q] = node(idx[0], idx[1], idx[2]);
            }
            f.parent_hex_id = 0; f.local_face_id = 0;
            m.faces.push_back(f);
        }
    }
    return m;
}

struct CalderonResult { double total, lids, sides; int nq; double ref_total; };

// Residuum (+1/2 T - D) a + G k mit k = +du/dn, ohne dichte Matrizen (Zeilen werden
// nacheinander gebildet). near_nsub > 0: disjunkte Paare mit d/s-Kriterium unterteilen
// (Prototyp, siehe Schritt C; 0 = Produktionsregel).
static CalderonResult calderon(const BoxMesh& cm, const V3& e, const V3& x0, int near_nsub = 0) {
    const int nq = static_cast<int>(cm.faces.size());
    const int nb = 4 * nq;
    std::vector<QuadGeom> geom(static_cast<size_t>(nq));
    for (int q = 0; q < nq; ++q) for (int i = 0; i < 4; ++i) geom[q].n[i] = cm.nodes[cm.faces[q].v[i]];
    auto Fu = [&](V3 p, V3 nrm) { const V3 d = p - x0; const V3 u = (1.0 / norm(d)) * e; return u - dot(u, nrm) * nrm; };
    auto Fdn = [&](V3 p, V3 nrm) {
        const V3 d = p - x0; const double r = norm(d);
        const V3 dn = (-dot(d, nrm) / (r * r * r)) * e; return dn - dot(dn, nrm) * nrm; };
    std::vector<double> a(static_cast<size_t>(nb)), k(static_cast<size_t>(nb));
    for (int q = 0; q < nq; ++q) {
        double ca[4], ck[4];
        project_to_rooftop(geom[q], Fu, ca);
        project_to_rooftop(geom[q], Fdn, ck);
        for (int i = 0; i < 4; ++i) { a[4 * q + i] = ca[i]; k[4 * q + i] = ck[i]; }
    }
    std::vector<double> Da(static_cast<size_t>(nb), 0.0), Gk(static_cast<size_t>(nb), 0.0), Ta(static_cast<size_t>(nb), 0.0);
    parallel_for(static_cast<size_t>(nq), [&](size_t p) {
        double np_[4][3];
        (void)np_;
        double3 P[4]; int gp[4];
        for (int i = 0; i < 4; ++i) { P[i] = cm.nodes[cm.faces[p].v[i]]; gp[i] = cm.faces[p].v[i]; }
        for (int q = 0; q < nq; ++q) {
            double3 Q[4]; int gq[4];
            for (int i = 0; i < 4; ++i) { Q[i] = cm.nodes[cm.faces[q].v[i]]; gq[i] = cm.faces[q].v[i]; }
            int shared = 0;
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) if (cm.faces[p].v[i] == cm.faces[q].v[j]) ++shared;
            double bg[4][4], bd[4][4];
            if (shared == 4) { integrate_singular_self_patch(P, bg); for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) bd[r][c] = 0.0; }
            else if (shared == 1 || shared == 2) { integrate_singular_quad_pair(P, gp, Q, gq, 0, bg); integrate_singular_quad_pair(P, gp, Q, gq, 1, bd); }
            else {
                const int lvl = near_nsub > 0 ? near_subdivision_level(P, Q, near_nsub) : 0;
                if (lvl > 0) { integrate_adjacent_quad_pair(P, Q, bg, lvl); integrate_adjacent_quad_pair_dlp(P, Q, bd, lvl); }
                else { integrate_regular_quad_pair(P, Q, bg); integrate_regular_quad_pair_dlp(P, Q, bd); }
            }
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) {
                Gk[4 * p + r] += bg[r][c] * k[4 * q + c];
                Da[4 * p + r] += bd[r][c] * a[4 * q + c];
            }
        }
        double Tq[4][4];
        integrate_boundary_mass_matrix(geom[p].n, Tq);
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) Ta[4 * p + r] += Tq[r][c] * a[4 * p + c];
    });
    double nr[3] = {0, 0, 0}, nn[3] = {0, 0, 0};   // 0 gesamt, 1 Deckel/Boden, 2 Seiten
    for (int q = 0; q < nq; ++q) {
        const V3 nrm = to_v3(evaluate_quad_normal(0, 0, geom[q].n));
        const int cls = std::fabs(nrm.z) > 0.99 ? 1 : 2;
        for (int r = 0; r < 4; ++r) {
            const int i = 4 * q + r;
            const double lin = 0.5 * Ta[i] - Da[i];
            const double res = lin + Gk[i];
            nr[0] += res * res; nn[0] += lin * lin; nr[cls] += res * res; nn[cls] += lin * lin;
        }
    }
    CalderonResult out;
    out.total = std::sqrt(nr[0] / nn[0]); out.lids = nn[1] > 0 ? std::sqrt(nr[1] / nn[1]) : 0;
    out.sides = nn[2] > 0 ? std::sqrt(nr[2] / nn[2]) : 0; out.nq = nq; out.ref_total = std::sqrt(nn[0]);
    return out;
}

static void run_boxes(bool table) {
    std::printf("[B2] Calderon-Aussenraumtest: duenner Quader 60 x 60 x 3 mm und Wuerfel 60 mm\n");
    const V3 e{0.7, -0.4, 0.5};
    std::vector<double> hs = table ? std::vector<double>{10.0, 5.0, 2.5} : std::vector<double>{10.0, 5.0};
    if (table) std::printf("\n| Koerper | Panelkante h [mm] | Deckel h/t | Quads | Residuum gesamt | Deckel/Boden | Seiten |\n|---|---:|---:|---:|---:|---:|---:|\n");
    double plate_total[3] = {}, cube_total[3] = {};
    int idx = 0;
    for (double h : hs) {
        const int n = static_cast<int>(std::lround(60.0 / h));
        const int nzp = std::max(1, static_cast<int>(std::lround(3.0 / h)));
        const BoxMesh plate = make_box(0.060, 0.060, 0.003, n, n, nzp);
        const BoxMesh cube = make_box(0.060, 0.060, 0.060, n, n, n);
        const V3 x0p{0.0301, 0.0293, 0.00151}, x0c{0.0301, 0.0293, 0.0302};
        const CalderonResult rp = calderon(plate, e, x0p), rc = calderon(cube, e, x0c);
        const CalderonResult rpn = calderon(plate, e, x0p, 8), rcn = calderon(cube, e, x0c, 8);
        plate_total[idx] = rp.total; cube_total[idx] = rc.total; ++idx;
        std::printf("      h = %.1f mm: Quader %d Quads: gesamt %.2f %%, Deckel/Boden %.2f %%, Seiten %.2f %%; Wuerfel %d Quads: %.2f %%\n",
                    h, rp.nq, 100 * rp.total, 100 * rp.lids, 100 * rp.sides, rc.nq, 100 * rc.total);
        if (table) {
            std::printf("| Quader 60x60x3 | %.1f | %.2f | %d | %.2f %% | %.2f %% | %.2f %% |\n", h, h / 3.0, rp.nq, 100 * rp.total, 100 * rp.lids, 100 * rp.sides);
            std::printf("| Wuerfel 60 | %.1f | - | %d | %.2f %% | %.2f %% | %.2f %% |\n", h, rc.nq, 100 * rc.total, 100 * rc.lids, 100 * rc.sides);
            std::printf("| Quader 60x60x3, --bem-near-subdiv=8 | %.1f | %.2f | %d | %.2f %% | %.2f %% | %.2f %% |\n", h, h / 3.0, rpn.nq, 100 * rpn.total, 100 * rpn.lids, 100 * rpn.sides);
            std::printf("| Wuerfel 60, --bem-near-subdiv=8 | %.1f | - | %d | %.2f %% | %.2f %% | %.2f %% |\n", h, rcn.nq, 100 * rcn.total, 100 * rcn.lids, 100 * rcn.sides);
        }
    }
    check(plate_total[0] < 5.0 && cube_total[0] < 5.0, "Residuen endlich und kleiner als 500 % (Aufbau stimmt)", std::max(plate_total[0], cube_total[0]), 5.0);
}

int main(int argc, char** argv) {
    const bool table = argc > 1 && std::strcmp(argv[1], "--table") == 0;
    std::printf("=== test_bem_near_field%s ===\n", table ? " --table" : "");
    run_pairs(table, table ? 24 : 8);
    run_boxes(table);
    std::printf("\n%s (%d Fehler)\n", g_fail ? "FEHLER" : "ALLE PRUEFUNGEN BESTANDEN", g_fail);
    return g_fail ? 1 : 0;
}
