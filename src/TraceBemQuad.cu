// =============================================================================
// Quadratur der Randoperatoren der Maxwell-Spur-Kopplung (Phase 1b, Schritt 2), Host-Code.
// Die Koordinatenabbildungen der Sauter-Schwab-Transformationen (gleiches Dreieck, gemeinsame Kante,
// gemeinsame Ecke) und die Permutationshelfer kommen unveraendert aus SauterSchwab.cuh /
// SingularQuadPair.cuh; neu sind die Integranden und die frei waehlbare Gauss-Ordnung.
// =============================================================================
#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>

#include "SauterSchwab.cuh"
#include "SingularQuadPair.cuh"
#include "TraceBemOps.h"

namespace trace {
namespace {

// ---- Gauss-Legendre auf [0,1], Ordnung n (Newton auf P_n) ----
struct Gauss01 {
    std::vector<double> x, w;
    explicit Gauss01(int n = 4) : x(n), w(n)
    {
        for (int i = 0; i < n; ++i) {
            double z = std::cos(M_PI * (i + 0.75) / (n + 0.5));
            double pp = 0.0;
            for (int it = 0; it < 100; ++it) {
                double p1 = 1.0, p2 = 0.0;
                for (int j = 0; j < n; ++j) { const double p3 = p2; p2 = p1; p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0); }
                pp = n * (z * p1 - p2) / (z * z - 1.0);
                const double z1 = z;
                z = z1 - p1 / pp;
                if (std::fabs(z - z1) < 1e-15) break;
            }
            // i-ter Knoten absteigend; auf [0,1] abbilden
            x[i] = 0.5 * (1.0 - z);
            w[i] = 1.0 / ((1.0 - z * z) * pp * pp);
        }
    }
};

// ---- Punkt auf dem Quad ----
struct QPoint {
    double3 r, n;
    double detJ;
    double3 W[4];   // kovariante Rooftop (lokale Kanten, ohne globales Vorzeichen)
    double N[4];    // bilineare Formfunktionen
};

inline QPoint eval_quad(const double3 nodes[4], double u, double v)
{
    QPoint p;
    evaluate_rooftop_basis(u, v, nodes, p.W, p.detJ);
    p.r = evaluate_quad_position(u, v, nodes);
    p.n = evaluate_quad_normal(u, v, nodes);
    p.N[0] = 0.25 * (1 - u) * (1 - v); p.N[1] = 0.25 * (1 + u) * (1 - v);
    p.N[2] = 0.25 * (1 + u) * (1 + v); p.N[3] = 0.25 * (1 - u) * (1 + v);
    return p;
}

inline double3 d3(double x, double y, double z) { return make_double3(x, y, z); }

// ---- Integrand: Beitrag eines Punktpaares zu allen Bloecken ----
inline void accumulate(const QPoint& a, const QPoint& b, double w, PairResult& out)
{
    const double3 d = sub3(a.r, b.r);
    const double R2 = dot3(d, d);
    if (R2 < 1e-30) return;
    const double R = std::sqrt(R2);
    const double f = w * a.detJ * b.detJ;
    const double G = 1.0 / (4.0 * M_PI * R);
    const double R3 = R2 * R;
    out.v0 += f * G;
    const double K = dot3(d, b.n) / (4.0 * M_PI * R3);
    for (int j = 0; j < 4; ++j) out.dt[j] += f * K * b.N[j];
    double3 A[4], B[4];
    for (int e = 0; e < 4; ++e) { A[e] = cross3(a.n, a.W[e]); B[e] = cross3(b.n, b.W[e]); }
    const double3 gradG = scale3(-1.0 / (4.0 * M_PI * R3), d);
    const double ngrad = dot3(a.n, gradG);
    for (int e = 0; e < 4; ++e) {
        const double wg = dot3(a.W[e], gradG);
        for (int g = 0; g < 4; ++g) {
            out.vr[e][g] += f * G * dot3(A[e], B[g]);
            out.vrf[e][g] += f * G * dot3(a.W[e], b.W[g]);
            out.cm[e][g] += f * (wg * dot3(a.n, b.W[g]) - dot3(a.W[e], b.W[g]) * ngrad);
        }
    }
}

struct QuadGeo {
    double3 x[4];
    int gid[4];
};

// ---- Getrennte Paare: unterteilte Tensor-Gauss-Regel ----
void integrate_regular(const QuadGeo& P, const QuadGeo& Q, const QuadParams& qp, const Gauss01& g, PairResult& out)
{
    // Unterteilung aus Quadgroesse und Abstand (neun Stuetzpunkte je Quad)
    auto diam = [](const QuadGeo& A) {
        double h = 0.0;
        for (int i = 0; i < 4; ++i) h = std::max(h, norm3(sub3(A.x[i], A.x[(i + 2) % 4])));
        return h;
    };
    double dmin = 1e300;
    const double ss[3] = {-1.0, 0.0, 1.0};
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) {
        const double3 pa = evaluate_quad_position(ss[a], ss[b], P.x);
        for (int c = 0; c < 3; ++c) for (int e = 0; e < 3; ++e)
            dmin = std::min(dmin, norm3(sub3(pa, evaluate_quad_position(ss[c], ss[e], Q.x))));
    }
    const double h = std::max(diam(P), diam(Q));
    int m = 1;
    if (dmin > 0.0) m = static_cast<int>(std::ceil(qp.near_c * h / dmin));
    m = std::max(1, std::min(m, qp.max_sub));

    const int n = static_cast<int>(g.x.size());
    // Punkte beider Quads vorab (m*n)^2 je Quad
    const int np = m * n;
    std::vector<QPoint> pp, qq;
    std::vector<double> pw, qw;
    auto fill = [&](const QuadGeo& A, std::vector<QPoint>& pts, std::vector<double>& wts) {
        const double cell = 2.0 / m;
        for (int ci = 0; ci < m; ++ci) for (int cj = 0; cj < m; ++cj)
            for (int a = 0; a < n; ++a) for (int b = 0; b < n; ++b) {
                const double u = -1.0 + cell * (ci + g.x[a]), v = -1.0 + cell * (cj + g.x[b]);
                pts.push_back(eval_quad(A.x, u, v));
                wts.push_back(cell * cell * g.w[a] * g.w[b]);
            }
    };
    fill(P, pp, pw);
    fill(Q, qq, qw);
    (void)np;
    for (size_t i = 0; i < pp.size(); ++i)
        for (size_t j = 0; j < qq.size(); ++j) accumulate(pp[i], qq[j], pw[i] * qw[j], out);
}

// ---- Gleiches Quad: Duffy-Regel in der Differenzvariablen h = y - x (Parameterraum [-1,1]^2) ----
// Die Sauter-Schwab-Abbildung fuer identische Dreiecke aus SauterSchwab.cuh (get_coincident_coords) hebt nur die
// Singularitaet im Ursprung auf, nicht die auf der ganzen Diagonalen x = y; gemessen: Fehler 2,8 % (ng = 4),
// 0,75 % (ng = 8) fuer int int 1/R ueber das Einheitsquadrat, algebraische Konvergenz. Hier statt dessen:
//   int_Q int_Q F(x,y) = int_{h in [-2,2]^2} int_{x in Q cap (Q-h)} F(x, x+h) dx dh.
// Fuer festes h ist das x-Gebiet ein Rechteck; das h-Quadrat wird durch seine Diagonalen in vier Dreiecke
// geteilt, in jedem gilt h = r (1, s) bzw. Vertauschung/Vorzeichen, dh = r dr ds (s in zwei Haelften). Der Faktor r kuerzt die
// 1/R-Singularitaet (R ~ r), der Integrand ist in (r, s, x) glatt.
void integrate_coincident(const QuadGeo& P, const Gauss01& g, PairResult& out)
{
    const int n = static_cast<int>(g.x.size());
    // Je Dreieck zwei Haelften s in [-1,0] und [0,1]: bei s = 0 wechselt das Vorzeichen der Nebenkomponente von h und
    // damit die Grenze des x-Rechtecks (Knick), jede Haelfte ist glatt.
    for (int side = 0; side < 4; ++side) for (int half = 0; half < 2; ++half)
        for (int i1 = 0; i1 < n; ++i1) for (int i2 = 0; i2 < n; ++i2) {
            const double r = 2.0 * g.x[i1], sv = (half == 0 ? -1.0 : 0.0) + g.x[i2];
            double h1, h2;
            if (side == 0)      { h1 = r;  h2 = r * sv; }
            else if (side == 1) { h1 = -r; h2 = r * sv; }
            else if (side == 2) { h2 = r;  h1 = r * sv; }
            else                { h2 = -r; h1 = r * sv; }
            const double lo1 = std::max(-1.0, -1.0 - h1), hi1 = std::min(1.0, 1.0 - h1);
            const double lo2 = std::max(-1.0, -1.0 - h2), hi2 = std::min(1.0, 1.0 - h2);
            const double wh = 2.0 * r * g.w[i1] * g.w[i2] * (hi1 - lo1) * (hi2 - lo2);   // dr = 2 dt1, ds = dt2
            for (int j1 = 0; j1 < n; ++j1) for (int j2 = 0; j2 < n; ++j2) {
                const double u = lo1 + (hi1 - lo1) * g.x[j1], v = lo2 + (hi2 - lo2) * g.x[j2];
                const QPoint a = eval_quad(P.x, u, v), b = eval_quad(P.x, u + h1, v + h2);
                accumulate(a, b, wh * g.w[j1] * g.w[j2], out);
            }
        }
}

// ---- Beruehrende Paare (gemeinsame Kante, gemeinsame Ecke): Sauter-Schwab ----
void integrate_touching(const QuadGeo& P, const QuadGeo& Q, bool same, const Gauss01& g, PairResult& out)
{
    const int n = static_cast<int>(g.x.size());
    for (int tp = 0; tp < 2; ++tp) for (int tq = 0; tq < 2; ++tq) {
        int lp[3], lq[3];
        sqp_triangle_nodes(tp, lp);
        sqp_triangle_nodes(tq, lq);
        int n_shared = 0, sp[2] = {0, 0}, sq[2] = {0, 0};
        const bool coincident = same && tp == tq;
        if (!coincident) {
            for (int a = 0; a < 3 && n_shared < 2; ++a)
                for (int b = 0; b < 3; ++b)
                    if (P.gid[lp[a]] == Q.gid[lq[b]]) { sp[n_shared] = a; sq[n_shared] = b; ++n_shared; break; }
        }
        int perm_p[3] = {0, 1, 2}, perm_q[3] = {0, 1, 2};
        if (!coincident) { sqp_build_permutation(sp, n_shared, perm_p); sqp_build_permutation(sq, n_shared, perm_q); }
        const int n_cases = coincident ? 6 : (n_shared == 2 ? 5 : (n_shared == 1 ? 2 : 1));
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) for (int l = 0; l < n; ++l) {
            const double xi = g.x[i], e1 = g.x[j], e2 = g.x[k], e3 = g.x[l];
            const double gw = g.w[i] * g.w[j] * g.w[k] * g.w[l];
            for (int c = 0; c < n_cases; ++c) {
                double x1, x2, y1, y2, jac;
                if (coincident) {
                    get_coincident_coords(c, xi, e1, e2, e3, x1, x2, y1, y2);
                    jac = xi * xi * xi * e1 * e1 * e2;
                } else if (n_shared == 2) get_common_edge_coords(c, xi, e1, e2, e3, x1, x2, y1, y2, jac);
                else if (n_shared == 1) get_common_vertex_coords(c, xi, e1, e2, e3, x1, x2, y1, y2, jac);
                else { x1 = xi; x2 = xi * e1; y1 = e2; y2 = e2 * e3; jac = xi * e2; }
                double up, vp, uq, vq;
                sqp_map_to_quad(x1 - x2, x2, tp, perm_p, up, vp);
                sqp_map_to_quad(y1 - y2, y2, tq, perm_q, uq, vq);
                const QPoint a = eval_quad(P.x, up, vp), b = eval_quad(Q.x, uq, vq);
                accumulate(a, b, gw * 16.0 * jac, out);
            }
        }
    }
}

int shared_nodes(const QuadGeo& P, const QuadGeo& Q)
{
    int s = 0;
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) if (P.gid[a] == Q.gid[b]) ++s;
    return s;
}

QuadGeo make_geo(const TraceSurface& S, int q)
{
    QuadGeo g;
    for (int i = 0; i < 4; ++i) {
        const Vec3& p = S.X[S.quads[q][i]];
        g.x[i] = make_double3(p.x, p.y, p.z);
        g.gid[i] = S.quads[q][i];
    }
    return g;
}

PairResult pair(const QuadGeo& P, const QuadGeo& Q, bool same, const QuadParams& qp, const Gauss01& gs, const Gauss01& gr)
{
    PairResult r;
    if (same) integrate_coincident(P, gs, r);
    else if (shared_nodes(P, Q) > 0) integrate_touching(P, Q, false, gs, r);
    else integrate_regular(P, Q, qp, gr, r);
    return r;
}

double quad_area(const QuadGeo& P)
{
    const Gauss01 g(6);
    double a = 0.0;
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        double3 W[4]; double dJ;
        evaluate_rooftop_basis(-1.0 + 2.0 * g.x[i], -1.0 + 2.0 * g.x[j], P.x, W, dJ);
        a += 4.0 * g.w[i] * g.w[j] * dJ;
    }
    return a;
}

} // namespace

PairResult integrate_quad_pair(const TraceSurface& S, int p, int q, const QuadParams& qp)
{
    const Gauss01 gs(qp.ng_sing), gr(qp.ng_reg);
    return pair(make_geo(S, p), make_geo(S, q), p == q, qp, gs, gr);
}

BoundaryOps assemble_boundary_ops(const TraceSurface& S, const QuadParams& qp, int nthreads)
{
    BoundaryOps O;
    O.F = S.num_quads(); O.V = S.num_nodes(); O.E = S.num_edges();
    const int F = O.F, V = O.V, E = O.E;
    O.V0.assign(static_cast<size_t>(F) * F, 0.0);
    O.Dt.assign(static_cast<size_t>(F) * V, 0.0);
    O.Mt.assign(static_cast<size_t>(F) * V, 0.0);
    O.Vrot.assign(static_cast<size_t>(E) * E, 0.0);
    O.Cm.assign(static_cast<size_t>(E) * E, 0.0);
    O.Mm.assign(static_cast<size_t>(E) * E, 0.0);
    O.VrotFlat.assign(static_cast<size_t>(E) * E, 0.0);
    O.M1.assign(static_cast<size_t>(V) * V, 0.0);
    O.area.assign(F, 0.0);

    std::vector<QuadGeo> geo(F);
    for (int q = 0; q < F; ++q) { geo[q] = make_geo(S, q); O.area[q] = quad_area(geo[q]); }

    const Gauss01 gs(qp.ng_sing), gr(qp.ng_reg), gm(6);
    // lokale Masse (Rooftop) und Q1-Masse
    for (int q = 0; q < F; ++q) {
        double mloc[4][4] = {}, nloc[4] = {}, m1loc[4][4] = {};
        for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
            const QPoint pt = eval_quad(geo[q].x, -1.0 + 2.0 * gm.x[i], -1.0 + 2.0 * gm.x[j]);
            const double w = 4.0 * gm.w[i] * gm.w[j] * pt.detJ;
            for (int a = 0; a < 4; ++a) {
                nloc[a] += w * pt.N[a];
                for (int b = 0; b < 4; ++b) { mloc[a][b] += w * dot3(pt.W[a], pt.W[b]); m1loc[a][b] += w * pt.N[a] * pt.N[b]; }
            }
        }
        for (int a = 0; a < 4; ++a) {
            O.Mt[static_cast<size_t>(q) * V + S.quads[q][a]] += nloc[a] / O.area[q];
            for (int b = 0; b < 4; ++b) {
                O.Mm[static_cast<size_t>(S.quad_edges[q][a]) * E + S.quad_edges[q][b]] += S.quad_sign[q][a] * S.quad_sign[q][b] * mloc[a][b];
                O.M1[static_cast<size_t>(S.quads[q][a]) * V + S.quads[q][b]] += m1loc[a][b];
            }
        }
    }

    if (nthreads <= 0) nthreads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::atomic<int> next(0);
    std::mutex mtx;
    auto worker = [&]() {
        std::vector<PairResult> row(F);
        for (;;) {
            const int p = next.fetch_add(1);
            if (p >= F) break;
            for (int q = 0; q < F; ++q) row[q] = pair(geo[p], geo[q], p == q, qp, gs, gr);
            std::lock_guard<std::mutex> lock(mtx);
            for (int q = 0; q < F; ++q) {
                const PairResult& r = row[q];
                O.V0[static_cast<size_t>(p) * F + q] += r.v0 / (O.area[p] * O.area[q]);
                for (int j = 0; j < 4; ++j) O.Dt[static_cast<size_t>(p) * V + S.quads[q][j]] += r.dt[j] / O.area[p];
                for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
                    const double s = S.quad_sign[p][a] * S.quad_sign[q][b];
                    const size_t idx = static_cast<size_t>(S.quad_edges[p][a]) * E + S.quad_edges[q][b];
                    O.Vrot[idx] += s * r.vr[a][b];
                    O.Cm[idx] += s * r.cm[a][b];
                    O.VrotFlat[idx] += s * r.vrf[a][b];
                }
            }
        }
    };
    std::vector<std::thread> th;
    for (int t = 0; t < nthreads; ++t) th.emplace_back(worker);
    for (auto& t : th) t.join();

    // exakte Symmetrie der Galerkin-Matrizen V0, Vrot, VrotFlat: Asymmetrie (Quadraturfehler) messen, dann (A + A^T)/2
    auto symmetrize = [](std::vector<double>& A, int n) {
        double d = 0.0, nn = 0.0;
        for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
            double& x = A[static_cast<size_t>(i) * n + j];
            double& y = A[static_cast<size_t>(j) * n + i];
            d += 2.0 * (x - y) * (x - y);
            const double m = 0.5 * (x + y);
            x = m; y = m;
        }
        for (double v : A) nn += v * v;
        return std::sqrt(d / nn);
    };
    O.asym_V0 = symmetrize(O.V0, F);
    O.asym_Vrot = symmetrize(O.Vrot, E);
    symmetrize(O.VrotFlat, E);
    return O;
}

FieldPoint evaluate_exterior(const TraceSurface& S, const std::vector<double>& lam, const std::vector<double>& at,
                             const std::vector<double>& g_p0, const std::vector<double>& nA_p0, const Vec3& xv, double mu0,
                             const ExteriorTerms& terms, int ng, int sub)
{
    const Gauss01 g(ng);
    const double3 x = make_double3(xv.x, xv.y, xv.z);
    double3 A = make_double3(0, 0, 0), B = make_double3(0, 0, 0);
    for (int q = 0; q < S.num_quads(); ++q) {
        const QuadGeo geo = make_geo(S, q);
        const double cell = 2.0 / sub;
        for (int ci = 0; ci < sub; ++ci) for (int cj = 0; cj < sub; ++cj)
            for (int a = 0; a < ng; ++a) for (int b = 0; b < ng; ++b) {
                const double u = -1.0 + cell * (ci + g.x[a]), v = -1.0 + cell * (cj + g.x[b]);
                const QPoint y = eval_quad(geo.x, u, v);
                const double w = cell * cell * g.w[a] * g.w[b] * y.detJ;
                const double3 d = sub3(x, y.r);
                const double R2 = dot3(d, d), R = std::sqrt(R2);
                const double Gv = 1.0 / (4.0 * M_PI * R);
                const double3 gradG = scale3(-1.0 / (4.0 * M_PI * R2 * R), d);
                double3 lam_y = make_double3(0, 0, 0), at_y = make_double3(0, 0, 0);
                for (int e = 0; e < 4; ++e) {
                    const double3 nw = cross3(y.n, y.W[e]);
                    const double s = S.quad_sign[q][e];
                    lam_y = add3(lam_y, scale3(s * lam[S.quad_edges[q][e]], nw));
                    at_y = add3(at_y, scale3(s * at[S.quad_edges[q][e]], nw));
                }
                // A = mu0 Psi(lambda) + curl Psi(n x A_t) - grad Psi(n.A);   curl Psi(phi) = int grad_x G x phi
                A = add3(A, scale3(w * terms.sign_lambda * mu0 * Gv, lam_y));
                A = add3(A, scale3(w * terms.sign_trace, cross3(gradG, at_y)));
                A = add3(A, scale3(-w * terms.sign_normal * nA_p0[q], gradG));
                // B = mu0 curl Psi(lambda) - grad Psi(n.B)
                B = add3(B, scale3(w * terms.sign_lambda * mu0, cross3(gradG, lam_y)));
                B = add3(B, scale3(-w * terms.sign_normal * g_p0[q], gradG));
            }
    }
    FieldPoint fp;
    fp.A.x = A.x; fp.A.y = A.y; fp.A.z = A.z;
    fp.B.x = B.x; fp.B.y = B.y; fp.B.z = B.z;
    return fp;
}

std::vector<double> project_on_rooftop(const TraceSurface& S, const std::function<Vec3(const Vec3&, const Vec3&)>& f, int ng)
{
    const Gauss01 g(ng);
    std::vector<double> b(S.num_edges(), 0.0);
    for (int q = 0; q < S.num_quads(); ++q) {
        const QuadGeo geo = make_geo(S, q);
        for (int i = 0; i < ng; ++i) for (int j = 0; j < ng; ++j) {
            const QPoint p = eval_quad(geo.x, -1.0 + 2.0 * g.x[i], -1.0 + 2.0 * g.x[j]);
            const double w = 4.0 * g.w[i] * g.w[j] * p.detJ;
            Vec3 x, n;
            x.x = p.r.x; x.y = p.r.y; x.z = p.r.z; n.x = p.n.x; n.y = p.n.y; n.z = p.n.z;
            const Vec3 fv = f(x, n);
            for (int e = 0; e < 4; ++e)
                b[S.quad_edges[q][e]] += S.quad_sign[q][e] * w * (fv.x * p.W[e].x + fv.y * p.W[e].y + fv.z * p.W[e].z);
        }
    }
    return b;
}

std::vector<double> edge_line_integrals(const TraceSurface& S, const std::function<Vec3(const Vec3&)>& A, int ng)
{
    const Gauss01 g(ng);
    std::vector<double> a(S.num_edges(), 0.0);
    for (int e = 0; e < S.num_edges(); ++e) {
        const Vec3 p0 = S.X[S.edges[e][0]], p1 = S.X[S.edges[e][1]];
        const double dx = p1.x - p0.x, dy = p1.y - p0.y, dz = p1.z - p0.z;
        for (int i = 0; i < ng; ++i) {
            Vec3 x;
            x.x = p0.x + g.x[i] * dx; x.y = p0.y + g.x[i] * dy; x.z = p0.z + g.x[i] * dz;
            const Vec3 av = A(x);
            a[e] += g.w[i] * (av.x * dx + av.y * dy + av.z * dz);
        }
    }
    return a;
}

std::vector<double> quad_average(const TraceSurface& S, const std::function<double(const Vec3&, const Vec3&)>& f, int ng)
{
    const Gauss01 g(ng);
    std::vector<double> avg(S.num_quads(), 0.0);
    for (int q = 0; q < S.num_quads(); ++q) {
        const QuadGeo geo = make_geo(S, q);
        double area = 0.0, sum = 0.0;
        for (int i = 0; i < ng; ++i) for (int j = 0; j < ng; ++j) {
            const QPoint p = eval_quad(geo.x, -1.0 + 2.0 * g.x[i], -1.0 + 2.0 * g.x[j]);
            const double w = 4.0 * g.w[i] * g.w[j] * p.detJ;
            Vec3 x, n;
            x.x = p.r.x; x.y = p.r.y; x.z = p.r.z; n.x = p.n.x; n.y = p.n.y; n.z = p.n.z;
            area += w; sum += w * f(x, n);
        }
        avg[q] = sum / area;
    }
    return avg;
}

} // namespace trace
