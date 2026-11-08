// =============================================================================
// BemNormalPart.h -- Zeile 2 mit vollstaendigen Daten (bem-normal-components)
// =============================================================================
//
// Reiner Host-Code (double, kein CUDA-Kernel), nur lesend. Nichts hier aendert Operatoren, Matrizen
// oder Loesung. Bericht: docs/design/bem-normal-components.md.
//
// HERLEITUNG (ausfuehrlich im Bericht)
//   Aussenraum Omega_e, n zeigt aus dem Leiter in Omega_e. Fuer u harmonisch in Omega_e und abklingend
//   gilt mit dem Rand-Normalenfeld nu = -n von Omega_e (Green)
//       u(x) = int [ -G d_n u + (d_n)_y G u ] dS_y,      G = 1 / (4 pi |x - y|).
//   Auf dem Rand (x in Gamma, glatte Umgebung):
//       1/2 u(x) - (D u)(x) + (G d_n u)(x) = 0,          (D u)(x) = int (x-y).n_y / (4 pi |x-y|^3) u(y) dS_y.
//   Das ist die Zeile 2 des Operators: (1/2 T - D) a + mu0 G k = 0 mit k = (d A / d n)/mu0, getestet mit
//   den Rooftops W_i. Fuer ein Vektorfeld A gilt sie komponentenweise, also fuer alle DREI kartesischen
//   Komponenten von A und d A / d n. Zerlegung der Quelldaten an der Quellflaeche (Normale n(y)):
//       A = A_tan + (A.n) n,      d_n A = (d_n A)_tan + ((d_n A).n) n.
//   Der Operator rechnet nur mit A_tan (Rooftops) und (d_n A)_tan (k). Die Normalanteile tragen bei:
//       N_i = 1/2 int W_i . (A.n) n dS  -  int_x W_i(x) . int_y kd(x,y) (A.n)(y) n(y)
//             + int_x W_i(x) . int_y kg(x,y) ((d_n A).n)(y) n(y).
//   Identitaetsterm: auf demselben Quad ist W_i(x) tangential, n(x) normal, also W_i(x).n(x) = 0
//   punktweise: der Term traegt nicht bei (wird mitgerechnet, ist 0 bis auf Rundung). Der Beitrag an
//   Facettenknicken steckt im D- und G-Term: W_i(x).n(y) ist dort fuer y auf dem Nachbarquad nicht 0.
//
// QUADRATUR (unabhaengig von der des Operators, genauer)
//   Quadpaare (p, q), p != q (Selbstterm fuer ebene Quads: kd = 0 und W_i(x).n(y) = 0):
//     fern  (Abstand der Mitten > 3 x groesste Diagonale): 4x4 Gauss je Quad,
//     mittel: 3x3 Zellen mit je 3x3 Gauss,
//     beruehrend (gemeinsamer Knoten): 8x8 Zellen mit je 3x3 Gauss.
//   Die Quadratur ist an einer geschlossenen Flaeche gegen die exakte Loesung gemessen
//   (tests/test_bem_normal_components.cu).
// =============================================================================
#ifndef BEM_NORMAL_PART_H
#define BEM_NORMAL_PART_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <vector>

#include "BemRowCheck.h"

namespace bem_normal {

using namespace bem_row_check;

struct PointSet {
    std::vector<V3> y, n;        // Punkt und aeussere Normale
    std::vector<double> w;       // Gewicht inkl. Flaechenelement
    std::vector<V3> W[4];        // Rooftop-Funktionen am Punkt
};

inline void gauss3(double x[3], double w[3]) {
    const double a = 0.7745966692414834;
    x[0] = -a; x[1] = 0.0; x[2] = a;
    w[0] = 5.0 / 9.0; w[1] = 8.0 / 9.0; w[2] = 5.0 / 9.0;
}

// m x m Zellen auf [-1,1]^2, je ng-Punkt-Gauss (ng = 3 oder 4)
inline PointSet make_points(const QuadGeom& q, int m, int ng) {
    PointSet ps;
    double gx[4], gw[4];
    if (ng == 3) gauss3(gx, gw); else gauss4(gx, gw);
    const double h = 2.0 / m;
    for (int ci = 0; ci < m; ++ci) for (int cj = 0; cj < m; ++cj)
        for (int a = 0; a < ng; ++a) for (int b = 0; b < ng; ++b) {
            const double u = -1.0 + h * (ci + 0.5 * (gx[a] + 1.0));
            const double v = -1.0 + h * (cj + 0.5 * (gx[b] + 1.0));
            double detJ; double3 W[4];
            evaluate_rooftop_basis(u, v, q.n, W, detJ);
            ps.y.push_back(quad_point(q.n, u, v));
            ps.n.push_back(outward_normal(q, u, v));
            ps.w.push_back(gw[a] * gw[b] * 0.25 * h * h * detJ);
            for (int i = 0; i < 4; ++i) ps.W[i].push_back(to_v3(W[i]));
        }
    return ps;
}

struct QuadSets {
    PointSet far, mid, tch;
    V3 center;
    double diam = 0.0;
};

// Dichten an den Punkten aller drei Mengen eines Quads
struct DensitySets {
    std::vector<V3> A_t[3], A_n[3], K_t[3], K_n[3];     // Index 0 far, 1 mid, 2 tch
};

// Quelldaten: A(p) und (d_n A)(p, n), jeweils der VOLLE Vektor.
using FieldFn = std::function<V3(V3 p)>;
using NormalDerivFn = std::function<V3(V3 p, V3 n)>;

struct Geometry {
    std::vector<QuadGeom> quads;
    std::vector<QuadSets> sets;
    std::vector<std::vector<int>> touching;   // je Quad: Quads mit gemeinsamem Knoten
};

inline Geometry build_geometry(const std::vector<QuadGeom>& quads) {
    Geometry g;
    g.quads = quads;
    const size_t nq = quads.size();
    g.sets.resize(nq);
    parallel_for(nq, [&](size_t i) {
        QuadSets& s = g.sets[i];
        s.far = make_points(quads[i], 1, 4);
        s.mid = make_points(quads[i], 3, 3);
        s.tch = make_points(quads[i], 8, 3);
        V3 c;
        for (int k = 0; k < 4; ++k) c = c + to_v3(quads[i].n[k]);
        s.center = 0.25 * c;
        double d = 0.0;
        for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b)
            d = std::max(d, norm(to_v3(quads[i].n[a]) - to_v3(quads[i].n[b])));
        s.diam = d;
    });
    // Beruehrung ueber gemeinsame Knotenkoordinaten (Quads sind nach BEM-Reihenfolge, Knoten als Koordinaten)
    g.touching.assign(nq, {});
    // einfache Hash-Zelle ueber Knotenkoordinaten
    struct Key { long long x, y, z; bool operator<(const Key& o) const {
        return x != o.x ? x < o.x : (y != o.y ? y < o.y : z < o.z); } };
    std::vector<std::pair<Key, int>> nodes;
    for (size_t i = 0; i < nq; ++i)
        for (int k = 0; k < 4; ++k) {
            const double3& p = quads[i].n[k];
            nodes.push_back({{std::llround(p.x * 1e9), std::llround(p.y * 1e9), std::llround(p.z * 1e9)},
                             static_cast<int>(i)});
        }
    std::sort(nodes.begin(), nodes.end(),
              [](const std::pair<Key, int>& a, const std::pair<Key, int>& b) {
                  if (a.first < b.first) return true;
                  if (b.first < a.first) return false;
                  return a.second < b.second; });
    for (size_t a = 0; a < nodes.size();) {
        size_t b = a;
        while (b < nodes.size() && !(nodes[a].first < nodes[b].first) && !(nodes[b].first < nodes[a].first)) ++b;
        for (size_t i = a; i < b; ++i) for (size_t j = a; j < b; ++j)
            if (nodes[i].second != nodes[j].second) g.touching[static_cast<size_t>(nodes[i].second)].push_back(nodes[j].second);
        a = b;
    }
    for (auto& t : g.touching) { std::sort(t.begin(), t.end()); t.erase(std::unique(t.begin(), t.end()), t.end()); }
    return g;
}

// Dichten an allen Punkten (parallel)
inline std::vector<DensitySets> evaluate_densities(const Geometry& g, const FieldFn& A, const NormalDerivFn& dnA) {
    const size_t nq = g.quads.size();
    std::vector<DensitySets> dens(nq);
    parallel_for(nq, [&](size_t i) {
        const PointSet* ps[3] = {&g.sets[i].far, &g.sets[i].mid, &g.sets[i].tch};
        for (int s = 0; s < 3; ++s) {
            const size_t np = ps[s]->y.size();
            dens[i].A_t[s].resize(np); dens[i].A_n[s].resize(np);
            dens[i].K_t[s].resize(np); dens[i].K_n[s].resize(np);
            for (size_t k = 0; k < np; ++k) {
                const V3 p = ps[s]->y[k], n = ps[s]->n[k];
                const V3 a = A(p), kk = dnA(p, n);
                const double an = dot(a, n), kn = dot(kk, n);
                dens[i].A_n[s][k] = an * n;
                dens[i].A_t[s][k] = a - an * n;
                dens[i].K_n[s][k] = kn * n;
                dens[i].K_t[s][k] = kk - kn * n;
            }
        }
    });
    return dens;
}

struct NormalPartResult {
    // Je Position und lokalem Rooftop i: Index pos*4+i
    std::vector<double> Tt, Tn;   // int W_i . A_tan,  int W_i . (A.n) n   (ohne Faktor 1/2)
    std::vector<double> Dt, Dn;   // D auf A_tan bzw. auf (A.n) n, getestet mit W_i
    std::vector<double> Gt, Gn;   // G auf (d_n A)_tan bzw. ((d_n A).n) n, getestet mit W_i
};

// Selbstterm von G auf den tangentialen Daten: int_p W_i(x) . int_p K_t(y) / (4 pi |x-y|) dS_y dS_x.
// Das Quad ist eben, W_i(x) tangential, also ist der Term fuer tangentiale Daten NICHT null (fuer D und fuer
// Normalanteile schon). Innere Integration in Polarkoordinaten um x (jedes Dreieck x, P_k, P_k+1):
// dS = rho drho dtheta hebt den Faktor 1/rho des Kerns auf.
inline void add_self_G_tangential(const Geometry& g, size_t p, const NormalDerivFn& dnA,
                                  std::vector<double>& Gt) {
    const QuadGeom& q = g.quads[p];
    double gx[4], gw[4];
    gauss4(gx, gw);
    double rx[6], rw[6];
    // 6-Punkt-Gauss-Legendre auf [-1,1]
    {
        const double xs[3] = {0.2386191860831969, 0.6612093864662645, 0.9324695142031521};
        const double ws[3] = {0.4679139345726910, 0.3607615730481386, 0.1713244923791704};
        for (int k = 0; k < 3; ++k) { rx[k] = -xs[2 - k]; rw[k] = ws[2 - k]; rx[3 + k] = xs[k]; rw[3 + k] = ws[k]; }
    }
    const V3 nrm = outward_normal(q, 0.0, 0.0);
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
        const double u = gx[a], v = gx[b];
        double detJ; double3 W[4];
        evaluate_rooftop_basis(u, v, q.n, W, detJ);
        const V3 x = quad_point(q.n, u, v);
        const double wx = gw[a] * gw[b] * detJ;
        V3 S;   // int K_t(y) / (4 pi rho) dS
        for (int e = 0; e < 4; ++e) {
            const V3 P0 = to_v3(q.n[e]), P1 = to_v3(q.n[(e + 1) % 4]);
            const V3 d0 = P0 - x, d1 = P1 - x;
            // Winkel in der Quadebene (Basis e1 = d0/|d0|, e2 = nrm x e1)
            const V3 e1 = (1.0 / norm(d0)) * d0;
            const V3 e2 = cross(nrm, e1);
            const double th1 = std::atan2(dot(d1, e2), dot(d1, e1));
            for (int it = 0; it < 6; ++it) {
                const double th = 0.5 * th1 * (rx[it] + 1.0);
                const double wth = 0.5 * std::fabs(th1) * rw[it];
                const V3 dir = std::cos(th) * e1 + std::sin(th) * e2;
                // Schnitt des Strahls x + rho dir mit der Geraden P0 P1
                const V3 ed = P1 - P0;
                // rho dir = P0 - x + t ed  ->  Loesung ueber Kreuzprodukt mit nrm
                const double rho_max = dot(cross(d0, ed), nrm) / dot(cross(dir, ed), nrm);
                for (int ir = 0; ir < 6; ++ir) {
                    const double rho = 0.5 * rho_max * (rx[ir] + 1.0);
                    const double wr = 0.5 * rho_max * rw[ir];
                    const V3 y = x + rho * dir;
                    V3 kk = dnA(y, nrm);
                    kk = kk - dot(kk, nrm) * nrm;
                    S = S + (wth * wr / (4.0 * kPi)) * kk;
                }
            }
        }
        for (int i = 0; i < 4; ++i) Gt[p * 4 + i] += wx * dot(to_v3(W[i]), S);
    }
}

// dnA_for_self: wenn gesetzt, wird der Selbstterm von G auf den tangentialen Daten ergaenzt (siehe oben).
inline NormalPartResult apply_kernels(const Geometry& g, const std::vector<DensitySets>& dens,
                                      const NormalDerivFn* dnA_for_self = nullptr) {
    const size_t nq = g.quads.size();
    NormalPartResult r;
    for (auto* v : {&r.Tt, &r.Tn, &r.Dt, &r.Dn, &r.Gt, &r.Gn}) v->assign(4 * nq, 0.0);
    const double inv4pi = 1.0 / (4.0 * kPi);
    parallel_for(nq, [&](size_t p) {
        const QuadSets& sp = g.sets[p];
        // Massen-Terme auf dem eigenen Quad (far-Menge)
        for (size_t k = 0; k < sp.far.y.size(); ++k)
            for (int i = 0; i < 4; ++i) {
                r.Tt[p * 4 + i] += sp.far.w[k] * dot(sp.far.W[i][k], dens[p].A_t[0][k]);
                r.Tn[p * 4 + i] += sp.far.w[k] * dot(sp.far.W[i][k], dens[p].A_n[0][k]);
            }
        std::vector<char> is_touch(nq, 0);
        for (int t : g.touching[p]) is_touch[static_cast<size_t>(t)] = 1;
        for (size_t q = 0; q < nq; ++q) {
            if (q == p) continue;
            const QuadSets& sq = g.sets[q];
            const double dist = norm(sp.center - sq.center);
            int lev;
            if (is_touch[q]) lev = 2;
            else if (dist > 3.0 * std::max(sp.diam, sq.diam)) lev = 0;
            else lev = 1;
            const PointSet& X = lev == 0 ? sp.far : (lev == 1 ? sp.mid : sp.tch);
            const PointSet& Y = lev == 0 ? sq.far : (lev == 1 ? sq.mid : sq.tch);
            const std::vector<V3>& At = dens[q].A_t[lev];
            const std::vector<V3>& An = dens[q].A_n[lev];
            const std::vector<V3>& Kt = dens[q].K_t[lev];
            const std::vector<V3>& Kn = dens[q].K_n[lev];
            for (size_t a = 0; a < X.y.size(); ++a) {
                V3 sDt, sDn, sGt, sGn;
                for (size_t b = 0; b < Y.y.size(); ++b) {
                    const V3 rv = X.y[a] - Y.y[b];
                    const double d2 = dot(rv, rv), d1 = std::sqrt(d2);
                    const double kd = inv4pi * dot(rv, Y.n[b]) / (d2 * d1) * Y.w[b];
                    const double kg = inv4pi / d1 * Y.w[b];
                    sDt = sDt + kd * At[b]; sDn = sDn + kd * An[b];
                    sGt = sGt + kg * Kt[b]; sGn = sGn + kg * Kn[b];
                }
                for (int i = 0; i < 4; ++i) {
                    const V3& Wi = X.W[i][a];
                    const double wx = X.w[a];
                    r.Dt[p * 4 + i] += wx * dot(Wi, sDt);
                    r.Dn[p * 4 + i] += wx * dot(Wi, sDn);
                    r.Gt[p * 4 + i] += wx * dot(Wi, sGt);
                    r.Gn[p * 4 + i] += wx * dot(Wi, sGn);
                }
            }
        }
    });
    if (dnA_for_self)
        parallel_for(nq, [&](size_t p) { add_self_G_tangential(g, p, *dnA_for_self, r.Gt); });
    return r;
}

}  // namespace bem_normal

#endif  // BEM_NORMAL_PART_H
