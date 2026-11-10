#ifndef TRACE_MESHES_H
#define TRACE_MESHES_H

// Testflaechen aus Quads fuer die Maxwell-Spur-Kopplung (Phase 1b): Wuerfel, Kugel (Cube-Sphere),
// Torus, Quader, Verschiebung/Vereinigung. Alle Quads sind nach aussen orientiert (siehe TraceSurface.h).
#include <array>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>
#include <utility>

#include "TraceSurface.h"

namespace trace {

struct QuadMesh {
    std::vector<Vec3> X;
    std::vector<std::array<int, 4>> quads;
};

inline Vec3 v3(double x, double y, double z) { Vec3 v; v.x = x; v.y = y; v.z = z; return v; }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return v3(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return v3(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3 operator*(double s, const Vec3& a) { return v3(s * a.x, s * a.y, s * a.z); }
inline double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b) { return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x); }
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }

// Knoten mit Rundungs-Schluessel zusammenfuehren
class NodeMerger {
public:
    explicit NodeMerger(QuadMesh& m, double tol = 1e-9) : m_(m), tol_(tol) {}
    int get(const Vec3& p) {
        const auto key = std::make_tuple(std::llround(p.x / tol_), std::llround(p.y / tol_), std::llround(p.z / tol_));
        auto it = map_.find(key);
        if (it != map_.end()) return it->second;
        const int id = static_cast<int>(m_.X.size());
        m_.X.push_back(p);
        map_[key] = id;
        return id;
    }
private:
    QuadMesh& m_;
    double tol_;
    std::map<std::tuple<long long, long long, long long>, int> map_;
};

// Dreht jedes Quad um, dessen Normale (in der Quadmitte) nicht von 'inside(p)' weg zeigt.
// inside(p) liefert einen Punkt im Innern nahe p (Mittelpunkt der Kugel, Kernkreispunkt beim Torus, ...).
template <class InsideFn>
void orient_outward(QuadMesh& m, InsideFn inside)
{
    for (auto& q : m.quads) {
        const Vec3 p0 = m.X[q[0]], p1 = m.X[q[1]], p2 = m.X[q[2]], p3 = m.X[q[3]];
        const Vec3 c = 0.25 * (p0 + p1 + p2 + p3);
        const Vec3 n = cross(p2 - p0, p3 - p1);            // ~ dr/du x dr/dv in der Quadmitte
        if (dot(n, c - inside(c)) < 0.0) std::swap(q[1], q[3]);
    }
}

// Quader [-a,a] x [-b,b] x [-c,c], nx x ny x nz Unterteilungen. Knoten werden bei Bedarf auf 'map' abgebildet.
template <class MapFn>
QuadMesh box_mesh(double a, double b, double c, int nx, int ny, int nz, MapFn map)
{
    QuadMesh m;
    NodeMerger nm(m);
    const int n[3] = {nx, ny, nz};
    const double ext[3] = {a, b, c};
    for (int axis = 0; axis < 3; ++axis)
        for (int sgn = -1; sgn <= 1; sgn += 2) {
            const int a1 = (axis + 1) % 3, a2 = (axis + 2) % 3;
            for (int i = 0; i < n[a1]; ++i)
                for (int j = 0; j < n[a2]; ++j) {
                    auto pt = [&](int s, int t) {
                        double cc[3];
                        cc[axis] = sgn * ext[axis];
                        cc[a1] = ext[a1] * (-1.0 + 2.0 * s / n[a1]);
                        cc[a2] = ext[a2] * (-1.0 + 2.0 * t / n[a2]);
                        return map(v3(cc[0], cc[1], cc[2]));
                    };
                    m.quads.push_back({nm.get(pt(i, j)), nm.get(pt(i + 1, j)), nm.get(pt(i + 1, j + 1)), nm.get(pt(i, j + 1))});
                }
        }
    orient_outward(m, [](const Vec3&) { return v3(0, 0, 0); });
    return m;
}

inline QuadMesh cube_mesh(double half, int n)
{
    return box_mesh(half, half, half, n, n, n, [](const Vec3& p) { return p; });
}

inline QuadMesh cuboid_mesh(double a, double b, double c, int nx, int ny, int nz)
{
    return box_mesh(a, b, c, nx, ny, nz, [](const Vec3& p) { return p; });
}

// Cube-Sphere (Wuerfelgitter auf die Kugel projiziert), 6 n^2 Quads
inline QuadMesh sphere_mesh(double radius, int n)
{
    return box_mesh(1.0, 1.0, 1.0, n, n, n, [radius](const Vec3& p) { return (radius / norm(p)) * p; });
}

// Torus: Hauptradius R (um die z-Achse), Nebenradius r; nu Schritte laengs (theta), nv Schritte im Meridian (phi).
// Knoten (i,j) = i*nv + j.
struct TorusMesh {
    QuadMesh mesh;
    int nu = 0, nv = 0;
    int node(int i, int j) const { return ((i % nu + nu) % nu) * nv + ((j % nv + nv) % nv); }
};

inline TorusMesh torus_mesh(double R, double r, int nu, int nv, Vec3 shift = Vec3())
{
    TorusMesh T;
    T.nu = nu; T.nv = nv;
    const double PI = 3.14159265358979323846;
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j) {
            const double th = 2.0 * PI * i / nu, ph = 2.0 * PI * j / nv;
            T.mesh.X.push_back(v3((R + r * std::cos(ph)) * std::cos(th), (R + r * std::cos(ph)) * std::sin(th), r * std::sin(ph)) + shift);
        }
    for (int i = 0; i < nu; ++i)
        for (int j = 0; j < nv; ++j)
            T.mesh.quads.push_back({T.node(i, j), T.node(i + 1, j), T.node(i + 1, j + 1), T.node(i, j + 1)});
    orient_outward(T.mesh, [&](const Vec3& c) {
        const Vec3 q = c - shift;
        const double rho = std::sqrt(q.x * q.x + q.y * q.y);
        return v3(R * q.x / rho, R * q.y / rho, 0.0) + shift;
    });
    return T;
}

inline QuadMesh merge_meshes(const QuadMesh& a, const QuadMesh& b)
{
    QuadMesh m = a;
    const int off = static_cast<int>(a.X.size());
    m.X.insert(m.X.end(), b.X.begin(), b.X.end());
    for (auto q : b.quads) { for (int& v : q) v += off; m.quads.push_back(q); }
    return m;
}

inline QuadMesh translate_mesh(QuadMesh m, const Vec3& d)
{
    for (auto& p : m.X) p = p + d;
    return m;
}

// Kante (a,b) einer Flaeche suchen: (Kantenindex, +1 wenn a->b der globalen Richtung entspricht, sonst -1). -1 als Index, wenn es sie nicht gibt.
inline std::pair<int, int> find_edge(const TraceSurface& S, int a, int b)
{
    for (int e = 0; e < S.num_edges(); ++e) {
        if (S.edges[e][0] == std::min(a, b) && S.edges[e][1] == std::max(a, b)) return {e, a < b ? +1 : -1};
    }
    return {-1, 0};
}

// Zyklus aus Knotenfolge (geschlossen)
inline std::vector<std::pair<int, int>> cycle_from_nodes(const TraceSurface& S, const std::vector<int>& nodes)
{
    std::vector<std::pair<int, int>> c;
    for (size_t i = 0; i < nodes.size(); ++i) c.push_back(find_edge(S, nodes[i], nodes[(i + 1) % nodes.size()]));
    return c;
}

} // namespace trace

#endif
