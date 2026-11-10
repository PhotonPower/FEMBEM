#ifndef TRACE_HEX_MESHES_H
#define TRACE_HEX_MESHES_H

// Hexaedernetze fuer die Tests der gekoppelten Maxwell-Spur-Formulierung (Phase 1b, Schritt 4); werden als
// ASCII-VTK geschrieben und ueber den Bestand (VtkReader) gelesen. Alle Hexaeder rechtshaendig (det J > 0).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "TraceMeshes.h"

namespace trace {

struct HexMesh {
    std::vector<Vec3> X;
    std::vector<std::array<int, 8>> hex;
    std::vector<int> comp;
};

// Ball vom Radius R aus n^3 Hexaedern: Wuerfel [-1,1]^3 radial auf die Kugel abgebildet, x = p |p|_inf / |p|_2 * R
inline HexMesh ball_hex(int n, double R)
{
    HexMesh m;
    auto id = [&](int i, int j, int k) { return (i * (n + 1) + j) * (n + 1) + k; };
    for (int i = 0; i <= n; ++i) for (int j = 0; j <= n; ++j) for (int k = 0; k <= n; ++k) {
        const Vec3 p = v3(-1.0 + 2.0 * i / n, -1.0 + 2.0 * j / n, -1.0 + 2.0 * k / n);
        const double pinf = std::max({std::fabs(p.x), std::fabs(p.y), std::fabs(p.z)});
        const double p2 = norm(p);
        m.X.push_back(p2 > 0 ? (R * pinf / p2) * p : p);
    }
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        m.hex.push_back({id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k),
                         id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1)});
        m.comp.push_back(1);
    }
    return m;
}

// Quader [x0,x1] x [y0,y1] x [z0,z1], nx x ny x nz Hexaeder
inline HexMesh box_hex(int nx, int ny, int nz, double x0, double x1, double y0, double y1, double z0, double z1, int comp)
{
    HexMesh m;
    auto id = [&](int i, int j, int k) { return (i * (ny + 1) + j) * (nz + 1) + k; };
    for (int i = 0; i <= nx; ++i) for (int j = 0; j <= ny; ++j) for (int k = 0; k <= nz; ++k)
        m.X.push_back(v3(x0 + (x1 - x0) * i / nx, y0 + (y1 - y0) * j / ny, z0 + (z1 - z0) * k / nz));
    for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j) for (int k = 0; k < nz; ++k) {
        m.hex.push_back({id(i, j, k), id(i + 1, j, k), id(i + 1, j + 1, k), id(i, j + 1, k),
                         id(i, j, k + 1), id(i + 1, j, k + 1), id(i + 1, j + 1, k + 1), id(i, j + 1, k + 1)});
        m.comp.push_back(comp);
    }
    return m;
}

// Ring mit Rechteckquerschnitt [rho0,rho1] x [z0,z1] um die z-Achse (nu Zellen in phi, nr in rho, nz in z); lokale Achsen (rho, phi, z)
inline HexMesh ring_hex(int nu, int nr, int nz, double rho0, double rho1, double z0, double z1, int comp, double theta_jitter = 0.0)
{
    HexMesh m;
    const double PI = 3.14159265358979323846;
    auto id = [&](int i, int j, int k) { return ((i % nu) * (nr + 1) + j) * (nz + 1) + k; };
    for (int i = 0; i < nu; ++i) for (int j = 0; j <= nr; ++j) for (int k = 0; k <= nz; ++k) {
        // ungleichmaessige Winkelteilung (deterministisch): bricht die knotenweise Aufhebung der Randbeitraege des azimutalen Stroms
        const unsigned hsh = (static_cast<unsigned>(i) * 2654435761u) >> 8;
        const double th = 2 * PI * (i + theta_jitter * ((hsh % 1000u) / 1000.0 - 0.5)) / nu, rho = rho0 + (rho1 - rho0) * j / nr, z = z0 + (z1 - z0) * k / nz;
        m.X.push_back(v3(rho * std::cos(th), rho * std::sin(th), z));
    }
    for (int i = 0; i < nu; ++i) for (int j = 0; j < nr; ++j) for (int k = 0; k < nz; ++k) {
        // xi = rho, eta = phi, zeta = z
        m.hex.push_back({id(i, j, k), id(i, j + 1, k), id(i + 1, j + 1, k), id(i + 1, j, k),
                         id(i, j, k + 1), id(i, j + 1, k + 1), id(i + 1, j + 1, k + 1), id(i + 1, j, k + 1)});
        m.comp.push_back(comp);
    }
    return m;
}

inline HexMesh merge_hex(const HexMesh& a, const HexMesh& b)
{
    HexMesh m = a;
    const int off = static_cast<int>(a.X.size());
    m.X.insert(m.X.end(), b.X.begin(), b.X.end());
    for (auto h : b.hex) { for (int& v : h) v += off; m.hex.push_back(h); }
    m.comp.insert(m.comp.end(), b.comp.begin(), b.comp.end());
    return m;
}

inline void write_vtk(const std::string& path, const HexMesh& m, const std::string& title)
{
    FILE* f = std::fopen(path.c_str(), "w");
    std::fprintf(f, "# vtk DataFile Version 2.0\n%s\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS %zu double\n", title.c_str(), m.X.size());
    for (const Vec3& p : m.X) std::fprintf(f, "%.17g %.17g %.17g\n", p.x, p.y, p.z);
    std::fprintf(f, "CELLS %zu %zu\n", m.hex.size(), 9 * m.hex.size());
    for (const auto& h : m.hex) { std::fprintf(f, "8"); for (int v : h) std::fprintf(f, " %d", v); std::fprintf(f, "\n"); }
    std::fprintf(f, "CELL_TYPES %zu\n", m.hex.size());
    for (size_t i = 0; i < m.hex.size(); ++i) std::fprintf(f, "12\n");
    std::fprintf(f, "CELL_DATA %zu\nSCALARS component_id int 1\nLOOKUP_TABLE default\n", m.hex.size());
    for (int c : m.comp) std::fprintf(f, "%d\n", c);
    std::fclose(f);
}

} // namespace trace

#endif
