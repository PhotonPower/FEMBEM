// =============================================================================
// ReferenceFieldDump.cu -- siehe include/ReferenceFieldDump.cuh
// =============================================================================

#include "ReferenceFieldDump.cuh"

#include <cstdio>
#include <iostream>

void reference_hex_center_fields(const double3 nd[8], const double3 Ah[8],
                                 double3& center, double out[6]) {
    center = make_double3(0.0, 0.0, 0.0);
    for (int c = 0; c < 6; ++c) out[c] = 0.0;
    for (int n = 0; n < 8; ++n) {
        center.x += 0.125 * nd[n].x; center.y += 0.125 * nd[n].y; center.z += 0.125 * nd[n].z;
        out[0] += 0.125 * Ah[n].x;   out[1] += 0.125 * Ah[n].y;   out[2] += 0.125 * Ah[n].z;
    }
    double3 gradN[8];
    double detJ = 0.0;
    evaluate_nodal_gradients(nd, 0.0, 0.0, 0.0, gradN, detJ);
    if (detJ <= 0.0) {
        for (int c = 0; c < 6; ++c) out[c] = 0.0;
        return;
    }
    for (int n = 0; n < 8; ++n) {
        // grad N x A
        out[3] += gradN[n].y * Ah[n].z - gradN[n].z * Ah[n].y;
        out[4] += gradN[n].z * Ah[n].x - gradN[n].x * Ah[n].z;
        out[5] += gradN[n].x * Ah[n].y - gradN[n].y * Ah[n].x;
    }
}

bool write_reference_field_dump(const std::string& path, const double3* nodes,
                                const Hexahedron* hexes, size_t num_hexes,
                                const int* body, const double* Ar, const double* Ai) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (f == nullptr) return false;
    std::fprintf(f, "elem,body,x,y,z,Arx,Ary,Arz,Aix,Aiy,Aiz,Brx,Bry,Brz,Bix,Biy,Biz\n");
    for (size_t h = 0; h < num_hexes; ++h) {
        double3 nd[8], ar[8], ai[8];
        for (int n = 0; n < 8; ++n) {
            const int v = hexes[h].v[n];
            nd[n] = nodes[v];
            ar[n] = make_double3(Ar[3 * v], Ar[3 * v + 1], Ar[3 * v + 2]);
            ai[n] = make_double3(Ai[3 * v], Ai[3 * v + 1], Ai[3 * v + 2]);
        }
        double3 c, c2;
        double r[6], i[6];
        reference_hex_center_fields(nd, ar, c, r);
        reference_hex_center_fields(nd, ai, c2, i);
        std::fprintf(f, "%zu,%d,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
                        "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e\n",
                     h, body[h], c.x, c.y, c.z, r[0], r[1], r[2], i[0], i[1], i[2],
                     r[3], r[4], r[5], i[3], i[4], i[5]);
    }
    std::fclose(f);
    return true;
}
