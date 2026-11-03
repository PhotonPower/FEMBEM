// =============================================================================
// test_reference_field_dump.cu -- Rundlauf des Felddumps im Referenzpfad
// =============================================================================
//
// write_reference_field_dump schreibt A_r, A_i, B_r, B_i je Hexaeder am
// Schwerpunkt. Geprueft wird:
//
//   1. Rundlauf: zwei verzerrte Hexaeder mit einem linearen Feld A = M x
//      (Q1 gibt lineare Felder exakt wieder). Aus der CSV gelesen muessen
//      Schwerpunkt, A am Schwerpunkt und B = rot A = (M_zy-M_yz, M_xz-M_zx,
//      M_yx-M_xy) auf 1e-8 relativ stimmen, Real- und Imaginaerteil getrennt.
//   2. Kopfzeile und Koerperkennung stehen wie im BEM-FEM-Dump.
//   3. Gegenproben: ein Dump mit vertauschten Phasen (A_r <-> A_i) und einer
//      mit vertauschten Koerperkennungen wird vom selben Vergleich ERKANNT.
//
// Braucht keine GPU.
// =============================================================================

#include "ReferenceFieldDump.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", what);
    if (!ok) ++g_fail;
}

struct Row { double v[17]; };   // elem,body,x,y,z,Ar(3),Ai(3),Br(3),Bi(3)

std::vector<Row> read_csv(const std::string& path, std::string& header) {
    std::ifstream in(path);
    std::getline(in, header);
    std::vector<Row> rows;
    std::string line;
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string tok;
        Row r{};
        int k = 0;
        while (std::getline(ss, tok, ',') && k < 17) r.v[k++] = std::atof(tok.c_str());
        rows.push_back(r);
    }
    return rows;
}

struct Model {
    std::vector<double3> nodes;
    std::vector<Hexahedron> hexes;
    std::vector<int> body;
    double Mr[3][3], Mi[3][3];
    std::vector<double> Ar, Ai;
};

double3 apply(const double M[3][3], double3 p) {
    return make_double3(M[0][0]*p.x + M[0][1]*p.y + M[0][2]*p.z,
                        M[1][0]*p.x + M[1][1]*p.y + M[1][2]*p.z,
                        M[2][0]*p.x + M[2][1]*p.y + M[2][2]*p.z);
}

Model make_model() {
    Model m;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(-0.1, 0.1), w(-1.0, 1.0);
    // 3 x 2 x 2 Knoten, zwei Hexaeder nebeneinander, Knoten verzerrt
    for (int k = 0; k < 2; ++k) for (int j = 0; j < 2; ++j) for (int i = 0; i < 3; ++i)
        m.nodes.push_back(make_double3(i + u(rng), j + u(rng), k + u(rng)));
    auto id = [](int i, int j, int k) { return i + 3 * (j + 2 * k); };
    for (int e = 0; e < 2; ++e) {
        Hexahedron H;
        const int c[8][3] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
        for (int n = 0; n < 8; ++n) H.v[n] = id(e + c[n][0], c[n][1], c[n][2]);
        m.hexes.push_back(H);
    }
    m.body = {2, 3};
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) { m.Mr[a][b] = w(rng); m.Mi[a][b] = w(rng); }
    for (const double3& p : m.nodes) {
        const double3 a = apply(m.Mr, p), b = apply(m.Mi, p);
        m.Ar.insert(m.Ar.end(), {a.x, a.y, a.z});
        m.Ai.insert(m.Ai.end(), {b.x, b.y, b.z});
    }
    return m;
}

double3 curl_of(const double M[3][3]) {
    return make_double3(M[2][1] - M[1][2], M[0][2] - M[2][0], M[1][0] - M[0][1]);
}

// Vergleich einer eingelesenen Datei mit dem erwarteten Feld; Rueckgabe:
// groesste relative Abweichung, Koerperkennung ok?
double compare(const Model& m, const std::vector<Row>& rows, bool& body_ok) {
    double worst = 0.0;
    body_ok = (rows.size() == m.hexes.size());
    if (!body_ok) return 1e30;
    for (size_t h = 0; h < rows.size(); ++h) {
        const Row& r = rows[h];
        if (static_cast<int>(r.v[1]) != m.body[h]) body_ok = false;
        double3 c = make_double3(0, 0, 0);
        for (int n = 0; n < 8; ++n) {
            const double3 p = m.nodes[m.hexes[h].v[n]];
            c.x += p.x / 8; c.y += p.y / 8; c.z += p.z / 8;
        }
        const double3 Ar = apply(m.Mr, c), Ai = apply(m.Mi, c);
        const double3 Br = curl_of(m.Mr), Bi = curl_of(m.Mi);
        const double exp[15] = {c.x, c.y, c.z, Ar.x, Ar.y, Ar.z, Ai.x, Ai.y, Ai.z,
                                Br.x, Br.y, Br.z, Bi.x, Bi.y, Bi.z};
        double num = 0.0, den = 0.0;
        for (int k = 0; k < 15; ++k) {
            const double d = r.v[2 + k] - exp[k];
            num += d * d; den += exp[k] * exp[k];
        }
        worst = std::fmax(worst, std::sqrt(num / den));
    }
    return worst;
}

} // namespace

int main() {
    const std::string path = "test_reference_field_dump.csv";
    Model m = make_model();

    std::printf("\nTest 1: Rundlauf, lineares Feld auf zwei verzerrten Hexaedern\n");
    bool ok = write_reference_field_dump(path, m.nodes.data(), m.hexes.data(),
                                         m.hexes.size(), m.body.data(),
                                         m.Ar.data(), m.Ai.data());
    check(ok, "Datei geschrieben");
    std::string header;
    std::vector<Row> rows = read_csv(path, header);
    bool body_ok = false;
    const double err = compare(m, rows, body_ok);
    std::printf("    groesste relative Abweichung %.3e\n", err);
    check(err < 1e-8, "A und B = rot A, Real- und Imaginaerteil, Schwerpunkt: Rundlauf");

    std::printf("\nTest 2: Kopfzeile und Koerperkennung\n");
    check(header == "elem,body,x,y,z,Arx,Ary,Arz,Aix,Aiy,Aiz,Brx,Bry,Brz,Bix,Biy,Biz",
          "Kopfzeile wie im BEM-FEM-Dump");
    check(body_ok, "Koerperkennung stimmt");

    std::printf("\nTest 3: Gegenproben\n");
    write_reference_field_dump(path, m.nodes.data(), m.hexes.data(), m.hexes.size(),
                               m.body.data(), m.Ai.data(), m.Ar.data());   // Phasen vertauscht
    rows = read_csv(path, header);
    const double err_sw = compare(m, rows, body_ok);
    std::printf("    vertauschte Phasen: Abweichung %.3e\n", err_sw);
    check(err_sw > 1e-2, "vertauschte Phasen werden erkannt");

    const int swapped[2] = {m.body[1], m.body[0]};
    write_reference_field_dump(path, m.nodes.data(), m.hexes.data(), m.hexes.size(),
                               swapped, m.Ar.data(), m.Ai.data());
    rows = read_csv(path, header);
    compare(m, rows, body_ok);
    check(!body_ok, "vertauschte Koerperkennungen werden erkannt");

    std::remove(path.c_str());
    std::printf("\n%s\n", g_fail == 0 ? "ALLE TESTS BESTANDEN" : "FEHLER");
    return g_fail == 0 ? 0 : 1;
}
