// =============================================================================
// Abnahmepruefung der Gradientenprojektion des Quellterms
//
//   cmake --build build --config Release --target test_source_projection
//   ./build/test_source_projection
//
// Reiner Host-Code, keine GPU.
//
// project_out_gradient() loest (G^T G) phi = G^T b und setzt b <- b - G phi.
// Das ist die orthogonale Projektion auf den Kern von G^T, und genau diese
// drei Eigenschaften muss sie haben:
//
//   1. Ein REINER Gradient verschwindet vollstaendig. b_e = f(kopf) - f(fuss)
//      liegt ganz im Bildraum von G, es darf nichts uebrigbleiben.
//   2. IDEMPOTENZ. Eine zweite Projektion darf nichts mehr aendern. Das ist
//      die definierende Eigenschaft einer Projektion und faellt sonst nicht
//      auf: ein Verfahren, das jedes Mal ein Stueck mehr wegnimmt, sieht beim
//      ersten Aufruf richtig aus.
//   3. ORTHOGONALITAET. ||G^T b||/||b|| geht auf Maschinengenauigkeit, und
//      ||b|| sinkt nur um den Gradientenanteil -- nicht mehr. Ein Verfahren,
//      das zu viel wegnimmt, erfuellt Punkt 1 und 3a und ist trotzdem falsch.
//
// Dazu: eine falsche Vektorlaenge muss werfen. Die 2n-Blockform hat
// 2*num_edges Eintraege, und dort waere je Haelfte zu projizieren -- ein
// stiller Fehlgriff dort ergaebe einen halb bereinigten Quellterm.
// =============================================================================

#include "FemDirectBC.h"
#include "Mesh.cuh"
#include "TopologyExtractor.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <vector>

struct TestMesh {
    std::vector<double3> pts;
    std::vector<std::vector<int>> tets;
    double h = 0.0;   // nominale Netzweite
};

static double frand(unsigned& seed) {
    seed = seed * 1103515245u + 12345u;
    return static_cast<double>((seed >> 8) & 0xFFFFFF) / 16777216.0;  // [0,1)
}

static double tet_volume(const double3& a, const double3& b,
                         const double3& c, const double3& d) {
    const double3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
    const double3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
    const double3 e3 = {d.x - a.x, d.y - a.y, d.z - a.z};
    return ((e1.y * e2.z - e1.z * e2.y) * e3.x
          + (e1.z * e2.x - e1.x * e2.z) * e3.y
          + (e1.x * e2.y - e1.y * e2.x) * e3.z) / 6.0;
}

// Der Kasten liegt bewusst NEBEN der z-Achse (x von 1 bis 2). Ein azimutales
// J_s = J * phi_hat ist auf der Achse undefiniert; dort greift im
// Produktionskernel die Abfrage r < 1e-12, und der Test wuerde eine
// Sonderfallbehandlung messen statt der Quadratur.
static TestMesh build_kuhn_mesh(int N, double distort, unsigned seed) {
    TestMesh m;
    const double3 lo = {1.0, -0.5, 0.0};
    const double3 hi = {2.0,  0.5, 1.0};
    m.h = 1.0 / N;

    const auto nid = [N](int i, int j, int k) {
        return (k * (N + 1) + j) * (N + 1) + i;
    };

    for (int k = 0; k <= N; ++k)
    for (int j = 0; j <= N; ++j)
    for (int i = 0; i <= N; ++i) {
        double3 p;
        p.x = lo.x + (hi.x - lo.x) * i / N;
        p.y = lo.y + (hi.y - lo.y) * j / N;
        p.z = lo.z + (hi.z - lo.z) * k / N;
        // Nur INNERE Knoten verschieben. Der Rand bleibt eben, sonst waere
        // "innerer Knoten" nicht mehr sauber vom Gebietsrand zu trennen.
        const bool interior = (i > 0 && i < N) && (j > 0 && j < N) && (k > 0 && k < N);
        if (interior && distort > 0.0) {
            p.x += distort * m.h * (2.0 * frand(seed) - 1.0);
            p.y += distort * m.h * (2.0 * frand(seed) - 1.0);
            p.z += distort * m.h * (2.0 * frand(seed) - 1.0);
        }
        m.pts.push_back(p);
    }

    // Die sechs Kuhn-Tetraeder als Eckindizes 0..7 des Einheitswuerfels,
    // wobei v = i + 2j + 4k. Jeder laeuft von 0 ueber zwei Zwischenecken
    // nach 7, einmal je Achsenreihenfolge.
    const int KUHN[6][4] = {
        {0, 1, 3, 7},   // x, y, z
        {0, 1, 5, 7},   // x, z, y
        {0, 2, 3, 7},   // y, x, z
        {0, 2, 6, 7},   // y, z, x
        {0, 4, 5, 7},   // z, x, y
        {0, 4, 6, 7},   // z, y, x
    };

    for (int k = 0; k < N; ++k)
    for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) {
        int corner[8];
        for (int c = 0; c < 8; ++c)
            corner[c] = nid(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));

        for (int t = 0; t < 6; ++t) {
            std::vector<int> e = {corner[KUHN[t][0]], corner[KUHN[t][1]],
                                  corner[KUHN[t][2]], corner[KUHN[t][3]]};
            // Positive Orientierung erzwingen. Fuer die kontravariante
            // Piola-Transformation waere ein negatives detJ zwar konsistent,
            // aber ein Netz mit gemischten Vorzeichen verdeckt echte Fehler.
            if (tet_volume(m.pts[e[0]], m.pts[e[1]], m.pts[e[2]], m.pts[e[3]]) < 0.0)
                std::swap(e[2], e[3]);
            m.tets.push_back(e);
        }
    }
    return m;
}

static void fill_mesh(Mesh& mesh, const TestMesh& tm) {
    mesh.num_nodes = tm.pts.size();
    mesh.nodes_initial.allocate(tm.pts.size());
    mesh.nodes_current.allocate(tm.pts.size());
    for (size_t i = 0; i < tm.pts.size(); ++i) {
        mesh.nodes_initial[i] = tm.pts[i];
        mesh.nodes_current[i] = tm.pts[i];
    }

    mesh.num_hexes = tm.tets.size();
    mesh.hex_nodes.allocate(tm.tets.size());
    mesh.hex_component_ids.allocate(tm.tets.size());
    mesh.elem_types.allocate(tm.tets.size());
    for (size_t i = 0; i < tm.tets.size(); ++i) {
        Hexahedron h;
        for (int k = 0; k < 8; ++k) h.v[k] = -1;
        for (int k = 0; k < 4; ++k) h.v[k] = tm.tets[i][k];
        mesh.hex_nodes[i] = h;
        mesh.hex_component_ids[i] = 0;
        mesh.elem_types[i] = ELEM_TET;
    }
}

// Knoten, die auf dem Gebietsrand liegen: alle Knoten einer Dreiecksflaeche,
// die nur EINMAL vorkommt. Selbst gezaehlt und nicht aus dem
// TopologyExtractor genommen -- der Test soll unabhaengig davon sein, was
// dort als Randflaeche gilt.
static std::vector<char> mark_boundary_nodes(const Mesh& mesh) {
    std::map<std::vector<int>, int> face_count;
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const Hexahedron H = mesh.hex_nodes[h];
        for (int f = 0; f < 4; ++f) {
            int loc[4];
            const int nn = element_local_face_nodes(ELEM_TET, f, loc);
            std::vector<int> key;
            for (int i = 0; i < nn; ++i) key.push_back(H.v[loc[i]]);
            std::sort(key.begin(), key.end());
            ++face_count[key];
        }
    }
    std::vector<char> on_boundary(mesh.num_nodes, 0);
    for (const auto& kv : face_count) {
        if (kv.second != 1) continue;
        for (int v : kv.first) on_boundary[v] = 1;
    }
    return on_boundary;
}


static int fails = 0;
static void ck(const char* n, bool ok, const char* x = "") {
    printf("  [%s] %s%s\n", ok ? "OK " : "FEHL", n, x); if (!ok) ++fails;
}

int main() {
    // Host-Test: ohne GPU lauffaehig. Ohne diese Zeile legt DeviceArray den
    // Speicher mit cudaMallocManaged an und bricht ohne Grafikkarte ab.
    bf::g_backend = bf::Backend::CPU;
    Mesh mesh;
    const TestMesh tm = build_kuhn_mesh(4, 0.25, 20260909u);
    fill_mesh(mesh, tm);
    TopologyExtractor::extract_edges(mesh);
    const size_t ne = mesh.num_edges;
    char buf[160];

    printf("\n== Gradientenprojektion des Quellterms ==\n");
    printf("  %zu Knoten, %zu Kanten\n", mesh.num_nodes, ne);

    // 1. Ein REINER GRADIENT muss vollstaendig verschwinden. b_e = f(kopf) -
    //    f(fuss) fuer ein beliebiges Knotenfeld f: das liegt ganz im
    //    Bildraum von G, die Projektion muss also b' = 0 liefern.
    {
        std::vector<double> f(mesh.num_nodes), b(ne);
        unsigned sd = 7u;
        for (auto& v : f) v = frand(sd) - 0.5;
        for (size_t e = 0; e < ne; ++e) {
            const int2 ed = mesh.edges[e];
            b[e] = f[ed.y] - f[ed.x];
        }
        const double n0 = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
        const auto r = fem_direct::project_out_gradient(mesh, b.data(), ne, nullptr, 1e-12, 5000);
        const double n1 = std::sqrt(std::inner_product(b.begin(), b.end(), b.begin(), 0.0));
        snprintf(buf, sizeof buf, "   (||b|| %.4g -> %.4g, %d CG-Iter)", n0, n1, r.cg_iters);
        ck("reiner Gradient wird vollstaendig entfernt", n1 < 1e-9 * n0, buf);
    }

    // 2. Ein DIVERGENZFREIES b muss unveraendert bleiben. Konstruiert als
    //    G^T-Kern: nimm ein beliebiges b, projiziere einmal -- das Ergebnis
    //    ist divergenzfrei. Eine ZWEITE Projektion darf nichts mehr tun.
    //    Das prueft die Idempotenz, die eine Projektion haben MUSS.
    {
        std::vector<double> b(ne);
        unsigned sd = 99u;
        for (auto& v : b) v = frand(sd) - 0.5;
        fem_direct::project_out_gradient(mesh, b.data(), ne, nullptr, 1e-13, 20000);
        std::vector<double> b1 = b;
        const auto r2 = fem_direct::project_out_gradient(mesh, b.data(), ne, nullptr, 1e-13, 20000);
        double diff = 0.0, nb = 0.0;
        for (size_t e = 0; e < ne; ++e) { diff += (b[e]-b1[e])*(b[e]-b1[e]); nb += b1[e]*b1[e]; }
        diff = std::sqrt(diff/nb);
        snprintf(buf, sizeof buf, "   (Aenderung %.3e, ||G^T b||/||b|| = %.3e)", diff, r2.rel_before);
        ck("idempotent: zweite Projektion aendert nichts", diff < 1e-9, buf);
    }

    // 3. Die eigentliche Zusage: ||G^T b||/||b|| geht auf Maschinengenauigkeit,
    //    und ||b|| sinkt nur wenig -- die Projektion ist ORTHOGONAL, entfernt
    //    also genau den Gradientenanteil und nicht mehr.
    {
        std::vector<double> b(ne, 0.0);
        unsigned sd = 4242u;
        // Quellterm-aehnlich: nur auf einem Teil der Kanten besetzt
        for (size_t e = 0; e < ne; ++e) if (frand(sd) < 0.2) b[e] = frand(sd) - 0.5;
        const auto r = fem_direct::project_out_gradient(mesh, b.data(), ne, nullptr, 1e-13, 20000);
        snprintf(buf, sizeof buf, "   (%.4e -> %.4e, %d CG-Iter)",
                 r.rel_before, r.rel_after, r.cg_iters);
        ck("||G^T b||/||b|| wird maschinengenau null", r.rel_after < 1e-12, buf);
        // Pythagoras: ||b||^2 = ||b'||^2 + ||G phi||^2
        const double drop = 1.0 - (r.b_norm_after/r.b_norm_before)*(r.b_norm_after/r.b_norm_before);
        snprintf(buf, sizeof buf, "   (||b|| %.4g -> %.4g, entfernter Anteil %.1f %%)",
                 r.b_norm_before, r.b_norm_after, 100*drop);
        ck("||b|| sinkt, orthogonale Zerlegung plausibel",
           r.b_norm_after < r.b_norm_before && r.b_norm_after > 0.5*r.b_norm_before, buf);
    }


    // 5. MIT MASKE: auf gesperrten Kanten muss b null BLEIBEN, und die
    //    Divergenzfreiheit muss das aushalten.
    //
    //    Das ist die Eigenschaft, um die es geht. Ohne Maske besetzt
    //    b <- b - G phi ALLE Kanten (phi loest ein globales
    //    Poisson-Problem), die spaetere Randbehandlung setzt die
    //    Dirichlet-Kanten wieder auf null, und genau das bringt die
    //    Divergenz an den Randknoten zurueck -- im Lauf gemessen 1,61e-05,
    //    wo die Projektion selbst 5,45e-16 gemeldet hatte.
    //
    //    Geprueft wird deshalb BEIDES: b ist null auf den gesperrten
    //    Kanten, UND G^T b ist null an allen Knoten, gerechnet ueber alle
    //    Kanten. Nur zusammen sagen die zwei etwas.
    {
        std::vector<char> blocked(ne, 0);
        // Als Randersatz: alle Kanten, die einen Knoten auf x = min beruehren
        double xmin = 1e30;
        for (size_t i = 0; i < mesh.num_nodes; ++i)
            xmin = std::fmin(xmin, mesh.nodes_current[i].x);
        long n_blocked = 0;
        for (size_t e = 0; e < ne; ++e) {
            const int2 ed = mesh.edges[e];
            if (std::fabs(mesh.nodes_current[ed.x].x - xmin) < 1e-12 ||
                std::fabs(mesh.nodes_current[ed.y].x - xmin) < 1e-12) {
                blocked[e] = 1; ++n_blocked;
            }
        }

        std::vector<double> b(ne, 0.0);
        unsigned sd = 31337u;
        for (size_t e = 0; e < ne; ++e)
            if (!blocked[e] && frand(sd) < 0.3) b[e] = frand(sd) - 0.5;

        const auto r = fem_direct::project_out_gradient(
            mesh, b.data(), ne, blocked.data(), 1e-13, 20000);

        double on_blocked = 0.0;
        for (size_t e = 0; e < ne; ++e) if (blocked[e]) on_blocked += std::fabs(b[e]);
        snprintf(buf, sizeof buf, "   (%ld gesperrte Kanten, Summe |b| dort %.3e)",
                 n_blocked, on_blocked);
        ck("b bleibt null auf gesperrten Kanten", on_blocked == 0.0, buf);

        // G^T b ueber ALLE Kanten -- so, wie es die Diagnose nach der
        // Randbehandlung rechnet.
        std::vector<double> gtb(mesh.num_nodes, 0.0);
        double nb = 0.0;
        for (size_t e = 0; e < ne; ++e) {
            const int2 ed = mesh.edges[e];
            gtb[ed.y] += b[e]; gtb[ed.x] -= b[e];
            nb += b[e]*b[e];
        }
        double g = 0.0;
        for (double v : gtb) g += v*v;
        const double rel = std::sqrt(g)/std::sqrt(nb);
        snprintf(buf, sizeof buf, "   (%.4e -> %.4e, %d CG-Iter)", r.rel_before, rel, r.cg_iters);
        ck("G^T b bleibt null, ueber ALLE Kanten gerechnet", rel < 1e-11, buf);
    }

    // 6. Falsche Laenge muss werfen, nicht still das Falsche tun -- die
    //    2n-Blockform hat 2*num_edges Eintraege, und dort waere die
    //    Projektion je Haelfte anzuwenden.
    {
        std::vector<double> b(2*ne, 1.0);
        bool threw = false;
        try { fem_direct::project_out_gradient(mesh, b.data(), 2*ne, nullptr, 1e-12, 100); }
        catch (const std::exception&) { threw = true; }
        ck("falsche Vektorlaenge wirft", threw);
    }

    printf("\n%s (%d Beanstandungen)\n", fails?"FEHLGESCHLAGEN":"ALLE PRUEFUNGEN BESTANDEN", fails);
    return fails?1:0;
}
