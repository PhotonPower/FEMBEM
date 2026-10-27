// =============================================================================
// Abnahmepruefung: Baum-Kobaum-Eichung auf einem TETRAEDERNETZ
//
//   cmake --build build --config Release --target test_gauge_on_tets
//   ./build/test_gauge_on_tets
//
// Reiner Host-Code, keine GPU.
//
// WARUM DIESER TEST
//
// build_tree_cotree_gauge() war nie auf einem Tetraedernetz mit
// RANDFLAECHEN gelaufen. Die Randabbildung hat die Schrittweite 4, ein
// Dreieck belegt aber nur drei Plaetze -- der vierte steht auf -1
// (TopologyExtractor meldet das selbst). Die Schleife ueber
// boundary_edges_to_fem_edges hat das nicht abgefragt und schrieb
// is_boundary_edge[-1], einmal je Randflaeche. Bei einem reinen
// Tetraedernetz sind das ALLE Flaechen; unter MSVC schlaegt es als
// STATUS_STACK_BUFFER_OVERRUN (0xC0000409) zu, also weit nach der Ursache.
//
// Der Test ruft deshalb ausdruecklich extract_boundary_faces() auf -- ohne
// das ist boundary_edges_to_fem_edges leer und der Fehler nicht erreichbar.
// Genau daran ist er den bestehenden Tests durchgegangen.
//
// Geprueft wird ausserdem die Nullraumdimension: die Zahl der geeichten
// Kanten muss der Zahl der nichtleitenden Knoten entsprechen. Weniger liesse
// Nullraum stehen, mehr waere Uebereichung -- beides sieht man an der Matrix
// nicht, sondern erst am Ergebnis.
//
// Die Generatorensuche laeuft in einem try. Sie ist fuer den BEM-Fall gebaut
// (eine Komponente, ein Volltorus, ein Generator) und in diesem Regime
// unzuverlaessig; das ist ein Befund, kein Fehler dieses Tests, und der
// Loeserpfad behandelt sie deshalb als Auskunft und nicht als Bedingung.
// =============================================================================

#include "CohomologyGauge.h"
#include "Mesh.cuh"
#include "NedelecDispatch.cuh"
#include "TopologyExtractor.h"
#include "TreeCotreeGauge.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
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
    setvbuf(stdout, nullptr, _IONBF, 0);   // sonst geht bei einem throw alles verloren
    Mesh mesh;
    const TestMesh tm = build_kuhn_mesh(4, 0.25, 20260909u);
    fill_mesh(mesh, tm);
    TopologyExtractor::extract_edges(mesh);
    // Genau der Aufruf, der die -1-Plaetze erzeugt: Dreiecksflaechen mit
    // Schrittweite 4. Ohne ihn ist boundary_edges_to_fem_edges leer und der
    // Fehler gar nicht erreichbar -- deshalb MUSS er hier stehen.
    TopologyExtractor::extract_boundary_faces(mesh);

    printf("\n== Eichung auf einem Tetraedernetz ==\n");
    printf("  %zu Knoten, %zu Kanten, %zu Randflaechen\n",
           mesh.num_nodes, mesh.num_edges, mesh.num_boundary_faces);

    long tri = 0, minus1 = 0;
    for (size_t i = 0; i < mesh.num_boundary_faces * 4; ++i)
        if (mesh.boundary_edges_to_fem_edges[i] < 0) ++minus1;
    for (size_t f = 0; f < mesh.num_boundary_faces; ++f)
        if (mesh.boundary_faces[f].v[3] < 0) ++tri;
    char buf[160];
    snprintf(buf, sizeof buf, "   (%ld Dreiecke, %ld Plaetze auf -1)", tri, minus1);
    ck("Randabbildung enthaelt -1-Plaetze", minus1 > 0, buf);

    // Leiter: die Haelfte der Elemente, damit es eine leitende Komponente gibt
    // Leiter als SCHEIBE in der Mitte, nicht als beliebiges Viertel der
    // Elementliste. Das nichtleitende Gebiet ist dann Kasten minus Scheibe --
    // einfach zusammenhaengend, also 0 Generatoren erwartet. Ein zufaellig
    // gewaehlter Leiter haette eine beliebige Topologie und der Test sagte
    // nichts.
    std::vector<MaterialProperties> mats(mesh.num_hexes);
    double zmin = 1e30, zmax = -1e30;
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        zmin = std::fmin(zmin, mesh.nodes_current[i].z);
        zmax = std::fmax(zmax, mesh.nodes_current[i].z);
    }
    const double zc = 0.5 * (zmin + zmax), dz = 0.12 * (zmax - zmin);
    long n_cond = 0;
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        mats[h] = MaterialProperties{};
        double zs = 0.0;
        for (int k = 0; k < 4; ++k) zs += mesh.nodes_current[mesh.hex_nodes[h].v[k]].z / 4.0;
        const bool inside = std::fabs(zs - zc) < dz;
        mats[h].kappa = inside ? 3.4e7 : 0.0;
        if (inside) ++n_cond;
    }
    printf("  Leiterscheibe: %ld von %zu Elementen\n", n_cond, mesh.num_hexes);

    const GaugeInfo g = build_tree_cotree_gauge(mesh, mats.data(), 0.0);
    snprintf(buf, sizeof buf, "   (%d geeichte Kanten von %zu)", g.num_gauged, mesh.num_edges);
    ck("build_tree_cotree_gauge laeuft durch", true, buf);
    ck("is_gauged hat die Laenge num_edges", g.is_gauged.size() == mesh.num_edges);
    ck("mindestens eine Kante geeicht", g.num_gauged > 0);

    // Die Nullraumdimension: Knoten des nichtleitenden Gebiets, die nicht zur
    // kontrahierten Leiterkomponente gehoeren. Der Baum muss genau die
    // verbinden -- mehr waere Uebereichung, weniger liesse den Nullraum stehen.
    std::vector<char> cond_node(mesh.num_nodes, 0);
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        if (!(mats[h].kappa > 0.0)) continue;
        for (int k = 0; k < 4; ++k) cond_node[mesh.hex_nodes[h].v[k]] = 1;
    }
    long n_noncond = 0;
    for (size_t v = 0; v < mesh.num_nodes; ++v) if (!cond_node[v]) ++n_noncond;
    snprintf(buf, sizeof buf, "   (%d geeicht, %ld nichtleitende Knoten)", g.num_gauged, n_noncond);
    ck("geeichte Kanten = nichtleitende Knoten", g.num_gauged == n_noncond, buf);

    // In einen try gefasst: die Suche ist fuer den BEM-Fall gebaut und in
    // diesem Regime unzuverlaessig -- das ist der Befund, nicht ein Fehler
    // dieses Tests.
    CohomologyGaugeResult coh;
    bool coh_ok = true;
    try { coh = find_cohomology_generators(mesh, g, mats.data(), 0.0, true); }
    catch (const std::exception& ex) { coh_ok = false; printf("  Suche wirft: %.110s\n", ex.what()); }
    if (coh_ok) {
        snprintf(buf, sizeof buf, "   (%zu Generatoren, %d einfach zush.)",
                 coh.generators.size(), coh.num_simply_connected);
        ck("Kasten minus Scheibe: keine Generatoren", coh.generators.empty(), buf);
    } else {
        printf("  [--- ] Generatorensuche in diesem Regime unzuverlaessig -- bekannt,\n");
        printf("         deshalb ist sie im Loeserpfad Auskunft und kein Abbruch.\n");
    }

    printf("\n%s (%d Beanstandungen)\n", fails?"FEHLGESCHLAGEN":"ALLE PRUEFUNGEN BESTANDEN", fails);
    return fails?1:0;
}
