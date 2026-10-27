// =============================================================================
// Prüft, dass die Baum-Kobaum-Eichung den Nullraum von A_sys = K + C/dt
// vollständig entfernt und dabei keine physikalische Kante fixiert.
//
//   cmake --build build --target test_tree_cotree_gauge
//   ./build/test_tree_cotree_gauge            (aus dem Repo-Wurzelverzeichnis)
//
// Der Nullraum besteht aus den diskreten Gradienten a = G·φ, für die C·a = 0
// gilt: φ ist auf jeder leitenden Komponente konstant und auf den übrigen Knoten
// frei. Die Eichung setzt a_e = 0 auf den Kanten eines Spannbaums über die
// nichtleitenden Kanten, wobei alle Leiterknoten zu einem geerdeten Knoten
// kontrahiert werden.
//
// Geprüft wird:
//
//   1. Keine leitende Kante ist geeicht. Dort ist A_sys regulär, eine Fixierung
//      würde echte Freiheitsgrade zerstören.
//   2. Die geeichten Kanten bilden einen Wald ohne Kreise. Ein Kreis hieße, dass
//      mehr fixiert wird als der Nullraum hergibt, das System wäre überbestimmt.
//   3. Der Baum verbindet jeden nichtleitenden Knoten mit der Erde. Nur dann ist
//      φ vollständig festgelegt.
//   4. INJEKTIVITÄT, der eigentliche Nachweis: für ein zufälliges φ wird
//      a = G·φ gebildet, daraus φ allein aus den Werten auf den Baumkanten
//      rekonstruiert und mit dem Original verglichen. Stimmen beide überein, ist
//      G auf den Baumkanten injektiv, also kann kein von null verschiedener
//      Gradient die Eichung überleben. Genau das heißt "Nullraum entfernt".
//   5. Die Zahl der geeichten Kanten entspricht der Nullraumdimension.
//
// Exit-Code 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "VtkReader.h"
#include "TopologyExtractor.h"
#include "TreeCotreeGauge.h"

#include <cstdio>
#include <vector>
#include <queue>
#include <cmath>
#include <numeric>
#include <functional>

int main(int argc, char** argv)
{
    const char* mesh_path = (argc > 1) ? argv[1] : "team28.vtk";

    Mesh mesh;
    VtkReader::read_hexahedral_mesh(mesh_path, mesh);
    TopologyExtractor::extract_edges(mesh);

    const int num_nodes = static_cast<int>(mesh.num_nodes);
    const int num_edges = static_cast<int>(mesh.num_edges);
    const int num_hexes = static_cast<int>(mesh.num_hexes);

    // Materialien wie in mainloop: Component 3 ist die Aluminiumplatte (leitend),
    // Component 1 und 2 sind die Spulen mit kappa = 0.
    std::vector<MaterialProperties> mats(num_hexes);
    for (int h = 0; h < num_hexes; ++h) {
        const int comp = mesh.hex_component_ids[h];
        mats[h].kappa = (comp == 3) ? 3.4e7 : 0.0;
        mats[h].mu_relative = 1.0;
    }

    const GaugeInfo g = build_tree_cotree_gauge(mesh, mats.data(), 0.0);

    printf("\n=============== Baum-Kobaum-Eichung ===============\n");
    printf("Knoten                        = %d\n", num_nodes);
    printf("Kanten                        = %d\n", num_edges);
    printf("davon leitend                 = %d\n", g.num_conductor_edges);
    printf("Leiterknoten                  = %d\n", g.num_conductor_nodes);
    printf("geeichte Kanten (Baumkanten)  = %d\n\n", g.num_gauged);

    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        printf("  [%s] %s\n", ok ? "OK  " : "FEHL", what);
        if (!ok) ++failures;
    };

    // ---- 1: keine leitende Kante geeicht ----
    {
        std::vector<char> cond_edge(num_edges, 0);
        for (int h = 0; h < num_hexes; ++h) {
            if (!(mats[h].kappa > 0.0)) continue;
            for (int e = 0; e < 12; ++e) cond_edge[mesh.hex_edges[h * 12 + e]] = 1;
        }
        int bad = 0;
        for (int e = 0; e < num_edges; ++e) if (g.is_gauged[e] && cond_edge[e]) ++bad;
        (void)0;
        printf("      leitende und gleichzeitig geeichte Kanten: %d\n", bad);
        check(bad == 0, "keine leitende Kante ist geeicht");
    }

    // Hilfsstrukturen: Leiterknoten, Union-Find unabhaengig vom Modul
    std::vector<char> cnode(num_nodes, 0);
    for (int h = 0; h < num_hexes; ++h) {
        if (!(mats[h].kappa > 0.0)) continue;
        for (int k = 0; k < 8; ++k) cnode[mesh.hex_nodes[h].v[k]] = 1;
    }
    std::vector<char> cedge(num_edges, 0);
    for (int h = 0; h < num_hexes; ++h) {
        if (!(mats[h].kappa > 0.0)) continue;
        for (int e = 0; e < 12; ++e) cedge[mesh.hex_edges[h * 12 + e]] = 1;
    }

    auto make_uf = [&](std::vector<int>& parent) {
        parent.resize(num_nodes);
        std::iota(parent.begin(), parent.end(), 0);
    };
    std::function<int(std::vector<int>&, int)> find =
        [](std::vector<int>& p, int x) {
            while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
            return x;
        };
    auto unite = [&](std::vector<int>& p, int a, int b) {
        a = find(p, a); b = find(p, b);
        if (a == b) return false;
        p[b] = a; return true;
    };

    // ---- 2: geeichte Kanten bilden einen Wald ohne Kreise ----
    // Referenz: Leiterknoten kontrahiert, dann die Baumkanten einzeln hinzu.
    {
        std::vector<int> p; make_uf(p);
        int ground = -1;
        for (int v = 0; v < num_nodes; ++v) {
            if (!cnode[v]) continue;
            if (ground < 0) ground = v; else unite(p, ground, v);
        }
        int cycles = 0;
        for (int e = 0; e < num_edges; ++e) {
            if (!g.is_gauged[e]) continue;
            const int2 ed = mesh.edges[e];
            if (!unite(p, ed.x, ed.y)) ++cycles;
        }
        printf("      Kreise unter den geeichten Kanten: %d\n", cycles);
        check(cycles == 0, "geeichte Kanten bilden einen Wald ohne Kreise");
    }

    // ---- 3 und 5: der Wald spannt den Eichgraphen auf ----
    // Eichgraph: Knoten mit kontrahierter Leitermenge, Kanten alle
    // nichtleitenden. Ein Spannwald hat V' - C' Kanten. Genau diese Zahl ist
    // auch die Nullraumdimension, denn phi ist pro Komponente bis auf eine
    // Konstante frei, und eine Konstante liefert G phi = 0, ist also kein
    // Nullvektor.
    //
    // Wichtig bei diesem Netz: es gibt kein Luftgebiet. Platte (z um 5 mm) und
    // Spulen (z = -26 mm) sind getrennte Netze, gekoppelt allein ueber das BEM.
    // Es existiert daher keine Kante vom Spulengebiet zur Platte, und jede
    // Spulenkomponente traegt eine eigene freie Konstante.
    int V_prime = 0, C_prime = 0, spanning = 0;
    {
        std::vector<int> p; make_uf(p);
        int ground = -1;
        for (int v = 0; v < num_nodes; ++v) {
            if (!cnode[v]) continue;
            if (ground < 0) ground = v; else unite(p, ground, v);
        }
        // Superknoten zaehlen: freie Knoten plus 1 fuer die Leitermenge
        int free_nodes = 0;
        for (int v = 0; v < num_nodes; ++v) if (!cnode[v]) ++free_nodes;
        V_prime = free_nodes + (ground >= 0 ? 1 : 0);

        for (int e = 0; e < num_edges; ++e) {
            if (cedge[e]) continue;
            const int2 ed = mesh.edges[e];
            if (unite(p, ed.x, ed.y)) ++spanning;
        }
        C_prime = V_prime - spanning;
        printf("      Eichgraph: %d Superknoten, %d Komponenten, Spannwald %d Kanten\n",
               V_prime, C_prime, spanning);
        printf("      davon getrennte nichtleitende Komponenten: %d\n",
               C_prime - (ground >= 0 ? 1 : 0));
        check(g.num_gauged == spanning,
              "geeichte Kanten bilden einen SPANNwald des Eichgraphen");
        check(g.num_gauged == V_prime - C_prime,
              "Anzahl geeichter Kanten gleich Nullraumdimension V' - C'");
    }

    // ---- 4: Injektivitaet, der eigentliche Nachweis ----
    // Fuer ein zufaelliges phi (null auf Leiterknoten) wird a = G phi gebildet,
    // daraus phi allein aus den Baumkanten rekonstruiert -- pro Komponente mit
    // frei gewaehlter Basis -- und geprueft, ob G phi_rec wieder genau a ergibt.
    // Gilt das, so kann kein von null verschiedener Gradient die Eichung
    // ueberleben: a|Baum = 0 erzwingt phi konstant pro Komponente und damit
    // a = 0 ueberall. Genau das heisst "Nullraum entfernt".
    {
        std::vector<double> phi(num_nodes, 0.0);
        unsigned seed = 987654321u;
        for (int v = 0; v < num_nodes; ++v) {
            if (cnode[v]) continue;
            seed = seed * 1103515245u + 12345u;
            phi[v] = ((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
        }
        std::vector<double> a(num_edges, 0.0);
        double a_norm = 0.0;
        for (int e = 0; e < num_edges; ++e) {
            const int2 ed = mesh.edges[e];
            const int lo = ed.x < ed.y ? ed.x : ed.y;
            const int hi = ed.x < ed.y ? ed.y : ed.x;
            a[e] = phi[hi] - phi[lo];
            a_norm += a[e] * a[e];
        }
        a_norm = std::sqrt(a_norm);

        std::vector<std::vector<int>> adj(num_nodes);
        for (int e = 0; e < num_edges; ++e) {
            if (!g.is_gauged[e]) continue;
            const int2 ed = mesh.edges[e];
            adj[ed.x].push_back(e);
            adj[ed.y].push_back(e);
        }
        std::vector<double> phi_rec(num_nodes, 0.0);
        std::vector<char> known(num_nodes, 0);
        std::queue<int> q;
        for (int v = 0; v < num_nodes; ++v) if (cnode[v]) { known[v] = 1; q.push(v); }
        int components_seeded = 0;
        for (;;) {
            while (!q.empty()) {
                const int v = q.front(); q.pop();
                for (int e : adj[v]) {
                    const int2 ed = mesh.edges[e];
                    const int w = (ed.x == v) ? ed.y : ed.x;
                    if (known[w]) continue;
                    const int lo = ed.x < ed.y ? ed.x : ed.y;
                    phi_rec[w] = (w == lo) ? (phi_rec[v] - a[e]) : (phi_rec[v] + a[e]);
                    known[w] = 1;
                    q.push(w);
                }
            }
            // naechste noch unbesuchte Komponente mit Basis 0 starten
            int seed_v = -1;
            for (int v = 0; v < num_nodes; ++v) if (!known[v]) { seed_v = v; break; }
            if (seed_v < 0) break;
            phi_rec[seed_v] = 0.0; known[seed_v] = 1; q.push(seed_v);
            ++components_seeded;
        }

        double max_err = 0.0;
        for (int e = 0; e < num_edges; ++e) {
            const int2 ed = mesh.edges[e];
            const int lo = ed.x < ed.y ? ed.x : ed.y;
            const int hi = ed.x < ed.y ? ed.y : ed.x;
            max_err = std::fmax(max_err, std::fabs((phi_rec[hi] - phi_rec[lo]) - a[e]));
        }
        printf("      ||a|| = %.4e, zusaetzlich geseedete Komponenten: %d\n",
               a_norm, components_seeded);
        printf("      max |G phi_rec - a| ueber alle Kanten = %.3e\n", max_err);
        check(max_err < 1e-12,
              "G aus den Baumkanten rekonstruierbar, Nullraum entfernt");
    }

    printf("\n=============== %s ===============\n\n",
           failures == 0 ? "BESTANDEN: Eichung entfernt den Nullraum"
                         : "FEHLGESCHLAGEN");
    return failures == 0 ? 0 : 1;
}
