#include "TreeCotreeGauge.h"

#include <algorithm>

#include <stdexcept>
#include <string>
#include <numeric>

namespace {

// Union-Find mit Pfadverkürzung und Vereinigung nach Rang. Deterministisch,
// solange die Kanten in fester Reihenfolge verarbeitet werden.
struct UnionFind {
    std::vector<int> parent;
    std::vector<int> rank_;

    explicit UnionFind(int n) : parent(n), rank_(n, 0) {
        std::iota(parent.begin(), parent.end(), 0);
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];   // Pfadverkürzung
            x = parent[x];
        }
        return x;
    }

    // true, wenn zwei bisher getrennte Mengen vereinigt wurden
    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rank_[a] < rank_[b]) { const int t = a; a = b; b = t; }
        parent[b] = a;
        if (rank_[a] == rank_[b]) ++rank_[a];
        return true;
    }
};

} // namespace

GaugeInfo build_tree_cotree_gauge(const Mesh& mesh,
                                  const MaterialProperties* materials,
                                  double kappa_threshold)
{
    if (materials == nullptr) {
        throw std::runtime_error("build_tree_cotree_gauge: materials ist nullptr.");
    }

    const int num_nodes = static_cast<int>(mesh.num_nodes);
    const int num_hexes = static_cast<int>(mesh.num_hexes);
    const int num_edges = static_cast<int>(mesh.num_edges);

    GaugeInfo out;
    out.num_edges = num_edges;
    out.is_gauged.assign(num_edges, 0);

    // -------------------------------------------------------------------------
    // 1. Leitende Elemente, daraus leitende Knoten und leitende Kanten
    // -------------------------------------------------------------------------
    std::vector<char> conductor_node(num_nodes, 0);
    std::vector<char> conductor_edge(num_edges, 0);

    for (int h = 0; h < num_hexes; ++h) {
        if (!(materials[h].kappa > kappa_threshold)) continue;

        // Nur die belegten Knotenplaetze: bei einem Tetraeder stehen v[4..7]
        // auf -1. Die Pruefung darunter faengt das laut ab, aber sie sollte
        // ueberhaupt nicht ausgeloest werden -- ein Tetraedernetz ist kein
        // Fehler.
        const int nn = element_num_nodes(mesh.elem_type(h));
        for (int k = 0; k < nn; ++k) {
            const int v = mesh.hex_nodes[h].v[k];
            if (v < 0 || v >= num_nodes) {
                throw std::runtime_error("build_tree_cotree_gauge: Knotenindex "
                    + std::to_string(v) + " von Hexaeder " + std::to_string(h)
                    + " liegt ausserhalb von [0," + std::to_string(num_nodes) + ").");
            }
            conductor_node[v] = 1;
        }
        const int ne = element_num_edges(mesh.elem_type(h));
        for (int e = 0; e < ne; ++e) {
            const int ge = mesh.hex_edges[h * 12 + e];
            if (ge < 0 || ge >= num_edges) {
                throw std::runtime_error("build_tree_cotree_gauge: Kantenindex "
                    + std::to_string(ge) + " liegt ausserhalb von [0,"
                    + std::to_string(num_edges) + ").");
            }
            conductor_edge[ge] = 1;
        }
    }

    for (int v = 0; v < num_nodes; ++v)  out.num_conductor_nodes += conductor_node[v];
    for (int e = 0; e < num_edges; ++e)  out.num_conductor_edges += conductor_edge[e];

    // -------------------------------------------------------------------------
    // 2. Union-Find; alle Leiterknoten vorab zu einer Menge kontrahieren
    // -------------------------------------------------------------------------
    // Grund: phi muss auf der leitenden Komponente konstant sein, dort gibt es
    // also keinen freien Gradientenanteil. Ohne die Kontraktion nähme der Baum
    // Verbindungen innerhalb des Leiters auf und fixierte mehr, als der
    // Nullraum hergibt.
    UnionFind uf(num_nodes);

    int ground = -1;
    for (int v = 0; v < num_nodes; ++v) {
        if (!conductor_node[v]) continue;
        if (ground < 0) ground = v;
        else            uf.unite(ground, v);
    }

    // Sonderfall: gar kein Leiter im Netz. Dann ist die Konstante der einzige
    // verbleibende Freiheitsgrad; ein beliebiger Knoten wird geerdet.
    if (ground < 0 && num_nodes > 0) {
        ground = 0;
    }

    // -------------------------------------------------------------------------
    // 3. Randkanten identifizieren
    // -------------------------------------------------------------------------
    // mesh.boundary_edges_to_fem_edges haelt 4 Kanten pro
    // Rand-Quad (siehe TopologyExtractor::extract_boundary_faces) -- markiert
    // jede FEM-Kante, die auf der BEM-Koppelflaeche liegt, damit Phase 1
    // unten sie meiden kann. ACHTUNG: mesh.num_boundary_faces ist im Mesh-
    // Konstruktor NICHT auf 0 initialisiert (siehe Mesh.cuh) -- ohne
    // vorherigen Aufruf von extract_boundary_faces() waere es unbelegter
    // Speicher. Deshalb ueber die DeviceArray-Groesse selbst absichern
    // (start bei 0, erst durch .allocate() in extract_boundary_faces()
    // veraendert) statt mesh.num_boundary_faces blind zu vertrauen --
    // test_tree_cotree_gauge.cu z.B. ruft extract_boundary_faces() nicht
    // auf und muss weiterhin funktionieren (dann exakt wie zuvor: keine
    // Randkanten bekannt, einphasiges Verhalten).
    std::vector<char> is_boundary_edge(num_edges, 0);
    if (mesh.boundary_edges_to_fem_edges.size() > 0) {
        // Die Schrittweite ist 4, aber ein DREIECK belegt nur drei Plaetze --
        // der vierte steht auf -1. TopologyExtractor meldet das selbst
        // ("triangles leave the fourth slot at -1"). Ohne die Abfrage
        // schreibt is_boundary_edge[-1] vor den Anfang des Vektors, und zwar
        // einmal je Randflaeche: bei einem reinen Tetraedernetz sind das ALLE
        // Flaechen. Unter MSVC schlaegt das als STATUS_STACK_BUFFER_OVERRUN
        // (0xC0000409) zu, also erst spaeter und ohne Bezug zur Ursache.
        //
        // Die obere Schranke wird zusaetzlich an der Feldgroesse gekappt:
        // num_boundary_faces stimmt zwar hier, aber die Groesse ist die
        // Wahrheit, und eine der beiden umsonst zu pruefen kostet nichts.
        const size_t n = std::min(mesh.num_boundary_faces * 4,
                                  mesh.boundary_edges_to_fem_edges.size());
        for (size_t i = 0; i < n; ++i) {
            const int e = mesh.boundary_edges_to_fem_edges[i];
            if (e < 0 || e >= num_edges) continue;   // Dreiecksflaeche
            is_boundary_edge[e] = 1;
        }
    }

    // -------------------------------------------------------------------------
    // 4. Zwei-Phasen-Spannbaum ueber die nichtleitenden Kanten (Guided Gauge)
    // -------------------------------------------------------------------------
    // Jeder gueltige Spannbaum ueber den nichtleitenden Kanten eicht korrekt
    // (Eichfreiheit) -- WELCHE Kanten dabei fest auf 0 gesetzt werden, ist
    // frei waehlbar, beeinflusst aber die Kopplung an die BEM-Randintegral-
    // gleichung: eine auf der Koppelflaeche geeichte Kante zwingt dort ein
    // hartes Null in ein tangentiales A-Feld, das die BEM-Gleichung sonst
    // frei/glatt vorfinden wuerde. Deshalb zwei Phasen statt einer
    // aufsteigenden Schleife ueber alle Kanten:
    //   Phase 1: nur innere (nicht-Rand) Kanten -- haelt die Eichung "tief
    //            im Volumen", die BEM-Koppelflaeche bleibt frei von
    //            kuenstlichen Nullen.
    //   Phase 2: Randkanten NUR dort, wo Phase 1 eine Zusammenhangs-
    //            komponente nicht vollstaendig verbinden konnte (z.B. wenn
    //            ein Teilgraph ausschliesslich ueber Randkanten zusammen-
    //            haengt). Dasselbe UnionFind uf laeuft ueber beide Phasen
    //            weiter, die Gesamtzahl der geeichten Kanten (= Nullraum-
    //            dimension) aendert sich dadurch nicht -- nur WELCHE
    //            Kanten es sind.
    // Aufsteigende Kantenindizes innerhalb jeder Phase: das Ergebnis bleibt
    // damit plattformunabhaengig reproduzierbar (wie zuvor).
    for (int e = 0; e < num_edges; ++e) {
        if (conductor_edge[e] || is_boundary_edge[e]) continue; // Leiter & Rand ueberspringen

        const int2 ed = mesh.edges[e];
        if (uf.unite(ed.x, ed.y)) {
            out.is_gauged[e] = 1;
            ++out.num_gauged;
        }
    }

    for (int e = 0; e < num_edges; ++e) {
        if (conductor_edge[e] || !is_boundary_edge[e]) continue; // nur noch Randkanten

        const int2 ed = mesh.edges[e];
        if (uf.unite(ed.x, ed.y)) {
            out.is_gauged[e] = 1;
            ++out.num_gauged;
        }
    }

    return out;
}
