#include "MeshRefinement.cuh"

#include "NodalHexahedron.cuh"

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

// -----------------------------------------------------------------------------
// Topologie des Referenzhexaeders in Indexkoordinaten
// -----------------------------------------------------------------------------
// Die 27 Knoten des verfeinerten Elternelements werden ueber (i,j,k) mit
// i,j,k in {0,1,2} adressiert, entsprechend xi,eta,zeta in {-1,0,1}. Der
// Eckknoten n des Elternelements liegt bei
//     i = sign_xi(n) + 1,  j = sign_eta(n) + 1,  k = sign_zeta(n) + 1
// Damit ist die Knotenreihenfolge der Kinder automatisch dieselbe Konvention
// wie die des Elternelements -- keine separate Tabelle, die falsch sein koennte.
inline int corner_i(int n) { return static_cast<int>(nodal_sign_xi(n))   + 1; }
inline int corner_j(int n) { return static_cast<int>(nodal_sign_eta(n))  + 1; }
inline int corner_k(int n) { return static_cast<int>(nodal_sign_zeta(n)) + 1; }

/// Kanonischer Schluessel: sortiertes Tupel von Elternknoten-IDs. Damit finden
/// Nachbarelemente denselben geteilten Knoten.
using Key = std::array<int, 8>;

Key make_key(std::vector<int> ids) {
    std::sort(ids.begin(), ids.end());
    Key k{};
    k.fill(-1);
    for (size_t i = 0; i < ids.size() && i < 8; ++i) k[i] = ids[i];
    return k;
}

// =============================================================================
// Tetraeder: uniforme Verfeinerung 1 -> 8 nach Freudenthal/Bey
// =============================================================================
//
// Vier Eckkinder an den vier Ecken, und was in der Mitte uebrigbleibt, ist ein
// OKTAEDER mit den sechs Kantenmitten als Ecken. Der wird in vier Tetraeder
// geteilt, und dafuer gibt es drei gleichberechtigte Moeglichkeiten -- eine je
// Raumdiagonale des Oktaeders:
//
//     m01-m23,   m02-m13,   m03-m12
//
// Das sind genau die Verbindungen der Mitten GEGENUEBERLIEGENDER Tetraederkanten.
//
// Die Wahl ist keine Formalie. Alle drei erzeugen gueltige Netze, aber ihre
// Elementguete unterscheidet sich erheblich, und bei wiederholter Verfeinerung
// pflanzt sich eine schlechte Wahl fort. Genommen wird deshalb die KUERZESTE
// Diagonale -- das ist Beys Regel und liefert unter den dreien die
// gleichmaessigsten Kinder.
//
// Wichtig fuer die Konformitaet: die Wahl haengt nur an den Kantenmitten des
// Elternelements, und die sind zwischen Nachbarn geteilt. Zwei Nachbarn treffen
// an ihrer gemeinsamen Flaeche also dieselbe Unterteilung -- die Flaeche wird
// beidseitig in dieselben vier Dreiecke geteilt, unabhaengig davon, welche
// Diagonale jeder im Inneren waehlt. Haengende Knoten koennen so nicht
// entstehen.
//
// Volumentreue: ein Tetraeder wird AFFIN abgebildet, die acht Kinder fuellen
// ihn deshalb EXAKT aus. Anders als beim Hexaeder, wo die trilineare Abbildung
// nur stueckweise angenaehert wird -- die Volumenprobe unten muss hier also auf
// Rundungsniveau herauskommen.

// Die sechs Kanten des Tetraeders in der Reihenfolge, die die Mitten-Indizes
// m[0..5] festlegt. Muss NICHT mit element_local_edge_nodes uebereinstimmen:
// hier geht es um Geometrie, nicht um Freiheitsgrade.
static const int TET_REF_EDGES[6][2] = {
    {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
};
// Die drei Diagonalen als Indexpaare in m[]: (m01,m23) (m02,m13) (m03,m12)
static const int TET_OCTA_DIAG[3][2] = { {0,5}, {1,4}, {2,3} };
// Der Aequator je Diagonale: die vier VERBLEIBENDEN Mitten als Viererzyklus.
//
// Zwei Bedingungen, und beide sind leicht zu verletzen: der Ring darf die
// Endpunkte seiner eigenen Diagonale nicht enthalten, und aufeinanderfolgende
// Ringpunkte duerfen kein Diagonalenpaar sein (sonst ist es kein Zyklus auf
// den Oktaederkanten). Hier standen einmal {0,2,5,4} zu Diagonale {1,4} und
// {0,1,5,3} zu {2,3} -- in beiden Faellen steckte ein Diagonalenendpunkt im
// eigenen Ring. Die vier Innenkinder deckten dann nur die Haelfte des
// Oktaeders ab, das Netz verlor ein Viertel seines Volumens und ueberlappte
// sich zugleich.
//
// Nicht aufgefallen waere es an einem Kuhn-Netz: dessen Elemente waehlen
// praktisch nie Diagonale 0, und genau die war als einzige richtig.
static const int TET_OCTA_RING[3][4] = {
    {1,2,4,3},   // zu Diagonale (0,5) = m01-m23
    {0,2,5,3},   // zu Diagonale (1,4) = m02-m13
    {0,1,5,4},   // zu Diagonale (2,3) = m03-m12
};

static double tet_signed_volume(const double3& a, const double3& b,
                                const double3& c, const double3& d) {
    const double3 e1 = {b.x-a.x, b.y-a.y, b.z-a.z};
    const double3 e2 = {c.x-a.x, c.y-a.y, c.z-a.z};
    const double3 e3 = {d.x-a.x, d.y-a.y, d.z-a.z};
    return ((e1.y*e2.z - e1.z*e2.y) * e3.x
          + (e1.z*e2.x - e1.x*e2.z) * e3.y
          + (e1.x*e2.y - e1.y*e2.x) * e3.z) / 6.0;
}

/// Legt ein Kind an und erzwingt positive Orientierung.
///
/// Das Vorzeichen von detJ geht in fem.cu bei der kontravarianten
/// Piola-Transformation durch eine Division; mathematisch ist ein negatives
/// detJ konsistent, aber ein Netz mit gemischten Vorzeichen verdeckt echte
/// Fehler und macht jede Volumensumme unlesbar.
static void push_tet(std::vector<Hexahedron>& out, std::vector<int>& out_cid,
                     const std::vector<double3>& pos, int cid,
                     int a, int b, int c, int d)
{
    if (tet_signed_volume(pos[a], pos[b], pos[c], pos[d]) < 0.0) std::swap(c, d);
    Hexahedron h;
    for (int i = 0; i < 8; ++i) h.v[i] = -1;
    h.v[0] = a; h.v[1] = b; h.v[2] = c; h.v[3] = d;
    out.push_back(h);
    out_cid.push_back(cid);
}

/// Summe der Tetraedervolumina. NICHT total_volume() aus dem Hexaederteil --
/// das wertet die trilineare Abbildung ueber acht Knoten aus und liefert auf
/// einem Tetraeder, dessen Knoten 4..7 auf -1 stehen, eine bedeutungslose
/// Zahl. Beim ersten Anlauf stand hier total_volume(), und die Volumenprobe
/// meldete -1.27 m^3 fuer ein Netz mit 1.0 m^3.
static double total_tet_volume(const Mesh& mesh) {
    double v = 0.0;
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const Hexahedron H = mesh.hex_nodes[h];
        v += tet_signed_volume(mesh.nodes_initial[H.v[0]], mesh.nodes_initial[H.v[1]],
                               mesh.nodes_initial[H.v[2]], mesh.nodes_initial[H.v[3]]);
    }
    return v;
}

void refine_tet_once(Mesh& mesh, size_t* out_old_edges, size_t* out_old_faces) {
    const size_t old_nodes = mesh.num_nodes;
    const size_t old_tets  = mesh.num_hexes;

    mesh.nodes_initial.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();
    mesh.hex_component_ids.prefetch_to_host();

    std::vector<double3> pos(old_nodes);
    for (size_t i = 0; i < old_nodes; ++i) pos[i] = mesh.nodes_initial[i];
    std::vector<Hexahedron> tet(old_tets);
    std::vector<int>        cid(old_tets);
    for (size_t h = 0; h < old_tets; ++h) {
        tet[h] = mesh.hex_nodes[h];
        cid[h] = mesh.hex_component_ids[h];
    }

    // Kantenmitten, geteilt ueber den kanonischen Schluessel -- sonst legte
    // jedes Element seine eigenen an, das Netz zerfiele in unverbundene
    // Stuecke und FEM rechnete stillschweigend Unsinn.
    std::map<Key, int> midpoint;
    std::vector<Hexahedron> new_tet;
    std::vector<int>        new_cid;
    new_tet.reserve(old_tets * 8);
    new_cid.reserve(old_tets * 8);

    long diag_count[3] = {0, 0, 0};

    for (size_t h = 0; h < old_tets; ++h) {
        const int v[4] = {tet[h].v[0], tet[h].v[1], tet[h].v[2], tet[h].v[3]};

        // Wachtposten. Ein Tetraeder MUSS vier gueltige Knoten haben; steht
        // dort -1 oder ein Index ausserhalb, greift der Zugriff unten ins
        // Leere und der Prozess stirbt ohne Meldung. Lieber hier eine Zeile,
        // die sagt welches Element und welcher Wert.
        for (int i = 0; i < 4; ++i) {
            if (v[i] < 0 || static_cast<size_t>(v[i]) >= pos.size()) {
                std::ostringstream e;
                e << "refine_tet_once: Element " << h << " hat den ungueltigen "
                     "Knotenindex v[" << i << "] = " << v[i]
                  << " (gueltig sind 0.." << pos.size() - 1 << "). Entweder ist "
                     "das Netz kein reines Tetraedernetz, oder elem_types passt "
                     "nicht zu hex_nodes.";
                throw std::runtime_error(e.str());
            }
        }

        int m[6];
        for (int e = 0; e < 6; ++e) {
            const int a = v[TET_REF_EDGES[e][0]];
            const int b = v[TET_REF_EDGES[e][1]];
            const Key key = make_key({a, b});
            auto it = midpoint.find(key);
            if (it != midpoint.end()) {
                m[e] = it->second;
            } else {
                const int id = static_cast<int>(pos.size());
                pos.push_back(make_double3(0.5 * (pos[a].x + pos[b].x),
                                           0.5 * (pos[a].y + pos[b].y),
                                           0.5 * (pos[a].z + pos[b].z)));
                midpoint.emplace(key, id);
                m[e] = id;
            }
        }

        // Vier Eckkinder
        push_tet(new_tet, new_cid, pos, cid[h], v[0], m[0], m[1], m[2]);
        push_tet(new_tet, new_cid, pos, cid[h], v[1], m[0], m[3], m[4]);
        push_tet(new_tet, new_cid, pos, cid[h], v[2], m[1], m[3], m[5]);
        push_tet(new_tet, new_cid, pos, cid[h], v[3], m[2], m[4], m[5]);

        // Oktaeder: kuerzeste Diagonale waehlen (Bey)
        int best = 0;
        double best_len = -1.0;
        for (int d = 0; d < 3; ++d) {
            const double3& p = pos[m[TET_OCTA_DIAG[d][0]]];
            const double3& q = pos[m[TET_OCTA_DIAG[d][1]]];
            const double len = (p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y)
                             + (p.z-q.z)*(p.z-q.z);
            if (best_len < 0.0 || len < best_len) { best_len = len; best = d; }
        }
        ++diag_count[best];

        const int d0 = m[TET_OCTA_DIAG[best][0]];
        const int d1 = m[TET_OCTA_DIAG[best][1]];
        const int* ring = TET_OCTA_RING[best];
        for (int k = 0; k < 4; ++k)
            push_tet(new_tet, new_cid, pos, cid[h],
                     d0, d1, m[ring[k]], m[ring[(k + 1) % 4]]);
    }

    mesh.num_nodes = pos.size();
    mesh.num_hexes = new_tet.size();

    mesh.nodes_initial.allocate(mesh.num_nodes);
    mesh.nodes_current.allocate(mesh.num_nodes);
    mesh.hex_nodes.allocate(mesh.num_hexes);
    mesh.hex_component_ids.allocate(mesh.num_hexes);
    mesh.elem_types.allocate(mesh.num_hexes);

    mesh.nodes_initial.prefetch_to_host();
    mesh.nodes_current.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();
    mesh.hex_component_ids.prefetch_to_host();
    mesh.elem_types.prefetch_to_host();

    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        mesh.nodes_initial[i] = pos[i];
        mesh.nodes_current[i] = pos[i];
    }
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        mesh.hex_nodes[h]         = new_tet[h];
        mesh.hex_component_ids[h] = new_cid[h];
        mesh.elem_types[h]        = ELEM_TET;
    }

    mesh.nodes_initial.prefetch_to_device();
    mesh.nodes_current.prefetch_to_device();
    mesh.hex_nodes.prefetch_to_device();
    mesh.hex_component_ids.prefetch_to_device();
    mesh.elem_types.prefetch_to_device();

    // Die Kantenmitten sind genau die Kanten des ALTEN Netzes -- damit ist die
    // Kantenzahl bekannt, ohne sie zu extrahieren. Flaechen zaehlt diese
    // Verfeinerung nicht mit; out_old_faces bleibt 0 und der Aufrufer sagt das.
    if (out_old_edges) *out_old_edges = midpoint.size();
    if (out_old_faces) *out_old_faces = 0;

    std::cout << "[Refine]   Oktaeder-Diagonale (kuerzeste je Element): "
              << diag_count[0] << " / " << diag_count[1] << " / " << diag_count[2]
              << "\n";
}

void refine_once(Mesh& mesh, size_t* out_old_edges, size_t* out_old_faces) {
    // Die Verfeinerung teilt jeden Hexaeder in acht und setzt dabei die
    // Trilinearitaet der Abbildung voraus. Ein Tetraeder oder Prisma wuerde
    // hier still als Hexaeder behandelt: die Knoten 4..7 stehen bei einem
    // Tetraeder auf -1, und daraus entstuenden Elemente mit ungueltigen
    // Knotennummern, ohne dass irgendetwas abbricht.
    if (!mesh.is_pure_hex()) {
        throw std::runtime_error(
            "refine_once: uniforme Verfeinerung gibt es nur fuer reine "
            "Hexaedernetze. Fuer ein Tetraeder- oder Prismennetz muss die "
            "Netzreihe fuer die Konvergenzstudie aus dem Netzgenerator kommen "
            "(--refine ist dort nicht anwendbar).");
    }

    const size_t old_nodes = mesh.num_nodes;
    const size_t old_hexes = mesh.num_hexes;

    mesh.nodes_initial.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();
    mesh.hex_component_ids.prefetch_to_host();

    // Alte Daten auf den Host uebernehmen -- die DeviceArrays werden neu angelegt.
    std::vector<double3>     pos(old_nodes);
    for (size_t i = 0; i < old_nodes; ++i) pos[i] = mesh.nodes_initial[i];
    std::vector<Hexahedron>  hex(old_hexes);
    std::vector<int>         cid(old_hexes);
    for (size_t h = 0; h < old_hexes; ++h) {
        hex[h] = mesh.hex_nodes[h];
        cid[h] = mesh.hex_component_ids[h];
    }

    std::map<Key, int> shared;      // Kantenmitten und Flaechenmitten
    std::vector<Hexahedron> new_hex;
    std::vector<int>        new_cid;
    new_hex.reserve(old_hexes * 8);
    new_cid.reserve(old_hexes * 8);

    // Die 12 Kanten und 6 Flaechen in Indexkoordinaten: fuer jeden der 27
    // Gitterpunkte wird bestimmt, welche Elternecken ihn aufspannen. Das
    // ergibt sich direkt aus der Zahl der Einsen in (i,j,k):
    //   0 Einsen -> Eckknoten (existiert schon)
    //   1 Eins   -> Kantenmitte, 2 Ecken
    //   2 Einsen -> Flaechenmitte, 4 Ecken
    //   3 Einsen -> Zellmitte, 8 Ecken (nie geteilt)
    for (size_t h = 0; h < old_hexes; ++h) {
        const Hexahedron& H = hex[h];
        double3 pc[8];
        for (int n = 0; n < 8; ++n) pc[n] = pos[H.v[n]];

        int g[3][3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    g[i][j][k] = -1;

        // Ecken eintragen
        for (int n = 0; n < 8; ++n) {
            g[corner_i(n)][corner_j(n)][corner_k(n)] = H.v[n];
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    if (g[i][j][k] >= 0) continue;   // Eckknoten

                    // Welche Elternecken spannen diesen Punkt auf? Alle Ecken,
                    // deren Indexkoordinaten in den nicht-mittigen Richtungen
                    // uebereinstimmen.
                    std::vector<int> parents;
                    for (int n = 0; n < 8; ++n) {
                        const int ci = corner_i(n), cj = corner_j(n), ck = corner_k(n);
                        if ((i != 1 && ci != i) || (j != 1 && cj != j) || (k != 1 && ck != k))
                            continue;
                        parents.push_back(H.v[n]);
                    }

                    const int ones = (i == 1) + (j == 1) + (k == 1);
                    int id = -1;
                    if (ones == 3) {
                        // Zellmitte: gehoert nur diesem Element, kein Teilen.
                        id = static_cast<int>(pos.size());
                        pos.push_back(nodal_map_to_physical(pc, 0.0, 0.0, 0.0));
                    } else {
                        const Key key = make_key(parents);
                        auto it = shared.find(key);
                        if (it != shared.end()) {
                            id = it->second;
                        } else {
                            id = static_cast<int>(pos.size());
                            pos.push_back(nodal_map_to_physical(
                                pc, static_cast<double>(i) - 1.0,
                                    static_cast<double>(j) - 1.0,
                                    static_cast<double>(k) - 1.0));
                            shared.emplace(key, id);
                        }
                    }
                    g[i][j][k] = id;
                }
            }
        }

        // Acht Kinder. Kind (a,b,c) belegt den Teilwuerfel [a,a+1]x[b,b+1]x[c,c+1]
        // im Indexraum; seine Ecke n sitzt bei a + (sign+1)/2 usw. Damit erbt
        // jedes Kind die Knotenreihenfolge und damit die Orientierung des
        // Elternelements.
        for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
                for (int c = 0; c < 2; ++c) {
                    Hexahedron child{};
                    for (int n = 0; n < 8; ++n) {
                        const int di = (static_cast<int>(nodal_sign_xi(n))   + 1) / 2;
                        const int dj = (static_cast<int>(nodal_sign_eta(n))  + 1) / 2;
                        const int dk = (static_cast<int>(nodal_sign_zeta(n)) + 1) / 2;
                        child.v[n] = g[a + di][b + dj][c + dk];
                    }
                    new_hex.push_back(child);
                    new_cid.push_back(cid[h]);   // Component ID wird vererbt
                }
            }
        }
    }

    // Neue Arrays anlegen und fuellen
    mesh.num_nodes = pos.size();
    mesh.num_hexes = new_hex.size();

    mesh.nodes_initial.allocate(mesh.num_nodes);
    mesh.nodes_current.allocate(mesh.num_nodes);
    mesh.hex_nodes.allocate(mesh.num_hexes);
    mesh.hex_component_ids.allocate(mesh.num_hexes);

    // elem_types muss mitwachsen. Bliebe es auf der alten Laenge, waere der
    // Zeiger weiterhin nicht null und die Assemblierung laese fuer die neuen
    // Elemente darueber hinaus. Der Wachtposten oben stellt sicher, dass hier
    // nur Hexaeder ankommen, also ist ELEM_HEX fuer alle richtig.
    if (mesh.elem_types.size() > 0) {
        mesh.elem_types.allocate(mesh.num_hexes);
        mesh.elem_types.prefetch_to_host();
        for (size_t i = 0; i < mesh.num_hexes; ++i) mesh.elem_types[i] = ELEM_HEX;
    }

    mesh.nodes_initial.prefetch_to_host();
    mesh.nodes_current.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();
    mesh.hex_component_ids.prefetch_to_host();

    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        mesh.nodes_initial[i] = pos[i];
        mesh.nodes_current[i] = pos[i];
    }
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        mesh.hex_nodes[h]         = new_hex[h];
        mesh.hex_component_ids[h] = new_cid[h];
    }

    mesh.nodes_initial.prefetch_to_device();
    mesh.nodes_current.prefetch_to_device();
    mesh.hex_nodes.prefetch_to_device();
    mesh.hex_component_ids.prefetch_to_device();

    // Nebenprodukt: der shared-Map enthaelt genau die Kantenmitten (2 Eltern)
    // und Flaechenmitten (4 Eltern) des ALTEN Netzes. Damit sind Kanten- und
    // Flaechenzahl des Ausgangsnetzes bekannt, ohne sie separat zu extrahieren --
    // und daraus laesst sich die Kantenzahl des verfeinerten Netzes vorhersagen,
    // also die Zahl der Nedelec-Freiheitsgrade des produktiven Pfads.
    if (out_old_edges || out_old_faces) {
        size_t e = 0, f = 0;
        for (const auto& kv : shared) {
            const int n = (kv.first[2] < 0) ? 2 : 4;   // 2 oder 4 belegte Plaetze
            if (n == 2) ++e; else ++f;
        }
        if (out_old_edges) *out_old_edges = e;
        if (out_old_faces) *out_old_faces = f;
    }
}

/// Summe der Elementvolumina, ueber die Q1-Quadratur. Dient der Kontrolle:
/// die Verfeinerung darf das Gesamtvolumen nur um den Betrag aendern, den die
/// stueckweise trilineare Annaeherung der trilinearen Elternabbildung
/// ausmacht -- bei affinen Elementen exakt null.
double total_volume(const Mesh& mesh) {
    mesh.nodes_current.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();

    double vol = 0.0;
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const Hexahedron H = mesh.hex_nodes[h];
        double3 nd[8];
        for (int n = 0; n < 8; ++n) nd[n] = mesh.nodes_current[H.v[n]];

        // 2x2x2 reicht nicht fuer das exakte Volumen eines verzerrten
        // Hexaeders; 3x3x3 ist hier genau genug fuer eine Kontrollgroesse.
        const int ng = nodal_gauss_count_n(3);
        for (int g = 0; g < ng; ++g) {
            double xi, eta, zeta, w;
            nodal_gauss_point_n(3, g, xi, eta, zeta, w);
            double3 grad[8]; double detJ;
            evaluate_nodal_gradients(nd, xi, eta, zeta, grad, detJ);
            vol += w * detJ;
        }
    }
    return vol;
}

} // namespace

void predict_refined_sizes(size_t num_nodes, size_t num_hexes, size_t num_edges,
                           size_t num_boundary_faces, size_t num_boundary_nodes,
                           int levels,
                           size_t* out_nodes, size_t* out_hexes,
                           size_t* out_boundary_faces, size_t* out_boundary_nodes)
{
    size_t nn = num_nodes, nh = num_hexes, ne = num_edges;
    size_t bf = num_boundary_faces, bn = num_boundary_nodes;

    for (int l = 0; l < levels; ++l) {
        // Innere Flaechen doppelt gezaehlt, Randflaechen einfach
        const size_t nf = (6 * nh + bf) / 2;
        // Randkanten aus Euler je geschlossener Randkomponente: E = V + F - 2.
        // Die Zahl der Komponenten faellt bei der Differenz nicht ins Gewicht,
        // sie wird hier mit 3 angenommen (Platte, zwei Spulen).
        const size_t be = bn + bf - 6;

        const size_t nn_new = nn + ne + nf + nh;
        const size_t bn_new = bn + be + bf;

        nn = nn_new;
        nh *= 8;
        // Jede Kante wird geteilt (2 neue), plus je Flaeche 4 und je Zelle 6
        // neue innere Kanten -- fuer die Vorhersage genuegt die Naeherung
        // ne_new = 2*ne + 4*nf + 6*nh_alt.
        ne = 2 * ne + 4 * nf + 6 * (nh / 8);
        bf *= 4;
        bn = bn_new;
    }

    if (out_nodes)           *out_nodes = nn;
    if (out_hexes)           *out_hexes = nh;
    if (out_boundary_faces)  *out_boundary_faces = bf;
    if (out_boundary_nodes)  *out_boundary_nodes = bn;
}

void refine_mesh(Mesh& mesh, int levels) {
    if (levels <= 0) return;

    // std::flush, nicht "\n". Bei einem Abbruch geht alles verloren, was noch
    // im Puffer steht, und stdout ist an einer Pipe vollgepuffert -- ein
    // Absturz hier saehe im Protokoll so aus, als waere er schon beim
    // Netzeinlesen passiert. Diese Zeilen grenzen ihn ein.
    std::cout << "[Refine] Eingang: " << mesh.num_hexes << " Elemente, "
              << mesh.num_nodes << " Knoten, " << levels << " Stufe(n)"
              << std::endl;

    // Der Typ wird EINMAL vorab bestimmt und entscheidet den Weg. Ein
    // gemischtes Netz bricht hier ab statt still etwas Falsches zu tun:
    // Hexaeder und Tetraeder teilen ihre gemeinsame Flaeche nach
    // unterschiedlichen Mustern -- der Hexaeder in vier Vierecke ueber eine
    // Flaechenmitte, der Tetraeder in vier Dreiecke ohne eine. An so einer
    // Flaeche entstuenden HAENGENDE KNOTEN: Knoten, die das eine Element
    // kennt und das andere nicht. Das Netz saehe unauffaellig aus, die
    // Assemblierung liefe durch, und das Feld waere an der Grenze schlicht
    // nicht gekoppelt.
    const int type = mesh.uniform_elem_type();
    std::cout << "[Refine] Elementtyp einheitlich: "
              << (type == ELEM_TET   ? "Tetraeder"
                : type == ELEM_HEX   ? "Hexaeder"
                : type == ELEM_PRISM ? "Prisma" : "nein / unbekannt")
              << std::endl;

    if (type == ELEM_HEX) {
        refine_hex_mesh(mesh, levels);
        std::cout << "[Refine] fertig." << std::endl;
        return;
    }
    if (type == ELEM_TET) {
        refine_tet_mesh(mesh, levels);
        std::cout << "[Refine] fertig." << std::endl;
        return;
    }

    // Ab hier ist es ein Fehler. Die Meldung nennt die Zusammensetzung, weil
    // "gemischt" allein dem Benutzer nicht sagt, was er hat.
    std::ostringstream e;
    e << "refine_mesh: uniforme Verfeinerung gibt es nur fuer REINE Tetraeder- "
         "oder Hexaedernetze. ";
    if (mesh.num_hexes == 0) {
        e << "Das Netz ist leer.";
    } else {
        long n[5] = {0, 0, 0, 0, 0};
        mesh.elem_types.prefetch_to_host();
        for (size_t i = 0; i < mesh.num_hexes; ++i) {
            const int t = mesh.elem_type(i);
            if (t == ELEM_TET)        ++n[0];
            else if (t == ELEM_HEX)   ++n[1];
            else if (t == ELEM_PRISM) ++n[2];
            else                      ++n[3];
        }
        e << "Vorliegend: " << n[0] << " Tetraeder, " << n[1] << " Hexaeder, "
          << n[2] << " Prismen, " << n[3] << " sonstige. ";
        if (n[2] > 0 || n[3] > 0) {
            e << "Fuer Prismen und Pyramiden ist keine Teilung umgesetzt. ";
        } else {
            e << "Bei einem gemischten Netz teilen die beiden Typen ihre "
                 "gemeinsame Flaeche unterschiedlich, es entstuenden haengende "
                 "Knoten und das Feld waere dort nicht gekoppelt. ";
        }
        e << "Die Netzreihe fuer die Konvergenzstudie muss dann aus dem "
             "Netzgenerator kommen.";
    }
    throw std::runtime_error(e.str());
}

void refine_tet_mesh(Mesh& mesh, int levels) {
    if (levels <= 0) return;

    const size_t n0 = mesh.num_nodes;
    const size_t h0 = mesh.num_hexes;
    mesh.nodes_initial.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();
    const double v0 = total_tet_volume(mesh);

    std::cout << "[Refine] Uniforme Verfeinerung, " << levels
              << " Stufe(n), je Stufe 1 -> 8 Kinder (4 Ecken + Oktaeder).\n";

    size_t pred_edges = 0;
    for (int l = 1; l <= levels; ++l) {
        const size_t tets_before = mesh.num_hexes;
        size_t old_e = 0, old_f = 0;
        refine_tet_once(mesh, &old_e, &old_f);
        // Kanten danach: jede alte Kante halbiert (2), dazu je altem Element
        // die Kanten des inneren Oktaeders. Der Oktaeder hat 12 Kanten plus
        // die gewaehlte Diagonale, davon liegen 4 je Flaeche auf geteilten
        // Flaechen -- die Zahl ist deshalb nur eine obere Schranke und wird
        // als solche gekennzeichnet.
        pred_edges = 2 * old_e + 13 * tets_before;
        std::cout << "[Refine]   Stufe " << l << ": " << mesh.num_hexes
                  << " Elemente, " << mesh.num_nodes << " Knoten"
                  << "   (Ausgangsnetz hatte " << old_e << " Kanten)"
                  << std::endl;
    }

    const double v1 = total_tet_volume(mesh);
    const double rel = (v0 != 0.0) ? std::abs(v1 - v0) / std::abs(v0) : 0.0;
    std::cout << "[Refine] Elemente " << h0 << " -> " << mesh.num_hexes
              << " (Faktor " << mesh.num_hexes / static_cast<double>(h0) << "), "
              << "Knoten " << n0 << " -> " << mesh.num_nodes << "\n";
    std::cout << "[Refine] Gesamtvolumen " << v0 << " -> " << v1
              << " m^3, relative Aenderung " << rel << "\n";

    // Beim Tetraeder ist das eine echte Pruefung, nicht nur eine Anzeige: die
    // Abbildung ist AFFIN, die acht Kinder fuellen das Elternelement also
    // EXAKT aus. Eine Aenderung oberhalb der Rundung bedeutet, dass ein Kind
    // falsch zusammengesetzt ist -- anders als beim Hexaeder, wo die
    // trilineare Abbildung stueckweise angenaehert wird und eine kleine
    // Aenderung normal ist.
    if (rel > 1e-10) {
        std::cout << "[Refine] WARNUNG: der Tetraeder wird AFFIN abgebildet, die "
                     "acht Kinder muessen ihn exakt ausfuellen. Eine relative "
                     "Volumenaenderung von " << rel << " ist zu gross fuer "
                     "Rundung und deutet auf eine falsch zusammengesetzte "
                     "Unterteilung hin.\n";
    } else {
        std::cout << "[Refine] Volumen exakt erhalten (affine Abbildung) -- die "
                     "Unterteilung ist konsistent.\n";
    }

    std::cout << "[Refine] Freiheitsgrade danach:  Kantenelemente hoechstens "
              << pred_edges << " (obere Schranke, geteilte Flaechenkanten sind "
                 "doppelt gezaehlt),  Knotenelemente " << 3 * mesh.num_nodes << "\n";
    std::cout << "[Refine] Die GEOMETRIE bleibt unveraendert -- Kantenmitten liegen "
                 "auf ihren Kanten, der polyedrische Rand wird nicht runder. "
                 "Verfeinert wird die Aufloesung des Feldes, nicht die der Form. "
                 "Der Abschneidefaktor des Luftkastens aendert sich dadurch NICHT.\n";
}

void refine_hex_mesh(Mesh& mesh, int levels) {
    if (levels <= 0) return;

    const size_t n0 = mesh.num_nodes;
    const size_t h0 = mesh.num_hexes;
    const double v0 = total_volume(mesh);

    std::cout << "[Refine] Uniforme Verfeinerung, " << levels
              << " Stufe(n), je Stufe 1 -> 8 Kinder (2x2x2).\n";

    size_t pred_edges = 0;
    for (int l = 1; l <= levels; ++l) {
        const size_t hexes_before = mesh.num_hexes;
        size_t old_e = 0, old_f = 0;
        refine_once(mesh, &old_e, &old_f);
        // Kanten des verfeinerten Netzes: jede alte Kante halbiert (2), je alte
        // Flaeche vier Speichen zur Flaechenmitte, je alte Zelle sechs zur
        // Zellmitte.
        pred_edges = 2 * old_e + 4 * old_f + 6 * hexes_before;
        std::cout << "[Refine]   Stufe " << l << ": " << mesh.num_hexes
                  << " Elemente, " << mesh.num_nodes << " Knoten"
                  << "   (Ausgangsnetz hatte " << old_e << " Kanten, "
                  << old_f << " Flaechen)\n";
    }

    const double v1 = total_volume(mesh);
    const double rel = (v0 != 0.0) ? std::abs(v1 - v0) / std::abs(v0) : 0.0;
    std::cout << "[Refine] Elemente " << h0 << " -> " << mesh.num_hexes
              << " (Faktor " << mesh.num_hexes / static_cast<double>(h0) << "), "
              << "Knoten " << n0 << " -> " << mesh.num_nodes << "\n";
    std::cout << "[Refine] Gesamtvolumen " << v0 << " -> " << v1
              << " m^3, relative Aenderung " << rel << "\n";
    if (rel > 1e-3) {
        std::cout << "[Refine] Hinweis: die Volumenaenderung kommt daher, dass die "
                     "Kinder die trilineare Abbildung des Elternelements stueckweise "
                     "annaehern. Bei stark verzerrten Elementen ist das normal; bei "
                     "affinen Elementen waere sie exakt null.\n";
    }
    // Was die beiden Loeserpfade daraus machen -- vor dem teuren Setup sichtbar.
    std::cout << "[Refine] Freiheitsgrade danach:  Kantenelemente (produktiv) "
              << pred_edges << ",  Knotenelemente (--reference) "
              << 3 * mesh.num_nodes << "\n";

    std::cout << "[Refine] Die GEOMETRIE bleibt unveraendert -- Kantenmitten liegen "
                 "auf ihren Kanten, der polyedrische Rand wird nicht runder. "
                 "Verfeinert wird die Auflaesung des Feldes, nicht die der Form.\n";
}
