#include "TopologyExtractor.h"
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <stdexcept>

void TopologyExtractor::extract_edges(Mesh& mesh) {
    if (mesh.num_hexes == 0) {
        throw std::runtime_error("TopologyExtractor Error: Mesh contains no hexahedrons.");
    }

    // -------------------------------------------------------------------------
    // Lokale Definition der 12 Kanten eines Hexaeders nach VTK-Standard
    // Die Kante 'e' (0 bis 11) geht von Knoten 'vA' nach 'vB'.
    // -------------------------------------------------------------------------
    // Die Tabellen stehen jetzt in Mesh.cuh (element_local_edge_nodes), weil
    // sie je Elementtyp verschieden sind und mit den Kantenelementen
    // uebereinstimmen MUESSEN. Fuer ein reines Hexaedernetz liefert
    // element_local_edge_nodes(ELEM_HEX, ...) genau die frueher hier stehende
    // Tabelle, das Verhalten ist also unveraendert.

    // Allokation der Mapping-Arrays auf dem Host (Unified Memory)
    mesh.hex_edges.allocate(12 * mesh.num_hexes);
    mesh.hex_edge_signs.allocate(12 * mesh.num_hexes);

    // Prefetching für schnellen Schreibzugriff durch die CPU
    mesh.hex_edges.prefetch_to_host();
    mesh.hex_edge_signs.prefetch_to_host();

    // Hash-Map und Vektor für globale eindeutige Kanten
    std::unordered_map<uint64_t, int> edge_map;
    std::vector<int2> unique_edges;

    // Empirischer Schätzwert für Hexaeder-Netze (ca. 3 Kanten pro Hex) zur Vermeidung von Reallokationen
    unique_edges.reserve(mesh.num_hexes * 3);

    // Iteration über alle Hexaeder im Netz
    for (size_t hex_id = 0; hex_id < mesh.num_hexes; ++hex_id) {

        Hexahedron current_hex = mesh.hex_nodes[hex_id];
        const int type    = mesh.elem_type(hex_id);
        const int n_edges = element_num_edges(type);

        // Ungenutzte Plaetze VORAB auf -1 setzen. Ein Tetraeder belegt nur 6
        // der 12 Plaetze; bliebe der Rest uninitialisiert, stuende dort der
        // Inhalt von cudaMallocManaged -- also Muell, den die Assemblierung
        // als gueltige Kantennummer lesen wuerde.
        for (int local_edge = n_edges; local_edge < 12; ++local_edge) {
            mesh.hex_edges[hex_id * 12 + local_edge] = -1;
            mesh.hex_edge_signs[hex_id * 12 + local_edge] = 0;
        }

        for (int local_edge = 0; local_edge < n_edges; ++local_edge) {

            int a, b;
            element_local_edge_nodes(type, local_edge, a, b);

            // Lokale Orientierung (Richtung) der Kante innerhalb des Elements
            int vA = current_hex.v[a];
            int vB = current_hex.v[b];

            // Globale (kanonische) Orientierung: immer von kleiner zu großer ID
            int v_min = std::min(vA, vB);
            int v_max = std::max(vA, vB);

            if (v_min == v_max) {
                throw std::runtime_error(
                    "TopologyExtractor Error: Degenerated edge found in element " + std::to_string(hex_id)
                );
            }

            // 64-Bit Perfect Hash generieren
            uint64_t edge_key = (static_cast<uint64_t>(v_max) << 32) | static_cast<uint32_t>(v_min);

            // Suchen oder Einfügen in die Hash-Map
            int global_edge_id;
            auto it = edge_map.find(edge_key);

            if (it == edge_map.end()) {
                // Kante existiert noch nicht: Neu anlegen
                global_edge_id = static_cast<int>(unique_edges.size());
                edge_map[edge_key] = global_edge_id;
                unique_edges.push_back(make_int2(v_min, v_max));
            }
            else {
                // Kante wurde bereits von einem benachbarten Hexaeder registriert
                global_edge_id = it->second;
            }

            // -----------------------------------------------------------------
            // Aufgabe 2.2: Speichern des Mappings und der relativen Orientierung
            // -----------------------------------------------------------------

            // Der flache 1D-Index für das 2D-Konstrukt [hex_id][local_edge]
            size_t flat_index = hex_id * 12 + local_edge;

            // 1. Zuweisung der globalen Kanten-ID
            mesh.hex_edges[flat_index] = global_edge_id;

            // 2. Bestimmung des Vorzeichens (Orientation Sign)
            // Wenn die lokale Kante (vA -> vB) in die gleiche Richtung zeigt wie die 
            // kanonische Kante (v_min -> v_max), ist das Vorzeichen +1, sonst -1.
            // (Da v_min immer der kleinere ist, prüfen wir einfach, ob vA kleiner als vB ist).
            if (vA < vB) {
                mesh.hex_edge_signs[flat_index] = 1;
            }
            else {
                mesh.hex_edge_signs[flat_index] = -1;
            }
        }
    }

    // Finale Übertragung der gesammelten globalen Kanten in das Mesh-Objekt
    mesh.num_edges = unique_edges.size();
    mesh.edges.allocate(mesh.num_edges);
    mesh.edges.prefetch_to_host();

    for (size_t i = 0; i < mesh.num_edges; ++i) {
        mesh.edges[i] = unique_edges[i];
    }

    std::cout << "[TopologyExtractor] Successfully extracted "
        << mesh.num_edges << " unique edges from "
        << mesh.num_hexes << " hexahedrons." << std::endl;
    {
        size_t n_tet = 0, n_prism = 0, n_hex = 0;
        for (size_t i = 0; i < mesh.num_hexes; ++i) {
            const int t = mesh.elem_type(i);
            if (t == ELEM_TET) ++n_tet; else if (t == ELEM_PRISM) ++n_prism; else ++n_hex;
        }
        std::cout << "[TopologyExtractor] Elements: " << n_hex << " hex, "
                  << n_prism << " prism, " << n_tet << " tet." << std::endl;
    }
    std::cout << "[TopologyExtractor] Generated element-to-edge mapping (stride 12, unused slots = -1) including orientation signs." << std::endl;
}

#include <array>

// -----------------------------------------------------------------------------
// Hilfsstrukturen für das 4-Knoten Hashing von Flächen
// -----------------------------------------------------------------------------
struct QuadKey {
    std::array<int, 4> v; // Sortierte Knoten-IDs

    // Gleichheitsoperator für die Map
    bool operator==(const QuadKey& other) const {
        return v[0] == other.v[0] && v[1] == other.v[1] &&
            v[2] == other.v[2] && v[3] == other.v[3];
    }
};

// Eigene Hash-Funktion für den QuadKey (kombiniert die Hashes der 4 IDs)
struct QuadKeyHasher {
    std::size_t operator()(const QuadKey& k) const {
        std::size_t h1 = std::hash<int>()(k.v[0]);
        std::size_t h2 = std::hash<int>()(k.v[1]);
        std::size_t h3 = std::hash<int>()(k.v[2]);
        std::size_t h4 = std::hash<int>()(k.v[3]);
        // Kombinieren (Bit-Shifting Methode aus Boost::hash_combine)
        h1 ^= h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        h1 ^= h3 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        h1 ^= h4 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

// -----------------------------------------------------------------------------
// Die eigentliche Extraktions-Methode
// -----------------------------------------------------------------------------
void TopologyExtractor::extract_boundary_faces(Mesh& mesh) {
    if (mesh.num_hexes == 0 || mesh.num_edges == 0) {
        throw std::runtime_error("TopologyExtractor Error: Hexahedrons or global edges not initialized. Call extract_edges first.");
    }

    // Die Flaechentabellen stehen jetzt in Mesh.cuh
    // (element_local_face_nodes), weil sie je Elementtyp verschieden sind.
    // Fuer ELEM_HEX liefert die Funktion genau die frueher hier stehende
    // Tabelle. Ein Tetraeder hat 4 Dreiecke, ein Prisma 2 Dreiecke und
    // 3 Vierecke; bei einer Dreiecksflaeche bleibt v[3] auf -1.

    // Struktur zum Speichern der Rohtopologie eines Quads
    struct RawQuadData {
        int original_v[4];    // Die echten globalen Knoten-IDs in korrekter Reihenfolge
        int parent_hex_id;    // Die ID des (ersten) gefundenen Hexaeders
        int local_face_id;    // Die lokale ID (0-5) in diesem Hexaeder
        int appearance_count; // Wie oft das Quad gefunden wurde (sollte 1 oder 2 sein)
    };

    // Die Hash-Map ordnet den sortierten QuadKey den originalen Quad-Daten zu.
    std::unordered_map<QuadKey, RawQuadData, QuadKeyHasher> face_map;

    // -------------------------------------------------------------------------
    // Schritt 1: Alle Quads hashen und zählen (Aufgabe 2.3)
    // -------------------------------------------------------------------------
    for (size_t hex_id = 0; hex_id < mesh.num_hexes; ++hex_id) {
        Hexahedron current_hex = mesh.hex_nodes[hex_id];

        const int type    = mesh.elem_type(hex_id);
        const int n_faces = element_num_faces(type);

        for (int local_face = 0; local_face < n_faces; ++local_face) {

            int fn[4];
            const int n_fv = element_local_face_nodes(type, local_face, fn);

            // Hole die globalen Knoten-IDs in CCW-Reihenfolge. Bei einem
            // Dreieck bleibt v3 auf -1; im sortierten Schluessel steht die -1
            // dann vorn und stoert nicht, weil Dreiecke und Vierecke nie
            // denselben Schluessel haben koennen.
            int v0 = current_hex.v[fn[0]];
            int v1 = current_hex.v[fn[1]];
            int v2 = current_hex.v[fn[2]];
            int v3 = (n_fv == 4) ? current_hex.v[fn[3]] : -1;

            // Erstelle einen kanonischen (sortierten) Schlüssel für die Hash-Map
            QuadKey key = { {v0, v1, v2, v3} };
            std::sort(key.v.begin(), key.v.end());

            // Eintragen oder Zähler erhöhen
            auto it = face_map.find(key);
            if (it == face_map.end()) {
                RawQuadData data;
                data.original_v[0] = v0;
                data.original_v[1] = v1;
                data.original_v[2] = v2;
                data.original_v[3] = v3;
                data.parent_hex_id = static_cast<int>(hex_id); // Hex-ID speichern
                data.local_face_id = local_face;               // Lokale Face-ID speichern
                data.appearance_count = 1;
                face_map[key] = data;
            }
            else {
                it->second.appearance_count++;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Schritt 2: Herausfiltern der Randflächen (count == 1)
    // -------------------------------------------------------------------------
    std::vector<Mesh::QuadFace> boundary_list;
    boundary_list.reserve(face_map.size() / 3); // Grobe Schätzung

    for (const auto& pair : face_map) {
        if (pair.second.appearance_count == 1) {
            Mesh::QuadFace b_face;
            b_face.v[0] = pair.second.original_v[0];
            b_face.v[1] = pair.second.original_v[1];
            b_face.v[2] = pair.second.original_v[2];
            b_face.v[3] = pair.second.original_v[3];
            b_face.parent_hex_id = pair.second.parent_hex_id; // Übertragen
            b_face.local_face_id = pair.second.local_face_id; // Übertragen

            boundary_list.push_back(b_face);
        }
        else if (pair.second.appearance_count > 2) {
            throw std::runtime_error("TopologyExtractor Error: Degenerated mesh. Face appears more than twice.");
        }
    }

    // -------------------------------------------------------------------------
    // Schritt 2b: Kanonische Sortierung -- ZWINGEND fuer Plattformunabhaengigkeit
    // -------------------------------------------------------------------------
    // Die Schleife oben iteriert ueber eine std::unordered_map. Deren
    // Iterationsreihenfolge ist implementierungsdefiniert: libstdc++ und die
    // MSVC-STL verwenden verschiedene Bucket-Zahlen und Wachstumsstrategien und
    // liefern damit verschiedene Reihenfolgen. boundary_faces war deshalb je
    // nach Compiler eine andere Permutation derselben 4136 Flaechen.
    //
    // Das ist nicht kosmetisch. ClusterTree::build_tree partitioniert nach
    // Schwerpunkten, und bei diesem Netz gibt es massenhaft exakte
    // Gleichstaende (auf der z-Achse nur 56 verschiedene Werte bei 4136 Quads).
    // Eine andere Eingabereihenfolge ergibt daher eine andere Blattpartition,
    // eine andere ACA/dense-Aufteilung und damit einen anderen komprimierten
    // Operator -- also andere Rechenergebnisse aus identischer Quelle.
    // Nachgewiesen mit tests/test_determinism_fingerprint: H1 (Knoten-
    // koordinaten) stimmte bitgleich, H2 (Randflaechen) wich ab.
    //
    // (parent_hex_id, local_face_id) ist pro Randflaeche eindeutig, weil eine
    // Randflaeche genau ein Nachbarhexaeder hat. Der Schluessel liefert
    // ausserdem eine netzgeordnete Durchlaufreihenfolge statt einer
    // Hash-Streuung.
    std::sort(boundary_list.begin(), boundary_list.end(),
        [](const Mesh::QuadFace& a, const Mesh::QuadFace& b) {
            if (a.parent_hex_id != b.parent_hex_id) return a.parent_hex_id < b.parent_hex_id;
            return a.local_face_id < b.local_face_id;
        });

    // Speicher für BEM-Flächen im Mesh allozieren
    mesh.num_boundary_faces = boundary_list.size();
    mesh.boundary_faces.allocate(mesh.num_boundary_faces);
    mesh.boundary_faces.prefetch_to_host();

    // -------------------------------------------------------------------------
    // Schritt 3: BEM-to-FEM Edge Mapping (Aufgabe 2.4)
    // -------------------------------------------------------------------------
    // Allokation der Mapping Arrays (4 Kanten pro Quad)
    mesh.boundary_edges_to_fem_edges.allocate(4 * mesh.num_boundary_faces);
    mesh.boundary_edge_signs.allocate(4 * mesh.num_boundary_faces);

    mesh.boundary_edges_to_fem_edges.prefetch_to_host();
    mesh.boundary_edge_signs.prefetch_to_host();

    // Um eine Kante in der globalen Kanten-Liste (mesh.edges) schnell zu finden, 
    // bauen wir eine lokale Hash-Map aus den bereits extrahierten FEM-Kanten.
    std::unordered_map<uint64_t, int> fem_edge_lookup;
    fem_edge_lookup.reserve(mesh.num_edges);
    for (size_t e = 0; e < mesh.num_edges; ++e) {
        int v_min = mesh.edges[e].x;
        int v_max = mesh.edges[e].y;
        uint64_t edge_key = (static_cast<uint64_t>(v_max) << 32) | static_cast<uint32_t>(v_min);
        fem_edge_lookup[edge_key] = static_cast<int>(e);
    }

    // Iteration über alle identifizierten Rand-Quads
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i) {
        mesh.boundary_faces[i] = boundary_list[i];

        // Ein Quad hat 4 Kanten: (0->1), (1->2), (2->3), (3->0);
        // ein Dreieck 3: (0->1), (1->2), (2->0).
        const int n_fv = boundary_list[i].num_nodes();
        const int quad_edge_to_nodes[4][2] = {
            {0, 1}, {1, 2}, {2, 3}, {3, 0}
        };
        const int tri_edge_to_nodes[3][2] = {
            {0, 1}, {1, 2}, {2, 0}
        };

        // Der vierte Platz bleibt bei einem Dreieck unbelegt. Er wird auf -1
        // gesetzt, nicht auf irgendeine Kante: wer ihn liest, soll das merken.
        if (n_fv == 3) {
            mesh.boundary_edges_to_fem_edges[i * 4 + 3] = -1;
            mesh.boundary_edge_signs[i * 4 + 3] = 0;
        }

        for (int local_edge = 0; local_edge < n_fv; ++local_edge) {
            const int a = (n_fv == 3) ? tri_edge_to_nodes[local_edge][0]
                                      : quad_edge_to_nodes[local_edge][0];
            const int b = (n_fv == 3) ? tri_edge_to_nodes[local_edge][1]
                                      : quad_edge_to_nodes[local_edge][1];
            int vA = boundary_list[i].v[a];
            int vB = boundary_list[i].v[b];

            // Wir bringen die Knoten wieder in die kanonische v_min < v_max Reihenfolge
            int v_min = std::min(vA, vB);
            int v_max = std::max(vA, vB);

            uint64_t edge_key = (static_cast<uint64_t>(v_max) << 32) | static_cast<uint32_t>(v_min);

            // Suchen der globalen FEM-Kante
            auto it = fem_edge_lookup.find(edge_key);
            if (it == fem_edge_lookup.end()) {
                throw std::runtime_error("TopologyExtractor Error: A boundary edge could not be found in the global FEM edge list. Mesh is corrupt.");
            }

            int global_fem_edge_id = it->second;

            // Flacher Array-Index: face_id * 4 + local_edge
            size_t flat_index = i * 4 + local_edge;

            // 1. Zuweisung des BEM-to-FEM Mappings
            mesh.boundary_edges_to_fem_edges[flat_index] = global_fem_edge_id;

            // 2. Bestimmung des Vorzeichens (Richtung der BEM-Kante bzgl. FEM-Kante)
            if (vA < vB) {
                mesh.boundary_edge_signs[flat_index] = 1;
            }
            else {
                mesh.boundary_edge_signs[flat_index] = -1;
            }
        }
    }

    std::cout << "[TopologyExtractor] Successfully extracted "
        << mesh.num_boundary_faces << " boundary faces." << std::endl;
    std::cout << "[TopologyExtractor] Generated boundary-to-FEM edge mapping (stride 4, triangles leave the fourth slot at -1)." << std::endl;
}

// =================================================================================
// Materialgrenzflaeche einer einzelnen Komponente (z.B. Aluminiumplatte)
// =================================================================================
void TopologyExtractor::extract_component_surface_faces(Mesh& mesh, int component_id) {
    if (mesh.num_hexes == 0) {
        throw std::runtime_error("TopologyExtractor Error: Hexahedrons not initialized.");
    }
    if (mesh.hex_component_ids.size() == 0) {
        throw std::runtime_error("TopologyExtractor Error: hex_component_ids not initialized "
                                  "(VtkReader muss die Component IDs vor diesem Aufruf gelesen haben).");
    }

    // Dieselbe VTK-Standard-Flaechendefinition wie in extract_boundary_faces
    // (lokale Normale zeigt per Rechte-Hand-Regel aus dem jeweiligen Hexaeder
    // heraus).
    // Die Flaechentabellen stehen in Mesh.cuh (element_local_face_nodes),
    // weil sie je Elementtyp verschieden sind.

    // Anders als in extract_boundary_faces() brauchen wir hier BEIDE Seiten
    // einer inneren Flaeche (Hex-ID + lokale Face-ID je Seite), um entscheiden
    // zu koennen, welche Seite zur gesuchten Komponente gehoert -- die
    // urspruengliche RawQuadData verwirft die zweite Seite bewusst, weil
    // extract_boundary_faces() nur die aeussere (einseitige) Randflaeche
    // braucht.
    struct RawQuadDataTwoSided {
        int hex_id[2]       = { -1, -1 };
        int local_face_id[2] = { -1, -1 };
        int appearance_count = 0;
    };

    std::unordered_map<QuadKey, RawQuadDataTwoSided, QuadKeyHasher> face_map;

    mesh.hex_component_ids.prefetch_to_host();

    for (size_t hex_id = 0; hex_id < mesh.num_hexes; ++hex_id) {
        Hexahedron current_hex = mesh.hex_nodes[hex_id];

        // Typabhaengig, wie in extract_boundary_faces. Mit der festen
        // Hexaedertabelle entstehen auf einem Tetraeder Phantomflaechen aus
        // den -1-Plaetzen; deren Schluessel kollidieren, und die Zaehlung
        // schlaegt als "Face appears more than twice" an. Genau daran ist der
        // zweite Lauf von --fem-direct gescheitert.
        const int etype   = mesh.elem_type(hex_id);
        const int n_faces = element_num_faces(etype);

        for (int local_face = 0; local_face < n_faces; ++local_face) {
            int fn[4];
            const int n_fv = element_local_face_nodes(etype, local_face, fn);

            int v0 = current_hex.v[fn[0]];
            int v1 = current_hex.v[fn[1]];
            int v2 = current_hex.v[fn[2]];
            int v3 = (n_fv == 4) ? current_hex.v[fn[3]] : -1;

            QuadKey key = { {v0, v1, v2, v3} };
            std::sort(key.v.begin(), key.v.end());

            auto it = face_map.find(key);
            if (it == face_map.end()) {
                RawQuadDataTwoSided data;
                data.hex_id[0] = static_cast<int>(hex_id);
                data.local_face_id[0] = local_face;
                data.appearance_count = 1;
                face_map[key] = data;
            } else if (it->second.appearance_count == 1) {
                it->second.hex_id[1] = static_cast<int>(hex_id);
                it->second.local_face_id[1] = local_face;
                it->second.appearance_count = 2;
            } else {
                throw std::runtime_error("TopologyExtractor Error: Degenerated mesh. Face appears more than twice.");
            }
        }
    }

    // Waehle je Flaeche die Seite aus, die zur gesuchten Komponente gehoert
    // (falls genau eine Seite passt) -- die aus deren eigener
    // local_face_to_nodes-Reihenfolge abgeleitete Normale zeigt dann bereits
    // korrekt von der Komponente weg.
    std::vector<Mesh::QuadFace> surface_list;
    surface_list.reserve(mesh.num_hexes / 10); // grobe Schaetzung

    for (const auto& pair : face_map) {
        const RawQuadDataTwoSided& data = pair.second;

        int chosen_side = -1;
        if (data.appearance_count == 1) {
            if (mesh.hex_component_ids[data.hex_id[0]] == component_id) {
                chosen_side = 0;
            }
        } else { // appearance_count == 2
            bool side0_is_component = (mesh.hex_component_ids[data.hex_id[0]] == component_id);
            bool side1_is_component = (mesh.hex_component_ids[data.hex_id[1]] == component_id);
            if (side0_is_component && !side1_is_component) {
                chosen_side = 0;
            } else if (side1_is_component && !side0_is_component) {
                chosen_side = 1;
            }
            // Beide oder keine Seite gehoeren zur Komponente: keine
            // Grenzflaeche der gesuchten Komponente, ueberspringen.
        }

        if (chosen_side < 0) continue;

        int hex_id = data.hex_id[chosen_side];
        int local_face = data.local_face_id[chosen_side];
        Hexahedron hex = mesh.hex_nodes[hex_id];

        int fn[4];
        const int n_fv = element_local_face_nodes(mesh.elem_type(hex_id),
                                                  local_face, fn);

        Mesh::QuadFace q_face;
        q_face.v[0] = hex.v[fn[0]];
        q_face.v[1] = hex.v[fn[1]];
        q_face.v[2] = hex.v[fn[2]];
        q_face.v[3] = (n_fv == 4) ? hex.v[fn[3]] : -1;
        q_face.parent_hex_id = hex_id;
        q_face.local_face_id = local_face;

        surface_list.push_back(q_face);
    }

    // Kanonische Sortierung -- aus denselben Determinismus-Gruenden wie bei
    // extract_boundary_faces() (siehe dortiger Kommentar): Iterationsreihenfolge
    // von std::unordered_map ist plattformabhaengig, (parent_hex_id,
    // local_face_id) ist pro Flaeche eindeutig.
    std::sort(surface_list.begin(), surface_list.end(),
        [](const Mesh::QuadFace& a, const Mesh::QuadFace& b) {
            if (a.parent_hex_id != b.parent_hex_id) return a.parent_hex_id < b.parent_hex_id;
            return a.local_face_id < b.local_face_id;
        });

    mesh.num_component_surface_faces = surface_list.size();
    mesh.component_surface_faces.allocate(mesh.num_component_surface_faces);
    mesh.component_surface_faces.prefetch_to_host();
    for (size_t i = 0; i < mesh.num_component_surface_faces; ++i) {
        mesh.component_surface_faces[i] = surface_list[i];
    }

    std::cout << "[TopologyExtractor] Materialgrenzflaeche fuer Component ID "
              << component_id << ": " << mesh.num_component_surface_faces
              << " Flaechen extrahiert (fuer Maxwell-Kraftintegration)." << std::endl;
}
