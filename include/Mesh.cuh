#ifndef MESH_CUH
#define MESH_CUH

#include <cuda_runtime.h>
#include "DeviceArray.cuh"

// Struktur für die Knoten-IDs eines Volumenelements.
//
// Der Name ist historisch: bis zum FEM-direkt-Pfad gab es ausschliesslich
// Hexaeder. Die acht Plaetze fassen alle vier Elementtypen erster Ordnung
// (Tetraeder 4, Pyramide 5, Prisma 6, Hexaeder 8); ungenutzte Plaetze stehen
// auf -1.
struct Hexahedron {
    int v[8];
};

// =============================================================================
// Elementtypen
// =============================================================================
// Die Werte sind absichtlich die VTK-Zelltypnummern. Damit entfaellt eine
// Uebersetzungstabelle zwischen Datei und Datenstruktur, und der Wert ist beim
// Lesen im Debugger sofort verstaendlich.
//
// RUECKWAERTSKOMPATIBILITAET: ist Mesh::elem_types nicht angelegt (Zeiger
// null), gilt jedes Element als Hexaeder. Alle bestehenden Pfade -- der
// transiente Loeser, BEM, MeshRefinement, die vorhandenen Tests -- verhalten
// sich damit bitgleich wie vorher.
#define ELEM_TET   10
#define ELEM_HEX   12
#define ELEM_PRISM 13

__host__ __device__ inline int element_num_nodes(int type) {
    if (type == ELEM_TET)   return 4;
    if (type == ELEM_PRISM) return 6;
    return 8;                                  // ELEM_HEX
}

__host__ __device__ inline int element_num_edges(int type) {
    if (type == ELEM_TET)   return 6;
    if (type == ELEM_PRISM) return 9;
    return 12;                                 // ELEM_HEX
}

__host__ __device__ inline int element_num_faces(int type) {
    if (type == ELEM_TET)   return 4;
    if (type == ELEM_PRISM) return 5;
    return 6;                                  // ELEM_HEX
}

// Zugriff auf den Typ mit der Vorgabe ELEM_HEX bei fehlendem Array. Auf dem
// Geraet wird derselbe Ausdruck ueber den rohen Zeiger benutzt.
__host__ __device__ inline int element_type_or_hex(const int* types, size_t idx) {
    return types ? types[idx] : ELEM_HEX;
}

// =============================================================================
// Lokale Topologie je Elementtyp
// =============================================================================
// EINE Quelle der Wahrheit fuer TopologyExtractor. Die Kantentabellen sind
// identisch zu denen in NedelecTetrahedron.cuh und NedelecPrism.cuh; laufen
// sie auseinander, entsteht kein Absturz, sondern eine falsche Kraft.
// tests/test_mixed_topology.cu prueft die Gleichheit.
__host__ __device__ inline void element_local_edge_nodes(
    int type, int e, int& vA, int& vB)
{
    if (type == ELEM_TET) {
        const int tbl[6][2] = {{0,1},{1,2},{2,0},{0,3},{1,3},{2,3}};
        vA = tbl[e][0]; vB = tbl[e][1]; return;
    }
    if (type == ELEM_PRISM) {
        const int tbl[9][2] = {{0,1},{1,2},{2,0},
                               {3,4},{4,5},{5,3},
                               {0,3},{1,4},{2,5}};
        vA = tbl[e][0]; vB = tbl[e][1]; return;
    }
    const int tbl[12][2] = {{0,1},{1,2},{2,3},{3,0},
                            {4,5},{5,6},{6,7},{7,4},
                            {0,4},{1,5},{2,6},{3,7}};
    vA = tbl[e][0]; vB = tbl[e][1];
}

// Lokale Flaeche f als bis zu 4 Knotenindizes. Rueckgabe ist die Knotenzahl
// (3 oder 4); bei einem Dreieck bleibt out[3] auf -1. Die Reihenfolge ist so
// gewaehlt, dass die Normale nach der Rechte-Hand-Regel aus dem Element HERAUS
// zeigt -- dieselbe Konvention wie bisher beim Hexaeder.
__host__ __device__ inline int element_local_face_nodes(int type, int f, int out[4]) {
    out[0] = out[1] = out[2] = out[3] = -1;

    if (type == ELEM_TET) {
        const int tbl[4][3] = {{0,2,1},{0,1,3},{1,2,3},{2,0,3}};
        for (int i = 0; i < 3; ++i) out[i] = tbl[f][i];
        return 3;
    }
    if (type == ELEM_PRISM) {
        if (f == 0) { out[0]=0; out[1]=2; out[2]=1; return 3; }   // Boden
        if (f == 1) { out[0]=3; out[1]=4; out[2]=5; return 3; }   // Deckel
        const int q[3][4] = {{0,1,4,3},{1,2,5,4},{2,0,3,5}};
        for (int i = 0; i < 4; ++i) out[i] = q[f-2][i];
        return 4;
    }
    const int tbl[6][4] = {{0,3,2,1},{4,5,6,7},{0,1,5,4},
                           {1,2,6,5},{2,3,7,6},{3,0,4,7}};
    for (int i = 0; i < 4; ++i) out[i] = tbl[f][i];
    return 4;
}

// Die zentrale Mesh-Klasse
struct Mesh {
    // -------------------------------------------------------------------------
    // Geometrie (Knoten)
    // -------------------------------------------------------------------------
    size_t num_nodes;
    DeviceArray<double3> nodes_initial; // Ursprungskoordinaten (t=0)
    DeviceArray<double3> nodes_current; // Aktuelle Koordinaten (t>0) für Bewegung

    // -------------------------------------------------------------------------
    // Topologie (Hexaeder)
    // -------------------------------------------------------------------------
    size_t num_hexes;
    DeviceArray<Hexahedron> hex_nodes; // 8 Knoten-IDs pro Hexaeder
    // Speichert die Physical Group / Component ID jedes Hexaeders
    DeviceArray<int> hex_component_ids;

    // Elementtyp je Element (ELEM_TET / ELEM_HEX / ELEM_PRISM). Bleibt bei
    // reinen Hexaedernetzen leer; element_type_or_hex() liefert dann ELEM_HEX.
    DeviceArray<int> elem_types;

    // Bequemer Host-Zugriff. Auf dem Geraet stattdessen
    // element_type_or_hex(elem_types.data(), i) benutzen.
    int elem_type(size_t i) const {
        return elem_types.size() ? elem_types[i] : ELEM_HEX;
    }

    // Wahr, wenn das Netz ausschliesslich aus Hexaedern besteht. Die BEM-Pfade
    // pruefen das, weil Sauter-Schwab auf Vierecks-Panels arbeitet.
    bool is_pure_hex() const {
        if (elem_types.size() == 0) return true;
        for (size_t i = 0; i < num_hexes; ++i)
            if (elem_types[i] != ELEM_HEX) return false;
        return true;
    }

    // Wahr, wenn das Netz ausschliesslich aus Tetraedern besteht.
    bool is_pure_tet() const {
        if (elem_types.size() == 0) return false;   // leer heisst Hexaeder
        if (num_hexes == 0) return false;
        for (size_t i = 0; i < num_hexes; ++i)
            if (elem_types[i] != ELEM_TET) return false;
        return true;
    }

    /// Der einheitliche Elementtyp des Netzes, oder -1 wenn es gemischt ist.
    ///
    /// Fuer Pfade, die einen Typ VORAUSSETZEN und bei einem gemischten Netz
    /// nicht still etwas anderes tun duerfen -- die uniforme Verfeinerung ist
    /// der Fall: sie teilt einen Hexaeder in 2x2x2 und einen Tetraeder nach
    /// einem ganz anderen Muster, und ein gemischtes Netz haette an den
    /// gemeinsamen Flaechen haengende Knoten.
    int uniform_elem_type() const {
        if (num_hexes == 0) return -1;
        if (elem_types.size() == 0) return ELEM_HEX;
        const int t0 = elem_types[0];
        for (size_t i = 1; i < num_hexes; ++i)
            if (elem_types[i] != t0) return -1;
        return t0;
    }

    // -------------------------------------------------------------------------
    // Topologie (Kanten / Edges)
    // -------------------------------------------------------------------------
    // = 0 statt nur im Konstruktor: der Konstruktor setzte num_nodes und
    // num_hexes, num_edges und num_boundary_faces aber NICHT. In der Praxis
    // werden sie vor dem ersten Lesen gesetzt, ein frisch gebautes Mesh
    // enthielt dort aber Stapelmuell -- und jede Pruefung der Form
    // "if (mesh.num_edges == 0) ... extract_edges() fehlt" war damit
    // wirkungslos. Aufgefallen in tests/test_fem_direct_preflight.
    size_t num_edges = 0;
    DeviceArray<int2> edges; // Globale Liste aller Kanten (v_min, v_max)

    Mesh() : num_nodes(0), num_hexes(0) {}

    // Zuweisungs- und Kopieroperatoren verbieten (Rule of Five)
    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;

    // -------------------------------------------------------------------------
    // FEM Spezifisch: Mapping Hexaeder -> globale Kanten
    // -------------------------------------------------------------------------
    // Array der Länge 12 * num_hexes.
    // Index-Berechnung: hex_id * 12 + lokaler_kanten_index
    DeviceArray<int> hex_edges;      
    
    // Array der Länge 12 * num_hexes.
    // Speichert +1, wenn lokale und globale Kante gleich orientiert sind, sonst -1.
    DeviceArray<int> hex_edge_signs;

    // -------------------------------------------------------------------------
    // BEM Spezifisch: Die Kopplungsfläche (Boundary Faces & Edges)
    // -------------------------------------------------------------------------
    struct QuadFace {
        int v[4];            // Die 4 globalen Knoten-IDs des Quads
        int parent_hex_id;   // Globale ID des Hexaeders, zu dem diese Fläche gehört
        int local_face_id;   // Lokale ID (0 bis 5) der Fläche innerhalb des Hexaeders

        // Bei einer DREIECKSFLAECHE (Tetraeder, Deck- und Bodenflaeche eines
        // Prismas) steht v[3] auf -1. Die BEM-Pfade vertragen das nicht,
        // Sauter-Schwab arbeitet auf Vierecks-Panels -- sie pruefen deshalb
        // Mesh::is_pure_hex(), bevor sie boundary_faces anfassen.
        __host__ __device__ int num_nodes() const { return v[3] < 0 ? 3 : 4; }
    };

    size_t num_boundary_faces = 0;
    DeviceArray<QuadFace> boundary_faces; // Die 4 Knoten-IDs jedes Rand-Quads

    // BEM-to-FEM Mapping:
    // Array der Länge 4 * num_boundary_faces.
    // Index-Berechnung: face_id * 4 + local_edge_index
    // Speichert für jede der 4 BEM-Quad-Kanten die globale FEM-Kanten-ID.
    DeviceArray<int> boundary_edges_to_fem_edges; 
    
    // Speichert die relative Orientierung der Kante auf dem BEM-Quad (+1 oder -1)
    DeviceArray<int> boundary_edge_signs;

    // -------------------------------------------------------------------------
    // Materialgrenzflaeche EINER Komponente (z.B. Aluminiumplatte) zu ihrer
    // Umgebung -- fuer eine korrekte Maxwell-Spannungstensor-Integration ueber
    // eine geschlossene Flaeche, die NUR den Koerper von Interesse umschliesst.
    // boundary_faces (oben) ist dafuer NICHT geeignet: das ist die aeussere
    // BEM-Kopplungsflaeche des GESAMTEN Simulationsgebiets (Flaechen mit genau
    // einem Nachbar-Hexaeder), nicht die Oberflaeche eines einzelnen Koerpers
    // im Inneren des Netzes (Flaechen zwischen zwei Hexaedern verschiedener
    // Component-ID). Siehe TopologyExtractor::extract_component_surface_faces().
    // Jede QuadFace ist so orientiert (parent_hex_id/local_face_id gehoert zur
    // Komponente selbst), dass die aus local_face_to_nodes abgeleitete Normale
    // von der Komponente WEG zeigt (siehe compute_maxwell_loads_kernel).
    size_t num_component_surface_faces = 0;
    DeviceArray<QuadFace> component_surface_faces;

};

#endif // MESH_CUH