// =============================================================================
// Abnahmepruefung: Topologie gemischter Netze (Tetraeder, Prisma, Hexaeder)
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_mixed_topology
//   ./build/test_mixed_topology
//
// Reiner Host-Code, keine GPU. Ruft den ECHTEN TopologyExtractor auf, nicht
// einen Nachbau.
//
// WAS HIER SCHIEFGEHEN KANN
// -------------------------
// Der FEM-direkt-Pfad bringt zum ersten Mal Elemente ins Netz, die nicht
// Hexaeder sind. Die Kanten- und Flaechentabellen sind je Typ verschieden,
// stehen aber an ZWEI Stellen: in Mesh.cuh (fuer die Topologie) und in den
// Nedelec-Headern (fuer die Basisfunktionen). Laufen sie auseinander, gibt es
// keinen Absturz -- es gibt eine falsche Kraft.
//
// Deshalb prueft dieser Test in dieser Reihenfolge:
//
//   1. Die Tabellen in Mesh.cuh sind identisch zu denen in
//      NedelecTetrahedron.cuh und NedelecPrism.cuh. Element fuer Element.
//   2. Ein einzelnes Element jedes Typs hat die erwartete Kantenzahl, und die
//      ungenutzten Plaetze der Schrittweite 12 stehen auf -1. Uninitialisiert
//      waeren sie Muell aus cudaMallocManaged, den die Assemblierung als
//      gueltige Kantennummer laese.
//   3. Ein Wuerfel, in 6 Tetraeder zerlegt: 8 Knoten, 19 Kanten, 12
//      Randdreiecke. Von Hand nachgerechnet.
//   4. Ein Hexaeder und ein Prisma an einer gemeinsamen Vierecksflaeche: die
//      geteilten Kanten muessen EINMAL gezaehlt werden. Das ist der Test auf
//      Konformitaet ueber die Typgrenze hinweg.
//   5. Rueckwaertskompatibilitaet: ein reines Hexaedernetz OHNE gesetztes
//      elem_types muss sich bitgleich verhalten wie vor der Aenderung.
// =============================================================================

#include "Mesh.cuh"
#include "TopologyExtractor.h"
#include "NedelecTetrahedron.cuh"
#include "NedelecPrism.cuh"

#include <cstdio>
#include <vector>
#include <set>
#include <string>

static int g_failures = 0;

static void check(const char* name, long got, long want) {
    const bool ok = (got == want);
    std::printf("  %-52s %8ld  (erwartet %8ld)  %s\n",
                name, got, want, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static void check_true(const char* name, bool ok) {
    std::printf("  %-52s %8s                      %s\n",
                name, ok ? "ja" : "nein", ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

// -----------------------------------------------------------------------------
// 1. Die Tabellen duerfen nicht auseinanderlaufen
// -----------------------------------------------------------------------------
static void test_tables_agree() {
    bool tet_ok = true;
    for (int e = 0; e < TET_NUM_EDGES; ++e) {
        int a1, b1, a2, b2;
        tet_local_edge_nodes(e, a1, b1);
        element_local_edge_nodes(ELEM_TET, e, a2, b2);
        if (a1 != a2 || b1 != b2) tet_ok = false;
    }
    check_true("Kantentabelle Tetraeder: Mesh.cuh == Nedelec", tet_ok);

    bool prism_ok = true;
    for (int e = 0; e < PRISM_NUM_EDGES; ++e) {
        int a1, b1, a2, b2;
        prism_local_edge_nodes(e, a1, b1);
        element_local_edge_nodes(ELEM_PRISM, e, a2, b2);
        if (a1 != a2 || b1 != b2) prism_ok = false;
    }
    check_true("Kantentabelle Prisma: Mesh.cuh == Nedelec", prism_ok);

    check("Kantenzahl Tetraeder", element_num_edges(ELEM_TET), TET_NUM_EDGES);
    check("Kantenzahl Prisma", element_num_edges(ELEM_PRISM), PRISM_NUM_EDGES);
    check("Kantenzahl Hexaeder", element_num_edges(ELEM_HEX), 12);
}

// -----------------------------------------------------------------------------
// Hilfe: Netz aus Knoten, Elementen und Typen bauen
// -----------------------------------------------------------------------------
static void build_mesh(Mesh& mesh,
                       const std::vector<double3>& pts,
                       const std::vector<std::vector<int>>& elems,
                       const std::vector<int>& types)
{
    mesh.num_nodes = pts.size();
    mesh.nodes_initial.allocate(pts.size());
    mesh.nodes_current.allocate(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        mesh.nodes_initial[i] = pts[i];
        mesh.nodes_current[i] = pts[i];
    }

    mesh.num_hexes = elems.size();
    mesh.hex_nodes.allocate(elems.size());
    mesh.hex_component_ids.allocate(elems.size());
    mesh.elem_types.allocate(types.size());
    for (size_t i = 0; i < elems.size(); ++i) {
        Hexahedron h;
        for (int k = 0; k < 8; ++k) h.v[k] = -1;
        for (size_t k = 0; k < elems[i].size(); ++k) h.v[k] = elems[i][k];
        mesh.hex_nodes[i] = h;
        mesh.hex_component_ids[i] = 0;
        mesh.elem_types[i] = types[i];
    }
}

static long count_unique_edges(const Mesh& mesh) {
    std::set<int> s;
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const int n = element_num_edges(mesh.elem_type(h));
        for (int e = 0; e < n; ++e) s.insert(mesh.hex_edges[h * 12 + e]);
    }
    return static_cast<long>(s.size());
}

static bool padding_is_minus_one(const Mesh& mesh) {
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const int n = element_num_edges(mesh.elem_type(h));
        for (int e = n; e < 12; ++e)
            if (mesh.hex_edges[h * 12 + e] != -1) return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// 3. Wuerfel aus 6 Tetraedern
// -----------------------------------------------------------------------------
// Die Kuhn-Zerlegung des Einheitswuerfels. Sie ist konform und benutzt keine
// zusaetzlichen Knoten. Erwartet: 8 Knoten, 12 Wuerfelkanten + 6
// Flaechendiagonalen + 1 Raumdiagonale = 19 Kanten, und 12 Randdreiecke
// (2 je Wuerfelseite).
static void test_cube_of_tets() {
    const std::vector<double3> pts = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    // Kuhn: alle Tetraeder teilen die Raumdiagonale 0-6
    const std::vector<std::vector<int>> elems = {
        {0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6},{0,5,1,6}
    };
    const std::vector<int> types(elems.size(), ELEM_TET);

    Mesh mesh;
    build_mesh(mesh, pts, elems, types);
    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);

    check("Wuerfel aus 6 Tetraedern: Kanten", static_cast<long>(mesh.num_edges), 19);
    check("  davon im Element-Mapping verwendet", count_unique_edges(mesh), 19);
    check("  Randflaechen", static_cast<long>(mesh.num_boundary_faces), 12);
    check_true("  Polsterung der Plaetze 6..11 auf -1", padding_is_minus_one(mesh));

    bool all_tri = true;
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i)
        if (mesh.boundary_faces[i].num_nodes() != 3) all_tri = false;
    check_true("  alle Randflaechen sind Dreiecke", all_tri);

    bool tri_pad = true;
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i)
        if (mesh.boundary_edges_to_fem_edges[i * 4 + 3] != -1) tri_pad = false;
    check_true("  vierte Randkante eines Dreiecks auf -1", tri_pad);

    check_true("  is_pure_hex() ist falsch", !mesh.is_pure_hex());

    // extract_component_surface_faces auf demselben Netz. Die Funktion war
    // beim ersten Lauf von --fem-direct die zweite Absturzstelle: sie benutzte
    // weiterhin die feste Hexaeder-Flaechentabelle, und die -1-Plaetze eines
    // Tetraeders ergaben Phantomflaechen mit kollidierenden Schluesseln
    // ("Face appears more than twice").
    //
    // Aufteilung: die drei Tetraeder 0..2 sind Komponente 3, der Rest ist 0.
    // Die Grenzflaeche zwischen beiden Haelften muss gefunden werden, ohne
    // dass etwas geworfen wird.
    for (size_t i = 0; i < mesh.num_hexes; ++i)
        mesh.hex_component_ids[i] = (i < 3) ? 3 : 0;

    bool threw = false;
    try {
        TopologyExtractor::extract_component_surface_faces(mesh, 3);
    } catch (const std::exception&) { threw = true; }
    check_true("  extract_component_surface_faces wirft nicht", !threw);
    check_true("  Grenzflaechen gefunden", mesh.num_component_surface_faces > 0);

    bool cs_tri = true;
    for (size_t i = 0; i < mesh.num_component_surface_faces; ++i)
        if (mesh.component_surface_faces[i].num_nodes() != 3) cs_tri = false;
    check_true("  Grenzflaechen sind Dreiecke", cs_tri);
}

// -----------------------------------------------------------------------------
// 4. Hexaeder und Prisma an gemeinsamer Vierecksflaeche
// -----------------------------------------------------------------------------
// Der Hexaeder belegt 0..7, das Prisma sitzt oben darauf und teilt sich mit ihm
// die Flaeche 4-5-6-7. Ein Prisma hat drei Vierecksflaechen; hier liegt seine
// Flaeche {0,1,4,3} des lokalen Musters auf der Hexaederdecke.
//
// Erwartet: Hexaeder 12 Kanten, Prisma 9 Kanten, davon 4 gemeinsam
// (die Kanten der geteilten Flaeche) -> 12 + 9 - 4 = 17 Kanten.
static void test_hex_prism_interface() {
    const std::vector<double3> pts = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},     // 0..3 Hexaederboden
        {0,0,1},{1,0,1},{1,1,1},{0,1,1},     // 4..7 Hexaederdecke = Prismaboden
        {0,0,2},{1,0,2}                      // 8,9  Prismadecke
    };
    // Prisma: Bodendreieck (4,5,6)? Nein -- es muss eine VIERECKSflaeche mit
    // dem Hexaeder teilen. Bodendreieck (4,5,8'), so dass die Vierecksflaeche
    // {0,1,4,3} des Prismas auf 4-5-...-... faellt. Konkret:
    //   Prisma-Knoten 0,1,2 = 4,5,6   (unteres Dreieck, liegt auf der Decke)
    //   Prisma-Knoten 3,4,5 = 7,8,9   -- das waere nicht konform.
    // Stattdessen ein Prisma, dessen VIERECKSflaeche {0,1,4,3} genau
    // 4-5-9-8 ist und das damit seitlich anschliesst, nicht oben:
    //   0=4, 1=5, 2=6, 3=8, 4=9, 5=6  -- entartet.
    //
    // Sauber und einfach: das Prisma teilt die Vierecksflaeche 4-5-6-7 NICHT,
    // sondern die Seitenflaeche des Hexaeders 1-2-6-5. Prisma:
    //   Bodendreieck  1, 2, 9      Deckdreieck  5, 6, 8
    // Damit ist {1,2,6,5} die Vierecksflaeche {0,1,4,3} des Prismas.
    const std::vector<std::vector<int>> elems = {
        {0,1,2,3,4,5,6,7},        // Hexaeder
        {1,2,9,5,6,8}             // Prisma, teilt 1-2-6-5
    };
    const std::vector<int> types = {ELEM_HEX, ELEM_PRISM};

    Mesh mesh;
    build_mesh(mesh, pts, elems, types);
    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);

    check("Hexaeder + Prisma: Kanten (12 + 9 - 4)",
          static_cast<long>(mesh.num_edges), 17);
    check_true("  Polsterung der Plaetze 9..11 auf -1", padding_is_minus_one(mesh));

    // 6 + 5 Flaechen, davon eine geteilt -> 9 Randflaechen
    check("  Randflaechen (11 - 2 geteilte Haelften)",
          static_cast<long>(mesh.num_boundary_faces), 9);

    long tri = 0, quad = 0;
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i)
        (mesh.boundary_faces[i].num_nodes() == 3 ? tri : quad)++;
    check("  davon Dreiecke", tri, 2);
    check("  davon Vierecke", quad, 7);
}

// -----------------------------------------------------------------------------
// 5. Rueckwaertskompatibilitaet
// -----------------------------------------------------------------------------
// Ein einzelner Hexaeder OHNE gesetztes elem_types. element_type_or_hex()
// muss ELEM_HEX liefern, und das Ergebnis muss dem eines ausdruecklich als
// ELEM_HEX markierten Netzes gleichen.
static void test_backwards_compatible() {
    const std::vector<double3> pts = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    const std::vector<std::vector<int>> elems = {{0,1,2,3,4,5,6,7}};

    Mesh a, b;
    build_mesh(a, pts, elems, {ELEM_HEX});

    // b bewusst OHNE elem_types
    b.num_nodes = pts.size();
    b.nodes_initial.allocate(pts.size());
    b.nodes_current.allocate(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        b.nodes_initial[i] = pts[i];
        b.nodes_current[i] = pts[i];
    }
    b.num_hexes = 1;
    b.hex_nodes.allocate(1);
    b.hex_component_ids.allocate(1);
    Hexahedron h;
    for (int k = 0; k < 8; ++k) h.v[k] = elems[0][k];
    b.hex_nodes[0] = h;
    b.hex_component_ids[0] = 0;

    TopologyExtractor::extract_edges(a);
    TopologyExtractor::extract_boundary_faces(a);
    TopologyExtractor::extract_edges(b);
    TopologyExtractor::extract_boundary_faces(b);

    check("Einzelner Hexaeder: Kanten", static_cast<long>(b.num_edges), 12);
    check("  Randflaechen", static_cast<long>(b.num_boundary_faces), 6);
    check_true("  is_pure_hex() ohne elem_types", b.is_pure_hex());

    bool identical = (a.num_edges == b.num_edges)
                  && (a.num_boundary_faces == b.num_boundary_faces);
    for (int i = 0; i < 12 && identical; ++i) {
        if (a.hex_edges[i] != b.hex_edges[i]) identical = false;
        if (a.hex_edge_signs[i] != b.hex_edge_signs[i]) identical = false;
    }
    check_true("  bitgleich zu ausdruecklichem ELEM_HEX", identical);
}

int main() {
    // Host-Test: ohne GPU lauffaehig. Ohne diese Zeile legt DeviceArray den
    // Speicher mit cudaMallocManaged an und bricht ohne Grafikkarte ab.
    bf::g_backend = bf::Backend::CPU;
    std::printf("\n== Tabellen ==\n");
    test_tables_agree();

    std::printf("\n== Wuerfel aus 6 Tetraedern ==\n");
    test_cube_of_tets();

    std::printf("\n== Hexaeder und Prisma an gemeinsamer Flaeche ==\n");
    test_hex_prism_interface();

    std::printf("\n== Rueckwaertskompatibilitaet ==\n");
    test_backwards_compatible();

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
