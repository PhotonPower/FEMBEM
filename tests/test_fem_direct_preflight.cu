// =============================================================================
// Abnahmepruefung: Vorbedingungen des FEM-direkt-Pfades (AP8)
//
//   cmake --build build --config Release --target test_fem_direct_preflight
//   ./build/test_fem_direct_preflight
//
// Reiner Host-Code, keine GPU.
//
// WORUM ES GEHT
// -------------
// Der FEM-direkt-Pfad hat Vorbedingungen, deren Verletzung KEINEN Fehler
// erzeugt, sondern ein falsches Ergebnis. Die wichtigste, und der Grund fuer
// dieses Modul:
//
//   Ohne mitvernetztes Luftgebiet ist mesh.boundary_faces die Oberflaeche der
//   LEITER, nicht die Aussenwand eines Luftkastens. A x n = 0 dort heisst "das
//   Feld endet an der Aluminiumplatte". Das Ergebnis ist eine saubere,
//   konvergierte, falsche Loesung -- die schlimmste Art Fehler.
//
// Pruefung 2 ist deshalb die eigentliche Rechtfertigung des Moduls: ein Netz
// ohne Luft MUSS abgelehnt werden.
//
// Die uebrigen Faelle sind die, bei denen gerechnet werden darf, aber mit
// Vorbehalt -- knapper Luftkasten, sehr grobes Netz -- und die deshalb als
// WARNUNG erscheinen und nicht als Abbruch. Die Unterscheidung ist wichtig:
// wer bei jeder Kleinigkeit abbricht, wird uebergangen.
// =============================================================================

#include "FemDirectRun.h"
#include "FemDirectBC.h"
#include "TopologyExtractor.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

static void check_true(const char* name, bool ok) {
    std::printf("  %-56s %-6s  %s\n", name, ok ? "ja" : "nein",
                ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static void check_int(const char* name, long got, long want) {
    const bool ok = (got == want);
    std::printf("  %-56s %6ld  (erwartet %ld)  %s\n", name, got, want,
                ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static bool mentions(const std::vector<std::string>& v, const char* needle) {
    for (const auto& s : v)
        if (s.find(needle) != std::string::npos) return true;
    return false;
}

// -----------------------------------------------------------------------------
// Netzbau: ein Wuerfelgitter aus Tetraedern, aussen Luft, innen Leiter
// -----------------------------------------------------------------------------
// nx Zellen je Richtung, Kantenlaenge h. Die innersten Zellen (Abstand >= 1
// vom Rand, wenn 'shell' gilt) sind Leiter, der Rest Luft.
static void build_cube_grid(Mesh& mesh, int nx, double h,
                            bool with_air, int conductor_margin)
{
    const int np = nx + 1;
    std::vector<double3> pts;
    pts.reserve(static_cast<size_t>(np) * np * np);
    auto nid = [&](int i, int j, int k) { return (k * np + j) * np + i; };
    for (int k = 0; k < np; ++k)
        for (int j = 0; j < np; ++j)
            for (int i = 0; i < np; ++i)
                pts.push_back(make_double3(i * h, j * h, k * h));

    struct Elem { int v[4]; int cid; };
    std::vector<Elem> elems;

    for (int k = 0; k < nx; ++k)
    for (int j = 0; j < nx; ++j)
    for (int i = 0; i < nx; ++i) {
        const int c[8] = {
            nid(i,j,k), nid(i+1,j,k), nid(i+1,j+1,k), nid(i,j+1,k),
            nid(i,j,k+1), nid(i+1,j,k+1), nid(i+1,j+1,k+1), nid(i,j+1,k+1)
        };
        // Kuhn-Zerlegung ueber die Raumdiagonale 0-6: konform, ohne neue Knoten
        const int kuhn[6][4] = {
            {0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6},{0,5,1,6}
        };
        const bool inner = (i >= conductor_margin && i < nx - conductor_margin &&
                            j >= conductor_margin && j < nx - conductor_margin &&
                            k >= conductor_margin && k < nx - conductor_margin);
        // with_air = false: alles Leiter (component_id 3), also kein Luftgebiet
        const int cid = with_air ? (inner ? 3 : 0) : 3;
        for (int t = 0; t < 6; ++t) {
            Elem e;
            for (int v = 0; v < 4; ++v) e.v[v] = c[kuhn[t][v]];
            e.cid = cid;
            elems.push_back(e);
        }
    }

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
    mesh.elem_types.allocate(elems.size());
    for (size_t i = 0; i < elems.size(); ++i) {
        Hexahedron hh;
        for (int v = 0; v < 8; ++v) hh.v[v] = -1;
        for (int v = 0; v < 4; ++v) hh.v[v] = elems[i].v[v];
        mesh.hex_nodes[i] = hh;
        mesh.hex_component_ids[i] = elems[i].cid;
        mesh.elem_types[i] = ELEM_TET;
    }

    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);
}

static std::vector<MaterialProperties> make_materials(const Mesh& mesh,
                                                      bool regularized)
{
    const double sigma_al = 3.526e7;
    std::vector<MaterialProperties> m(mesh.num_hexes);
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const bool air = (mesh.hex_component_ids[h] == 0);
        m[h].kappa = air ? (regularized ? 1e-8 * sigma_al : 0.0) : sigma_al;
        m[h].mu_relative = 1.0;
    }
    return m;
}

// -----------------------------------------------------------------------------
// 1. Der gute Fall darf nicht beanstandet werden
// -----------------------------------------------------------------------------
static void test_good_case() {
    Mesh mesh;
    build_cube_grid(mesh, 6, 1.0, true, 2);   // Leiter 2x2x2 in der Mitte
    const auto mat = make_materials(mesh, true);

    // Leiter spannen 2 Zellen, Luft 6 -> Faktor 3
    const auto rep = fem_direct::preflight_fem_direct(mesh, mat.data(), 0, 2.0);
    std::printf("%s", rep.to_string().c_str());

    check_true("guter Fall: keine Beanstandung", rep.ok());
    check_true("  keine Warnung", rep.warnings.empty());
    check_true("  Luftelemente gezaehlt", rep.num_air_elements > 0);
    check_true("  Leiterelemente gezaehlt", rep.num_conductor_elements > 0);
    check_true("  Abschneidefaktor 3 (Luft 6 Zellen, Leiter 2)",
               rep.truncation_factor_min > 2.9 && rep.truncation_factor_min < 3.1);
    check_int("  Blocksystemzeilen = 2 * Kanten",
              static_cast<long>(rep.block_system_rows),
              2 * static_cast<long>(mesh.num_edges));
    check_true("  freie Freiheitsgrade plus Rand = alle Kanten",
               rep.num_free_dofs + rep.num_dirichlet_edges == mesh.num_edges);
}

// -----------------------------------------------------------------------------
// 2. DER Fall: kein Luftgebiet
// -----------------------------------------------------------------------------
static void test_no_air_is_rejected() {
    Mesh mesh;
    build_cube_grid(mesh, 4, 1.0, false, 0);   // alles Leiter
    const auto mat = make_materials(mesh, true);

    const auto rep = fem_direct::preflight_fem_direct(mesh, mat.data(), 0, 2.0);

    check_true("kein Luftgebiet: abgelehnt", !rep.ok());
    check_true("  Meldung nennt die Leiteroberflaeche",
               mentions(rep.problems, "LEITEROBERFLAECHE"));
    check_true("  Meldung verweist auf den BEM-FEM-Pfad",
               mentions(rep.problems, "BEM-FEM"));
    check_int("  Luftelemente", static_cast<long>(rep.num_air_elements), 0);
}

// -----------------------------------------------------------------------------
// 3. Fehlende Regularisierung
// -----------------------------------------------------------------------------
static void test_missing_regularization() {
    Mesh mesh;
    build_cube_grid(mesh, 6, 1.0, true, 2);
    const auto mat = make_materials(mesh, false);   // Luft mit kappa = 0

    const auto rep = fem_direct::preflight_fem_direct(mesh, mat.data(), 0, 2.0);
    check_true("kappa = 0 in der Luft: abgelehnt", !rep.ok());
    check_true("  Meldung nennt regularize_nonconducting_regions",
               mentions(rep.problems, "regularize_nonconducting_regions"));

    // Nach der Regularisierung muss derselbe Fall durchgehen
    auto fixed = mat;
    fem_direct::regularize_nonconducting_regions(fixed.data(), mesh, 3.526e7, 1e-8);
    const auto rep2 = fem_direct::preflight_fem_direct(mesh, fixed.data(), 0, 2.0);
    check_true("  nach der Regularisierung sauber", rep2.ok());
}

// -----------------------------------------------------------------------------
// 4. Knapper Luftkasten: Warnung, kein Abbruch
// -----------------------------------------------------------------------------
static void test_tight_air_box_warns() {
    Mesh mesh;
    build_cube_grid(mesh, 6, 1.0, true, 1);   // Leiter 4 Zellen, Luft 6 -> 1.5
    const auto mat = make_materials(mesh, true);

    const auto rep = fem_direct::preflight_fem_direct(mesh, mat.data(), 0, 2.0);
    check_true("knapper Kasten: darf gerechnet werden", rep.ok());
    check_true("  aber mit Warnung", !rep.warnings.empty());
    check_true("  Warnung nennt den Abschneidefaktor",
               mentions(rep.warnings, "Abschneidefaktor"));
    check_true("  Faktor liegt bei 1.5",
               rep.truncation_factor_min > 1.4 && rep.truncation_factor_min < 1.6);
}

// -----------------------------------------------------------------------------
// 5. Netz ohne Topologie und leeres Netz
// -----------------------------------------------------------------------------
static void test_guards() {
    {
        Mesh mesh;
        const auto rep = fem_direct::preflight_fem_direct(mesh, nullptr);
        check_true("leeres Netz: abgelehnt", !rep.ok());
    }
    {
        // Netz ohne extract_edges
        Mesh mesh;
        mesh.num_nodes = 4;
        mesh.nodes_initial.allocate(4);
        mesh.nodes_current.allocate(4);
        const double3 p[4] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
        for (int i = 0; i < 4; ++i) { mesh.nodes_initial[i] = p[i]; mesh.nodes_current[i] = p[i]; }
        mesh.num_hexes = 1;
        mesh.hex_nodes.allocate(1);
        mesh.hex_component_ids.allocate(1);
        mesh.elem_types.allocate(1);
        Hexahedron h; for (int v = 0; v < 8; ++v) h.v[v] = -1;
        for (int v = 0; v < 4; ++v) h.v[v] = v;
        mesh.hex_nodes[0] = h;
        mesh.hex_component_ids[0] = 0;
        mesh.elem_types[0] = ELEM_TET;

        const auto rep = fem_direct::preflight_fem_direct(mesh, nullptr);
        check_true("Netz ohne extract_edges: abgelehnt", !rep.ok());
        check_true("  Meldung nennt extract_edges",
                   mentions(rep.problems, "extract_edges"));
    }
    {
        // Materialien nicht uebergeben -> Warnung, aber kein Abbruch
        Mesh mesh;
        build_cube_grid(mesh, 6, 1.0, true, 2);
        const auto rep = fem_direct::preflight_fem_direct(mesh, nullptr, 0, 2.0);
        check_true("ohne Materialien: rechenbar", rep.ok());
        check_true("  aber mit Warnung", mentions(rep.warnings, "Regularisierung"));
    }
}

int main() {
    // Host-Test: ohne GPU lauffaehig. Ohne diese Zeile legt DeviceArray den
    // Speicher mit cudaMallocManaged an und bricht ohne Grafikkarte ab.
    bf::g_backend = bf::Backend::CPU;
    std::printf("\n== Guter Fall ==\n");
    test_good_case();

    std::printf("\n== Kein Luftgebiet (der stille Fehler) ==\n");
    test_no_air_is_rejected();

    std::printf("\n== Fehlende Regularisierung ==\n");
    test_missing_regularization();

    std::printf("\n== Knapper Luftkasten ==\n");
    test_tight_air_box_warns();

    std::printf("\n== Wachtposten ==\n");
    test_guards();

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
