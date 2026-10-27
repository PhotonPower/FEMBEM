// =============================================================================
// Abnahmepruefung: Dirichlet-Rand und Regularisierung (AP4)
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_dirichlet_bc
//   ./build/test_dirichlet_bc
//
// Reiner Host-Code, keine GPU.
//
// WORUM ES GEHT
// -------------
// Der iterative Pfad behandelt Eichung und Randbedingung als Projektion
// P*A*P + (I-P), angewendet im Operator. cuDSS bekommt aber eine MATRIX, keine
// apply()-Funktion. Beides muss also in die CSR-Werte hinein.
//
// Das Einarbeiten einer Randbedingung durch Nullen von Zeile und Spalte ist
// die Art Umformung, die "offensichtlich richtig" aussieht und es dann nicht
// ist -- typische Fehler sind eine vergessene Spalte (Matrix wird
// unsymmetrisch), ein vergessener Korrekturterm auf der rechten Seite (nur bei
// inhomogenen Werten), oder ein Diagonalwert, der die Skalierung zerstoert.
//
// DER ENTSCHEIDENDE TEST
// ----------------------
// Pruefung 4 loest zweimal: einmal das volle System nach dem Einarbeiten,
// einmal das auf die freien Freiheitsgrade reduzierte System. Beide muessen
// auf den freien Zeilen dasselbe liefern, und auf den festgehaltenen genau
// null. Das ist der Nachweis, dass die Umformung aequivalent ist, und nicht
// nur, dass sie plausibel aussieht.
//
// HAND NACHGERECHNET
// ------------------
// Pruefung 5 nimmt den Kuhn-Wuerfel aus 6 Tetraedern. Er hat 19 Kanten: 12
// Wuerfelkanten, 6 Flaechendiagonalen, 1 Raumdiagonale. Alle 18 erstgenannten
// liegen auf der Oberflaeche, nur die Raumdiagonale 0-6 liegt innen. Erwartet
// sind also 18 festgehaltene und genau 1 freier Freiheitsgrad.
// =============================================================================

#include "FemDirectBC.h"
#include "TopologyExtractor.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

static int g_failures = 0;

static void check(const char* name, long got, long want) {
    const bool ok = (got == want);
    std::printf("  %-52s %10ld  (erwartet %8ld)  %s\n",
                name, got, want, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static void check_num(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  %-52s %10.3e  (Grenze %8.1e)  %s\n",
                name, err, tol, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static void check_true(const char* name, bool ok) {
    std::printf("  %-52s %10s                      %s\n",
                name, ok ? "ja" : "nein", ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

// -----------------------------------------------------------------------------
// Kleine dichte Hilfen fuer die Aequivalenzpruefung
// -----------------------------------------------------------------------------
struct Dense {
    int n;
    std::vector<double> a;
    Dense(int n_) : n(n_), a(static_cast<size_t>(n_) * n_, 0.0) {}
    double& operator()(int i, int j) { return a[static_cast<size_t>(i) * n + j]; }
    double  operator()(int i, int j) const { return a[static_cast<size_t>(i) * n + j]; }
};

// Gauss mit Spaltenpivotisierung. Nur fuer die winzigen Testsysteme hier.
static bool dense_solve(Dense m, std::vector<double> b, std::vector<double>& x) {
    const int n = m.n;
    std::vector<int> perm(n);
    for (int i = 0; i < n; ++i) perm[i] = i;

    for (int k = 0; k < n; ++k) {
        int piv = k;
        for (int i = k + 1; i < n; ++i)
            if (std::fabs(m(i, k)) > std::fabs(m(piv, k))) piv = i;
        if (std::fabs(m(piv, k)) < 1e-300) return false;
        if (piv != k) {
            for (int j = 0; j < n; ++j) std::swap(m(k, j), m(piv, j));
            std::swap(b[k], b[piv]);
        }
        for (int i = k + 1; i < n; ++i) {
            const double f = m(i, k) / m(k, k);
            if (f == 0.0) continue;
            for (int j = k; j < n; ++j) m(i, j) -= f * m(k, j);
            b[i] -= f * b[k];
        }
    }
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double s = b[i];
        for (int j = i + 1; j < n; ++j) s -= m(i, j) * x[j];
        x[i] = s / m(i, i);
    }
    return true;
}

// -----------------------------------------------------------------------------
// Ein kleines symmetrisch positiv definites CSR-System von Hand
// -----------------------------------------------------------------------------
// Bandmatrix, 6x6: Diagonale 4..9, Nebendiagonalen -1, dazu ein Eintrag
// (0,5)/(5,0) damit das Muster nicht rein tridiagonal ist.
struct TestCsr {
    int n = 6;
    std::vector<int> row_offsets, col_indices;
    std::vector<double> values;

    TestCsr() {
        Dense d(6);
        for (int i = 0; i < 6; ++i) d(i, i) = 4.0 + i;
        for (int i = 0; i + 1 < 6; ++i) { d(i, i + 1) = -1.0; d(i + 1, i) = -1.0; }
        d(0, 5) = -0.5; d(5, 0) = -0.5;

        row_offsets.push_back(0);
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (d(i, j) == 0.0) continue;
                col_indices.push_back(j);
                values.push_back(d(i, j));
            }
            row_offsets.push_back(static_cast<int>(col_indices.size()));
        }
    }

    Dense to_dense() const {
        Dense d(n);
        for (int r = 0; r < n; ++r)
            for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
                d(r, col_indices[k]) = values[k];
        return d;
    }
};

// -----------------------------------------------------------------------------
// 1 bis 4: Struktur, Symmetrie, Skalierung, Aequivalenz
// -----------------------------------------------------------------------------
static void test_elimination() {
    TestCsr csr;
    const Dense before = csr.to_dense();

    // Freiheitsgrade 1 und 4 festhalten
    std::vector<char> fixed(6, 0);
    fixed[1] = 1;
    fixed[4] = 1;

    std::vector<double> rhs = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const std::vector<double> rhs_before = rhs;

    const auto rep = fem_direct::apply_homogeneous_dirichlet(
        csr.n, csr.row_offsets.data(), csr.col_indices.data(),
        csr.values.data(), rhs.data(), fixed);

    check("festgehaltene Freiheitsgrade", static_cast<long>(rep.num_constrained), 2);
    check("freie Freiheitsgrade", static_cast<long>(rep.num_free), 4);

    // Diagonalwert automatisch = Mittel der freien Diagonale: (4+6+7+9)/4 = 6.5
    check_num("Diagonalwert automatisch = 6.5",
              std::fabs(rep.diag_value - 6.5), 1e-14);

    const Dense after = csr.to_dense();

    // Symmetrie
    double asym = 0.0;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            asym = std::fmax(asym, std::fabs(after(i, j) - after(j, i)));
    check_num("Symmetrie erhalten", asym, 0.0);

    // Festgehaltene Zeilen: nur die Diagonale
    bool rows_clean = true, cols_clean = true;
    for (int i = 0; i < 6; ++i) {
        if (!fixed[i]) continue;
        for (int j = 0; j < 6; ++j) {
            if (i == j) { if (after(i, j) != rep.diag_value) rows_clean = false; }
            else        { if (after(i, j) != 0.0) rows_clean = false; }
        }
        for (int r = 0; r < 6; ++r)
            if (r != i && after(r, i) != 0.0) cols_clean = false;
    }
    check_true("festgehaltene Zeilen nur Diagonale", rows_clean);
    check_true("festgehaltene Spalten genullt", cols_clean);

    // Freier Block unberuehrt, und zwar bitgleich
    bool free_block_identical = true;
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            if (!fixed[i] && !fixed[j] && before(i, j) != after(i, j))
                free_block_identical = false;
    check_true("freier Block bitgleich unberuehrt", free_block_identical);

    // Rechte Seite: festgehaltene null, freie unveraendert
    bool rhs_ok = true;
    for (int i = 0; i < 6; ++i) {
        if (fixed[i]) { if (rhs[i] != 0.0) rhs_ok = false; }
        else          { if (rhs[i] != rhs_before[i]) rhs_ok = false; }
    }
    check_true("rechte Seite korrekt", rhs_ok);

    // -------------------------------------------------------------------------
    // Der Aequivalenznachweis
    // -------------------------------------------------------------------------
    std::vector<double> x_full;
    check_true("volles System loesbar", dense_solve(after, rhs, x_full));

    // Reduziertes System nur ueber die freien Freiheitsgrade
    std::vector<int> free_idx;
    for (int i = 0; i < 6; ++i) if (!fixed[i]) free_idx.push_back(i);
    const int nf = static_cast<int>(free_idx.size());
    Dense red(nf);
    std::vector<double> bred(nf);
    for (int i = 0; i < nf; ++i) {
        bred[i] = rhs_before[free_idx[i]];
        for (int j = 0; j < nf; ++j)
            red(i, j) = before(free_idx[i], free_idx[j]);
    }
    std::vector<double> x_red;
    check_true("reduziertes System loesbar", dense_solve(red, bred, x_red));

    double diff = 0.0;
    for (int i = 0; i < nf; ++i)
        diff = std::fmax(diff, std::fabs(x_full[free_idx[i]] - x_red[i]));
    check_num("Loesung gleich der des reduzierten Systems", diff, 1e-13);

    double fixed_val = 0.0;
    for (int i = 0; i < 6; ++i)
        if (fixed[i]) fixed_val = std::fmax(fixed_val, std::fabs(x_full[i]));
    check_num("festgehaltene Loesungswerte genau null", fixed_val, 0.0);

    // Ausdruecklicher Diagonalwert
    TestCsr csr2;
    std::vector<double> rhs2(6, 1.0);
    const auto rep2 = fem_direct::apply_homogeneous_dirichlet(
        csr2.n, csr2.row_offsets.data(), csr2.col_indices.data(),
        csr2.values.data(), rhs2.data(), fixed, 12345.0);
    check_num("ausdruecklicher Diagonalwert wird uebernommen",
              std::fabs(rep2.diag_value - 12345.0), 0.0);
}

// -----------------------------------------------------------------------------
// 5: Kuhn-Wuerfel, von Hand nachgerechnet
// -----------------------------------------------------------------------------
static void test_cube_boundary() {
    const std::vector<double3> pts = {
        {0,0,0},{1,0,0},{1,1,0},{0,1,0},
        {0,0,1},{1,0,1},{1,1,1},{0,1,1}
    };
    const std::vector<std::vector<int>> elems = {
        {0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6},{0,5,1,6}
    };

    Mesh mesh;
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
        Hexahedron h;
        for (int k = 0; k < 8; ++k) h.v[k] = -1;
        for (size_t k = 0; k < elems[i].size(); ++k) h.v[k] = elems[i][k];
        mesh.hex_nodes[i] = h;
        mesh.hex_component_ids[i] = 0;
        mesh.elem_types[i] = ELEM_TET;
    }

    TopologyExtractor::extract_edges(mesh);
    TopologyExtractor::extract_boundary_faces(mesh);

    const auto fixed = fem_direct::collect_dirichlet_edges(mesh);

    long n_fixed = 0;
    for (char c : fixed) n_fixed += (c != 0);
    check("Kuhn-Wuerfel: Kanten", static_cast<long>(mesh.num_edges), 19);
    check("  festgehalten (alles ausser der Raumdiagonale)", n_fixed, 18);
    check("  frei", static_cast<long>(mesh.num_edges) - n_fixed, 1);

    // Die einzige freie Kante MUSS die Raumdiagonale 0-6 sein
    int free_edge = -1;
    for (size_t e = 0; e < fixed.size(); ++e) if (!fixed[e]) free_edge = static_cast<int>(e);
    bool is_diagonal = false;
    if (free_edge >= 0) {
        const int a = mesh.edges[free_edge].x, b = mesh.edges[free_edge].y;
        is_diagonal = (a == 0 && b == 6);
    }
    check_true("  freie Kante ist die Raumdiagonale 0-6", is_diagonal);
}

// -----------------------------------------------------------------------------
// 6: Regularisierung
// -----------------------------------------------------------------------------
static void test_regularization() {
    Mesh mesh;
    mesh.num_hexes = 5;
    mesh.hex_nodes.allocate(5);
    mesh.elem_types.allocate(5);
    for (int i = 0; i < 5; ++i) mesh.elem_types[i] = ELEM_TET;

    const double sigma_al = 3.526e7;
    std::vector<MaterialProperties> mat(5);
    mat[0].kappa = sigma_al;   // Platte
    mat[1].kappa = 0.0;        // Luft
    mat[2].kappa = 0.0;        // Luft
    mat[3].kappa = sigma_al;   // Spule
    mat[4].kappa = 0.0;        // Luft

    const size_t changed = fem_direct::regularize_nonconducting_regions(
        mat.data(), mesh, sigma_al, 1e-8);

    check("regularisierte Elemente", static_cast<long>(changed), 3);
    check_num("sigma_Luft = 1e-8 * sigma_Al",
              std::fabs(mat[1].kappa - 1e-8 * sigma_al) / (1e-8 * sigma_al), 1e-15);
    check_true("Leiter unberuehrt",
               mat[0].kappa == sigma_al && mat[3].kappa == sigma_al);

    // Ein zweiter Aufruf darf nichts mehr finden: die Luft leitet jetzt.
    const size_t again = fem_direct::regularize_nonconducting_regions(
        mat.data(), mesh, sigma_al, 1e-8);
    check("zweiter Aufruf aendert nichts", static_cast<long>(again), 0);

    // Unsinnige Werte muessen abbrechen, nicht still durchlaufen
    bool threw_eps = false, threw_sigma = false;
    try { fem_direct::regularize_nonconducting_regions(mat.data(), mesh, sigma_al, 0.0); }
    catch (const std::exception&) { threw_eps = true; }
    try { fem_direct::regularize_nonconducting_regions(mat.data(), mesh, -1.0, 1e-8); }
    catch (const std::exception&) { threw_sigma = true; }
    check_true("epsilon = 0 bricht ab", threw_eps);
    check_true("negatives sigma_Referenz bricht ab", threw_sigma);
}

// -----------------------------------------------------------------------------
// 7: falsche Laengen muessen abbrechen
// -----------------------------------------------------------------------------
static void test_guards() {
    TestCsr csr;
    std::vector<char> wrong(3, 0);
    bool threw = false;
    try {
        fem_direct::apply_homogeneous_dirichlet(
            csr.n, csr.row_offsets.data(), csr.col_indices.data(),
            csr.values.data(), nullptr, wrong);
    } catch (const std::exception&) { threw = true; }
    check_true("Markierung falscher Laenge bricht ab", threw);

    // rhs == nullptr ist erlaubt
    std::vector<char> fixed(6, 0);
    fixed[2] = 1;
    bool ok = true;
    try {
        fem_direct::apply_homogeneous_dirichlet(
            csr.n, csr.row_offsets.data(), csr.col_indices.data(),
            csr.values.data(), nullptr, fixed);
    } catch (const std::exception&) { ok = false; }
    check_true("rhs == nullptr ist erlaubt", ok);
}

int main() {
    // Host-Test: ohne GPU lauffaehig. Ohne diese Zeile legt DeviceArray den
    // Speicher mit cudaMallocManaged an und bricht ohne Grafikkarte ab.
    bf::g_backend = bf::Backend::CPU;
    std::printf("\n== Einarbeiten in die CSR-Werte ==\n");
    test_elimination();

    std::printf("\n== Randkanten des Kuhn-Wuerfels ==\n");
    test_cube_boundary();

    std::printf("\n== Regularisierung des Nullraums ==\n");
    test_regularization();

    std::printf("\n== Wachtposten ==\n");
    test_guards();

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
