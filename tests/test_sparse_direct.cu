// =============================================================================
// Abnahmepruefung: Musterpruefung und Verhalten ohne cuDSS (AP6)
//
//   cmake --build build --config Release --target test_sparse_direct
//   ./build/test_sparse_direct
//
// Reiner Host-Code. Laeuft MIT und OHNE cuDSS und prueft in beiden Faellen
// etwas Sinnvolles.
//
// WAS HIER GEPRUEFT WERDEN KANN UND WAS NICHT
// -------------------------------------------
// Die Zerlegung selbst braucht cuDSS und eine GPU; dieser Test kann sie nicht
// nachrechnen. Was er prueft, ist der Teil, der ohne Bibliothek auskommt und
// erfahrungsgemaess die meisten Fehler enthaelt:
//
//   1. validate_csr(). cuDSS prueft das Eingabemuster nicht vollstaendig, und
//      ein Loeser auf fehlerhaftem CSR liefert keine Fehlermeldung, sondern
//      Zahlen. Deshalb pruefen wir vorher selbst -- und deshalb muss die
//      Pruefung ihrerseits geprueft sein, samt der Faelle, die sie NICHT
//      beanstanden darf.
//
//   2. Das Zusammenspiel mit AP5: das von build_block_structure erzeugte
//      2n-Muster und das daraus gezogene obere Dreieck muessen beide durch
//      validate_csr gehen. Das ist die eigentliche Schnittstelle zwischen den
//      beiden Bausteinen, und ein Fehler darin faellt sonst erst auf der GPU
//      auf.
//
//   3. Ohne cuDSS muessen alle Methoden mit einer verstaendlichen Meldung
//      werfen, statt zu einem Linkfehler oder einem Absturz zu fuehren.
//      Mit cuDSS wird die Version ausgegeben.
// =============================================================================

#include "SparseDirectSolver.h"
#include "HarmonicBlockSystem.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

static void expect_ok(const char* name, const std::string& problem) {
    const bool ok = problem.empty();
    std::printf("  %-50s %-8s  %s\n", name, ok ? "sauber" : "beanstandet",
                ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) {
        std::printf("      unerwartete Beanstandung: %s\n", problem.c_str());
        ++g_failures;
    }
}

static void expect_rejected(const char* name, const std::string& problem) {
    const bool ok = !problem.empty();
    std::printf("  %-50s %-8s  %s\n", name, ok ? "erkannt" : "durchgelassen",
                ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
    else std::printf("      Meldung: %.100s\n", problem.c_str());
}

static void check_true(const char* name, bool ok) {
    std::printf("  %-50s %-8s  %s\n", name, ok ? "ja" : "nein",
                ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

// -----------------------------------------------------------------------------
// 1. Ein gueltiges Muster darf NICHT beanstandet werden
// -----------------------------------------------------------------------------
static void test_valid_pattern() {
    // 4x4, tridiagonal mit Diagonale
    const std::vector<int> ro = {0, 2, 5, 8, 10};
    const std::vector<int> ci = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3};
    expect_ok("gueltiges tridiagonales Muster",
              fem_direct::validate_csr(4, 10, ro.data(), ci.data()));

    // Oberes Dreieck: Zeilen sind kuerzer, Diagonale aber vorhanden
    const std::vector<int> tro = {0, 2, 4, 6, 7};
    const std::vector<int> tci = {0, 1, 1, 2, 2, 3, 3};
    expect_ok("oberes Dreieck (kurze Zeilen, Diagonale da)",
              fem_direct::validate_csr(4, 7, tro.data(), tci.data()));
}

// -----------------------------------------------------------------------------
// 2. Jede Fehlerart muss erkannt werden
// -----------------------------------------------------------------------------
static void test_broken_patterns() {
    {   // Spalten nicht aufsteigend -- der gefaehrlichste Fall, weil cuDSS ihn
        // teilweise toleriert und dann falsch rechnet
        const std::vector<int> ro = {0, 2, 4};
        const std::vector<int> ci = {1, 0, 0, 1};
        expect_rejected("Spalten nicht aufsteigend",
                        fem_direct::validate_csr(2, 4, ro.data(), ci.data()));
    }
    {   // Doppelter Spaltenindex (nicht STRENG aufsteigend)
        const std::vector<int> ro = {0, 2, 4};
        const std::vector<int> ci = {0, 0, 0, 1};
        expect_rejected("doppelter Spaltenindex in einer Zeile",
                        fem_direct::validate_csr(2, 4, ro.data(), ci.data()));
    }
    {   // Fehlende Diagonale
        const std::vector<int> ro = {0, 1, 2};
        const std::vector<int> ci = {1, 0};
        expect_rejected("fehlende Diagonale",
                        fem_direct::validate_csr(2, 2, ro.data(), ci.data()));
        expect_ok("  dieselbe ohne Diagonalforderung",
                  fem_direct::validate_csr(2, 2, ro.data(), ci.data(), false));
    }
    {   // Spaltenindex ausserhalb
        const std::vector<int> ro = {0, 2, 4};
        const std::vector<int> ci = {0, 5, 0, 1};
        expect_rejected("Spaltenindex ausserhalb [0, n)",
                        fem_direct::validate_csr(2, 4, ro.data(), ci.data()));
    }
    {   // Einsbasiert statt nullbasiert -- der klassische Fehler bei CSR
        const std::vector<int> ro = {1, 3, 5};
        const std::vector<int> ci = {1, 2, 1, 2};
        expect_rejected("einsbasierte Offsets (CUDSS_BASE_ZERO erwartet)",
                        fem_direct::validate_csr(2, 4, ro.data(), ci.data()));
    }
    {   // nnz passt nicht zu row_offsets
        const std::vector<int> ro = {0, 2, 4};
        const std::vector<int> ci = {0, 1, 0, 1};
        expect_rejected("nnz passt nicht zu row_offsets",
                        fem_direct::validate_csr(2, 7, ro.data(), ci.data()));
    }
    {   // leere Zeile
        const std::vector<int> ro = {0, 1, 1, 2};
        const std::vector<int> ci = {0, 2};
        expect_rejected("leere Zeile",
                        fem_direct::validate_csr(3, 2, ro.data(), ci.data()));
    }
    {   // Nicht monotone Offsets. Die Daten sind bewusst so gewaehlt, dass
        // WIRKLICH die Monotonieprueung anschlaegt und nicht vorher eine
        // andere: Zeile 0 (k = 0..1) ist sauber aufsteigend und hat eine
        // Diagonale, und row_offsets[2] == nnz stimmt. Erst Zeile 1 faellt
        // auf, weil ihr Ende vor ihrem Anfang liegt.
        const std::vector<int> ro = {0, 2, 1};
        const std::vector<int> ci = {0, 1};
        const std::string msg = fem_direct::validate_csr(2, 1, ro.data(), ci.data());
        expect_rejected("nicht monotone row_offsets", msg);
        check_true("  und zwar durch die Monotoniepruefung",
                   msg.find("monoton") != std::string::npos);
    }
    expect_rejected("num_rows = 0", fem_direct::validate_csr(0, 0, nullptr, nullptr));
    {
        const std::vector<int> ro = {0, 1};
        const std::vector<int> ci = {0};
        expect_rejected("col_indices == nullptr",
                        fem_direct::validate_csr(1, 1, ro.data(), nullptr));
    }
}

// -----------------------------------------------------------------------------
// 3. Schnittstelle zu AP5: die Blockform muss durch die Pruefung gehen
// -----------------------------------------------------------------------------
static void test_block_form_is_valid_input() {
    // Kleines Muster wie in test_block_system, aber mit Diagonale in jeder Zeile
    const int n = 5;
    std::vector<std::vector<int>> pat = {
        {0, 1, 4}, {0, 1, 2}, {1, 2, 3, 4}, {2, 3}, {0, 2, 4}
    };
    std::vector<int> ro, ci;
    ro.push_back(0);
    for (int r = 0; r < n; ++r) {
        for (int c : pat[r]) ci.push_back(c);
        ro.push_back(static_cast<int>(ci.size()));
    }
    expect_ok("Eingabemuster selbst",
              fem_direct::validate_csr(n, static_cast<int>(ci.size()),
                                       ro.data(), ci.data()));

    std::vector<int> ro2, ci2;
    fem_direct::build_block_structure(n, ro.data(), ci.data(), ro2, ci2);
    expect_ok("2n-Blockmuster (fuer CUDSS_MTYPE_GENERAL)",
              fem_direct::validate_csr(2 * n, static_cast<int>(ci2.size()),
                                       ro2.data(), ci2.data()));

    // Werte fuellen und Dreieck ziehen
    std::vector<double> K(ci.size()), C(ci.size());
    for (size_t r = 0; r < pat.size(); ++r)
        for (int k = ro[r]; k < ro[r + 1]; ++k) {
            K[k] = (ci[k] == static_cast<int>(r)) ? 10.0 + r : -1.0;
            C[k] = (ci[k] == static_cast<int>(r)) ? 1e-3 : -1e-4;
        }
    std::vector<double> v(ci2.size());
    fem_direct::fill_block_values(n, ro.data(), ci.data(), K.data(), C.data(),
                                  2.0 * 3.14159265358979 * 50.0,
                                  fem_direct::BlockForm::Symmetric, v.data());
    std::vector<int> tro, tci;
    std::vector<double> tv;
    fem_direct::extract_upper_triangle(2 * n, ro2.data(), ci2.data(), v.data(),
                                       tro, tci, tv);
    expect_ok("oberes Dreieck (fuer CUDSS_MTYPE_SYMMETRIC)",
              fem_direct::validate_csr(2 * n, static_cast<int>(tci.size()),
                                       tro.data(), tci.data()));
}

// -----------------------------------------------------------------------------
// 4. Verhalten mit und ohne cuDSS
// -----------------------------------------------------------------------------
static void test_availability() {
    const bool have = fem_direct::SparseDirectSolver::available();
    std::printf("  cuDSS eingebunden: %s%s%s\n",
                have ? "ja, Version " : "nein",
                have ? fem_direct::SparseDirectSolver::version().c_str() : "",
                have ? "" : " (mit -DFEMBEM_WITH_CUDSS=ON neu konfigurieren)");

    fem_direct::SparseDirectSolver solver;
    fem_direct::SolverOptions opt;
    opt.deterministic = true;
    solver.set_options(opt);      // muss in beiden Faellen gehen

    if (!have) {
        // Ohne cuDSS: verstaendliche Meldung statt Absturz
        bool threw = false;
        std::string msg;
        try {
            solver.factorize();
        } catch (const std::exception& e) { threw = true; msg = e.what(); }
        check_true("ohne cuDSS wirft factorize()", threw);
        check_true("  Meldung nennt FEMBEM_WITH_CUDSS",
                   msg.find("FEMBEM_WITH_CUDSS") != std::string::npos);
        check_true("  Meldung nennt BUILD.md",
                   msg.find("BUILD.md") != std::string::npos);
    } else {
        // Mit cuDSS: ein ungueltiges Muster muss VOR dem
        // Bibliotheksaufruf abgefangen werden
        std::vector<int> ro = {0, 2, 4};
        std::vector<int> ci = {1, 0, 0, 1};   // nicht aufsteigend
        std::vector<double> v = {1, 2, 3, 4};
        bool threw = false;
        std::string msg;
        try {
            solver.analyze(2, 4, ro.data(), ci.data(), v.data(),
                           fem_direct::MatrixKind::General);
        } catch (const std::exception& e) { threw = true; msg = e.what(); }
        check_true("ungueltiges Muster wird in analyze() abgefangen", threw);
        check_true("  Meldung nennt das Muster",
                   msg.find("CSR-Muster") != std::string::npos);

        // solve() ohne Faktorisierung
        bool threw2 = false;
        try {
            double b = 1.0, x = 0.0;
            solver.solve(&b, &x);
        } catch (const std::exception&) { threw2 = true; }
        check_true("solve() ohne Faktorisierung wirft", threw2);
    }
}

int main() {
    std::printf("\n== Gueltige Muster (duerfen nicht beanstandet werden) ==\n");
    test_valid_pattern();

    std::printf("\n== Fehlerhafte Muster (muessen erkannt werden) ==\n");
    test_broken_patterns();

    std::printf("\n== Schnittstelle zur Blockform aus AP5 ==\n");
    test_block_form_is_valid_input();

    std::printf("\n== Verfuegbarkeit ==\n");
    test_availability();

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
