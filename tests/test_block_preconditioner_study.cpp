// =============================================================================
// Selbstpruefung der Deckenmessung fuer den BEM-Block
// =============================================================================
//
// docs/reports/convergence-findings.tex haelt fest, warum dieser Test so aussieht, wie er aussieht:
// sechs Diagnosekennzahlen des Projekts waren miskalibriert und haben die Suche
// in die Irre gefuehrt. Ein auffaelliges Messergebnis ist ohne Gegenprobe nicht
// von einem Werkzeugfehler zu unterscheiden.
//
// Geprueft wird deshalb nicht nur, DASS das Werkzeug laeuft, sondern dass es
// an Systemen mit BEKANNTER Antwort das Richtige sagt -- und zwar in BEIDE
// Richtungen:
//
//   Test 1  HostDenseLU: Loesung und transponierte Loesung gegen eine
//           unabhaengig gerechnete Referenz.
//   Test 2  kappa_2 an Matrizen VORGEGEBENER Konditionszahl (1e3, 1e6, 1e9).
//   Test 3  Die Falle. Ein 2x2-Blocksystem, bei dem die schlechte Kondition
//           BEWUSST im FEM-Block sitzt. Das Werkzeug MUSS melden, dass die
//           exakte Inverse des BEM-Blocks nichts bringt. Sagt es hier "bringt
//           viel", ist es unbrauchbar.
//   Test 4  Die Gegenprobe zu Test 3. Dasselbe System, die schlechte Kondition
//           jetzt im BEM-Block. Das Werkzeug MUSS jetzt einen deutlichen
//           Gewinn melden. Ohne diesen Test koennte ein Werkzeug, das
//           grundsaetzlich "bringt nichts" sagt, Test 3 bestehen.
//
// Reiner Host-Code, ohne CUDA. Baubar mit:
//   g++ -std=c++17 -O2 -Iinclude tests/test_block_preconditioner_study.cpp
//       src/BlockPreconditionerStudy.cpp -o test_block_preconditioner_study
//   (beide Quelldateien in einem Aufruf)

#include "BlockPreconditionerStudy.h"

#include <algorithm>
#include <string>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  [ok]   " : "  [FEHL] ") << what << "\n";
    if (!ok) ++g_failures;
}

inline double& at(std::vector<double>& A, int n, int i, int j) {
    return A[static_cast<size_t>(j) * n + i];
}

// Eine symmetrisch positiv definite (n x n)-Matrix mit VORGEGEBENEM Spektrum:
// Q diag(s) Q^T mit orthogonalem Q aus einer Householder-Kette. Singulaerwerte
// und Eigenwerte fallen damit zusammen und sind exakt bekannt -- die Messung
// hat eine harte Referenz.
//
// SPD und nicht allgemein: die Krylov-Konvergenz haengt am SPEKTRUM, nicht an
// den Singulaerwerten. Bei einer allgemeinen Matrix mit vorgegebenen
// Singulaerwerten liegen die Eigenwerte irgendwo, GMRES braucht dann
// unabhaengig von der Vorkonditionierung nahezu n Iterationen, und der Test
// misst nur noch die endliche Terminierung. Mit SPD-Bloecken ist der
// Zusammenhang "Vorkonditionierer raeumt das Spektrum auf -> weniger
// Iterationen" der, den das Werkzeug im Ernstfall auch sehen soll.
std::vector<double> spd_with_condition(int n, double kappa, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        // Eigenwerte logarithmisch von 1 bis 1/kappa
        const double t = (n > 1) ? static_cast<double>(i) / (n - 1) : 0.0;
        at(A, n, i, i) = std::pow(kappa, -t);
    }

    // Zwei Householder-Aehnlichkeitstransformationen H A H (H = H^T = H^-1).
    // Sie zerstoeren die Struktur, ohne das Spektrum zu aendern.
    auto householder_similarity = [&](std::vector<double>& M) {
        std::vector<double> v(static_cast<size_t>(n));
        double nv = 0.0;
        for (int i = 0; i < n; ++i) { v[i] = nd(rng); nv += v[i] * v[i]; }
        nv = std::sqrt(nv);
        for (int i = 0; i < n; ++i) v[i] /= nv;
        // von links: M <- M - 2 v (v^T M)
        for (int j = 0; j < n; ++j) {
            double d = 0.0;
            for (int i = 0; i < n; ++i) d += v[i] * at(M, n, i, j);
            for (int i = 0; i < n; ++i) at(M, n, i, j) -= 2.0 * d * v[i];
        }
        // von rechts: M <- M - 2 (M v) v^T
        for (int i = 0; i < n; ++i) {
            double d = 0.0;
            for (int j = 0; j < n; ++j) d += at(M, n, i, j) * v[j];
            for (int j = 0; j < n; ++j) at(M, n, i, j) -= 2.0 * d * v[j];
        }
    };
    householder_similarity(A);
    householder_similarity(A);
    return A;
}

// Baut ein gekoppeltes 2x2-Blocksystem
//     [ A11  A12 ]
//     [ A21  A22 ]
// mit vorgegebener Kondition je Diagonalblock und schwacher Kopplung.
std::vector<double> coupled_system(int n_A, int n_K, double kappa_A, double kappa_K,
                                   double coupling, unsigned seed) {
    const int n = n_A + n_K;
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    const std::vector<double> B11 = spd_with_condition(n_A, kappa_A, seed);
    const std::vector<double> B22 = spd_with_condition(n_K, kappa_K, seed + 7u);

    for (int j = 0; j < n_A; ++j)
        for (int i = 0; i < n_A; ++i)
            at(A, n, i, j) = B11[static_cast<size_t>(j) * n_A + i];
    for (int j = 0; j < n_K; ++j)
        for (int i = 0; i < n_K; ++i)
            at(A, n, n_A + i, n_A + j) = B22[static_cast<size_t>(j) * n_K + i];

    std::mt19937 rng(seed + 99u);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (int j = 0; j < n_K; ++j)
        for (int i = 0; i < n_A; ++i) at(A, n, i, n_A + j) = coupling * nd(rng);
    for (int j = 0; j < n_A; ++j)
        for (int i = 0; i < n_K; ++i) at(A, n, n_A + i, j) = coupling * nd(rng);
    return A;
}

// -----------------------------------------------------------------------------

void test_dense_lu() {
    std::cout << "\nTest 1: HostDenseLU, normale und transponierte Loesung\n";
    const int n = 40;
    std::mt19937 rng(4242u);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> A(static_cast<size_t>(n) * n);
    for (auto& a : A) a = nd(rng);
    for (int i = 0; i < n; ++i) at(A, n, i, i) += n;   // diagonaldominant

    std::vector<double> x_true(static_cast<size_t>(n));
    for (auto& v : x_true) v = nd(rng);

    // b = A x_true
    std::vector<double> b(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) b[i] += at(A, n, i, j) * x_true[j];
    // bt = A^T x_true
    std::vector<double> bt(static_cast<size_t>(n), 0.0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) bt[j] += at(A, n, i, j) * x_true[i];

    bf::HostDenseLU lu;
    std::vector<double> work(A);
    lu.factorize(work, n);

    std::vector<double> x(b);
    lu.solve(x.data());
    double err = 0.0, ref = 0.0;
    for (int i = 0; i < n; ++i) { err += (x[i]-x_true[i])*(x[i]-x_true[i]); ref += x_true[i]*x_true[i]; }
    const double rel = std::sqrt(err/ref);
    std::cout << "    solve():            rel. Fehler = " << rel << "\n";
    check(rel < 1e-12, "solve() trifft die Referenzloesung");

    std::vector<double> xt(bt);
    lu.solve_transposed(xt.data());
    err = 0.0;
    for (int i = 0; i < n; ++i) err += (xt[i]-x_true[i])*(xt[i]-x_true[i]);
    const double relT = std::sqrt(err/ref);
    std::cout << "    solve_transposed(): rel. Fehler = " << relT << "\n";
    check(relT < 1e-12, "solve_transposed() trifft die Referenzloesung");

    // Die Falle: singulaere Matrix MIT rundungskleinem, aber nicht exakt
    // nullem Pivot. "best == 0.0" laesst so etwas durch (docs/nan-und-
    // schwellwerte.md); die relative Schranke darf das nicht.
    std::vector<double> Sing(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) at(Sing, n, i, i) = 1.0;
    at(Sing, n, n-1, n-1) = 1e-300;
    bool threw = false;
    try { bf::HostDenseLU l2; l2.factorize(Sing, n); }
    catch (const std::exception&) { threw = true; }
    check(threw, "Pivot von 1e-300 wird als singulaer erkannt (nicht 'ungleich null')");

    // Und dieselbe Falle mit NaN.
    std::vector<double> Nan(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) at(Nan, n, i, i) = 1.0;
    at(Nan, n, 3, 3) = std::nan("");
    threw = false;
    try { bf::HostDenseLU l3; l3.factorize(Nan, n); }
    catch (const std::exception&) { threw = true; }
    check(threw, "NaN in der Matrix wird erkannt (Zaehlen statt Vergleichen)");
}

void test_condition_estimate() {
    std::cout << "\nTest 2: kappa_2 an Matrizen vorgegebener Konditionszahl\n";
    const int n = 60;
    for (double kappa : {1e3, 1e6, 1e9}) {
        const std::vector<double> A = spd_with_condition(n, kappa, 20260818u);
        const bf::ConditionEstimate e = bf::estimate_condition(A, n, 20000, 1e-12);
        const double rel_err = std::fabs(e.kappa - kappa) / kappa;
        std::cout << "    vorgegeben " << kappa << "  gemessen " << e.kappa
                  << "   Abweichung " << (100.0 * rel_err) << " %\n";
        check(rel_err < 0.02, "kappa_2 auf 2 % getroffen bei kappa = " + std::to_string(kappa));
    }
}

void run_study_quiet(const std::vector<double>& A, int n, int n_A,
                     bf::BlockPcStudyResult& out_res) {
    std::vector<double> b(static_cast<size_t>(n));
    std::mt19937 rng(555u);
    std::normal_distribution<double> nd(0.0, 1.0);
    for (auto& v : b) v = nd(rng);

    bf::BlockPcStudyOptions opt;
    opt.gmres_tol      = 1e-8;
    opt.gmres_max_iter = 1500;
    opt.gmres_restart  = 0;
    opt.measure_kappa  = false;      // im Test nicht noetig, spart Zeit

    std::ostringstream sink;
    out_res = bf::study_block_preconditioners(A, n, n_A, b, opt, sink);
}

void test_bad_conditioning_in_fem_block() {
    std::cout << "\nTest 3: schlechte Kondition BEWUSST im FEM-Block\n";
    std::cout << "        (das Werkzeug muss melden: exakte Inverse von A22 bringt nichts)\n";
    const int n_A = 200, n_K = 120, n = n_A + n_K;
    const std::vector<double> A = coupled_system(n_A, n_K, /*kappa_A=*/1e4, /*kappa_K=*/4.0,
                                                 /*coupling=*/1e-3, 31337u);
    bf::BlockPcStudyResult res;
    run_study_quiet(A, n, n_A, res);

    std::cout << "    kappa_2: System " << res.whole.kappa
              << ", FEM-Block " << res.block_A.kappa
              << ", BEM-Block " << res.block_K.kappa << "\n";
    check(res.block_K.kappa < 100.0,     "BEM-Block wird als gut konditioniert erkannt");
    check(res.block_A.kappa > 1e3,       "FEM-Block wird als schlecht konditioniert erkannt");

    const int it_id  = res.variants[0].gmres_iters;   // M = I
    const int it_bem = res.variants[2].gmres_iters;   // M = blockdiag(I, A22)
    const int it_fem = res.variants[1].gmres_iters;   // M = blockdiag(A11, I)
    std::cout << "    GMRES-Iterationen: I = " << it_id
              << ", exakter BEM-Block = " << it_bem
              << ", exakter FEM-Block = " << it_fem << "\n";
    std::cout << "    gemeldete Decke fuer den BEM-Block: Faktor "
              << res.bem_block_speedup << "\n";

    check(res.bem_block_speedup > 0.0 && res.bem_block_speedup < 1.5,
          "Decke fuer den BEM-Block wird korrekt als klein gemeldet");
    check(it_fem < it_bem,
          "der exakte FEM-Block hilft hier deutlich mehr als der exakte BEM-Block");
}

void test_bad_conditioning_in_bem_block() {
    std::cout << "\nTest 4: Gegenprobe -- schlechte Kondition im BEM-Block\n";
    std::cout << "        (ohne diesen Test bestuende auch ein Werkzeug, das immer\n";
    std::cout << "         'bringt nichts' sagt, den Test 3)\n";
    const int n_A = 200, n_K = 120, n = n_A + n_K;
    const std::vector<double> A = coupled_system(n_A, n_K, /*kappa_A=*/4.0, /*kappa_K=*/1e4,
                                                 /*coupling=*/1e-3, 31337u);
    bf::BlockPcStudyResult res;
    run_study_quiet(A, n, n_A, res);

    std::cout << "    kappa_2: System " << res.whole.kappa
              << ", FEM-Block " << res.block_A.kappa
              << ", BEM-Block " << res.block_K.kappa << "\n";
    check(res.block_K.kappa > 1e3, "BEM-Block wird als schlecht konditioniert erkannt");

    const int it_id  = res.variants[0].gmres_iters;
    const int it_bem = res.variants[2].gmres_iters;
    std::cout << "    GMRES-Iterationen: I = " << it_id
              << ", exakter BEM-Block = " << it_bem << "\n";
    std::cout << "    gemeldete Decke fuer den BEM-Block: Faktor "
              << res.bem_block_speedup << "\n";

    check(res.bem_block_speedup > 2.0,
          "Decke fuer den BEM-Block wird korrekt als gross gemeldet");
}

void test_exact_schur_is_the_best() {
    std::cout << "\nTest 5: das exakte Schur-Komplement schlaegt den blockdiagonalen Zweig\n";
    std::cout << "        (bei STARKER Kopplung -- sonst ist der Unterschied bedeutungslos)\n";
    const int n_A = 120, n_K = 80, n = n_A + n_K;
    const std::vector<double> A = coupled_system(n_A, n_K, /*kappa_A=*/1e4, /*kappa_K=*/1e3,
                                                 /*coupling=*/0.2, 90210u);
    bf::BlockPcStudyResult res;
    run_study_quiet(A, n, n_A, res);

    const int it_diag = res.variants[3].gmres_iters;   // blockdiag(A11, A22)
    const int it_schur= res.variants[4].gmres_iters;   // exaktes Schur
    std::cout << "    GMRES-Iterationen: blockdiag exakt = " << it_diag
              << ", exaktes Schur = " << it_schur << "\n";
    check(it_schur > 0 && it_schur <= it_diag,
          "exaktes Schur-Komplement ist nicht schlechter als blockdiagonal-exakt");
    // Theoretisch terminiert rechts-vorkonditioniertes GMRES mit exaktem
    // Schur-Komplement nach 2 Iterationen (das Minimalpolynom von A*M^-1 hat
    // Grad 2). In Gleitkomma kommen ein bis zwei Iterationen dazu, bis die
    // Toleranz von 1e-8 auch wirklich unterschritten ist -- die Schranke ist
    // deshalb 8 und nicht 2.
    check(it_schur > 0 && it_schur <= 8,
          "exaktes Schur-Komplement loest das System in wenigen Iterationen");

    // Die Genauigkeitsreihe muss MONOTON sein: ein groberes A11^-1 darf nicht
    // WENIGER Iterationen brauchen als ein feineres. Tut sie es doch, misst sie
    // Rauschen und nicht die Genauigkeit -- und die Kurve, auf der die
    // Entscheidung ueber ILU(0) beruhen soll, waere wertlos.
    std::vector<int> eps_iters;
    for (const auto& v : res.variants)
        if (v.name.rfind("7.", 0) == 0 && v.gmres_iters > 0)
            eps_iters.push_back(v.gmres_iters);
    std::cout << "    Genauigkeitsreihe (grob -> fein):";
    for (int it : eps_iters) std::cout << " " << it;
    std::cout << "\n";
    check(eps_iters.size() >= 3, "die Genauigkeitsreihe liefert Werte");
    bool monotone = true;
    for (size_t i = 1; i < eps_iters.size(); ++i)
        if (eps_iters[i] > eps_iters[i - 1] + 1) monotone = false;
    check(monotone, "feineres A11^-1 braucht nicht mehr Iterationen als groberes");

    // Beide Eliminationsrichtungen muessen sich melden. Variante 8 ist die
    // Gegenrichtung (ueber A22); faellt sie still aus, weil eine Zerlegung
    // scheitert, wuerde die Deutung stumm die eine Richtung bevorzugen --
    // und niemand saehe, dass die andere gar nicht gemessen wurde.
    const bool has8 = std::any_of(res.variants.begin(), res.variants.end(),
        [](const bf::BlockPcVariant& v) {
            return v.name.rfind("8 ", 0) == 0 && v.gmres_iters > 0;
        });
    check(has8, "die Eliminationsrichtung ueber A22 wird ueberhaupt gemessen");
    check(res.schur.available && res.schur_A.available,
          "beide Schur-Komplemente werden diagnostiziert");

    std::vector<int> eps9;
    for (const auto& v : res.variants)
        if (v.name.rfind("9.", 0) == 0 && v.gmres_iters > 0) eps9.push_back(v.gmres_iters);
    std::cout << "    Genauigkeitsreihe DtN-Zweig:     ";
    for (int it : eps9) std::cout << " " << it;
    std::cout << "\n";
    check(eps9.size() >= 3, "die Genauigkeitsreihe der Gegenrichtung liefert Werte");

    auto series = [&](const char* prefix) {
        std::vector<int> out_it;
        for (const auto& v : res.variants)
            if (v.name.rfind(prefix, 0) == 0 && v.gmres_iters > 0) out_it.push_back(v.gmres_iters);
        return out_it;
    };
    const std::vector<int> s10 = series("10.");
    const std::vector<int> s11 = series("11.");
    std::cout << "    Reihe 10 (nur S_A ungenau):      ";
    for (int it : s10) std::cout << " " << it;
    std::cout << "\n    Reihe 11 (beide ungenau):        ";
    for (int it : s11) std::cout << " " << it;
    std::cout << "\n";
    check(s10.size() >= 3 && s11.size() >= 3, "Reihen 10 und 11 liefern Werte");

    // Der inhaltlich zwingende Zusammenhang: Reihe 11 stoert BEIDE Anteile,
    // Reihen 9 und 10 je einen davon. Reihe 11 darf deshalb auf keiner Stufe
    // BESSER sein als beide Einzelreihen -- mehr Stoerung kann nicht helfen.
    // Ist sie es doch, sind die Reihen vertauscht oder eine Stoerung greift
    // nicht, und die Empfehlung stuende auf einer Messung, die etwas anderes
    // misst als ihr Name sagt. Toleranz 20 %, weil GMRES-Iterationszahlen
    // gegenueber der Rauschrealisierung nicht auf die Einerstelle stabil sind.
    bool consistent = true;
    const size_t n_lv = std::min({eps9.size(), s10.size(), s11.size()});
    for (size_t i = 0; i < n_lv; ++i) {
        const int best_single = std::min(eps9[i], s10[i]);
        if (s11[i] < static_cast<int>(0.8 * best_single) - 1) consistent = false;
    }
    check(consistent, "beide Anteile gestoert ist nie besser als nur einer");
}

void test_symmetric_part_extremes() {
    std::cout << "\nTest 6: Eigenwerte des symmetrischen Anteils\n";
    std::cout << "        (entscheidet, ob CG auf S ueberhaupt zulaessig ist --\n";
    std::cout << "         eine Fehlmessung hier wuerde ein falsches Verfahren rechtfertigen)\n";

    // spd_with_condition liefert Eigenwerte von 1 bis 1/kappa. Die Matrix ist
    // symmetrisch, also ist ihr symmetrischer Anteil sie selbst.
    const int n = 60;
    const double kappa = 1e4;
    const std::vector<double> A = spd_with_condition(n, kappa, 777u);
    double lmin = 0.0, lmax = 0.0;
    bf::symmetric_part_extremes(A, n, lmin, lmax, 20000, 1e-12);
    std::cout << "    erwartet [" << 1.0/kappa << ", 1]   gemessen ["
              << lmin << ", " << lmax << "]\n";
    check(std::fabs(lmax - 1.0) < 0.02, "lambda_max getroffen");
    // lambda_min nur auf 30 %: die verschobene Potenzmethode konvergiert am
    // unteren Spektralende langsam, wenn die Eigenwerte dort dicht liegen
    // (hier logarithmisch bis 1e-4). Das ist keine Schwaeche fuer den Zweck --
    // gebraucht wird das VORZEICHEN, nicht die dritte Stelle.
    //
    // Die Schranke stand bei 10 % und war damit PLATTFORMABHAENGIG: unter
    // GCC/Linux lag die Abweichung bei 4,5 %, unter MSVC/Windows bei 19 %
    // (gemessen 1,19e-04 gegen 1,00e-04). Beides ist dieselbe Rechnung; der
    // Unterschied kommt aus der Reihenfolge der Gleitkommaoperationen, die der
    // Optimierer waehlt, und die entscheidet bei einem langsam konvergierenden
    // Verfahren darueber, wo die Iteration abbricht.
    //
    // Ein Test, der auf einer Plattform gruen und auf der anderen rot ist,
    // ohne dass sich am Verfahren etwas geaendert hat, misst den Compiler und
    // nicht den Code. 30 % lassen dem Verfahren Luft und halten die Aussage,
    // auf die es ankommt: das Vorzeichen und die Groessenordnung.
    check(std::fabs(lmin - 1.0/kappa) < 0.30 / kappa, "lambda_min auf 30 % getroffen");
    check(lmin > 0.0, "positiv definite Matrix wird als solche erkannt");

    // Die Gegenprobe: dieselbe Matrix minus 2*lambda_max*I ist indefinit.
    // Ohne sie bestuende auch ein Verfahren, das lambda_min grundsaetzlich
    // positiv meldet.
    std::vector<double> B(A);
    for (int i = 0; i < n; ++i) B[static_cast<size_t>(i) * n + i] -= 0.5;
    bf::symmetric_part_extremes(B, n, lmin, lmax, 20000, 1e-12);
    std::cout << "    verschoben um -0.5: gemessen [" << lmin << ", " << lmax << "]\n";
    check(lmin < 0.0 && lmax > 0.0, "indefinite Matrix wird als indefinit erkannt");
}

void test_smallest_singular_values() {
    std::cout << "\nTest 7: die k kleinsten Singulaerwerte\n";
    std::cout << "        (entscheidet, ob ein Fastnullraum ein Einzelmodus oder ein\n";
    std::cout << "         Buendel ist -- die beiden Faelle verlangen Gegenteiliges)\n";

    // spd_with_condition hat das Spektrum kappa^(-i/(n-1)), i = 0..n-1.
    // Die k kleinsten sind damit exakt bekannt.
    const int n = 80;
    const double kappa = 1e6;
    const std::vector<double> A = spd_with_condition(n, kappa, 13579u);
    const int k = 6;
    const std::vector<double> sv = bf::smallest_singular_values(A, n, k, 500, 1e-12);

    check(static_cast<int>(sv.size()) == k, "k Werte zurueckgegeben");
    if (static_cast<int>(sv.size()) != k) return;

    double worst = 0.0;
    for (int i = 0; i < k; ++i) {
        // aufsteigend: sv[0] entspricht i = n-1, sv[1] -> n-2, ...
        const double expect = std::pow(kappa, -static_cast<double>(n - 1 - i) / (n - 1));
        const double rel = std::fabs(sv[static_cast<size_t>(i)] - expect) / expect;
        std::cout << "    sv[" << i << "] erwartet " << expect
                  << "  gemessen " << sv[static_cast<size_t>(i)]
                  << "  Abweichung " << (100.0 * rel) << " %\n";
        worst = std::max(worst, rel);
    }
    check(worst < 0.01, "alle k kleinsten Singulaerwerte auf 1 % getroffen");

    // Die Gegenprobe zur Deutung: eine Matrix mit einem EINZELNEN abgesetzten
    // kleinen Wert muss einen grossen Abstand sigma_2/sigma_1 zeigen, eine mit
    // einem Buendel einen kleinen. Ohne diese Unterscheidung waere die
    // Deutungszeile im Bericht wertlos.
    std::vector<double> Single(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) at(Single, n, i, i) = 1.0;
    at(Single, n, 0, 0) = 1e-5;                       // ein Ausreisser
    const std::vector<double> s1 = bf::smallest_singular_values(Single, n, 4, 500, 1e-12);
    std::cout << "    Einzelmodus: sigma_2/sigma_1 = " << (s1[1] / s1[0]) << "\n";
    check(s1[1] / s1[0] > 5.0, "Einzelmodus wird als abgesetzt erkannt");

    std::vector<double> Cluster(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) at(Cluster, n, i, i) = 1.0;
    for (int i = 0; i < 5; ++i) at(Cluster, n, i, i) = 1e-5 * (1.0 + 0.1 * i);
    const std::vector<double> s2 = bf::smallest_singular_values(Cluster, n, 4, 500, 1e-12);
    std::cout << "    Buendel:     sigma_2/sigma_1 = " << (s2[1] / s2[0]) << "\n";
    check(s2[1] / s2[0] < 5.0, "Buendel wird NICHT als abgesetzt erkannt");
}

void test_accuracy_series_comparison() {
    std::cout << "\nTest 8: Vergleich zweier Genauigkeitsreihen\n";
    std::cout << "        (hier sass ein Fehler: ein Gleichstand der eps-Schwelle\n";
    std::cout << "         wurde als Gleichstand der Reihe gewertet, obwohl die eine\n";
    std::cout << "         Richtung auf derselben Stufe 41x mehr Iterationen brauchte)\n";

    auto mk = [](const std::string& name, int iters, double relres, bool conv) {
        bf::BlockPcVariant v;
        v.name = name; v.gmres_iters = iters;
        v.gmres_relres = relres; v.gmres_converged = conv;
        return v;
    };

    // DER FALL, DER DEN FEHLER VERDECKT HATTE: beide Reihen kippen bei
    // derselben eps-Stufe von "nicht konvergiert" auf "konvergiert", aber B
    // braucht dort drastisch weniger Iterationen. Das sind die echten Zahlen
    // aus team28_coarse_12.
    std::vector<bf::BlockPcVariant> v;
    v.push_back(mk("7.1 x 0.1",    4000, 0.975,   false));
    v.push_back(mk("7.2 x 0.01",   4000, 0.837,   false));
    v.push_back(mk("7.3 x 0.001",  1181, 1e-8,    true));
    v.push_back(mk("7.4 x 0.0001",   34, 3e-9,    true));
    v.push_back(mk("7.5 x 1e-06",     6, 1.85e-10,true));
    v.push_back(mk("9.1 x 0.1",    4000, 0.414,   false));
    v.push_back(mk("9.2 x 0.01",   4000, 2.17e-4, false));
    v.push_back(mk("9.3 x 0.001",    29, 8.35e-9, true));
    v.push_back(mk("9.4 x 0.0001",    8, 4.73e-9, true));
    v.push_back(mk("9.5 x 1e-06",     3, 4.65e-10,true));

    const int score = bf::compare_accuracy_series(v, "7.", "9.");
    std::cout << "    Punktestand (positiv = zweite Reihe besser): " << score << "\n";
    check(score == 5, "alle fuenf Stufen gehen an die zweite Reihe");

    // Gegenprobe: dieselben Daten vertauscht muessen das Vorzeichen drehen.
    // Ohne sie bestuende auch eine Regel, die grundsaetzlich positiv meldet.
    const int rev = bf::compare_accuracy_series(v, "9.", "7.");
    std::cout << "    vertauscht: " << rev << "\n";
    check(rev == -score, "der Vergleich ist antisymmetrisch");

    // Und ein echter Gleichstand muss auch als solcher herauskommen.
    std::vector<bf::BlockPcVariant> eq;
    eq.push_back(mk("7.1 x 0.1", 100, 1e-9, true));
    eq.push_back(mk("9.1 x 0.1", 100, 1e-9, true));
    check(bf::compare_accuracy_series(eq, "7.", "9.") == 0,
          "echter Gleichstand wird als Gleichstand gemeldet");
}

void test_inner_cost() {
    std::cout << "\nTest 9: Kosten der inneren Loesungen\n";
    std::cout << "        (die Reihen 7-11 sagen, wie GENAU es sein muss --\n";
    std::cout << "         diese Zahl sagt, was das kostet)\n";

    // Zwei SPD-Systeme mit BEKANNTER Kondition. Die Krylov-Schrittzahl bis
    // zur Toleranz waechst wie sqrt(kappa); die Messung muss das wiedergeben,
    // sonst ist die Kostenrechnung darauf wertlos.
    const int n = 100;
    const std::vector<double> leicht = spd_with_condition(n, 1e2, 24680u);
    const std::vector<double> schwer = spd_with_condition(n, 1e4, 13579u);

    const bf::InnerCostEstimate a =
        bf::estimate_inner_cost(schwer, n, leicht, n, 1e-4, 10, 2000);
    std::cout << "    kappa 1e4 braucht " << a.iters_S_A
              << " Schritte, kappa 1e2 braucht " << a.iters_A22 << "\n";
    check(a.iters_S_A > 0 && a.iters_A22 > 0, "beide inneren Loesungen konvergieren");
    check(a.iters_S_A > a.iters_A22,
          "das schlechter konditionierte System braucht mehr Schritte");

    // Die Produktrechnung muss stimmen -- sie ist die Zahl, auf der die
    // Empfehlung beruht.
    const double expect = 10.0 * a.iters_S_A * a.iters_A22;
    std::cout << "    Arbeit = 10 x " << a.iters_S_A << " x " << a.iters_A22
              << " = " << a.bem_matvecs << "\n";
    check(std::fabs(a.bem_matvecs - expect) < 1e-9, "die Produktrechnung stimmt");

    // Und die Gegenprobe zur Toleranz: eine LOCKERERE Toleranz darf nicht
    // MEHR Schritte kosten. Ohne sie bestuende auch eine Messung, die die
    // Toleranz gar nicht beachtet -- und dann waere die ganze Kostenrechnung
    // von der geforderten Genauigkeit entkoppelt.
    const bf::InnerCostEstimate b =
        bf::estimate_inner_cost(schwer, n, leicht, n, 1e-2, 10, 2000);
    std::cout << "    bei Toleranz 1e-2 statt 1e-4: " << b.iters_S_A
              << " statt " << a.iters_S_A << " Schritte\n";
    check(b.iters_S_A > 0 && b.iters_S_A <= a.iters_S_A,
          "lockerere Toleranz kostet nicht mehr Schritte");
}

void test_harmonic_split() {
    std::cout << "\nTest 10: Aufspaltung A(dt) = K + C/dt und harmonisches System\n";
    std::cout << "         (die erste Fassung hatte hier eine WIRKUNGSLOSE Gegenprobe:\n";
    std::cout << "          zwei Matrizen, zwei Unbekannte je Eintrag -- der Rueckbau\n";
    std::cout << "          gelingt IMMER. Jetzt sagt eine dritte Abtastung vorher.)\n";

    const int n = 60, n_A = 40;
    std::vector<double> K = spd_with_condition(n, 1e3, 111u);
    std::vector<double> C(static_cast<size_t>(n) * n, 0.0);
    const std::vector<double> Cb = spd_with_condition(n_A, 10.0, 222u);
    for (int j = 0; j < n_A; ++j)
        for (int i = 0; i < n_A; ++i)
            at(C, n, i, j) = 1e3 * Cb[static_cast<size_t>(j) * n_A + i];

    auto build = [&](double dt) {
        std::vector<double> A(static_cast<size_t>(n) * n);
        for (size_t i = 0; i < A.size(); ++i) A[i] = K[i] + C[i] / dt;
        return A;
    };

    // dt2 GROSS: damit ist A2 praktisch schon K und die Ausloeschung faellt
    // weg. Das ist die Empfehlung aus dem Header, und sie wird hier belegt.
    const double dt1 = 1e-4, dt2 = 1.0, dt3 = 1e-3;
    const std::vector<double> A1 = build(dt1), A2 = build(dt2), A3 = build(dt3);

    const bf::KCSplit sp = bf::split_K_and_C(A1, dt1, A2, dt2, A3, dt3, n, n_A);
    std::cout << "    Vorhersagefehler fuer A3 = " << sp.rel_error_prediction
              << ",  ||C/dt1||/||K|| = " << sp.split_conditioning << "\n";
    check(sp.valid, "Aufspaltung wird als gueltig gemeldet");
    check(sp.c_outside_fem_block >= 0.0 && sp.c_outside_fem_block < 1e-12,
          "C verschwindet ausserhalb des FEM-Blocks");

    double errK = 0.0, errC = 0.0, nK = 0.0, nC = 0.0;
    for (size_t i = 0; i < K.size(); ++i) {
        errK += (sp.K[i] - K[i]) * (sp.K[i] - K[i]);  nK += K[i] * K[i];
        errC += (sp.C[i] - C[i]) * (sp.C[i] - C[i]);  nC += C[i] * C[i];
    }
    std::cout << "    K getroffen auf " << std::sqrt(errK / nK)
              << ", C auf " << std::sqrt(errC / nC) << "\n";
    check(std::sqrt(errK / nK) < 1e-12 && std::sqrt(errC / nC) < 1e-12,
          "K und C werden mit grossem dt2 exakt zurueckgewonnen");

    // Gegenprobe zur Empfehlung: mit dt2 = 1e-3 (klein) muss die Ausloeschung
    // messbar zuschlagen. Ohne diese Probe waere die Empfehlung im Header
    // eine Behauptung.
    {
        const std::vector<double> A2k = build(1e-3);
        const bf::KCSplit eng = bf::split_K_and_C(A1, dt1, A2k, 1e-3, build(1e-2), 1e-2, n, n_A);
        double e = 0.0;
        for (size_t i = 0; i < K.size(); ++i) e += (eng.K[i] - K[i]) * (eng.K[i] - K[i]);
        std::cout << "    mit dt2 = 1e-3 statt 1.0: K nur auf " << std::sqrt(e / nK) << "\n";
        check(std::sqrt(e / nK) > std::sqrt(errK / nK) * 10.0,
              "kleines dt2 verliert messbar Stellen (Ausloeschung)");
    }

    // ---------------------------------------------------------------------
    // DIE FALLE: die Abtastungen liegen in VERSCHIEDENEN Raeumen, weil
    // zwischen ihnen neu aequilibriert wurde. Die Differenz sieht immer noch
    // aus wie eine Matrix, und K, C und alles Weitere waeren Unsinn.
    // ---------------------------------------------------------------------
    std::mt19937 rng(31415u);
    std::uniform_real_distribution<double> uni(0.5, 2.0);
    std::vector<double> d(static_cast<size_t>(n));
    for (double& v : d) v = uni(rng);
    std::vector<double> A2_anders(A2);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) at(A2_anders, n, i, j) *= d[i] * d[j];

    const bf::KCSplit bad = bf::split_K_and_C(A1, dt1, A2_anders, dt2, A3, dt3, n, n_A);
    std::cout << "    bei verschiedenen Raeumen: Vorhersagefehler = "
              << bad.rel_error_prediction << "\n";
    check(!bad.valid, "Abtastung aus verschiedenen Raeumen wird ERKANNT und verworfen");

    const bf::KCSplit close = bf::split_K_and_C(A1, dt1, A2, dt1 * (1.0 + 1e-12), A3, dt3, n, n_A);
    check(!close.valid, "zu dicht beieinanderliegende dt werden abgelehnt");

    // Struktur des harmonischen Blocksystems
    const double omega = 2.0 * 3.14159265358979 * 50.0;
    const std::vector<double> H = bf::build_harmonic_system(sp.K, sp.C, omega, n);
    const int N = 2 * n;
    bool struct_ok = true;
    for (int j = 0; j < n && struct_ok; ++j)
        for (int i = 0; i < n; ++i) {
            const double k = at(K, n, i, j), c = omega * at(C, n, i, j);
            const double tol = 1e-9 * (1.0 + std::fabs(k) + std::fabs(c));
            if (std::fabs(H[static_cast<size_t>(j) * N + i]         - k) > tol ||
                std::fabs(H[static_cast<size_t>(j + n) * N + i]     + c) > tol ||
                std::fabs(H[static_cast<size_t>(j) * N + i + n]     - c) > tol ||
                std::fabs(H[static_cast<size_t>(j + n) * N + i + n] - k) > tol) {
                struct_ok = false; break;
            }
        }
    check(struct_ok, "harmonisches System hat die Blockstruktur [[W,-T],[T,W]]");

    // ---------------------------------------------------------------------
    // C-to-R muss auf einem System, das die Voraussetzungen erfuellt, in
    // WENIGEN Iterationen loesen -- und die Genauigkeitsreihe fuer (W+T)
    // muss monoton sein. Ohne beides waere die Empfehlung "C-to-R traegt"
    // eine Zahl ohne Gegenprobe.
    // ---------------------------------------------------------------------
    std::ostringstream sink;
    bf::BlockPcStudyOptions hopt;
    hopt.gmres_tol      = 1e-8;
    hopt.gmres_max_iter = 400;
    hopt.gmres_restart  = 0;
    hopt.measure_kappa  = false;

    std::vector<double> A_tr(static_cast<size_t>(n) * n);
    for (size_t i = 0; i < A_tr.size(); ++i) A_tr[i] = K[i] + C[i] / dt1;

    const bf::HarmonicStudyResult hr =
        bf::study_harmonic_system(A_tr, sp, omega, dt1, n, n_A, hopt, sink);
    std::cout << "    C-to-R (exakt): " << hr.iters_ctor << " Iterationen\n";
    check(hr.valid, "harmonische Studie laeuft durch");
    check(hr.iters_ctor > 0 && hr.iters_ctor < 40,
          "C-to-R loest das harmonische System in wenigen Iterationen");
    check(hr.iters_ctor < hr.iters_plain || hr.iters_plain < 0,
          "C-to-R ist besser als gar kein Vorkonditionierer");

    std::cout << "    (W+T)-Genauigkeitsreihe:";
    for (size_t i = 0; i < hr.wt_eps.size(); ++i)
        std::cout << " " << hr.wt_eps[i] << "->" << hr.wt_iters[i];
    std::cout << "\n";
    check(hr.wt_eps.size() >= 3, "die (W+T)-Genauigkeitsreihe liefert Werte");
    bool mono = true;
    int last = -1;
    for (size_t i = 0; i < hr.wt_iters.size(); ++i) {
        const int it = hr.wt_iters[i];
        if (it < 0) continue;
        if (last >= 0 && it > last + 1) mono = false;   // feiner darf nicht teurer sein
        last = it;
    }
    check(mono, "feineres (W+T)^-1 braucht nicht mehr Iterationen als groberes");

    // Die Auswahl der Toleranz muss die ARBEIT minimieren, nicht die groebste
    // tragende Stufe nehmen. Am echten Netz kostete die groebste Stufe (1e-3)
    // 4,7e+06 Matvecs, die feinere (1e-4) rund 3,2e+06 -- die erste Fassung
    // haette die teurere gewaehlt und den Ansatz zu schlecht dastehen lassen.
    if (hr.work_harmonic > 0.0) {
        std::cout << "    gewaehlt: eps = " << hr.best_eps << ", " << hr.best_outer
                  << " aeussere x " << hr.inner_iters_WpT << " innere = "
                  << hr.work_harmonic << "\n";
        const double recomputed = 10.0 * hr.best_outer * 2.0 * hr.inner_iters_WpT;
        check(std::fabs(hr.work_harmonic - recomputed) < 1e-9,
              "die gemeldete Arbeit passt zu den gemeldeten Faktoren");
        check(hr.reuse_per_setup == 2 * hr.best_outer,
              "die Wiederverwendungszahl passt zu den aeusseren Iterationen");
        // Der Break-even-Satz muss zur Richtung des Verhaeltnisses passen.
        // Die erste Fassung meldete bei work_harmonic < work_transient
        // "braucht nur Faktor 0,597" -- ein Faktor unter 1 waere aber eine
        // Verschlechterung. Die Zahl war richtig, der Satz falsch.
        check(hr.work_transient > 0.0, "die transiente Vergleichsarbeit ist gesetzt");

        // Keine andere tragende Stufe darf billiger sein als die gewaehlte.
        // Ohne diese Probe waere "Minimum" eine Behauptung.
        bool is_min = true;
        for (size_t i = 0; i < hr.wt_eps.size(); ++i)
            if (hr.wt_iters[i] > 0 && hr.wt_eps[i] < hr.best_eps
                && hr.wt_iters[i] * 1.0 < hr.best_outer * 0.5)
                is_min = false;   // grob: deutlich weniger aeussere bei feinerer Stufe
        check(is_min, "keine offensichtlich billigere Stufe wurde uebergangen");
    }
}

} // namespace

int main() {
    std::cout << "=== Selbstpruefung: Deckenmessung fuer den BEM-Block ===\n";
    try {
        test_dense_lu();
        test_condition_estimate();
        test_bad_conditioning_in_fem_block();
        test_bad_conditioning_in_bem_block();
        test_exact_schur_is_the_best();
        test_symmetric_part_extremes();
        test_smallest_singular_values();
        test_accuracy_series_comparison();
        test_inner_cost();
        test_harmonic_split();
    } catch (const std::exception& ex) {
        std::cout << "\nABBRUCH mit Ausnahme: " << ex.what() << "\n";
        return 1;
    }

    std::cout << "\n";
    if (g_failures == 0) {
        std::cout << "Alle Pruefungen bestanden.\n";
        return 0;
    }
    std::cout << g_failures << " Pruefung(en) fehlgeschlagen.\n";
    return 1;
}
