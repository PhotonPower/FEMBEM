// =============================================================================
// Abnahmepruefung: Dirichlet im n-Raum gegen Dirichlet in der 2n-Blockform
//
//   cmake --build build --config Release --target test_harmonic_dirichlet_split
//   ./build/test_harmonic_dirichlet_split
//
// Reiner Host-Code, keine GPU, kein Netz.
//
// WORUM ES GEHT
// -------------
// Der FEM-direkt-Pfad kennt zwei Wege zum harmonischen System:
//
//   2n-FORM   Die Matrix H = [[K, -wC], [wC, K]] wird explizit gebaut, und die
//             Dirichlet-Bedingung A x n = 0 wird DORT eingearbeitet: Zeile und
//             Spalte nullen, Diagonale auf einen Wert innerhalb der freien
//             Diagonale. Das ist der Weg zum Direktloeser.
//
//   n-RAUM    Im AMS-Modus (--fem-direct-nullspace=ams) gibt es keine
//             2n-Matrix. solve_harmonic() wendet H matrixfrei an, ueber
//             W = K + wC und C:
//
//                 y_r = W x_r - wC (x_r + x_i)
//                 y_i = W x_i + wC (x_r - x_i)
//
//             Festgehalten werden muss deshalb in W UND in C, jeweils im
//             n-Raum.
//
// DIE FALLGRUBE
// -------------
// Die beiden Matrizen brauchen VERSCHIEDENE Diagonalwerte auf den
// festgehaltenen Kanten:
//
//     W_ee = d,    (wC)_ee = 0
//
// Denn dann lautet die festgehaltene Zeile [d, 0] beziehungsweise [0, d] -- der
// Real- und der Imaginaerteil dieser Kante werden unabhaengig voneinander auf
// null gesetzt, wie es die Randbedingung verlangt.
//
// Bekaeme C dort denselben Wert d wie W (also das, was
// apply_homogeneous_dirichlet() von sich aus tut), stuende dort
//
//     [d - w*d, -w*d]
//
// -- eine Gleichung, die den Imaginaerteil der Randkante an ihren Realteil
// KOPPELT, statt beide festzuhalten. Das System bliebe loesbar, es waere nur
// ein anderes. Sichtbar wuerde es allein als seltsame Iterationszahl.
//
// Genau diese Fallgrube prueft dieser Test, und zwar an der Form, die der
// Loeser wirklich anwendet: die drei Terme oben, nicht eine nachgebaute
// Matrix.
//
// WAS ABSICHTLICH NICHT VERGLICHEN WIRD
// -------------------------------------
// Der Diagonalwert d selbst. Er ist frei waehlbar (irgendetwas in der
// Groessenordnung der freien Diagonale), und die beiden Wege bestimmen ihn aus
// verschiedenen Matrizen: die 2n-Form aus den K_ii, der n-Raum aus den
// (K + wC)_ii. Auf den festgehaltenen Kanten unterscheiden sich die beiden
// Systeme deshalb, und das ist richtig so -- auf den FREIEN
// Freiheitsgraden muss die Loesung identisch sein, und nur das ist geprueft.
//
// EXIT-CODE 0, wenn alle Pruefungen bestehen, sonst 1.
// =============================================================================

#include "FemDirectBC.h"
#include "HarmonicBlockSystem.h"

#include <algorithm>   // std::fill -- MSVC zieht das nicht transitiv mit
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int    N     = 60;
const     double OMEGA = 2.0 * 3.14159265358979323846 * 50.0;

/// Ein symmetrisches Muster mit Diagonale und aufsteigenden Spaltenindizes je
/// Zeile -- genau die Voraussetzung, die HarmonicBlockSystem.h nennt.
/// Gekoppelt wird an i-3, i-1, i, i+1, i+3, soweit vorhanden.
void build_pattern(std::vector<int>& ro, std::vector<int>& ci) {
    ro.assign(N + 1, 0);
    ci.clear();
    const int off[5] = {-3, -1, 0, 1, 3};
    for (int i = 0; i < N; ++i) {
        ro[i] = static_cast<int>(ci.size());
        for (int k = 0; k < 5; ++k) {
            const int j = i + off[k];
            if (j >= 0 && j < N) ci.push_back(j);
        }
    }
    ro[N] = static_cast<int>(ci.size());
}

/// y = M x fuer eine Host-CSR.
void spmv(const std::vector<int>& ro, const std::vector<int>& ci,
          const std::vector<double>& v, const std::vector<double>& x,
          std::vector<double>& y) {
    y.assign(x.size(), 0.0);
    for (int i = 0; i + 1 < static_cast<int>(ro.size()); ++i) {
        double s = 0.0;
        for (int k = ro[i]; k < ro[i + 1]; ++k) s += v[k] * x[ci[k]];
        y[i] = s;
    }
}

}  // namespace

int main() {
    // Host-Test: ohne GPU lauffaehig. Ohne diese Zeile legt DeviceArray den
    // Speicher mit cudaMallocManaged an und bricht ohne Grafikkarte ab.
    bf::g_backend = bf::Backend::CPU;
    std::printf("\n=== Dirichlet: n-Raum gegen 2n-Blockform ===\n\n");
    int failures = 0;

    std::vector<int> ro, ci;
    build_pattern(ro, ci);
    const size_t nnz = ci.size();

    // K symmetrisch und diagonaldominant, C symmetrisch und NUR auf der
    // unteren Haelfte besetzt -- das ist die Struktur des echten Problems:
    // sigma = 0 in Luft und Spulen, sigma > 0 nur in der Platte.
    std::mt19937 rng(20260910);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<double> K(nnz, 0.0), C(nnz, 0.0);
    auto conductor = [](int i) { return i < N / 2; };

    for (int i = 0; i < N; ++i) {
        for (int k = ro[i]; k < ro[i + 1]; ++k) {
            const int j = ci[k];
            if (j < i) continue;                    // spaeter gespiegelt
            const double kv = (i == j) ? (6.0 + uni(rng)) : uni(rng);
            const double cv = (conductor(i) && conductor(j))
                            ? ((i == j) ? (3.0e-3 + 1.0e-4 * uni(rng))
                                        : 1.0e-4 * uni(rng))
                            : 0.0;
            K[k] = kv; C[k] = cv;
            // Spiegeln: Eintrag (j, i) suchen.
            for (int k2 = ro[j]; k2 < ro[j + 1]; ++k2) {
                if (ci[k2] == i) { K[k2] = kv; C[k2] = cv; break; }
            }
        }
    }

    // Jede siebte Kante ist Randkante.
    std::vector<char> fixed_n(N, 0);
    int n_fixed = 0;
    for (int i = 0; i < N; ++i) if (i % 7 == 0) { fixed_n[i] = 1; ++n_fixed; }

    std::vector<double> b_r(N);
    for (int i = 0; i < N; ++i) b_r[i] = uni(rng);
    // b_i = 0, wie im Lauf (Quellterm bei t = T/4 ausgewertet).

    // -------------------------------------------------------------------------
    // Weg A: explizite 2n-Form
    // -------------------------------------------------------------------------
    std::vector<int> ro2, ci2;
    fem_direct::build_block_structure(N, ro.data(), ci.data(), ro2, ci2);
    std::vector<double> v2(ci2.size());
    fem_direct::fill_block_values(N, ro.data(), ci.data(), K.data(), C.data(),
                                  OMEGA, fem_direct::BlockForm::General, v2.data());
    std::vector<double> rhs2(2 * N, 0.0);
    fem_direct::fill_block_rhs(N, b_r.data(), nullptr,
                               fem_direct::BlockForm::General, rhs2.data());

    std::vector<char> fixed_2n(2 * N, 0);
    for (int i = 0; i < N; ++i) { fixed_2n[i] = fixed_n[i]; fixed_2n[N + i] = fixed_n[i]; }
    const auto dA = fem_direct::apply_homogeneous_dirichlet(
        2 * N, ro2.data(), ci2.data(), v2.data(), rhs2.data(), fixed_2n);

    // -------------------------------------------------------------------------
    // Weg B: n-Raum, genau wie in fem.cu
    // -------------------------------------------------------------------------
    std::vector<double> W(nnz), Cd(C);
    for (size_t k = 0; k < nnz; ++k) W[k] = K[k] + OMEGA * C[k];
    std::vector<double> bB(b_r);

    const auto dB = fem_direct::apply_homogeneous_dirichlet(
        N, ro.data(), ci.data(), W.data(), bB.data(), fixed_n);
    fem_direct::apply_homogeneous_dirichlet(
        N, ro.data(), ci.data(), Cd.data(), nullptr, fixed_n);
    for (int i = 0; i < N; ++i) {
        if (!fixed_n[i]) continue;
        for (int k = ro[i]; k < ro[i + 1]; ++k) if (ci[k] == i) Cd[k] = 0.0;
    }

    std::printf("  Diagonalwerte: 2n-Form %.6g, n-Raum (in W) %.6g\n",
                dA.diag_value, dB.diag_value);
    std::printf("  %d von %d Kanten festgehalten\n\n", n_fixed, N);

    // -------------------------------------------------------------------------
    // Pruefung 1: rechte Seite
    // -------------------------------------------------------------------------
    {
        double worst = 0.0;
        for (int i = 0; i < N; ++i) {
            worst = std::fmax(worst, std::fabs(rhs2[i] - bB[i]));
            worst = std::fmax(worst, std::fabs(rhs2[N + i]));   // b_i bleibt null
        }
        const bool ok = worst == 0.0;
        std::printf("  [%s] rechte Seite identisch (Abweichung %.2e)\n",
                    ok ? " OK " : "FEHL", worst);
        if (!ok) ++failures;
    }

    // -------------------------------------------------------------------------
    // Pruefung 2: die matrixfreie Anwendung stimmt auf den FREIEN Zeilen
    // -------------------------------------------------------------------------
    // Verglichen wird an einem Zufallsvektor, nicht eintragsweise an den
    // Matrizen: der Loeser wendet H an, er baut sie nicht. Ein Fehler in der
    // Anwendung -- ein Vorzeichen, ein vergessener Term -- schlaegt hier durch,
    // eine eintragsweise Probe an einer nachgebauten Matrix koennte ihn
    // verfehlen.
    {
        std::vector<double> x_r(N), x_i(N);
        for (int i = 0; i < N; ++i) { x_r[i] = uni(rng); x_i[i] = uni(rng); }
        // Die festgehaltenen Freiheitsgrade sind Teil der Loesung und dort
        // null; ein Vektor, der sie besetzt, wuerde die beiden Systeme ueber
        // ihre unterschiedlichen Diagonalwerte auseinanderziehen.
        for (int i = 0; i < N; ++i) if (fixed_n[i]) { x_r[i] = 0.0; x_i[i] = 0.0; }

        std::vector<double> x2(2 * N);
        for (int i = 0; i < N; ++i) { x2[i] = x_r[i]; x2[N + i] = x_i[i]; }
        std::vector<double> yA;
        spmv(ro2, ci2, v2, x2, yA);

        // Weg B, wie HarmonicOperator::apply() es rechnet.
        std::vector<double> sum(N), dif(N);
        for (int i = 0; i < N; ++i) { sum[i] = x_r[i] + x_i[i]; dif[i] = x_r[i] - x_i[i]; }
        std::vector<double> Wxr, Wxi, Csum, Cdif;
        spmv(ro, ci, W, x_r, Wxr);
        spmv(ro, ci, W, x_i, Wxi);
        spmv(ro, ci, Cd, sum, Csum);
        spmv(ro, ci, Cd, dif, Cdif);

        double scale = 0.0;
        for (double v : yA) scale = std::fmax(scale, std::fabs(v));
        double worst_free = 0.0;
        for (int i = 0; i < N; ++i) {
            if (fixed_n[i]) continue;
            const double yr = Wxr[i] - OMEGA * Csum[i];
            const double yi = Wxi[i] + OMEGA * Cdif[i];
            worst_free = std::fmax(worst_free, std::fabs(yr - yA[i]));
            worst_free = std::fmax(worst_free, std::fabs(yi - yA[N + i]));
        }
        const double rel = (scale > 0.0) ? worst_free / scale : worst_free;

        // Nicht maschinengenau, und das ist erklaerbar: Weg B rechnet K als
        // (K + wC) - wC, also durch Subtraktion. Bei wC/K um 3e-3 ist die
        // Ausloeschung mild, aber vorhanden. 1e-12 relativ ist reichlich Luft
        // dafuer und immer noch zehn Dekaden unter jedem Vorzeichenfehler.
        const bool ok = rel < 1e-12;
        std::printf("  [%s] H x stimmt auf den freien Zeilen (relativ %.2e, "
                    "Schwelle 1e-12)\n", ok ? " OK " : "FEHL", rel);
        if (!ok) ++failures;
    }

    // -------------------------------------------------------------------------
    // Pruefung 3: die festgehaltenen Zeilen sind reine Diagonalzeilen
    // -------------------------------------------------------------------------
    // Das ist die Pruefung auf die Fallgrube. Sie schlaegt an, sobald C auf
    // seiner Diagonale einen Wert ungleich null behaelt -- dann steht in der
    // Randzeile ein Kopplungsterm zwischen Real- und Imaginaerteil.
    {
        std::vector<double> x_r(N, 0.0), x_i(N, 0.0);
        double worst_cross = 0.0, min_diag = 1e300;
        for (int e = 0; e < N; ++e) {
            if (!fixed_n[e]) continue;

            // Einheitsvektor auf dem REALTEIL der Randkante e.
            std::fill(x_r.begin(), x_r.end(), 0.0);
            std::fill(x_i.begin(), x_i.end(), 0.0);
            x_r[e] = 1.0;

            std::vector<double> sum(N), dif(N);
            for (int i = 0; i < N; ++i) { sum[i] = x_r[i] + x_i[i]; dif[i] = x_r[i] - x_i[i]; }
            std::vector<double> Wxr, Wxi, Csum, Cdif;
            spmv(ro, ci, W, x_r, Wxr);
            spmv(ro, ci, W, x_i, Wxi);
            spmv(ro, ci, Cd, sum, Csum);
            spmv(ro, ci, Cd, dif, Cdif);

            for (int i = 0; i < N; ++i) {
                const double yr = Wxr[i] - OMEGA * Csum[i];
                const double yi = Wxi[i] + OMEGA * Cdif[i];
                if (i == e) {
                    min_diag = std::fmin(min_diag, std::fabs(yr));
                    // Der Imaginaerteil DERSELBEN Kante ist der Kopplungsterm,
                    // auf den es ankommt.
                    worst_cross = std::fmax(worst_cross, std::fabs(yi));
                } else {
                    worst_cross = std::fmax(worst_cross, std::fabs(yr));
                    worst_cross = std::fmax(worst_cross, std::fabs(yi));
                }
            }
        }
        const bool ok = (worst_cross == 0.0) && (min_diag > 0.0);
        std::printf("  [%s] Randzeilen sind diagonal: groesster Nebeneintrag %.2e, "
                    "kleinste Diagonale %.6g\n",
                    ok ? " OK " : "FEHL", worst_cross, min_diag);
        if (!ok) ++failures;
    }

    // -------------------------------------------------------------------------
    // Pruefung 4: mit der falschen C-Diagonale MUSS Pruefung 3 scheitern
    // -------------------------------------------------------------------------
    // Ohne diese Gegenprobe waere Pruefung 3 nicht belegt scharf: sie koennte
    // aus einem ganz anderen Grund bestehen. Hier wird der Fehler absichtlich
    // eingebaut -- C behaelt den Diagonalwert, den
    // apply_homogeneous_dirichlet() von sich aus setzt -- und der
    // Kopplungsterm muss auftauchen.
    {
        std::vector<double> C_bad(C);
        const auto db = fem_direct::apply_homogeneous_dirichlet(
            N, ro.data(), ci.data(), C_bad.data(), nullptr, fixed_n);

        int e = -1;
        for (int i = 0; i < N; ++i) if (fixed_n[i]) { e = i; break; }

        std::vector<double> x_r(N, 0.0), x_i(N, 0.0);
        x_r[e] = 1.0;
        std::vector<double> dif(N);
        for (int i = 0; i < N; ++i) dif[i] = x_r[i] - x_i[i];
        std::vector<double> Wxi, Cdif;
        spmv(ro, ci, W, x_i, Wxi);
        spmv(ro, ci, C_bad, dif, Cdif);
        const double cross = Wxi[e] + OMEGA * Cdif[e];

        const bool ok = std::fabs(cross) > 1e-6;
        std::printf("  [%s] Gegenprobe: mit C-Diagonale %.6g entsteht der "
                    "Kopplungsterm %.6g (er MUSS auftauchen)\n",
                    ok ? " OK " : "FEHL", db.diag_value, cross);
        if (!ok) ++failures;
    }

    std::printf("\n");
    if (failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", failures);
    return 1;
}
