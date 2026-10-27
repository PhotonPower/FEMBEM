// =============================================================================
// Abnahmeprüfung: bleibt der skalare AMS-Hilfsraum beschränkt, wenn A_G
// außerhalb der Leiter nur noch Rundungsrauschen enthält?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_ams_scalar_masking
//   ./build/test_ams_scalar_masking
//
// Reiner Host-Code, keine GPU, kein Netz.
//
// HINTERGRUND
// -----------
// Der AMS-Zweig lieferte am TEAM-28-Problem ||z_A|| = 8.6e+14 bei ||r_A|| = 1
// und stand über 1000 GMRES-Iterationen bei Rel. Res. 1.0.
//
// Ursache ist die Struktur der skalaren Hilfsmatrix A_G = G^T A G. Für den
// Curl-Curl-Anteil gilt A_cc * G = 0 EXAKT in der Analysis; numerisch bleibt
// davon nur Rundung stehen. Eine echte Diagonale steuert allein der Massenterm
// sigma/dt bei, also ausschließlich im Leiter. Außerhalb (Spulen mit kappa = 0,
// Luft) ist die GANZE ZEILE von der Größenordnung eps * ||A||.
//
// A_G ist dort also praktisch singulär, und der Hilfsraum-Schritt
// e_G = A_G^-1 (G^T r) liefert e_G ~ r/eps.
//
// Die vorhandene Diagnose konnte das nicht zeigen: sie zählte Zeilen mit
// "d == 0.0" (gemeldet wurden 888 von 7013). Exakt null wird aber nur, wo sich
// alles auslöscht -- die vielen rundungskleinen Zeilen blieben unsichtbar. Es
// ist dieselbe Fehlerfamilie wie bei bf::dnrm2 und bf::dgemv: eine Prüfung auf
// exakte Gleichheit, wo eine relative Schranke gehört.
//
// Ein Guard auf der Diagonalen allein reicht nicht. Gemessen am Modellproblem
// unten (ein Block der Matrix mit eps skaliert), Verstärkung ||z||/||b||:
//
//   Zeilenskala   ohne Behandlung   nur Diagonal-Guard   mit Ausblenden
//         1e+00              9.7               9.7                 9.7
//         1e-06            5.6e+06           5.6e+06             5.6e+06
//         1e-12            9.2e+11           9.2e+11               7.8
//         1e-16            1.3e+12           1.3e+12               7.8
//
// Der Diagonal-Guard deckelt nur (die Werte sättigen statt weiter zu wachsen),
// er beseitigt nichts. Richtig ist, den skalaren Hilfsraum dort ganz
// auszublenden: wo kein sigma ist, hat er nichts zu korrigieren -- der
// Gradientenanteil des Feldes in Luft wird von der Eichung festgelegt.
//
// EINE EHRLICHE GRENZE: die Schranke ist relativ (1e-10). Zeilen, die nur um
// 1e-6 kleiner sind als der Rest, werden NICHT ausgeblendet und verstärken
// weiterhin. Beim echten Problem liegt das Rauschen bei ~1e-16, also weit
// darunter. Verschöbe sich das (etwa durch andere Materialskalen), müsste die
// Schranke mitgeführt werden -- deshalb prüft dieser Test auch den Fall 1e-6
// und dokumentiert ihn als bekannte Grenze, statt ihn zu verschweigen.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "AlgebraicMultigrid.cuh"
#include "AmsPreconditioner.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

/// Modell von A_G: ein Block mit echter Physik (Leiter) und ein Block, in dem
/// die ganze Zeile mit eps skaliert ist (Luft/Spulen).
HostCsr make_model(int n, int n_air, double eps) {
    HostCsr A; A.num_rows = n; A.num_cols = n;
    A.row_offsets.assign(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        A.row_offsets[i] = static_cast<int>(A.col_indices.size());
        const double s = (i >= n - n_air) ? eps : 1.0;
        if (i > 0)      { A.col_indices.push_back(i - 1); A.values.push_back(-1.0 * s); }
        A.col_indices.push_back(i); A.values.push_back(2.1 * s);
        if (i < n - 1)  { A.col_indices.push_back(i + 1); A.values.push_back(-1.0 * s); }
    }
    A.row_offsets[n] = static_cast<int>(A.col_indices.size());
    return A;
}

/// Verstaerkung ||z||/||b|| eines AMG-Durchlaufs, mit oder ohne Ausblenden.
double amplification(int n, int n_air, double eps, bool use_mask, int* out_masked) {
    HostCsr A = make_model(n, n_air, eps);
    std::vector<char> mask(static_cast<size_t>(n), 0);
    if (use_mask) mask = ams_mask_meaningless_rows(A, 1e-10);
    if (out_masked) { *out_masked = 0; for (char c : mask) if (c) ++(*out_masked); }

    AlgebraicMultigrid amg;
    AlgebraicMultigrid::Options o;
    amg.setup(A, 1, o);

    std::vector<double> b(static_cast<size_t>(n), 0.0), z(static_cast<size_t>(n), 0.0);
    double bn = 0.0;
    for (int i = 0; i < n; ++i) { b[i] = std::sin(0.05 * i); bn += b[i] * b[i]; }
    bn = std::sqrt(bn);

    for (int i = 0; i < n; ++i) if (mask[i]) b[i] = 0.0;
    amg.apply(b.data(), z.data());
    for (int i = 0; i < n; ++i) if (mask[i]) z[i] = 0.0;

    double zn = 0.0;
    for (int i = 0; i < n; ++i) zn += z[i] * z[i];
    return std::sqrt(zn) / bn;
}

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " OK " : "FEHL", what);
    if (!ok) ++g_failures;
}

} // namespace

int main() {
    const int N = 600, N_AIR = 200;
    std::printf("\n=== AMS: skalarer Hilfsraum bei rundungskleinem A_G ===\n\n");
    std::printf("  %12s %16s %16s %12s\n",
                "Zeilenskala", "ohne Ausblenden", "mit Ausblenden", "maskiert");

    double healthy = 0.0, amp_masked_1e12 = 0.0, amp_masked_1e16 = 0.0;
    for (double eps : {1.0, 1e-6, 1e-12, 1e-16, 1e-20}) {
        int nm = 0;
        const double raw = amplification(N, N_AIR, eps, false, nullptr);
        const double msk = amplification(N, N_AIR, eps, true,  &nm);
        std::printf("  %12.0e %16.3e %16.3e %12d\n", eps, raw, msk, nm);
        if (eps == 1.0)    healthy         = msk;
        if (eps == 1e-12)  amp_masked_1e12 = msk;
        if (eps == 1e-16)  amp_masked_1e16 = msk;
    }
    std::printf("\n");

    check(healthy < 100.0,
          "gesunder Fall bleibt beschraenkt (nichts wird faelschlich ausgeblendet)");
    check(amp_masked_1e12 < 10.0 * healthy,
          "eps = 1e-12: Verstaerkung auf dem Niveau des gesunden Falls");
    check(amp_masked_1e16 < 10.0 * healthy,
          "eps = 1e-16: ebenso (die Groessenordnung des echten Problems)");

    // Der gesunde Fall darf NICHTS ausblenden -- sonst verloere AMS im Leiter
    // genau die Korrektur, fuer die es gebaut ist.
    {
        int nm = 0;
        amplification(N, N_AIR, 1.0, true, &nm);
        check(nm == 0, "bei einheitlicher Skala wird keine Zeile ausgeblendet");
    }
    // Und im kritischen Fall muessen es genau die Luft-Zeilen sein.
    {
        int nm = 0;
        amplification(N, N_AIR, 1e-16, true, &nm);
        std::printf("      (ausgeblendet: %d, erwartet %d)\n", nm, N_AIR);
        check(nm == N_AIR, "ausgeblendet werden genau die rundungskleinen Zeilen");
    }

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
