// =============================================================================
// Abnahmeprüfung: verhält sich der blocktriangulare Vorkonditionierer korrekt?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_block_triangular_consistency
//   ./build/test_block_triangular_consistency
//
// Läuft im CPU-Modus, braucht keine GPU und kein Netz -- der Operator wird mit
// einer kleinen synthetischen FEM-Matrix, zwei Rand-Quads, einem dichten
// G_BEM-Block, Eichung UND Kohomologie-Projektoren aufgebaut.
//
// Geprüfte Invarianten
// --------------------
// (1) r_K == 0  =>  blocktriangular == blockdiagonal, EXAKT.
//     Denn dann ist z_K = S~^-1 * 0 = 0 und damit T*z_K = 0; der einzige
//     Unterschied zwischen den Zweigen verschwindet. Diese Invariante ist
//     nicht akademisch: sie ist genau die Lage in GMRES-Iteration 0, weil die
//     rechte Seite des K-Blocks konstruktionsbedingt null ist (siehe fem.cu).
//
//     Diese Prüfung hat einen echten Fehler gefunden. Eine frühere Fassung
//     projizierte die Eichung auf die SUMME r_A + T*z_K statt nur auf den
//     Korrekturterm. Mathematisch ist das dasselbe (P ist linear und r_A ist
//     bereits geeicht), numerisch nicht: die Kohomologie-Anteile von r_A
//     wurden ein zweites Mal herausgerechnet. Ergebnis war ||z_A|| = 1.3956
//     statt 1.6609 -- ein stiller Faktor 0.84 auf dem FEM-Block, ohne jede
//     Fehlermeldung. Ohne Kohomologie-Projektoren im Testaufbau wäre er
//     unsichtbar geblieben.
//
// (2) r_K != 0  =>  z_K ist endlich, ungleich null und von plausibler
//     Grössenordnung (im äquilibrierten Raum hat S~ die Diagonale 1, also
//     ||z_K|| ~ ||r_K||).
//
// (3) Die Ausgabe ist in beiden Zweigen durchgehend endlich.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "CoupledSolver.cuh"
#include "Mesh.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

namespace bf { extern Backend g_backend; }

static double nrm(const double* v, int n) {
    double s = 0; for (int i = 0; i < n; ++i) s += v[i]*v[i]; return std::sqrt(s);
}
static bool has_nan(const double* v, int n) {
    for (int i = 0; i < n; ++i) if (std::isnan(v[i]) || std::isinf(v[i])) return true;
    return false;
}

int main() {
    bf::g_backend = bf::Backend::CPU;

    const int NF = 7;          // FEM-DOFs
    const int NFACES = 2;
    const int NB = NFACES * 4; // BEM-DOFs

    // --- FEM-Matrix: tridiagonal, diagonaldominant ---------------------------
    std::vector<int> row(NF + 1, 0), col;
    std::vector<double> val;
    for (int i = 0; i < NF; ++i) {
        if (i > 0)      { col.push_back(i - 1); val.push_back(-1.0); }
        col.push_back(i); val.push_back(4.0 + 0.1 * i);
        if (i < NF - 1) { col.push_back(i + 1); val.push_back(-1.0); }
        row[i + 1] = (int)col.size();
    }

    CoupledMVOperator op(NF);
    op.setup_fem_matrix(NF, (int)val.size(), row.data(), col.data(), val.data());

    // --- BEM-Kopplung --------------------------------------------------------
    std::vector<int> map  = {0,1,2,3, 3,4,5,6};
    std::vector<int> sgn  = {1,-1,1,1, -1,1,1,-1};
    op.setup_bem_coupling(NB, map.data(), sgn.data());

    std::vector<double3> nodes = {
        {0.00,0.00,0.00},{1.00,0.05,0.00},{1.10,1.00,0.10},
        {0.05,0.95,0.00},{2.10,0.10,0.05},{2.00,1.05,0.00}
    };
    std::vector<Mesh::QuadFace> faces(NFACES);
    faces[0].v[0]=0; faces[0].v[1]=1; faces[0].v[2]=2; faces[0].v[3]=3;
    faces[1].v[0]=1; faces[1].v[1]=4; faces[1].v[2]=5; faces[1].v[3]=2;
    op.setup_bem_mesh(NFACES, faces.data(), nodes.data());

    // --- G_BEM als ein dichter SPD-Block (Column-Major) ----------------------
    std::vector<double> G(NB * NB, 0.0);
    for (int i = 0; i < NB; ++i)
        for (int j = 0; j < NB; ++j)
            G[j * NB + i] = (i == j) ? 8.0e8 : 1.0e8 / (1.0 + std::abs(i - j));
    op.add_dense_block(NB, NB, 0, 0, G.data());

    // --- Eichung: DOF 2 und 5 geeicht ---------------------------------------
    std::vector<char> gauged(NF, 0);
    gauged[2] = 1; gauged[5] = 1;
    op.setup_gauge(gauged.data());

    // Zwei Kohomologie-Projektoren wie im echten Lauf (auf Baumkoordinaten null)
    std::vector<double> c0(NF, 0.0), c1(NF, 0.0);
    c0[0] = 1.0; c0[1] = -0.5; c0[3] = 0.7;
    c1[4] = 1.0; c1[6] = 0.3;
    double n0 = 0, n1 = 0;
    for (int i = 0; i < NF; ++i) { n0 += c0[i]*c0[i]; n1 += c1[i]*c1[i]; }
    op.add_cohomology_projector(c0.data(), n0);
    op.add_cohomology_projector(c1.data(), n1);

    op.setup_equilibration();
    op.compute_ilu();

    const int N = op.get_total_dofs();
    std::printf("\nget_total_dofs = %d (FEM %d + BEM %d)\n\n", N, NF, NB);

    // Ein Krylov-artiger Eingangsvektor: FEM-Anteil nichttrivial, K-Anteil 0
    // (genau die Lage in GMRES-Iteration 0, wo b_K == 0 ist).
    std::vector<double> in(N, 0.0), out_diag(N, 0.0), out_tri(N, 0.0);
    for (int i = 0; i < NF; ++i) in[i] = 0.3 * (i + 1) - 1.1;
    in[2] = 0.0; in[5] = 0.0;   // geeichte Eintraege

    op.apply_preconditioner(in.data(), out_diag.data());
    std::printf("blockdiagonal   : ||z|| = %.6e  ||z_A|| = %.6e  ||z_K|| = %.6e  %s\n",
                nrm(out_diag.data(), N), nrm(out_diag.data(), NF),
                nrm(out_diag.data() + NF, NB),
                has_nan(out_diag.data(), N) ? "<<< NaN/Inf" : "");

    op.set_block_triangular_preconditioner(true, 10, 1e-2);
    // INVARIANTE: bei r_K == 0 muss der blocktriangulaere Zweig exakt dasselbe
    // liefern wie der blockdiagonale -- T*z_K ist dann null.
    op.apply_preconditioner(in.data(), out_tri.data());
    std::printf("blocktriangulaer: ||z|| = %.6e  ||z_A|| = %.6e  ||z_K|| = %.6e  %s\n",
                nrm(out_tri.data(), N), nrm(out_tri.data(), NF),
                nrm(out_tri.data() + NF, NB),
                has_nan(out_tri.data(), N) ? "<<< NaN/Inf" : "");

    int failures = 0;
    {
        double d = 0;
        for (int i = 0; i < N; ++i) d = std::fmax(d, std::fabs(out_tri[i] - out_diag[i]));
        const bool ok = d <= 1e-14 * nrm(out_diag.data(), N);
        std::printf("\n  [%s] r_K = 0 => blocktriangulaer == blockdiagonal "
                    "(max. Abweichung %.3e)\n", ok ? " OK " : "FEHL", d);
        if (!ok) ++failures;
        const bool nz = nrm(out_diag.data(), NF) > 1e-12;
        std::printf("  [%s] der Vorkonditionierer liefert ueberhaupt etwas\n",
                    nz ? " OK " : "FEHL");
        if (!nz) ++failures;
        if (has_nan(out_diag.data(), N) || has_nan(out_tri.data(), N)) {
            std::printf("  [FEHL] Ausgabe enthaelt NaN/Inf\n"); ++failures;
        } else {
            std::printf("  [ OK ] Ausgabe ist durchgehend endlich\n");
        }
    }

    // Zweiter Test: K-Anteil ungleich 0 (spaetere Krylov-Vektoren)
    for (int i = 0; i < NB; ++i) in[NF + i] = 0.2 * ((i % 3) - 1);
    std::fill(out_tri.begin(), out_tri.end(), 0.0);
    op.apply_preconditioner(in.data(), out_tri.data());
    std::printf("\nmit r_K != 0    : ||z|| = %.6e  ||z_A|| = %.6e  ||z_K|| = %.6e  %s\n\n",
                nrm(out_tri.data(), N), nrm(out_tri.data(), NF),
                nrm(out_tri.data() + NF, NB),
                has_nan(out_tri.data(), N) ? "<<< NaN/Inf" : "");
    {
        const double rK = nrm(in.data() + NF, NB);
        const double zK = nrm(out_tri.data() + NF, NB);
        const bool ok = zK > 1e-12 && std::isfinite(zK) && zK < 1e3 * rK;
        std::printf("  [%s] r_K != 0 => z_K endlich, ungleich null, "
                    "plausible Groessenordnung (||z_K||/||r_K|| = %.3f)\n",
                    ok ? " OK " : "FEHL", zK / rK);
        if (!ok) ++failures;
        if (has_nan(out_tri.data(), N)) { std::printf("  [FEHL] NaN/Inf\n"); ++failures; }
    }

    // -----------------------------------------------------------------------
    // Rueckfall auf die Identitaet, wenn das innere CG nichts erreicht
    // -----------------------------------------------------------------------
    //
    // Diese Pruefung hat einen echten Fehler gefunden. Brach das innere CG in
    // Iteration 0 ab, blieb z_K ~ 1e-17 stehen. Das ist NICHT "so gut wie
    // nichts tun", sondern deutlich schlechter: der blockdiagonale Zweig setzt
    // z_K = r_K. Mit z_K = 0 bekommt der K-Block ueberhaupt keine
    // Vorkonditionierung, und der aeussere GMRES stagnierte am TEAM-28-Problem
    // bei Rel. Res. 0.996 statt bei 0.13.
    //
    // Erzwungen wird der Abbruch hier mit einem INDEFINITEN G-Block: dann ist
    // p^T S~ p <= 0 schon im ersten Schritt, und solve_schur_block muss
    // aussteigen.
    {
        std::vector<double> G_indef(static_cast<size_t>(NB) * NB, 0.0);
        for (int i = 0; i < NB; ++i)
            for (int j = 0; j < NB; ++j)
                // Negativ definit: dann ist p^T S~ p < 0 bereits im ersten
                // Schritt, und solve_schur_block muss aussteigen. Ein bloss
                // alternierendes Vorzeichen genuegt nicht -- p^T S~ p kann dann
                // trotzdem positiv sein (nachgemessen).
                G_indef[static_cast<size_t>(j) * NB + i] =
                    (i == j) ? -8.0e8 : -1.0e8 / (1.0 + std::abs(i - j));

        CoupledMVOperator op3(NF);
        op3.setup_fem_matrix(NF, (int)val.size(), row.data(), col.data(), val.data());
        op3.setup_bem_coupling(NB, map.data(), sgn.data());
        op3.setup_bem_mesh(NFACES, faces.data(), nodes.data());
        op3.add_dense_block(NB, NB, 0, 0, G_indef.data());
        op3.setup_gauge(gauged.data());
        op3.setup_equilibration();
        op3.compute_ilu();
        op3.set_block_triangular_preconditioner(true, 10, 1e-2);

        std::vector<double> in3(N, 0.0), out3(N, 0.0);
        for (int i = 0; i < NF; ++i) in3[i] = 0.3 * (i + 1) - 1.1;
        in3[2] = 0.0; in3[5] = 0.0;
        for (int i = 0; i < NB; ++i) in3[NF + i] = 0.2 * ((i % 3) - 1);

        op3.apply_preconditioner(in3.data(), out3.data());

        double d = 0.0, sc = 0.0;
        for (int i = 0; i < NB; ++i) {
            d  = std::fmax(d,  std::fabs(out3[NF + i] - in3[NF + i]));
            sc = std::fmax(sc, std::fabs(in3[NF + i]));
        }
        const bool ok = d <= 1e-14 * std::fmax(sc, 1.0);
        std::printf("  [%s] indefinites S~ => z_K = r_K (Identitaet), "
                    "max. Abweichung %.3e\n", ok ? " OK " : "FEHL", d);
        if (!ok) ++failures;
    }

    std::printf("\n");
    if (failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", failures);
    return 1;
}
