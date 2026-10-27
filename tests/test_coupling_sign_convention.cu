// =============================================================================
// Abnahmeprüfung: stimmen die Vorzeichen der Kopplungs-Massenmatrix [T]?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_coupling_sign_convention
//   ./build/test_coupling_sign_convention               (Linux)
//   .\build\Release\test_coupling_sign_convention.exe   (Windows)
//
// Reiner Host-Code, keine GPU, kein Netz, keine Datei. Der Test ruft die
// __host__ __device__ Kernelrümpfe aus include/CouplingKernels.cuh direkt auf.
//
// HINTERGRUND
// -----------
// Der gekoppelte Operator (CoupledMVOperator::apply_physical) hat die Gestalt
//
//     ( A_sys           -T      ) (A)   (RHS)
//     ( -0.5*T - D_BEM  +mu_0*G ) (K) = ( 0 )
//
// Der blocktriangulare Vorkonditionierer (--blocktri) löst dazu
//
//     P = ( A~   -T )   =>   A~ z_A = r_A - (-T) z_K = r_A + T z_K
//         (  0   S~ )
//
// Der Operator braucht also -T, der Vorkonditionierer +T -- und zwar mit
// EXAKT derselben Matrix T, inklusive derselben Vorzeichenbehandlung der
// Kantenorientierung (sign[r]*sign[c]*T_loc[r][c], einmal angewendet).
//
// Sind die beiden nicht exakt entgegengesetzt, ist der Vorkonditionierer
// stillschweigend falsch: er konvergiert dann nicht schlechter mit einer
// Fehlermeldung, sondern verschlechtert einfach die Konvergenz -- genau die
// Sorte Fehler, die man beim Debuggen der Konvergenz zuletzt verdächtigt.
// Deshalb wird die Beziehung hier als Invariante festgehalten:
//
//     apply_T_matrix_kernel_body(x_A = 0, x_K)|_y_A
//   + apply_T_to_A_kernel_body(x_K)|_y_A            == 0   für jedes x_K.
//
// Geprüft wird zusätzlich, dass
//   (1) T tatsächlich nur über y_A wirkt, wenn x_A = 0 ist (der K-Block darf
//       dann keinen Beitrag bekommen -- sonst wäre die Blockstruktur anders
//       als der Vorkonditionierer annimmt),
//   (2) die Orientierungsvorzeichen konsistent durchschlagen: dreht man ALLE
//       sign[] um, ändert sich nichts (sign[r]*sign[c] ist invariant),
//   (3) T symmetrisch ist -- der Rand-Massenmatrix-Anteil muss es sein, und
//       genau deshalb ist die verbleibende Unsymmetrie der Kopplung (-T gegen
//       -0.5*T) eine Eigenschaft der FORMULIERUNG und kein Assemblierfehler.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "CouplingKernels.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " OK " : "FEHL", what);
    if (!ok) ++g_failures;
}

// Zwei leicht verzerrte Quads, die sich eine Kante teilen -- die Verzerrung
// stellt sicher, dass T_loc nicht zufällig eine Vielfache der Einheitsmatrix
// ist, das gemeinsame Paar, dass der atomare Scatter nach y_A wirklich
// akkumuliert.
struct Fixture {
    static constexpr int num_faces = 2;
    static constexpr int num_bem_dofs = num_faces * 4;
    static constexpr int num_fem_dofs = 7;   // eine Kante wird geteilt

    std::vector<double3> nodes;
    std::vector<Mesh::QuadFace> faces;
    std::vector<int> map;
    std::vector<int> sign;

    Fixture() {
        nodes = {
            {0.00, 0.00, 0.00}, {1.00, 0.05, 0.00},
            {1.10, 1.00, 0.10}, {0.05, 0.95, 0.00},
            {2.10, 0.10, 0.05}, {2.00, 1.05, 0.00}
        };
        faces.resize(num_faces);
        faces[0].v[0] = 0; faces[0].v[1] = 1; faces[0].v[2] = 2; faces[0].v[3] = 3;
        faces[1].v[0] = 1; faces[1].v[1] = 4; faces[1].v[2] = 5; faces[1].v[3] = 2;

        // Quad 0 belegt die FEM-Kanten 0..3, Quad 1 die Kanten 3..6 -- Kante 3
        // ist die gemeinsame. Vorzeichen bewusst gemischt.
        map  = {0, 1, 2, 3,   3, 4, 5, 6};
        sign = {1, -1, 1, 1,  -1, 1, 1, -1};
    }
};

// y_A aus dem Operator-Rumpf, mit x_A = 0.
std::vector<double> operator_yA(const Fixture& f, const std::vector<double>& x_K,
                                const std::vector<int>& sign) {
    std::vector<double> x_A(Fixture::num_fem_dofs, 0.0);
    std::vector<double> y_A(Fixture::num_fem_dofs, 0.0);
    std::vector<double> y_K(Fixture::num_bem_dofs, 0.0);
    for (int i = 0; i < Fixture::num_faces; ++i) {
        apply_T_matrix_kernel_body(i, Fixture::num_faces, f.faces.data(), f.nodes.data(),
                                   f.map.data(), sign.data(),
                                   x_A.data(), x_K.data(), y_A.data(), y_K.data());
    }
    return y_A;
}

// y_K aus dem Operator-Rumpf, mit x_A = 0.
std::vector<double> operator_yK(const Fixture& f, const std::vector<double>& x_K) {
    std::vector<double> x_A(Fixture::num_fem_dofs, 0.0);
    std::vector<double> y_A(Fixture::num_fem_dofs, 0.0);
    std::vector<double> y_K(Fixture::num_bem_dofs, 0.0);
    for (int i = 0; i < Fixture::num_faces; ++i) {
        apply_T_matrix_kernel_body(i, Fixture::num_faces, f.faces.data(), f.nodes.data(),
                                   f.map.data(), f.sign.data(),
                                   x_A.data(), x_K.data(), y_A.data(), y_K.data());
    }
    return y_K;
}

// y_A aus dem Vorkonditionierer-Rumpf.
std::vector<double> precond_yA(const Fixture& f, const std::vector<double>& x_K,
                               const std::vector<int>& sign) {
    std::vector<double> y_A(Fixture::num_fem_dofs, 0.0);
    for (int i = 0; i < Fixture::num_faces; ++i) {
        apply_T_to_A_kernel_body(i, Fixture::num_faces, f.faces.data(), f.nodes.data(),
                                 f.map.data(), sign.data(), x_K.data(), y_A.data());
    }
    return y_A;
}

double max_abs(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::fmax(m, std::fabs(x));
    return m;
}

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
    return m;
}

} // namespace

int main() {
    std::printf("\n=== Vorzeichenkonvention der Kopplungs-Massenmatrix [T] ===\n\n");

    Fixture f;

    // Mehrere Testvektoren, damit die Invariante nicht zufaellig fuer einen
    // einzelnen Vektor gilt (z.B. weil er im Kern von T liegt).
    std::vector<std::vector<double>> probes = {
        {1, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1, 1},
        {0.3, -1.7, 2.9, 0.11, -0.5, 4.2, -3.1, 0.07}
    };

    double worst_sum = 0.0;
    double worst_scale = 0.0;
    for (const auto& x_K : probes) {
        const std::vector<double> op = operator_yA(f, x_K, f.sign);
        const std::vector<double> pc = precond_yA(f, x_K, f.sign);

        std::vector<double> sum(Fixture::num_fem_dofs);
        for (int i = 0; i < Fixture::num_fem_dofs; ++i) sum[i] = op[i] + pc[i];

        worst_sum   = std::fmax(worst_sum, max_abs(sum));
        worst_scale = std::fmax(worst_scale, max_abs(pc));
    }
    std::printf("  groesster Beitrag |T*x_K|            : %.6e\n", worst_scale);
    std::printf("  groesstes |Operator + Vorkond.|      : %.6e\n\n", worst_sum);

    check(worst_scale > 1e-12,
          "T wirkt ueberhaupt (Testvektoren liegen nicht im Kern)");
    check(worst_sum <= 1e-14 * std::fmax(worst_scale, 1.0),
          "Operator (-T) und Vorkonditionierer (+T) sind exakt entgegengesetzt");

    // (1) Mit x_A = 0 darf der K-Block nichts abbekommen -- der 0.5*T-Anteil
    //     haengt ausschliesslich an x_A.
    {
        const std::vector<double> y_K = operator_yK(f, probes.back());
        check(max_abs(y_K) == 0.0,
              "x_A = 0 erzeugt keinen Beitrag im K-Block (Blockstruktur wie angenommen)");
    }

    // (2) Globale Vorzeichenumkehr aller Kantenorientierungen darf nichts
    //     aendern: der Faktor ist sign[r]*sign[c], nicht sign[r].
    {
        std::vector<int> flipped = f.sign;
        for (int& s : flipped) s = -s;

        const std::vector<double> a = precond_yA(f, probes.back(), f.sign);
        const std::vector<double> b = precond_yA(f, probes.back(), flipped);
        check(max_abs_diff(a, b) <= 1e-14 * std::fmax(max_abs(a), 1.0),
              "sign[r]*sign[c] wird EINMAL angewendet (globale Umkehr ist wirkungslos)");
    }

    // (3) T selbst ist symmetrisch -- die Unsymmetrie der Kopplung stammt
    //     ausschliesslich aus dem Faktor 1/2 der K-Gleichung, nicht aus T.
    {
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < Fixture::num_faces; ++i) {
            double3 q[4];
            for (int k = 0; k < 4; ++k) q[k] = f.nodes[f.faces[i].v[k]];
            double T_loc[4][4];
            integrate_boundary_mass_matrix(q, T_loc);
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) {
                    worst = std::fmax(worst, std::fabs(T_loc[r][c] - T_loc[c][r]));
                    scale = std::fmax(scale, std::fabs(T_loc[r][c]));
                }
            }
        }
        std::printf("  groesste Abweichung |T_rc - T_cr|   : %.6e (Skala %.6e)\n\n",
                    worst, scale);
        check(worst <= 1e-14 * std::fmax(scale, 1.0),
              "die lokale Rand-Massenmatrix T_loc ist symmetrisch");
    }

    std::printf("\n");
    if (g_failures == 0) {
        std::printf("Alle Pruefungen bestanden.\n\n");
        return 0;
    }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
