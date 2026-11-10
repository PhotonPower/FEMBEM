// =============================================================================
// T5 (Phase 1b, Schritt 3): Torus (R = 3, r = 1), Kreisstrom I = 1 auf dem Kernkreis im Innern (Radius 3, z = 0).
// Aussenfeld analytisch (Schleifenformel mit elliptischen Integralen, gegen Biot-Savart-Summe geprueft), mu0 = 1.
//   [S0]  Schleifenformel gegen Biot-Savart / Neumann-Summe
//   [S1]  exakte Spuren: c_Meridian (Zirkulation von H auf dem Meridian) = I, c_Laengs = 0
//   [S2]  Zeilen (S2), (S3) mit exakten Spuren: Residuen mit Konvergenzordnung; Gegenprobe c_Meridian := 0
//   [S3]  (psi, c) aus (S2), (S3) bei exaktem a (Aussen-Dirichlet-Neumann-Abbildung): c_Meridian -> I, c_Laengs -> 0
// Reiner Host-Test.
// =============================================================================
#include <Eigen/Dense>
#include <cmath>
#include <cstdio>
#include <vector>

#include "TraceBemOps.h"
#include "TraceMeshes.h"
#include "TraceSystem.h"

using namespace trace;

static int g_fail = 0;
static void check(bool ok, const char* what, double value, double limit)
{
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? "<" : ">=", limit);
    if (!ok) ++g_fail;
}
static void check_ge(bool ok, const char* what, double value, double limit)
{
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? ">=" : "<", limit);
    if (!ok) ++g_fail;
}
static const double PI = 3.14159265358979323846;
static const double MU0 = 1.0, CUR = 1.0, LOOP_R = 3.0;
static double order(double a, double b) { return std::log(a / b) / std::log(2.0); }

// Feld der Kreisschleife (Radius a um die z-Achse, Strom I in +phi-Richtung)
static void loop_fields(const Vec3& x, Vec3& A, Vec3& B)
{
    const double a = LOOP_R, I = CUR;
    const double rho = std::sqrt(x.x * x.x + x.y * x.y), z = x.z;
    const double q2 = (a + rho) * (a + rho) + z * z, d2 = (a - rho) * (a - rho) + z * z;
    const double k2 = 4.0 * a * rho / q2, k = std::sqrt(k2);
    const double K = std::comp_ellint_1(k), E = std::comp_ellint_2(k);
    const double Bz = MU0 * I / (2 * PI) / std::sqrt(q2) * (K + (a * a - rho * rho - z * z) / d2 * E);
    const double Br = MU0 * I / (2 * PI) * z / (rho * std::sqrt(q2)) * (-K + (a * a + rho * rho + z * z) / d2 * E);
    const double Ap = MU0 * I / (PI * k) * std::sqrt(a / rho) * ((1.0 - 0.5 * k2) * K - E);
    const double c = x.x / rho, s = x.y / rho;
    B = v3(Br * c, Br * s, Bz);
    A = v3(-Ap * s, Ap * c, 0.0);
}
static Vec3 loopA(const Vec3& x) { Vec3 A, B; loop_fields(x, A, B); return A; }
static Vec3 loopH(const Vec3& x) { Vec3 A, B; loop_fields(x, A, B); return (1.0 / MU0) * B; }

struct Torus {
    TorusMesh T;
    TraceSurface S;
    BoundaryOps O;
    DerivedOps D;
    std::vector<std::pair<int, int>> meridian, longitudinal;
};

static Torus build(int nu, int nv, double jitter)
{
    Torus t;
    t.T = torus_mesh(3.0, 1.0, nu, nv);
    if (jitter > 0.0) {   // Knoten zufaellig verschieben (bricht die Achsensymmetrie des Netzes), Betrag jitter * h, h = Meridian-Kantenlaenge
        const double h = 2.0 * PI * 1.0 / nv;
        unsigned st = 12345u;
        auto rnd = [&]() { st = st * 1664525u + 1013904223u; return (static_cast<double>(st >> 8) / 16777216.0) - 0.5; };
        for (Vec3& p : t.T.mesh.X) { p.x += 2 * jitter * h * rnd(); p.y += 2 * jitter * h * rnd(); p.z += 2 * jitter * h * rnd(); }
    }
    t.S = build_trace_surface(t.T.mesh.X, t.T.mesh.quads);
    compute_harmonic_fields(t.S);
    std::vector<int> mer, lon;
    // Meridian in Richtung fallender phi: Zirkulation von H = +I fuer den Strom in +phi-Richtung (Rechte-Hand-Regel)
    for (int j = nv - 1; j >= 0; --j) mer.push_back(t.T.node(0, j));
    for (int i = 0; i < nu; ++i) lon.push_back(t.T.node(i, 0));
    t.meridian = cycle_from_nodes(t.S, mer);
    t.longitudinal = cycle_from_nodes(t.S, lon);
    normalize_harmonic_to_cycles(t.S, 0, {t.meridian, t.longitudinal});
    t.O = assemble_boundary_ops(t.S);
    t.D = derive_ops(t.S, t.O);
    return t;
}

int main()
{
    std::printf("[S0] Schleifenformel gegen Biot-Savart-/Neumann-Summe (4000 Segmente)\n");
    {
        const Vec3 pts[] = {v3(2.0, 0.5, 0.8), v3(-3.5, 1.0, -0.7), v3(0.2, 4.1, 1.0)};
        double eb = 0, ea = 0;
        for (const Vec3& x : pts) {
            Vec3 A, B;
            loop_fields(x, A, B);
            const int ns = 4000;
            Vec3 Bs = v3(0, 0, 0), As = v3(0, 0, 0);
            for (int i = 0; i < ns; ++i) {
                const double ph = 2 * PI * (i + 0.5) / ns;
                const Vec3 y = v3(LOOP_R * std::cos(ph), LOOP_R * std::sin(ph), 0.0);
                const Vec3 dl = (2 * PI * LOOP_R / ns) * v3(-std::sin(ph), std::cos(ph), 0.0);
                const Vec3 d = x - y;
                const double r = norm(d);
                Bs = Bs + (MU0 * CUR / (4 * PI * r * r * r)) * cross(dl, d);
                As = As + (MU0 * CUR / (4 * PI * r)) * dl;
            }
            eb = std::max(eb, norm(B - Bs) / norm(Bs));
            ea = std::max(ea, norm(A - As) / norm(As));
        }
        check(eb < 1e-9 && ea < 1e-9, "Schleifenformel (B und A) gegen direkte Summe", std::max(eb, ea), 1e-9);
    }

    for (double jitter : {0.0, 0.15}) {
    std::printf("[S1-S3] Torus 12x6, 24x12, 48x24%s\n", jitter > 0 ? ", Knoten zufaellig verschoben (0.15 h; bricht die Achsensymmetrie)" : "");
    double r2p = 0, r3p = 0, cmp = 0, clp = 0;
    const int sizes[3][2] = {{12, 6}, {24, 12}, {48, 24}};
    for (int k = 0; k < 3; ++k) {
        const Torus t = build(sizes[k][0], sizes[k][1], jitter);
        const TraceSurface& S = t.S;
        const int E = S.num_edges(), V = S.num_nodes();
        // exakte Spuren: a_e = int_e A.dl, c_e = int_e H.dl (Koeffizienten von w = H_t = sum c_e W_e), flux = R a
        const auto av = edge_line_integrals(S, loopA), cv = edge_line_integrals(S, loopH);
        const Vec a = Eigen::Map<const Vec>(av.data(), E), c = Eigen::Map<const Vec>(cv.data(), E);
        std::vector<double> ca(c.data(), c.data() + E);
        const double circ_mer = cycle_circulation(t.meridian, c.data()), circ_lon = cycle_circulation(t.longitudinal, c.data());
        std::printf("  Torus %dx%d (%d Quads):\n", sizes[k][0], sizes[k][1], S.num_quads());
        std::printf("    [S1] Zirkulation von H: Meridian %.12f (I = %.1f), Laengs %.3e\n", circ_mer, CUR, circ_lon);
        check(std::fabs(circ_mer - CUR) < 1e-8 && std::fabs(circ_lon) < 1e-8, "c_Meridian = I, c_Laengs = 0 fuer die exakten Spuren (Ampere, Quadraturgenauigkeit)", std::fabs(circ_mer - CUR) + std::fabs(circ_lon), 1e-8);
        // Zerlegung c = -G psi + H ch + Rest; Rest = Verletzung von R c = 0 (diskreter Fehler)
        const Mat HtH = t.D.H.transpose() * t.D.H;
        const Vec ch = HtH.ldlt().solve(t.D.H.transpose() * c);
        const Mat GtG = t.D.G.transpose() * t.D.G;
        const Vec psi = (GtG + Mat::Ones(V, V) / V).ldlt().solve(-t.D.G.transpose() * (c - t.D.H * ch));   // Konstante auf Mittelwert 0 fixiert
        const double rest = (c - t.D.H * ch + t.D.G * psi).norm() / c.norm();
        const double Rc = (t.D.R * c).norm() / c.norm();
        std::printf("         Zerlegung c = -G psi + H c_h: c_h = (%.10f, %.3e), |R c|/|c| = %.2e, Rest/|c| = %.2e\n", ch[0], ch[1], Rc, rest);

        // (S2), (S3) mit exakten Spuren
        const Vec flux = t.D.R * a;
        const Vec Wpsi = t.D.W * psi;
        const Vec row2 = t.D.N.transpose() * flux - MU0 * (Wpsi + t.D.Z * ch);
        const Vec row3 = t.D.Q.transpose() * a - MU0 * (t.D.Z.transpose() * psi + t.D.Y * ch);
        const double s2 = (MU0 * (Wpsi + t.D.Z * ch)).norm(), s3 = std::max(MU0 * (t.D.Z.transpose() * psi + t.D.Y * ch).norm(), t.D.Q.transpose().operator*(a).norm());
        const double r2 = row2.norm() / s2, r3 = row3.norm() / s3;
        std::printf("    [S2] Residuum (S2) %.3e, (S3) %.3e", r2, r3);
        if (r2p > 0) std::printf("   Ordnung (S2) %.2f, (S3) %.2f", order(r2p, r2), order(r3p, r3));
        std::printf("\n");
        // Gegenprobe c_Meridian := 0
        Vec ch0 = ch; ch0[0] = 0.0;
        const Vec row2b = t.D.N.transpose() * flux - MU0 * (Wpsi + t.D.Z * ch0);
        const Vec row3b = t.D.Q.transpose() * a - MU0 * (t.D.Z.transpose() * psi + t.D.Y * ch0);
        const double r2b = row2b.norm() / s2, r3b = row3b.norm() / s3;
        std::printf("         Gegenprobe c_Meridian := 0: (S2) %.3e, (S3) %.3e\n", r2b, r3b);
        check(r3b > 10 * r3 && r2b > 10 * r2, "Gegenprobe: c_Meridian = 0 erzeugt ein um >= Faktor 10 groesseres Residuum", std::min(r3b / r3, r2b / r2), 10.0);
        // Gegenprobe: Q-Block mit falschem Vorzeichen des Crot-Anteils
        const Mat Qwrong = (0.5 * as_matrix(t.O.Mrot, E, E) - as_matrix(t.O.Crot, E, E)) * t.D.H;
        const double r3w = (Qwrong.transpose() * a - MU0 * (t.D.Z.transpose() * psi + t.D.Y * ch)).norm() / s3;
        std::printf("         Gegenprobe Vorzeichen von C in Q: (S3) %.3e\n", r3w);
        check(r3w > 10 * r3, "Gegenprobe: falsches Vorzeichen von C in Q", r3w / r3, 10.0);

        // [S3] Loesung des Randblocks bei exaktem a: (psi, c) aus (S2), (S3)
        //   [ -mu0 W  -mu0 Z ] [psi]   [ -N^T R a ]
        //   [ -mu0 Z^T -mu0 Y ] [c  ] = [ -Q^T a  ]   (psi nur modulo Konstanten, per Pseudoinverse mit Mittelwert 0)
        const int nH = S.nH;
        Mat B(V + nH, V + nH);
        B.topLeftCorner(V, V) = MU0 * t.D.W + Mat::Ones(V, V) / V;    // Konstante festlegen (W 1 = 0)
        B.topRightCorner(V, nH) = MU0 * t.D.Z;
        B.bottomLeftCorner(nH, V) = MU0 * t.D.Z.transpose();
        B.bottomRightCorner(nH, nH) = MU0 * t.D.Y;
        Vec rhs(V + nH);
        rhs.head(V) = t.D.N.transpose() * flux;
        rhs.tail(nH) = t.D.Q.transpose() * a;
        const Vec sol = B.partialPivLu().solve(rhs);
        const Vec ch_s = sol.tail(nH);
        const double em = std::fabs(ch_s[0] - CUR), el = std::fabs(ch_s[1]);
        std::printf("    [S3] c aus (S2),(S3) bei exaktem a: c_Meridian = %.8f (Fehler %.3e), c_Laengs = %.3e", ch_s[0], em, el);
        if (cmp > 0) std::printf("   Ordnung c_Meridian %.2f, c_Laengs %.2f", order(cmp, em), order(clp, el));
        std::printf("\n");
        // Gegenprobe: ohne (S3)-Zeilen (nur S2 mit c = 0) ist psi falsch -- hier: Loesung ohne harmonische Spalten
        Mat B0 = B.topLeftCorner(V, V);
        const Vec psi0 = B0.partialPivLu().solve(rhs.head(V));
        std::printf("         Gegenprobe ohne harmonische Spalten: |psi - psi_exakt|/|psi_exakt| = %.3e (mit Spalten: %.3e)\n",
                    (psi0 - psi).norm() / psi.norm(), (sol.head(V) - psi).norm() / psi.norm());
        check((psi0 - psi).norm() > 5 * (sol.head(V) - psi).norm(), "Gegenprobe: ohne harmonische Spalten ist psi um >= Faktor 5 schlechter", (psi0 - psi).norm() / (sol.head(V) - psi).norm(), 5.0);
        if (k == 2) {
            check(em < 5e-3 && el < (jitter > 0 ? 1e-3 : 5e-3), "feinstes Netz: c_Meridian = I, c_Laengs = 0 aus (S2),(S3)", std::max(em, el), 5e-3);
            check_ge(order(cmp, em) >= 1.5, "c_Meridian: Konvergenzordnung >= 1.5", order(cmp, em), 1.5);
            // verschobene Knoten: die Flaeche ist keine Verfeinerung einer glatten (Normalen weichen um O(0.15) ab); (S2) konvergiert dort langsamer (gemessen ~0.8)
            check(r2 < (jitter > 0 ? 0.1 : 0.05) && r3 < 0.05, "Residuen (S2),(S3) klein", std::max(r2, r3), jitter > 0 ? 0.1 : 0.05);
            if (jitter == 0.0) check_ge(order(r2p, r2) >= 1.5 && order(r3p, r3) >= 1.5, "Residuen: Konvergenzordnung >= 1.5", std::min(order(r2p, r2), order(r3p, r3)), 1.5);
            else check_ge(order(r3p, r3) >= 1.5, "Residuum (S3): Konvergenzordnung >= 1.5", order(r3p, r3), 1.5);
        }
        r2p = r2; r3p = r3; cmp = em; clp = std::max(el, 1e-300);
    }
    }
    std::printf(g_fail == 0 ? "ALLE TESTS BESTANDEN\n" : "FEHLGESCHLAGEN: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
