// =============================================================================
// Abnahmeprüfung: wirken die Vorkonditionierer am ÄUSSEREN Löser?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_preconditioner_end_to_end
//   ./build/test_preconditioner_end_to_end
//
// Reiner Host-Code, keine GPU, kein Netz von der Platte. Läuft in Sekunden.
//
// WARUM ES DIESEN TEST GIBT
// -------------------------
// Alle bisherigen Prüfungen messen Teilgrößen: den Konvergenzfaktor des AMG
// auf seiner eigenen Hilfsmatrix, die Symmetrie von G_BEM, das Vorzeichen von
// T. Keine davon beantwortet die Frage, auf die es ankommt -- ob der
// Vorkonditionierer den äußeren Löser schneller macht.
//
// Das ist nicht theoretisch. Beim Ersetzen von Pi^T A Pi durch den
// Alpha-Poisson-Operator fiel der AMG-Konvergenzfaktor des vektoriellen
// Hilfsraums von 0.93 auf 0.74, die Hierarchie wurde gesund
// (Operatorkomplexität 1.09 -> 2.40) -- und der äußere GMRES stagnierte
// danach bei Rel. Res. 0.99 statt bei 0.025. Eine deutliche Verschlechterung,
// die keine der bestehenden Prüfungen sehen konnte, weil der Operator gegen
// sich selbst geprüft wurde.
//
// Der Grund war die relative Skalierung: AMS ist eine SUMME aus
// Kantenglätter, skalarem und vektoriellem Hilfsraum. Ein einzelner Summand,
// der um Größenordnungen verstellt ist, macht das Ganze unbrauchbar -- eine
// EINHEITLICHE Skalierung des gesamten Vorkonditionierers wäre dagegen
// wirkungslos, GMRES ist dagegen invariant.
//
// WAS HIER GEMESSEN WIRD
// ----------------------
// Ein 4x4x4-Hexaedernetz, untere Hälfte Leiter (kappa > 0), obere Hälfte Luft
// (kappa = 0) -- also mit derselben strukturellen Eigenheit wie das echte
// Problem: der Curl-Curl-Operator ist in der Luft singulär, weil dort der
// Massenterm fehlt. Darauf ein Nédélec-Kantensystem
//
//     A = nu * K_curl + (kappa/dt) * M
//
// und die Zahl der GMRES-Iterationen bis 1e-8, ohne und mit ILU(0).
//
// Die Prüfung ist bewusst grob: ILU muss BESSER sein als gar kein
// Vorkonditionierer. Ein Test, der eine Verschlechterung um Faktor 40 fängt,
// muss keine zehn Prozent auflösen -- und eine enge Schranke würde bei jeder
// harmlosen Änderung fehlschlagen und dann abgeschaltet.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "Backend.cuh"
#include "CoupledSolver.cuh"
#include "NedelecHexahedron.cuh"
#include "Mesh.cuh"

#include <cmath>
#include <cstdio>
#include <array>
#include <algorithm>
#include <map>
#include <random>
#include <vector>

namespace bf { extern Backend g_backend; }

namespace {

int g_failures = 0;

constexpr double MU0   = 4.0e-7 * M_PI;
constexpr double NU    = 1.0 / MU0;
constexpr double KAPPA = 3.4e7;
constexpr double DT    = 1.0e-4;

// Die lokale Kantennummerierung MUSS zu evaluate_nedelec_basis passen (siehe
// den Kommentar dort). Wird sie hier anders gewählt, misst der Test etwas
// anderes als der Löser rechnet.
const int LOCAL_EDGE[12][2] = {
    {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
};

struct Grid {
    int n;                                  // Zellen je Richtung
    std::vector<double3> nodes;
    std::vector<Hexahedron> hexes;
    std::vector<std::array<int,12>> hex_edges;
    std::vector<std::array<int,12>> hex_signs;
    int num_edges = 0;
    std::vector<int2> edge_nodes;           // (min, max) je globaler Kante
};

/// Fuellt ein Mesh-Objekt so, wie AmsPreconditioner::setup() es erwartet.
/// Ohne das laesst sich AMS im Test nicht aufsetzen -- und ohne AMS misst der
/// Test genau den Vorkonditionierer nicht, um dessentwillen er gebaut wurde.
void fill_mesh(const Grid& g, Mesh& mesh) {
    mesh.num_nodes = g.nodes.size();
    mesh.num_hexes = g.hexes.size();
    mesh.num_edges = static_cast<size_t>(g.num_edges);

    mesh.nodes_initial.allocate(mesh.num_nodes);
    mesh.nodes_current.allocate(mesh.num_nodes);
    for (size_t i = 0; i < mesh.num_nodes; ++i) {
        mesh.nodes_initial.data()[i] = g.nodes[i];
        mesh.nodes_current.data()[i] = g.nodes[i];
    }
    mesh.hex_nodes.allocate(mesh.num_hexes);
    for (size_t i = 0; i < mesh.num_hexes; ++i) mesh.hex_nodes.data()[i] = g.hexes[i];

    mesh.edges.allocate(mesh.num_edges);
    for (int e = 0; e < g.num_edges; ++e) mesh.edges.data()[e] = g.edge_nodes[e];
}

/// Zell-Seitenverhaeltnis des Testnetzes.
///
/// Das Netz war zuerst ein Wuerfel, also isotrop. Damit hat es die
/// entscheidende Eigenheit des echten Problems nicht abgebildet: die
/// Aluminiumplatte im TEAM-28-Netz ist 2 mm dick bei 126 mm Durchmesser und
/// hat ein Zell-Seitenverhaeltnis von 4.5 im Median.
///
/// Das ist keine Kleinigkeit. Der Wechsel des Staerkemaszstabs in der
/// Aggregation (diagonal -> zeilenrelativ) verbessert den AMG-Faktor bei 4.5
/// von 0.774 auf 0.439, kostet im isotropen Fall aber leicht (0.237 -> 0.273).
/// Auf dem Wuerfelnetz sah die Aenderung deshalb wie eine Verschlechterung aus
/// (28 -> 32 Iterationen), waehrend sie am echten Problem der Gewinn ist.
///
/// Ein Testnetz, das die Eigenheit nicht hat, um die es geht, misst die
/// falsche Sache -- und haette hier fast zu einer zurueckgenommenen
/// Verbesserung gefuehrt.
constexpr double CELL_ASPECT = 4.5;

Grid build_grid(int n) {
    Grid g; g.n = n;
    const int np = n + 1;
    const double zext = 1.0 / CELL_ASPECT;      // duenn in z, wie die Platte
    auto nid = [&](int i,int j,int k){ return (k*np + j)*np + i; };
    g.nodes.resize(static_cast<size_t>(np)*np*np);
    for (int k = 0; k < np; ++k)
    for (int j = 0; j < np; ++j)
    for (int i = 0; i < np; ++i)
        g.nodes[nid(i,j,k)] = make_double3(i/(double)n, j/(double)n, zext*k/(double)n);

    for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
        Hexahedron h;
        h.v[0]=nid(i,j,k);     h.v[1]=nid(i+1,j,k);
        h.v[2]=nid(i+1,j+1,k); h.v[3]=nid(i,j+1,k);
        h.v[4]=nid(i,j,k+1);   h.v[5]=nid(i+1,j,k+1);
        h.v[6]=nid(i+1,j+1,k+1); h.v[7]=nid(i,j+1,k+1);
        g.hexes.push_back(h);
    }

    // Globale Kanten ueber das Knotenpaar (min, max). Das Vorzeichen haelt
    // fest, ob die lokale Kante der globalen Richtung folgt -- genau wie
    // TopologyExtractor es tut.
    std::map<std::pair<int,int>, int> edge_id;
    g.hex_edges.resize(g.hexes.size());
    g.hex_signs.resize(g.hexes.size());
    for (size_t h = 0; h < g.hexes.size(); ++h) {
        for (int e = 0; e < 12; ++e) {
            const int a = g.hexes[h].v[LOCAL_EDGE[e][0]];
            const int b = g.hexes[h].v[LOCAL_EDGE[e][1]];
            const auto key = std::make_pair(std::min(a,b), std::max(a,b));
            auto it = edge_id.find(key);
            if (it == edge_id.end()) it = edge_id.emplace(key, g.num_edges++).first;
            if (static_cast<int>(g.edge_nodes.size()) <= it->second)
                g.edge_nodes.resize(static_cast<size_t>(it->second) + 1);
            g.edge_nodes[it->second] = make_int2(key.first, key.second);
            g.hex_edges[h][e] = it->second;
            g.hex_signs[h][e] = (a < b) ? 1 : -1;
        }
    }
    return g;
}

/// A = nu * K_curl + (kappa/dt) * M, als CSR ueber die Kanten.
/// kappa haengt am Hexaeder: untere Haelfte Leiter, obere Luft.
void assemble(const Grid& g, std::vector<int>& row, std::vector<int>& col,
              std::vector<double>& val)
{
    const int ne = g.num_edges;
    std::vector<std::map<int,double>> rows(static_cast<size_t>(ne));

    const double gp = 0.5773502691896257;
    const double pts[8][3] = {
        {-gp,-gp,-gp},{ gp,-gp,-gp},{-gp, gp,-gp},{ gp, gp,-gp},
        {-gp,-gp, gp},{ gp,-gp, gp},{-gp, gp, gp},{ gp, gp, gp}
    };

    for (size_t h = 0; h < g.hexes.size(); ++h) {
        double3 nd[8];
        for (int i = 0; i < 8; ++i) nd[i] = g.nodes[g.hexes[h].v[i]];

        // Zellindex k aus der Reihenfolge zurueckrechnen: untere Haelfte leitet.
        const int k = static_cast<int>(h) / (g.n * g.n);
        const double kappa = (k < g.n/2) ? KAPPA : 0.0;

        double Ae[12][12] = {};
        for (int q = 0; q < 8; ++q) {
            double3 W[12], curlW[12];
            evaluate_nedelec_basis(pts[q][0], pts[q][1], pts[q][2], W);
            evaluate_nedelec_curl (pts[q][0], pts[q][1], pts[q][2], curlW);

            double J[3][3], invJ[3][3], detJ;
            compute_jacobian(nd, pts[q][0], pts[q][1], pts[q][2], J, invJ, detJ);
            if (!(std::fabs(detJ) > 1e-30)) continue;

            // Covariante Abbildung fuer W, contravariante fuer curl W.
            double3 Wp[12], Cp[12];
            for (int i = 0; i < 12; ++i) {
                Wp[i] = make_double3(
                    invJ[0][0]*W[i].x + invJ[1][0]*W[i].y + invJ[2][0]*W[i].z,
                    invJ[0][1]*W[i].x + invJ[1][1]*W[i].y + invJ[2][1]*W[i].z,
                    invJ[0][2]*W[i].x + invJ[1][2]*W[i].y + invJ[2][2]*W[i].z);
                Cp[i] = make_double3(
                    (J[0][0]*curlW[i].x + J[0][1]*curlW[i].y + J[0][2]*curlW[i].z)/detJ,
                    (J[1][0]*curlW[i].x + J[1][1]*curlW[i].y + J[1][2]*curlW[i].z)/detJ,
                    (J[2][0]*curlW[i].x + J[2][1]*curlW[i].y + J[2][2]*curlW[i].z)/detJ);
            }
            const double w = std::fabs(detJ);
            for (int i = 0; i < 12; ++i)
                for (int j = 0; j < 12; ++j)
                    Ae[i][j] += w * (NU * (Cp[i].x*Cp[j].x + Cp[i].y*Cp[j].y + Cp[i].z*Cp[j].z)
                                   + (kappa/DT) * (Wp[i].x*Wp[j].x + Wp[i].y*Wp[j].y + Wp[i].z*Wp[j].z));
        }

        for (int i = 0; i < 12; ++i)
            for (int j = 0; j < 12; ++j) {
                const double s = g.hex_signs[h][i] * g.hex_signs[h][j];
                rows[g.hex_edges[h][i]][g.hex_edges[h][j]] += s * Ae[i][j];
            }
    }

    // Kanten ohne Massenterm und ohne Curl-Beitrag (reine Luft, Nullraum)
    // bekommen eine kleine Diagonale, sonst ist das System exakt singulaer und
    // die Frage nach der Iterationszahl waere sinnlos. Das entspricht der
    // Eichung im echten Loeser, nur viel gruber.
    double dmax = 0.0;
    for (int i = 0; i < ne; ++i) {
        auto it = rows[i].find(i);
        if (it != rows[i].end()) dmax = std::fmax(dmax, std::fabs(it->second));
    }
    for (int i = 0; i < ne; ++i) rows[i][i] += 1e-6 * dmax;

    row.assign(static_cast<size_t>(ne) + 1, 0);
    col.clear(); val.clear();
    for (int i = 0; i < ne; ++i) {
        row[i] = static_cast<int>(col.size());
        for (const auto& kv : rows[i]) { col.push_back(kv.first); val.push_back(kv.second); }
    }
    row[ne] = static_cast<int>(col.size());
}

struct Result { int iters; double res; double conv_vector; };

void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? " OK " : "FEHL", what);
    if (!ok) ++g_failures;
}

} // namespace

int main() {
    bf::g_backend = bf::Backend::CPU;
    std::printf("\n=== Vorkonditionierer am aeusseren GMRES (End-to-End) ===\n\n");

    const int n = 4;
    Grid g = build_grid(n);
    std::vector<int> row, col; std::vector<double> val;
    assemble(g, row, col, val);
    const int N = g.num_edges;
    std::printf("  Netz %dx%dx%d (Zell-Seitenverhaeltnis %.1f wie die Platte), "
                "%d Kanten, %zu Nichtnullen\n",
                n, n, n, CELL_ASPECT, N, val.size());
    std::printf("  untere Haelfte Leiter (kappa = %.1e), obere Luft (kappa = 0)\n\n", KAPPA);

    // Rechte Seite: divergenzfrei ist hier nicht noetig, es geht um die
    // Iterationszahl. Fester Startwert, damit der Test reproduzierbar ist.
    std::mt19937 rng(2024);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    std::vector<double> b(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) b[i] = uni(rng);

    Mesh mesh;
    fill_mesh(g, mesh);

    // Materialkoeffizienten je Hexaeder, exakt die der Assemblierung oben:
    // nu ueberall 1/mu_0, beta = kappa/dt nur in der unteren Haelfte.
    std::vector<double> nu_per_hex(g.hexes.size(), NU);
    std::vector<double> beta_per_hex(g.hexes.size(), 0.0);
    for (size_t h = 0; h < g.hexes.size(); ++h) {
        const int k = static_cast<int>(h) / (g.n * g.n);
        beta_per_hex[h] = (k < g.n/2) ? (KAPPA / DT) : 0.0;
    }

    enum class Pc { None, Ilu, AmsDefault, AmsAlphaPoisson };

    auto run = [&](Pc pc) {
        CoupledMVOperator op(N);
        op.setup_fem_matrix(N, static_cast<int>(val.size()),
                            row.data(), col.data(), val.data());
        op.setup_equilibration();
        op.compute_ilu();
        op.set_ilu_bypass(pc == Pc::None);
        if (pc == Pc::AmsDefault || pc == Pc::AmsAlphaPoisson) {
            op.set_ams_use_alpha_poisson(pc == Pc::AmsAlphaPoisson);
            op.set_ams_alpha_coefficients(nu_per_hex.data(), beta_per_hex.data());
            op.setup_ams(mesh);
        }

        std::vector<double> rhs(b), x(static_cast<size_t>(N), 0.0);
        op.scale_by_D(rhs.data());

        GMRESSolver solver(N, 50);
        solver.solve(op, rhs.data(), x.data(), 1e-8, 400);
        Result r;
        r.iters = solver.last_iterations();
        r.res   = solver.last_relative_residual();
        r.conv_vector = op.ams_vector_convergence();
        return r;
    };

    const auto no_pc  = run(Pc::None);
    const auto ilu_pc = run(Pc::Ilu);
    const auto ams_pc = run(Pc::AmsDefault);
    const auto ams_ap = run(Pc::AmsAlphaPoisson);

    std::printf("\n  %-34s %6s %12s\n", "Vorkonditionierer", "Iter.", "Rel. Res.");
    std::printf("  %-34s %6d %12.2e\n", "keiner",              no_pc.iters,  no_pc.res);
    std::printf("  %-34s %6d %12.2e\n", "ILU(0)",              ilu_pc.iters, ilu_pc.res);
    std::printf("  %-34s %6d %12.2e\n", "AMS (Pi^T A Pi)",     ams_pc.iters, ams_pc.res);
    std::printf("  %-34s %6d %12.2e\n", "AMS (Alpha-Poisson)", ams_ap.iters, ams_ap.res);
    std::printf("\n");

    check(ilu_pc.res <= 1e-8, "ILU(0): GMRES erreicht die Toleranz");
    check(ilu_pc.iters < no_pc.iters,
          "ILU(0) braucht weniger Iterationen als gar kein Vorkonditionierer");
    check(ams_pc.res <= 1e-8, "AMS (Vorgabe): GMRES erreicht die Toleranz");
    check(ams_pc.iters < no_pc.iters,
          "AMS (Vorgabe) braucht weniger Iterationen als gar kein Vorkonditionierer");

    // Der Alpha-Poisson-Weg, jetzt mit nu UND sigma/dt.
    //
    // Vorgeschichte: eine erste Fassung ohne Massenterm und mit nu = 1 machte
    // den aeusseren GMRES unbrauchbar (400 Iterationen, Rel. Res. 0.545 statt
    // 28 Iterationen). Dieser Test hat das reproduziert -- und war damit der
    // erste, der einen solchen Einbruch ueberhaupt sehen konnte.
    check(ams_ap.res <= 1e-8, "AMS (Alpha-Poisson): GMRES erreicht die Toleranz");
    // Schranke bewusst 1.25 und nicht "<=".
    //
    // Der Zweck dieser Pruefung ist, einen EINBRUCH zu fangen: die erste
    // Fassung des Alpha-Poisson-Operators brauchte 400 Iterationen statt 28
    // und erreichte die Toleranz gar nicht. Gemessen werden hier 43 gegen 42 --
    // ein Unterschied von einer Iteration ist bei einem iterativen Verfahren
    // kein Signal, und eine Schranke, die darauf anspringt, wird beim naechsten
    // Rauschen abgeschaltet statt gelesen.
    check(ams_ap.iters <= 1.25 * ams_pc.iters + 2,
          "AMS (Alpha-Poisson) bricht nicht ein gegenueber Pi^T A Pi");
    check(ams_ap.conv_vector < ams_pc.conv_vector,
          "der vektorielle Hilfsraum konvergiert mit Alpha-Poisson besser");

    std::printf("      (vektorieller Hilfsraum: %.2f mit Pi^T A Pi, "
                "%.2f mit Alpha-Poisson)\n", ams_pc.conv_vector, ams_ap.conv_vector);

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
