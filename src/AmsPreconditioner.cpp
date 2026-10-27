#include "AmsPreconditioner.cuh"

#include <cmath>
#include <iostream>
#include <limits>
#include <random>

#include "HostLinalg.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// Blendet die Zeilen aus A_G aus, in denen die skalare Hilfsmatrix keine
// Information traegt, und liefert die Maske zurueck.
//
// Warum das noetig ist -- und warum ein Diagonal-Guard allein nicht reicht:
//
// A_G = G^T A G. Fuer den Curl-Curl-Anteil gilt A_cc * G = 0 EXAKT in der
// Analysis, numerisch bleibt davon nur Rundung stehen. Eine echte Diagonale
// steuert allein der Massenterm sigma/dt bei -- also ausschliesslich im
// Leiter. Ausserhalb (Spulen mit kappa = 0, Luft) ist die GANZE ZEILE von der
// Groessenordnung eps * ||A||, nicht nur die Diagonale.
//
// Damit ist A_G dort praktisch singulaer, und der Hilfsraum-Schritt
// e_G = A_G^-1 (G^T r) liefert e_G ~ r/eps. Nachgemessen an einem
// Modellproblem mit derselben Struktur (ein Block der ganzen Matrix mit eps
// skaliert):
//
//   Zeilenskala   Verstaerkung ||z||/||b||
//         1e+00        9.7
//         1e-06        5.6e+06
//         1e-12        9.2e+11
//         1e-16        1.3e+12
//
// Ein Guard auf der Diagonalen (inv_diag = 0 statt 1/eps) deckelt das nur, er
// beseitigt es nicht -- die Verstaerkung saettigt bei 1e12 statt weiter zu
// wachsen. Das erklaert die gemessenen ||z_A|| = 8.6e+14 bei ||r_A|| = 1.
//
// Richtig ist, den skalaren Hilfsraum dort ganz auszublenden: wo kein sigma
// ist, hat er nichts zu korrigieren. Der Gradientenanteil des Feldes in Luft
// wird von der Eichung festgelegt, nicht vom AMS. Umgesetzt als symmetrische
// Elimination -- Zeile und Spalte des betroffenen Knotens werden geleert, die
// Diagonale auf 1 gesetzt -- und in apply() wird die rechte Seite dort
// genullt. Der Korrekturterm G * e_G ist damit an diesen Knoten exakt null.
std::vector<char> ams_mask_meaningless_rows(HostCsr& M, double ratio) {
    const int n = M.num_rows;
    std::vector<double> diag(static_cast<size_t>(n), 0.0);
    double dmax = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int k = M.row_offsets[i]; k < M.row_offsets[i+1]; ++k) {
            if (M.col_indices[k] == i) { diag[i] = M.values[k]; break; }
        }
        dmax = std::fmax(dmax, std::fabs(diag[i]));
    }

    std::vector<char> masked(static_cast<size_t>(n), 0);
    if (dmax <= 0.0) return masked;          // gar keine Diagonale: nichts zu tun
    const double tiny = ratio * dmax;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(diag[i]) <= tiny) masked[i] = 1;
    }

    for (int i = 0; i < n; ++i) {
        for (int k = M.row_offsets[i]; k < M.row_offsets[i+1]; ++k) {
            const int j = M.col_indices[k];
            if (masked[i] || masked[j]) {
                // Zeile UND Spalte leeren; die Diagonale der ausgeblendeten
                // Knoten wird zu 1, damit das AMG dort e_G = r_G rechnet --
                // und r_G wird in apply() genullt, also e_G = 0.
                M.values[k] = (i == j) ? 1.0 : 0.0;
            }
        }
    }
    return masked;
}

// Misst, wie gut ein AMG-Zyklus SEIN EIGENES System loest.
//
// Das ist die Frage, die ueber den Hilfsraum entscheidet, und sie wurde bisher
// nirgends gestellt. Ersatzweise wurde auf Level-Groessen und
// Operatorkomplexitaet geschaut -- beide sagen darueber nichts:
//
//   * "21039 -> 948" klingt nach absurder Vergroeberung, sind aber bei bs = 3
//     genau 316 Aggregate aus 7013 Knoten. Ein 3x3x3-Aggregat im
//     Hexaedernetz ist 27-fach, also Lehrbuch fuer 3D-Aggregation.
//   * Die Warnschwelle 1.3 fuer die Operatorkomplexitaet stammt aus einem
//     Modellproblem mit viel sanfterer Vergroeberung ueber mehr Ebenen. Bei
//     27-facher Vergroeberung und vier Ebenen ist ein Wert um 1.1 zu erwarten
//     und kein Alarmzeichen.
//
// Was zaehlt, ist der Konvergenzfaktor: um welchen Faktor faellt das Residuum
// je V-Zyklus? Als Vorkonditionierer taugt ein AMG ab etwa 0.5; Werte nahe 1.0
// bedeuten, dass der Hilfsraum nichts beitraegt -- unabhaengig davon, wie die
// Hierarchie aussieht.
//
// Gemessen wird auf einer Zufalls-rechten-Seite, weil es um den Operator geht
// und nicht um eine bestimmte Anregung.
static double measure_amg_convergence(const AlgebraicMultigrid& amg, const HostCsr& A,
                                      int num_cycles, unsigned seed,
                                      const std::vector<char>* mask)
{
    const int n = A.num_rows;
    if (n <= 0 || num_cycles <= 0) return -1.0;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    std::vector<double> b(static_cast<size_t>(n), 0.0);
    std::vector<double> r(static_cast<size_t>(n), 0.0);
    std::vector<double> e(static_cast<size_t>(n), 0.0);
    std::vector<double> Ax(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) b[i] = uni(rng);

    // Die Maske MUSS hier genauso angewandt werden wie in apply().
    //
    // Ohne sie misst die Kennzahl etwas, das im Betrieb nie vorkommt: eine
    // rechte Seite, die auch auf den ausgeblendeten Knoten besetzt ist. Dort
    // ist die Matrix die Identitaet, und der gedaempfte Jacobi-Glaetter
    // reduziert den Fehler pro Durchgang nur um den Faktor (1 - omega) --
    // omega stammt aber aus der Gershgorin-Schranke ueber ALLE Zeilen und ist
    // entsprechend klein. Die 3905 ausgeblendeten Zeilen wuerden den
    // gemessenen Faktor also nach oben ziehen, obwohl sie im Betrieb gar nicht
    // angeregt werden.
    //
    // Das ist ein Fehler in der Kennzahl, nicht im Verfahren -- und genau die
    // Sorte, die diese Arbeit schon mehrfach in die Irre geschickt hat.
    if (mask != nullptr) {
        for (int i = 0; i < n; ++i) if ((*mask)[i]) b[i] = 0.0;
    }

    auto res_norm = [&]() {
        apply_host_csr(A, x.data(), Ax.data());
        double s = 0.0;
        for (int i = 0; i < n; ++i) { r[i] = b[i] - Ax[i]; s += r[i] * r[i]; }
        return std::sqrt(s);
    };

    const double r0 = res_norm();
    if (!(r0 > 0.0)) return -1.0;

    double rk = r0;
    for (int c = 0; c < num_cycles; ++c) {
        amg.apply(r.data(), e.data());
        if (mask != nullptr) {
            for (int i = 0; i < n; ++i) if ((*mask)[i]) e[i] = 0.0;
        }
        for (int i = 0; i < n; ++i) x[i] += e[i];
        const double rn = res_norm();
        if (!std::isfinite(rn)) return std::numeric_limits<double>::infinity();
        rk = rn;
    }
    // Geometrisches Mittel des Faktors je Zyklus.
    return std::pow(rk / r0, 1.0 / static_cast<double>(num_cycles));
}

void AmsPreconditioner::setup(const HostCsr& A_edge, const Mesh& mesh, double shift) {
    m_num_edges = A_edge.num_rows;
    if (m_num_edges != static_cast<int>(mesh.num_edges)) {
        throw std::runtime_error("AmsPreconditioner::setup: A hat " +
            std::to_string(m_num_edges) + " Zeilen, das Netz " +
            std::to_string(mesh.num_edges) + " Kanten.");
    }

    m_G  = build_discrete_gradient(mesh);
    m_Pi = build_nodal_vector_interpolation(mesh);

    double max_G = 0.0, max_Pi = 0.0;
    // Die Zahl der rundungskleinen Diagonalen wird jetzt tatsaechlich abgeholt
    // -- der Parameter existierte, wurde aber mit nullptr aufgerufen. Genau
    // diese Zahl, nicht die der EXAKT leeren, sagt etwas ueber A_G aus.
    HostCsr A_G_raw  = build_auxiliary_nodal_matrix(A_edge, mesh, &m_tiny_G, -1.0, 1e-10, &max_G);
    const auto n2e   = build_node_to_edges(mesh);
    // Grundlage der AMG-Hierarchie im vektoriellen Hilfsraum.
    //
    // ACHTUNG -- DER ALPHA-POISSON-WEG IST DERZEIT ABGESCHALTET.
    //
    // Er tut als AMG-Grundlage genau das, was er soll: am echten Netz gemessen
    // faellt der Konvergenzfaktor von 0.93 auf 0.74, und die Hierarchie wird
    // gesund (21039 -> 2547 -> 978 -> 651 -> 600 -> 597, Operatorkomplexitaet
    // 2.40 statt 1.09).
    //
    // Als VORKONDITIONIERER ist er trotzdem unbrauchbar: der aeussere GMRES
    // stagnierte damit bei Rel. Res. 0.99 statt bei 0.025. Der Grund ist die
    // Skalierung, und er ist elementar.
    //
    // Der Hilfsraum-Schritt loest e = A_Pi^-1 (Pi^T r). Wird dafuer eine ANDERE
    // Matrix benutzt, muss sie zu A_Pi = Pi^T A Pi spektral aequivalent sein --
    // nicht bloss von aehnlicher Struktur. Der hier gebaute Operator ist es
    // nicht, gleich doppelt:
    //
    //   * Er ist mit nu = 1 assembliert, waehrend der Curl-Curl-Anteil von A
    //     mit nu = 1/mu_0 = 7.96e+05 gewichtet ist. Allein das sind fast sechs
    //     Groessenordnungen.
    //   * Ihm fehlt der Massenterm sigma/dt vollstaendig. Im Leiter ist der mit
    //     3.4e+07 / 1e-04 = 3.4e+11 der dominierende Beitrag; in Luft ist er
    //     null. Ein einzelner Skalierungsfaktor kann diesen Unterschied
    //     zwischen Leiter und Luft prinzipiell nicht ausgleichen.
    //
    // Zum Brauchbarmachen fehlt also nicht viel, aber es fehlt etwas Konkretes:
    // nu je Hexaeder und sigma/dt je Hexaeder muessen bis hierher
    // durchgereicht werden (setup() bekommt derzeit nur A_edge, mesh und
    // shift). Dann waere A_alpha = Integral nu grad.grad + Integral (sigma/dt)
    // phi phi zu assemblieren -- also Steifigkeit UND Masse, mit den echten
    // Koeffizienten. Erst das ist der Operator, den die AMS-Theorie meint.
    //
    // Bis dahin bleibt der alte Weg die Vorgabe. Umschalten ueber
    // set_use_alpha_poisson(true); die Kennzahl zum Vergleichen ist
    // vector_convergence().
    //
    // Pi^T A Pi (build_auxiliary_vector_matrix) erbt den Nullraum des
    // Curl-Curl-Anteils und ist dort, wo sigma = 0 ist, in einem ganzen
    // UNTERRAUM entartet -- gemessener AMG-Konvergenzfaktor 0.93, der
    // Hilfsraum trug also nichts bei. Statt dessen der knotenweise
    // Vektor-Poisson-Operator mit nu (in hypre:
    // HYPRE_AMSSetAlphaPoissonMatrix), der ueberall positiv definit ist.
    //
    // Pi selbst bleibt unveraendert als Transferoperator; ersetzt wird nur die
    // Matrix, auf der die Hierarchie aufgebaut wird. Ueber
    // set_use_alpha_poisson(false) laesst sich der alte Weg zum Vergleich
    // zuschalten -- die Kennzahl dafuer ist vector_convergence().
    HostCsr A_Pi_raw = m_use_alpha_poisson
        ? build_alpha_poisson_matrix(mesh, m_nu_per_hex, m_beta_per_hex)
        : build_auxiliary_vector_matrix(A_edge, m_Pi, mesh, n2e, &max_Pi);
    if (m_use_alpha_poisson && (m_nu_per_hex == nullptr || m_beta_per_hex == nullptr)) {
        std::cout << "[AMS] WARNUNG: Alpha-Poisson ist eingeschaltet, aber "
                     "set_alpha_coefficients() wurde nicht (vollstaendig) aufgerufen.\n"
                     "[AMS]   Ohne nu und sigma/dt ist der Operator NICHT spektral "
                     "aequivalent zu Pi^T A Pi. Gemessen hat das den aeusseren GMRES\n"
                     "[AMS]   von Rel. Res. 0.025 auf 0.99 gebracht -- der Hilfsraum wird "
                     "dann schaedlich, nicht bloss wirkungslos.\n";
    }
    if (m_use_alpha_poisson) {
        max_Pi = 0.0;
        for (double v : A_Pi_raw.values) max_Pi = std::fmax(max_Pi, std::fabs(v));
    }
    m_ratio_Pi_over_G = (max_G > 0.0) ? (max_Pi / max_G) : 0.0;

    // AMG statt ILU(0): die Hilfsprobleme sind Knoten-Laplace-Probleme, und
    // dort entkoppelt AMG die Iterationszahl von der Problemgroesse, waehrend
    // ILU(0) mit ihr waechst. A_Pi hat drei Freiheitsgrade je Knoten -- die
    // Aggregation muss deshalb auf KNOTENebene laufen, sonst koennten
    // Komponenten verschiedener Richtungen in dasselbe Aggregat geraten und
    // der Nahnullraum der konstanten Vektorfelder waere nicht mehr darstellbar.
    // Leere Zeilen zaehlen, BEVOR das AMG darauf losgeht -- das ist die
    // Kennzahl, an der eine kollabierte Vergroeberung erkennbar wird.
    auto count_empty = [](const HostCsr& M) {
        int e = 0;
        for (int i = 0; i < M.num_rows; ++i) {
            double d = 0.0;
            for (int k = M.row_offsets[i]; k < M.row_offsets[i+1]; ++k)
                if (M.col_indices[k] == i) { d = M.values[k]; break; }
            if (d == 0.0) ++e;
        }
        return e;
    };
    m_empty_G  = count_empty(A_G_raw);
    m_empty_Pi = count_empty(A_Pi_raw);

    // Ausblenden VOR dem AMG-Aufbau -- sonst baut das AMG seine Hierarchie auf
    // Rundungsrauschen auf, und genau daran ist die Vergroeberung kollabiert
    // (Operatorkomplexitaet 1.10 statt der 2.96 bis 6.98 des Modellproblems).
    m_mask_G = ams_mask_meaningless_rows(A_G_raw, 1e-10);
    m_masked_G = 0;
    for (char c : m_mask_G) if (c) ++m_masked_G;

    AlgebraicMultigrid::Options o;

    o.theta = bf::amg_theta();
    o.diagonal_shift = shift;
    m_amg_G.setup(A_G_raw, 1, o);
    m_amg_Pi.setup(A_Pi_raw, 3, o);

    m_lv_G.clear();  for (int l = 0; l < m_amg_G.num_levels(); ++l)  m_lv_G.push_back(m_amg_G.level_size(l));
    m_lv_Pi.clear(); for (int l = 0; l < m_amg_Pi.num_levels(); ++l) m_lv_Pi.push_back(m_amg_Pi.level_size(l));

    // Jacobi-Glaetter im Kantenraum. Dieselbe relative Schranke wie im AMG
    // (siehe fill_inv_diag in src/AlgebraicMultigrid.cpp): "ungleich null" ist
    // als Pruefung wertlos, entscheidend ist "gross genug im Verhaeltnis zum
    // groessten Diagonaleintrag". Im Kantenraum ist die Diagonale zwar
    // gutartig, aber die Konvention soll ueberall dieselbe sein -- sonst
    // repariert man dieselbe Falle spaeter noch einmal.
    m_inv_diag_edge.assign(static_cast<size_t>(m_num_edges), 0.0);
    {
        double dmax_edge = 0.0;
        for (int i = 0; i < m_num_edges; ++i) {
            for (int k = A_edge.row_offsets[i]; k < A_edge.row_offsets[i + 1]; ++k) {
                if (A_edge.col_indices[k] == i) {
                    dmax_edge = std::fmax(dmax_edge, std::fabs(A_edge.values[k]));
                    break;
                }
            }
        }
        const double tiny_edge = 1e-10 * dmax_edge;
        for (int i = 0; i < m_num_edges; ++i) {
            for (int k = A_edge.row_offsets[i]; k < A_edge.row_offsets[i + 1]; ++k) {
                if (A_edge.col_indices[k] == i) {
                    const double d = A_edge.values[k];
                    m_inv_diag_edge[i] = (std::fabs(d) > tiny_edge) ? (1.0 / d) : 0.0;
                    break;
                }
            }
        }
    }

    // Der skalare Raum wird MIT Maske gemessen, weil apply() ihn so benutzt.
    // Der vektorielle hat keine Maske.
    m_conv_G  = measure_amg_convergence(m_amg_G,  A_G_raw,  5, 12345u, &m_mask_G);
    m_conv_Pi = measure_amg_convergence(m_amg_Pi, A_Pi_raw, 5, 67890u, nullptr);

    // Zum Vergleich auch ohne Maske -- die Differenz zeigt, wie viel des
    // fruehen Werts von 0.72 blosses Messartefakt war.
    m_conv_G_unmasked = measure_amg_convergence(m_amg_G, A_G_raw, 5, 12345u, nullptr);

    m_rg.assign(static_cast<size_t>(A_G_raw.num_rows), 0.0);
    m_eg.assign(static_cast<size_t>(A_G_raw.num_rows), 0.0);
    m_rp.assign(static_cast<size_t>(A_Pi_raw.num_rows), 0.0);
    m_ep.assign(static_cast<size_t>(A_Pi_raw.num_rows), 0.0);
    m_edge_tmp.assign(static_cast<size_t>(m_num_edges), 0.0);
}

void AmsPreconditioner::apply(const double* r, double* z) const {
    const int ne = m_num_edges;

    // 1. Glaetter im Kantenraum
    for (int i = 0; i < ne; ++i) z[i] = m_inv_diag_edge[i] * r[i];

    // 2. Skalarer Hilfsraum: z += G A_G^-1 G^T r
    //    Die ausgeblendeten Knoten (kein sigma, siehe mask_meaningless_rows)
    //    bekommen rechte Seite 0 und liefern damit e_G = 0 -- der
    //    Korrekturterm ist dort exakt null statt r/eps.
    apply_gradient_transpose(m_G, r, m_rg.data(), static_cast<int>(m_rg.size()));
    for (size_t i = 0; i < m_rg.size(); ++i) if (m_mask_G[i]) m_rg[i] = 0.0;
    m_amg_G.apply(m_rg.data(), m_eg.data());
    for (size_t i = 0; i < m_eg.size(); ++i) if (m_mask_G[i]) m_eg[i] = 0.0;
    apply_gradient(m_G, m_eg.data(), m_edge_tmp.data());
    for (int i = 0; i < ne; ++i) z[i] += m_edge_tmp[i];

    // 3. Vektorieller Hilfsraum: z += Pi A_Pi^-1 Pi^T r
    //    Pi^T r per Streuung ueber die Zeilen von Pi.
    std::fill(m_rp.begin(), m_rp.end(), 0.0);
    for (int e = 0; e < ne; ++e) {
        const double re = r[e];
        if (re == 0.0) continue;
        for (int k = m_Pi.row_offsets[e]; k < m_Pi.row_offsets[e + 1]; ++k) {
            m_rp[m_Pi.col_indices[k]] += m_Pi.values[k] * re;
        }
    }
    m_amg_Pi.apply(m_rp.data(), m_ep.data());
    apply_host_csr(m_Pi, m_ep.data(), m_edge_tmp.data());
    for (int i = 0; i < ne; ++i) z[i] += m_edge_tmp[i];
}
