#include "DiscreteGradient.cuh"
#include "NodalHexahedron.cuh"
#include "NodalDispatch.cuh"   // typabhaengige Knotengradienten und Quadratur

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>

HostCsr build_discrete_gradient(const Mesh& mesh) {
    HostCsr G;
    G.num_rows = static_cast<int>(mesh.num_edges);
    G.num_cols = static_cast<int>(mesh.num_nodes);
    G.row_offsets.resize(G.num_rows + 1);
    G.col_indices.reserve(static_cast<size_t>(G.num_rows) * 2);
    G.values.reserve(static_cast<size_t>(G.num_rows) * 2);

    mesh.edges.prefetch_to_host();

    for (int e = 0; e < G.num_rows; ++e) {
        G.row_offsets[e] = static_cast<int>(G.col_indices.size());
        const int2 ed = mesh.edges[e];
        // Spaltenindizes je Zeile aufsteigend -- bf::csr_ilu0 und die
        // Dreieckssubstitutionen setzen das voraus, und mesh.edges haelt
        // (v_min, v_max), die Reihenfolge stimmt also bereits.
        if (ed.x < ed.y) {
            G.col_indices.push_back(ed.x); G.values.push_back(-1.0);
            G.col_indices.push_back(ed.y); G.values.push_back(+1.0);
        } else {
            G.col_indices.push_back(ed.y); G.values.push_back(+1.0);
            G.col_indices.push_back(ed.x); G.values.push_back(-1.0);
        }
    }
    G.row_offsets[G.num_rows] = static_cast<int>(G.col_indices.size());
    return G;
}

HostCsr build_auxiliary_nodal_matrix(const HostCsr& A, const Mesh& mesh,
                                     int* out_num_tiny_diagonal,
                                     double reference_scale,
                                     double tiny_threshold,
                                     double* out_max_abs)
{
    const int nn = static_cast<int>(mesh.num_nodes);
    if (A.num_rows != static_cast<int>(mesh.num_edges)) {
        throw std::runtime_error("build_auxiliary_nodal_matrix: A hat " +
                                 std::to_string(A.num_rows) + " Zeilen, erwartet "
                                 + std::to_string(mesh.num_edges) + " (Kantenzahl).");
    }

    mesh.edges.prefetch_to_host();

    // Aufsammeln in einer Map je Zeile. Der Speicher ist unkritisch (rund
    // 27 Nachbarn je Knoten), und der Code bleibt nachvollziehbar; ein
    // handoptimierter Dreifachproduktalgorithmus waere hier verfruehte
    // Optimierung.
    std::vector<std::map<int, double>> rows(static_cast<size_t>(nn));

    for (int e = 0; e < A.num_rows; ++e) {
        const int2 ee = mesh.edges[e];
        // Vorzeichen von G in Zeile e: -1 bei ee.x, +1 bei ee.y
        const int    ei[2] = { ee.x, ee.y };
        const double es[2] = { -1.0, +1.0 };

        for (int k = A.row_offsets[e]; k < A.row_offsets[e + 1]; ++k) {
            const int    f   = A.col_indices[k];
            const double val = A.values[k];
            if (val == 0.0) continue;
            const int2 ff = mesh.edges[f];
            const int    fj[2] = { ff.x, ff.y };
            const double fs[2] = { -1.0, +1.0 };

            for (int a = 0; a < 2; ++a) {
                for (int b = 0; b < 2; ++b) {
                    rows[ei[a]][fj[b]] += es[a] * fs[b] * val;
                }
            }
        }
    }

    HostCsr An;
    An.num_rows = nn;
    An.num_cols = nn;
    An.row_offsets.resize(nn + 1);
    for (int i = 0; i < nn; ++i) {
        An.row_offsets[i] = static_cast<int>(An.col_indices.size());
        for (const auto& kv : rows[i]) {          // std::map -> aufsteigend sortiert
            An.col_indices.push_back(kv.first);
            An.values.push_back(kv.second);
        }
    }
    An.row_offsets[nn] = static_cast<int>(An.col_indices.size());

    if (out_max_abs != nullptr) {
        double m = 0.0;
        for (double v : An.values) m = std::fmax(m, std::fabs(v));
        *out_max_abs = m;
    }

    if (out_num_tiny_diagonal != nullptr) {
        std::vector<double> diag(static_cast<size_t>(nn), 0.0);
        double dmax = 0.0;
        for (int i = 0; i < nn; ++i) {
            for (int k = An.row_offsets[i]; k < An.row_offsets[i + 1]; ++k) {
                if (An.col_indices[k] == i) { diag[i] = An.values[k]; break; }
            }
            dmax = std::fmax(dmax, std::fabs(diag[i]));
        }
        // Bezugsgroesse von aussen, sonst notduerftig das eigene Maximum.
        const double ref = (reference_scale > 0.0) ? reference_scale
                                                   : (dmax > 0.0 ? dmax : 1.0);
        int tiny = 0;
        for (int i = 0; i < nn; ++i) {
            if (std::fabs(diag[i]) <= tiny_threshold * ref) ++tiny;
        }
        *out_num_tiny_diagonal = tiny;
    }

    return An;
}

void apply_gradient(const HostCsr& G, const double* x_nodes, double* y_edges) {
    for (int e = 0; e < G.num_rows; ++e) {
        double acc = 0.0;
        for (int k = G.row_offsets[e]; k < G.row_offsets[e + 1]; ++k) {
            acc += G.values[k] * x_nodes[G.col_indices[k]];
        }
        y_edges[e] = acc;
    }
}

void apply_gradient_transpose(const HostCsr& G, const double* x_edges,
                              double* y_nodes, int num_nodes)
{
    for (int i = 0; i < num_nodes; ++i) y_nodes[i] = 0.0;
    for (int e = 0; e < G.num_rows; ++e) {
        const double xe = x_edges[e];
        if (xe == 0.0) continue;
        for (int k = G.row_offsets[e]; k < G.row_offsets[e + 1]; ++k) {
            y_nodes[G.col_indices[k]] += G.values[k] * xe;
        }
    }
}

void apply_host_csr(const HostCsr& A, const double* x, double* y) {
    for (int i = 0; i < A.num_rows; ++i) {
        double acc = 0.0;
        for (int k = A.row_offsets[i]; k < A.row_offsets[i + 1]; ++k) {
            acc += A.values[k] * x[A.col_indices[k]];
        }
        y[i] = acc;
    }
}

// =============================================================================
// Vektorieller Hilfsraum
// =============================================================================

std::vector<std::vector<int>> build_node_to_edges(const Mesh& mesh) {
    mesh.edges.prefetch_to_host();
    std::vector<std::vector<int>> n2e(mesh.num_nodes);
    for (size_t e = 0; e < mesh.num_edges; ++e) {
        const int2 ed = mesh.edges[e];
        n2e[ed.x].push_back(static_cast<int>(e));
        n2e[ed.y].push_back(static_cast<int>(e));
    }
    return n2e;
}

HostCsr build_nodal_vector_interpolation(const Mesh& mesh) {
    mesh.edges.prefetch_to_host();
    mesh.nodes_current.prefetch_to_host();

    HostCsr Pi;
    Pi.num_rows = static_cast<int>(mesh.num_edges);
    Pi.num_cols = 3 * static_cast<int>(mesh.num_nodes);
    Pi.row_offsets.resize(Pi.num_rows + 1);
    Pi.col_indices.reserve(static_cast<size_t>(Pi.num_rows) * 6);
    Pi.values.reserve(static_cast<size_t>(Pi.num_rows) * 6);

    for (int e = 0; e < Pi.num_rows; ++e) {
        Pi.row_offsets[e] = static_cast<int>(Pi.col_indices.size());
        const int2 ed = mesh.edges[e];
        const double3 p1 = mesh.nodes_current[ed.x];
        const double3 p2 = mesh.nodes_current[ed.y];
        // Kantenvektor x_n2 - x_n1, halbiert (Trapezregel)
        const double t[3] = { 0.5 * (p2.x - p1.x),
                              0.5 * (p2.y - p1.y),
                              0.5 * (p2.z - p1.z) };
        // Spaltenindizes je Zeile aufsteigend: der kleinere Knoten zuerst.
        const int na = (ed.x < ed.y) ? ed.x : ed.y;
        const int nb = (ed.x < ed.y) ? ed.y : ed.x;
        for (int c = 0; c < 3; ++c) { Pi.col_indices.push_back(3*na + c); Pi.values.push_back(t[c]); }
        for (int c = 0; c < 3; ++c) { Pi.col_indices.push_back(3*nb + c); Pi.values.push_back(t[c]); }
    }
    Pi.row_offsets[Pi.num_rows] = static_cast<int>(Pi.col_indices.size());
    return Pi;
}

HostCsr build_auxiliary_vector_matrix(const HostCsr& A, const HostCsr& Pi,
                                      const Mesh& mesh,
                                      const std::vector<std::vector<int>>& node_to_edges,
                                      double* out_max_abs)
{
    const int nn  = static_cast<int>(mesh.num_nodes);
    const int ndf = 3 * nn;

    HostCsr Ap;
    Ap.num_rows = ndf;
    Ap.num_cols = ndf;
    Ap.row_offsets.assign(ndf + 1, 0);

    // Streuvektor und Trefferliste: O(1) je Beitrag statt O(log n) in einer Map.
    std::vector<double> scratch(static_cast<size_t>(ndf), 0.0);
    std::vector<int>    touched;
    touched.reserve(1024);
    std::vector<char>   is_touched(static_cast<size_t>(ndf), 0);

    for (int n = 0; n < nn; ++n) {
        for (int c = 0; c < 3; ++c) {
            const int row = 3 * n + c;
            Ap.row_offsets[row] = static_cast<int>(Ap.col_indices.size());

            // Alle Kanten, die an Knoten n haengen, tragen zu Zeile (n,c) bei.
            for (int e : node_to_edges[n]) {
                // Pi[e][row] heraussuchen
                double pie = 0.0;
                for (int k = Pi.row_offsets[e]; k < Pi.row_offsets[e + 1]; ++k) {
                    if (Pi.col_indices[k] == row) { pie = Pi.values[k]; break; }
                }
                if (pie == 0.0) continue;

                for (int ka = A.row_offsets[e]; ka < A.row_offsets[e + 1]; ++ka) {
                    const int    f  = A.col_indices[ka];
                    const double av = A.values[ka];
                    if (av == 0.0) continue;
                    const double coef = pie * av;
                    for (int kp = Pi.row_offsets[f]; kp < Pi.row_offsets[f + 1]; ++kp) {
                        const int    col = Pi.col_indices[kp];
                        const double add = coef * Pi.values[kp];
                        if (!is_touched[col]) { is_touched[col] = 1; touched.push_back(col); }
                        scratch[col] += add;
                    }
                }
            }

            std::sort(touched.begin(), touched.end());
            for (int col : touched) {
                Ap.col_indices.push_back(col);
                Ap.values.push_back(scratch[col]);
                scratch[col] = 0.0;
                is_touched[col] = 0;
            }
            touched.clear();
        }
    }
    Ap.row_offsets[ndf] = static_cast<int>(Ap.col_indices.size());

    if (out_max_abs != nullptr) {
        double m = 0.0;
        for (double v : Ap.values) m = std::fmax(m, std::fabs(v));
        *out_max_abs = m;
    }
    return Ap;
}

// =============================================================================
// Knotenweiser Vektor-Poisson-Operator fuer den vektoriellen AMS-Hilfsraum
// =============================================================================
//
// Warum nicht Pi^T A Pi, wie bisher?
//
// Gemessen wurde der Konvergenzfaktor des AMG je V-Zyklus, auf einer
// Zufalls-rechten-Seite (siehe measure_amg_convergence in
// src/AmsPreconditioner.cpp):
//
//     3D-Laplace (Bezugswert)   0.30
//     skalarer Hilfsraum        0.72
//     vektorieller Hilfsraum    0.93
//
// 0.93 heisst: der vektorielle Hilfsraum traegt praktisch nichts bei. Die
// AMG-Implementierung ist dabei nachweislich gesund -- auf dem 3D-Laplace
// erreicht sie 0.30, und zwar netzunabhaengig
// (tests/test_amg_convergence.cu). Es liegt also am Operator.
//
// A_Pi = Pi^T A Pi erbt den Nullraum des Curl-Curl-Anteils. Wo sigma = 0 ist
// (Spulen mit kappa = 0 und Luft, hier 3905 von 7013 Knoten), traegt nur der
// Massenterm eine definite Struktur bei -- und den gibt es dort nicht.
// Entartet ist dabei nicht eine Zeile, sondern ein UNTERRAUM: die
// Gradientenanteile im knotenweisen Vektorfeld. Beim skalaren Hilfsraum liess
// sich das durch Ausblenden ganzer Zeilen loesen, hier nicht -- glatte
// Aggregation kann so einen Nahnullraum nicht darstellen.
//
// Der uebliche Ausweg (in hypre: HYPRE_AMSSetAlphaPoissonMatrix) ist, den
// Hilfsraum-Operator NICHT aus A abzuleiten, sondern direkt als
// knotenweisen Poisson-Operator mit dem alpha-Koeffizienten nu = 1/mu:
//
//     A_alpha[i][j] = Integral nu * grad(phi_i) . grad(phi_j) dV
//
// Der ist ueberall positiv definit -- nu ist auch in Luft positiv (1/mu_0) --
// und damit genau die Sorte Operator, fuer die AMG gebaut ist. Pi bleibt
// unveraendert als Transferoperator; nur die Matrix, auf der die Hierarchie
// aufgebaut wird, wird ersetzt.
//
// Die drei Komponenten koppeln in diesem Operator NICHT (alpha ist skalar).
// Die Matrix ist also blockdiagonal aus drei identischen Kopien des
// skalaren Poisson-Operators; genau so wird sie hier aufgebaut.
//
// ZUR SKALIERUNG -- hier steckte ein Fehler, der den Operator unbrauchbar
// gemacht hat, und die Lehre daraus ist allgemein:
//
// Der Hilfsraum-Schritt loest e = A_Pi^-1 (Pi^T r). Eine Ersatzmatrix muss dafuer
// SPEKTRAL AEQUIVALENT zu Pi^T A Pi sein, nicht bloss von aehnlicher Struktur.
// Eine erste Fassung assemblierte mit nu = 1 und ohne Massenterm; der aeussere
// GMRES stagnierte damit bei Rel. Res. 0.99 statt 0.025.
//
// Dabei ist wichtig, WARUM das schadet: eine EINHEITLICHE Fehlskalierung waere
// wirkungslos, GMRES ist gegen B -> c*B invariant. AMS ist aber eine SUMME aus
// Kantenglaetter, skalarem und vektoriellem Hilfsraum -- ein einzelner
// Summand, der um Groessenordnungen daneben liegt, uebertoent die anderen. Und
// der fehlende Massenterm ist gar keine Konstante: im Leiter ist sigma/dt der
// dominierende Beitrag, in Luft ist er null. Diesen Unterschied kann kein
// Skalierungsfaktor ausgleichen.
//
// Deshalb werden jetzt BEIDE Koeffizienten mitgefuehrt:
//   nu_per_hex    = 1/mu je Hexaeder   (Steifigkeit, ueberall positiv)
//   beta_per_hex  = sigma/dt je Hexaeder (Masse, nur im Leiter)
// nullptr bedeutet 1 bzw. 0.
HostCsr build_alpha_poisson_matrix(const Mesh& mesh, const double* nu_per_hex,
                                   const double* beta_per_hex)
{
    const int nn = static_cast<int>(mesh.num_nodes);
    const int nh = static_cast<int>(mesh.num_hexes);

    mesh.hex_nodes.prefetch_to_host();
    // Die AKTUELLEN Koordinaten, nicht die Ausgangskoordinaten: die Platte
    // bewegt sich, und der Hilfsraum-Operator muss zur assemblierten Matrix
    // desselben Zeitschritts passen.
    mesh.nodes_current.prefetch_to_host();
    if (mesh.elem_types.size()) mesh.elem_types.prefetch_to_host();
    bf::sync();

    // Elementtypen: leeres Array heisst "alles Hexaeder" (siehe Mesh.cuh). Der
    // Zeiger wird EINMAL geholt und dann ueber element_type_or_hex() gelesen,
    // damit der Hexaederfall bitgleich bleibt wie vor dem Dispatch.
    const int* types = mesh.elem_types.size() ? mesh.elem_types.data() : nullptr;

    // Erst die Kopplungen sammeln (Knoten -> Menge der Nachbarknoten inkl.
    // sich selbst), dann werteweise akkumulieren.
    std::vector<std::map<int, double>> rows(static_cast<size_t>(nn));

    for (int h = 0; h < nh; ++h) {
        const Hexahedron& hex = mesh.hex_nodes.data()[h];
        const int type = element_type_or_hex(types, static_cast<size_t>(h));
        const int nv   = element_num_nodes(type);

        // NUR die belegten Plaetze lesen. Auf hex.v[nv..7] steht bei
        // Tetraedern und Prismen eine -1, und mesh.nodes_current[-1] ist der
        // Zugriff, der hier vorher stattfand.
        double3 nd[8];
        for (int i = 0; i < nv; ++i) nd[i] = mesh.nodes_current.data()[hex.v[i]];
        for (int i = nv; i < 8; ++i) nd[i] = make_double3(0.0, 0.0, 0.0);

        const double nu   = (nu_per_hex   != nullptr) ? nu_per_hex[h]   : 1.0;
        const double beta = (beta_per_hex != nullptr) ? beta_per_hex[h] : 0.0;

        double ke[8][8] = {};
        // Gelumpte Masse: nur die Diagonale. Dass die Zeilensummen-Diagonale
        // spektral aequivalent zur konsistenten Massenmatrix ist, ist ein
        // Standardresultat -- und die Zeilensumme laesst sich hier direkt
        // ausrechnen, weil die Formfunktionen eine Zerlegung der Eins bilden:
        // Summe_j Integral phi_i phi_j = Integral phi_i. Das gilt fuer alle
        // drei Elementtypen, die Lumpung ist also nicht hexaederspezifisch.
        // Fuer AMG ist die gelumpte Variante ausserdem robuster als die
        // konsistente.
        double me[8] = {};
        // Dieselbe Quadratur wie die Kantenassemblierung: 2x2x2-Gauss auf dem
        // Hexaeder (Gewicht 1), 4 Punkte auf dem Tetraeder (Gewicht 1/24, die
        // Summe ist das Referenzvolumen 1/6), 6 auf dem Prisma. Fuer den
        // Hexaeder liefert element_quadrature_point() Punkt fuer Punkt
        // dieselben Werte in derselben Reihenfolge wie das frueher hier
        // benutzte nodal_gauss_point() -- der bestehende transiente Pfad
        // rechnet also bitgleich weiter.
        const int nq = element_num_quad_points(type);
        for (int g = 0; g < nq; ++g) {
            double xi, eta, zeta, w;
            element_quadrature_point(type, g, xi, eta, zeta, w);

            double3 gradN[8];
            double detJ = 0.0;
            element_nodal_gradients(type, nd, xi, eta, zeta, gradN, detJ);

            // Ein entartetes Element traegt nichts bei. Die Pruefung ist
            // relativ gedacht -- detJ hat die Dimension eines Volumens, und
            // "ungleich null" waere hier wertlos (siehe
            // docs/design/nan-and-thresholds.md).
            if (!(std::fabs(detJ) > 1e-30)) continue;

            const double vol   = w * std::fabs(detJ);
            const double scale = nu * vol;
            for (int i = 0; i < nv; ++i) {
                for (int j = 0; j < nv; ++j) {
                    ke[i][j] += scale * (gradN[i].x * gradN[j].x +
                                         gradN[i].y * gradN[j].y +
                                         gradN[i].z * gradN[j].z);
                }
            }
            if (beta != 0.0) {
                double N[8];
                element_shape_functions(type, xi, eta, zeta, N);
                for (int i = 0; i < nv; ++i) me[i] += beta * vol * N[i];
            }
        }

        for (int i = 0; i < nv; ++i) {
            for (int j = 0; j < nv; ++j) {
                rows[hex.v[i]][hex.v[j]] += ke[i][j];
            }
            rows[hex.v[i]][hex.v[i]] += me[i];
        }
    }

    // Blockdiagonal auf 3 Komponenten aufziehen: Zeile 3n+c koppelt nur an
    // Spalten 3m+c. Die Reihenfolge (Knoten-Major, Komponente-Minor) muss zu
    // Pi passen -- dort ist die Spalte 3*node + component.
    HostCsr A;
    A.num_rows = 3 * nn;
    A.num_cols = 3 * nn;
    A.row_offsets.assign(static_cast<size_t>(3 * nn) + 1, 0);

    for (int n = 0; n < nn; ++n) {
        for (int c = 0; c < 3; ++c) {
            const int row = 3 * n + c;
            A.row_offsets[row] = static_cast<int>(A.col_indices.size());
            for (const auto& kv : rows[n]) {
                A.col_indices.push_back(3 * kv.first + c);
                A.values.push_back(kv.second);
            }
        }
    }
    A.row_offsets[3 * nn] = static_cast<int>(A.col_indices.size());
    return A;
}
