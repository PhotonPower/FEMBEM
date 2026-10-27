#include "FemDirectBC.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace fem_direct {

// -----------------------------------------------------------------------------
std::vector<char> collect_dirichlet_edges(const Mesh& mesh) {
    if (mesh.num_edges == 0) {
        throw std::runtime_error(
            "collect_dirichlet_edges: keine Kanten. extract_edges() zuerst aufrufen.");
    }
    if (mesh.num_boundary_faces == 0) {
        throw std::runtime_error(
            "collect_dirichlet_edges: keine Randflaechen. extract_boundary_faces() "
            "zuerst aufrufen. Ein Netz ohne Rand kann es nicht geben -- fuer ein "
            "abgeschnittenes Luftgebiet muss es eine Aussenwand geben.");
    }

    mesh.boundary_edges_to_fem_edges.prefetch_to_host();

    std::vector<char> constrained(mesh.num_edges, 0);
    for (size_t f = 0; f < mesh.num_boundary_faces; ++f) {
        for (int k = 0; k < 4; ++k) {
            const int e = mesh.boundary_edges_to_fem_edges[f * 4 + k];
            // -1 = vierter Platz einer Dreiecksflaeche.
            if (e < 0) continue;
            if (e >= static_cast<int>(mesh.num_edges)) {
                throw std::runtime_error(
                    "collect_dirichlet_edges: Randkante " + std::to_string(e) +
                    " liegt ausserhalb der globalen Kantenliste. Topologie ist defekt.");
            }
            constrained[e] = 1;
        }
    }
    return constrained;
}

// -----------------------------------------------------------------------------
DirichletReport apply_homogeneous_dirichlet(
    int num_rows,
    const int* row_offsets,
    const int* col_indices,
    double* values,
    double* rhs,
    const std::vector<char>& is_constrained,
    double diag_value)
{
    if (static_cast<size_t>(num_rows) != is_constrained.size()) {
        throw std::runtime_error(
            "apply_homogeneous_dirichlet: Zeilenzahl " + std::to_string(num_rows) +
            " passt nicht zur Markierung der Laenge " +
            std::to_string(is_constrained.size()) + ".");
    }

    DirichletReport rep;

    // -------------------------------------------------------------------------
    // Schritt 1: Diagonalwert bestimmen, BEVOR etwas veraendert wird
    // -------------------------------------------------------------------------
    double diag_sum = 0.0;
    rep.diag_min_free = 0.0;
    rep.diag_max_free = 0.0;
    bool first_free = true;

    for (int r = 0; r < num_rows; ++r) {
        if (is_constrained[r]) { ++rep.num_constrained; continue; }
        ++rep.num_free;
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
            if (col_indices[k] != r) continue;
            const double d = std::fabs(values[k]);
            diag_sum += d;
            if (first_free) {
                rep.diag_min_free = rep.diag_max_free = d;
                first_free = false;
            } else {
                rep.diag_min_free = std::min(rep.diag_min_free, d);
                rep.diag_max_free = std::max(rep.diag_max_free, d);
            }
            break;
        }
    }

    if (diag_value > 0.0) {
        rep.diag_value = diag_value;
    } else if (rep.num_free > 0 && diag_sum > 0.0) {
        rep.diag_value = diag_sum / static_cast<double>(rep.num_free);
    } else {
        // Alles festgehalten oder eine leere Matrix. 1.0 ist dann so gut wie
        // jeder andere Wert; ein Fehler ist es nicht, aber es ist ein Zeichen
        // dafuer, dass etwas anderes nicht stimmt.
        rep.diag_value = 1.0;
    }

    // -------------------------------------------------------------------------
    // Schritt 2: Zeilen und Spalten nullen
    // -------------------------------------------------------------------------
    // In EINEM Durchgang ueber alle Eintraege. Ein Eintrag (r, c) faellt weg,
    // sobald r ODER c festgehalten ist; die Diagonale einer festgehaltenen
    // Zeile bekommt den Diagonalwert. Damit bleibt die Matrix symmetrisch,
    // sofern sie es vorher war -- Voraussetzung fuer CUDSS_MTYPE_SYMMETRIC.
    for (int r = 0; r < num_rows; ++r) {
        const bool row_fixed = is_constrained[r] != 0;
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
            const int c = col_indices[k];
            const bool col_fixed = is_constrained[c] != 0;

            if (!row_fixed && !col_fixed) continue;   // freier Block, unberuehrt

            if (r == c && row_fixed) {
                values[k] = rep.diag_value;
            } else {
                if (values[k] != 0.0) ++rep.entries_zeroed;
                values[k] = 0.0;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Schritt 3: rechte Seite
    // -------------------------------------------------------------------------
    // Der vorgeschriebene Wert ist null, deshalb genuegt rhs[i] = 0 fuer die
    // festgehaltenen Zeilen. Bei einem Wert ungleich null muesste hier
    // zusaetzlich rhs[j] -= A(j,i) * wert fuer alle freien Zeilen j stehen --
    // und zwar VOR dem Nullen der Spalte.
    if (rhs) {
        for (int r = 0; r < num_rows; ++r) {
            if (is_constrained[r]) rhs[r] = 0.0;
        }
    }

    return rep;
}

// -----------------------------------------------------------------------------
size_t regularize_nonconducting_regions(
    MaterialProperties* materials,
    const Mesh& mesh,
    double sigma_reference,
    double epsilon)
{
    if (!materials) {
        throw std::runtime_error("regularize_nonconducting_regions: materials ist null.");
    }
    if (!(sigma_reference > 0.0)) {
        throw std::runtime_error(
            "regularize_nonconducting_regions: sigma_reference muss positiv sein "
            "(erwartet die Leitfaehigkeit des Leiters, z. B. 3.526e7 fuer Aluminium).");
    }
    if (!(epsilon > 0.0) || !(epsilon < 1.0)) {
        throw std::runtime_error(
            "regularize_nonconducting_regions: epsilon muss in (0, 1) liegen. "
            "Sinnvoll ist 1e-8; die Abnahme verlangt Unempfindlichkeit zwischen "
            "1e-6 und 1e-10.");
    }

    const double sigma_air = epsilon * sigma_reference;
    size_t changed = 0;

    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        // Nur echte Nichtleiter anfassen. Ein Element mit vorhandener
        // Leitfaehigkeit bleibt unberuehrt, auch wenn sie sehr klein ist --
        // sonst wuerde ein absichtlich schwach leitendes Material still
        // ueberschrieben.
        if (materials[h].kappa > 0.0) continue;
        materials[h].kappa = sigma_air;
        ++changed;
    }

    return changed;
}


// =============================================================================
// Gradientenanteil aus dem Quellterm herausprojizieren
// =============================================================================
// Herleitung und Begruendung im Kopf der Deklaration in FemDirectBC.h.
//
// Hier nur die zwei Punkte, die die Umsetzung bestimmen:
//
// 1. G^T G ist SINGULAER. Konstantes phi liegt im Kern, je
//    Zusammenhangskomponente des Kantengraphen eines. Die rechte Seite G^T b
//    ist dazu automatisch orthogonal -- jede Kante traegt +b_e am Kopf und
//    -b_e am Fuss bei, die Summe ueber alle Knoten ist also exakt null. Das
//    CG bleibt aber nur stabil, wenn der konstante Anteil auch WAEHREND der
//    Iteration herausgenommen wird, weil Rundung ihn sonst wieder einbringt
//    und er dann mit dem Eigenwert 0 beliebig anwaechst.
//
//    Herausgenommen wird global, nicht je Komponente: das Luftgebiet ist
//    zusammenhaengend und umschliesst alles, ein Netz aus mehreren
//    unverbundenen Stuecken hat hier andere Probleme.
//
// 2. Kein Vorkonditionierer. Der Graph-Laplace eines Tetraedernetzes hat
//    kappa ~ h^-2, das sind bei 158238 Kanten einige Hundert Iterationen --
//    Sekunden. Ein ILU oder AMG davor waere Aufwand fuer eine Groesse, die
//    neben der Faktorisierung nicht ins Gewicht faellt.
GradientProjectionReport project_out_gradient(const Mesh& mesh,
                                              double* b, size_t n,
                                              const char* blocked_edge,
                                              double tol, int max_iter)
{
    if (b == nullptr) throw std::runtime_error("project_out_gradient: b ist null.");
    if (n != mesh.num_edges) {
        throw std::runtime_error(
            "project_out_gradient: n = " + std::to_string(n) +
            " passt nicht zu mesh.num_edges = " + std::to_string(mesh.num_edges) +
            ". Erwartet wird der Quellterm im FEM-Kantenraum, nicht die "
            "2n-Blockform -- dort waere die Projektion je Haelfte anzuwenden.");
    }

    const size_t nn = mesh.num_nodes;
    GradientProjectionReport rep;

    const auto is_free = [&](size_t e) {
        return blocked_edge == nullptr || blocked_edge[e] == 0;
    };

    // Gesperrte Kanten sofort auf null. Die Randbehandlung tut das spaeter
    // ohnehin; hier vorweggenommen, damit der gemeldete Wert von
    // ||G^T b||/||b|| der ist, der nach der Randbehandlung wirklich gilt.
    for (size_t e = 0; e < mesh.num_edges; ++e)
        if (!is_free(e)) b[e] = 0.0;

    // Knoten, die an mindestens einer FREIEN Kante haengen. Alle anderen
    // fallen aus dem Teilgraphen: ihre Laplace-Zeile ist leer, und ohne
    // diesen Ausschluss brachte die Konstantenbereinigung unten ihnen einen
    // Wert ein, den keine Gleichung mehr korrigiert.
    std::vector<char> active(nn, 0);
    for (size_t e = 0; e < mesh.num_edges; ++e) {
        if (!is_free(e)) continue;
        const int2 ed = mesh.edges[e];
        active[ed.x] = 1;
        active[ed.y] = 1;
    }
    size_t n_active = 0;
    for (char a : active) if (a) ++n_active;
    if (n_active == 0) {
        rep.converged = true;
        return rep;
    }

    // G_F^T x -- nur freie Kanten tragen bei
    const auto apply_Gt = [&](const double* x, std::vector<double>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (size_t e = 0; e < mesh.num_edges; ++e) {
            if (!is_free(e)) continue;
            const int2 ed = mesh.edges[e];
            out[ed.y] += x[e];
            out[ed.x] -= x[e];
        }
    };
    // G_F p -- auf gesperrten Kanten null, damit b dort null BLEIBT
    const auto apply_G = [&](const std::vector<double>& p, std::vector<double>& out) {
        for (size_t e = 0; e < mesh.num_edges; ++e) {
            if (!is_free(e)) { out[e] = 0.0; continue; }
            const int2 ed = mesh.edges[e];
            out[e] = p[ed.y] - p[ed.x];
        }
    };
    // Konstanten herausnehmen -- siehe Punkt 1 oben. Nur ueber die aktiven
    // Knoten, und inaktive werden hart auf null gehalten.
    const auto deflate = [&](std::vector<double>& v) {
        double mean = 0.0;
        for (size_t i = 0; i < nn; ++i) if (active[i]) mean += v[i];
        mean /= static_cast<double>(n_active);
        for (size_t i = 0; i < nn; ++i) v[i] = active[i] ? (v[i] - mean) : 0.0;
    };
    const auto dot = [](const std::vector<double>& a, const std::vector<double>& c) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * c[i];
        return s;
    };

    std::vector<double> rhs(nn), phi(nn, 0.0), r(nn), p(nn), Ap(nn);
    std::vector<double> tmp_e(mesh.num_edges);

    for (size_t e = 0; e < mesh.num_edges; ++e) rep.b_norm_before += b[e] * b[e];
    rep.b_norm_before = std::sqrt(rep.b_norm_before);

    apply_Gt(b, rhs);
    deflate(rhs);
    double g0 = std::sqrt(dot(rhs, rhs));
    rep.rel_before = (rep.b_norm_before > 0.0) ? g0 / rep.b_norm_before : 0.0;

    if (!(g0 > 0.0)) {          // schon divergenzfrei, nichts zu tun
        rep.rel_after = rep.rel_before;
        rep.b_norm_after = rep.b_norm_before;
        rep.converged = true;
        return rep;
    }

    // CG auf (G^T G) phi = G^T b
    r = rhs;
    p = r;
    double rr = dot(r, r);
    for (int it = 0; it < max_iter; ++it) {
        apply_G(p, tmp_e);
        apply_Gt(tmp_e.data(), Ap);
        deflate(Ap);
        const double pAp = dot(p, Ap);
        if (!(pAp > 0.0)) break;      // Kernrichtung erreicht, mehr geht nicht
        const double alpha = rr / pAp;
        for (size_t i = 0; i < nn; ++i) {
            phi[i] += alpha * p[i];
            r[i]   -= alpha * Ap[i];
        }
        const double rr_new = dot(r, r);
        rep.cg_iters = it + 1;
        rep.cg_rel_res = std::sqrt(rr_new) / g0;
        if (rep.cg_rel_res < tol) { rep.converged = true; break; }
        const double beta = rr_new / rr;
        for (size_t i = 0; i < nn; ++i) p[i] = r[i] + beta * p[i];
        rr = rr_new;
    }

    // b <- b - G_F phi. Auf gesperrten Kanten ist tmp_e null, b bleibt dort
    // also null -- das ist die ganze Wirkung der Maske.
    apply_G(phi, tmp_e);
    for (size_t e = 0; e < mesh.num_edges; ++e) b[e] -= tmp_e[e];

    for (size_t e = 0; e < mesh.num_edges; ++e) rep.b_norm_after += b[e] * b[e];
    rep.b_norm_after = std::sqrt(rep.b_norm_after);
    apply_Gt(b, rhs);
    deflate(rhs);
    rep.rel_after = (rep.b_norm_after > 0.0)
                        ? std::sqrt(dot(rhs, rhs)) / rep.b_norm_after : 0.0;
    return rep;
}

}  // namespace fem_direct
