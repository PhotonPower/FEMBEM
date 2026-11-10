#ifndef TRACE_BEM_OPS_H
#define TRACE_BEM_OPS_H

// =============================================================================
// Randoperatoren der Maxwell-Spur-Kopplung, dicht, Double, Host (Phase 1b, Schritt 2)
// =============================================================================
// Theorie: docs/theory/maxwell-trace-coupling.md, Abschnitte 3.3, 5.2, 6.1. Alle Bloecke Galerkin.
// Die Quadratur steht in src/TraceBemQuad.cu (Sauter-Schwab-Transformationen aus SauterSchwab.cuh /
// SingularQuadPair.cuh, mit frei waehlbarer Gauss-Ordnung); diese Schnittstelle ist reines C++.
//
// Bloecke (F = Quads, V = Knoten, E = Kanten; zeilenweise Speicherung):
//   V0   F x F  (1/(|p||q|)) int_p int_q G                           P0-Einfachschicht (g = (R a)/|q| je Quad)
//   Dt   F x V  (1/|p|) int_p int_q (x-y).n_y/(4 pi |x-y|^3) N_j(y)   Doppelschicht, P0 x Q1
//   Mt   F x V  (1/|p|) int_p N_j                                    Masse, P0 x Q1
//   Vrot E x E  int int G (n_x x W_e(x)).(n_y x W_f(y))               gedrehte Rooftop-Einfachschicht
//   Cm   E x E  int int W_e(x).[ n_x x (grad_x G x W_f(y)) ]          Maxwell-Doppelschicht, = <W_e, n x C W_f>
//   Mm   E x E  int W_e.W_f                                          Rooftop-Masse
// W_e = kovariante Rooftop-Funktion der globalen Kante e (Kantenintegral 1, Orientierung vmin->vmax).
// G(x,y) = 1/(4 pi |x-y|),  grad_x G = -(x-y)/(4 pi |x-y|^3).
// =============================================================================

#include <functional>
#include <vector>

#include "TraceSurface.h"

namespace trace {

struct QuadParams {
    int ng_sing = 8;      // Gauss-Punkte je Richtung in den singulaeren Regeln (Sauter-Schwab bzw. Duffy-Regel fuer das gleiche Quad; 4D: ng^4 je Teilgebiet)
    int ng_reg = 4;       // Gauss-Punkte je Richtung und Teilquad fuer getrennte Paare
    int max_sub = 4;      // maximale Unterteilung je Richtung bei nahen getrennten Paaren
    double near_c = 1.0;  // Unterteilung m = ceil(near_c * h / d), h = Quadgroesse, d = Abstand der Quads
};

struct BoundaryOps {
    int F = 0, V = 0, E = 0;
    std::vector<double> area;                // |q|
    std::vector<double> V0, Dt, Mt, Vrot, Cm, Mm;
    std::vector<double> VrotFlat;   // V mit koplanarem Skalarprodukt W_e.W_f statt (n x W_e).(n x W_f): NUR Gegenprobe (T3)
    std::vector<double> M1;         // V x V, Q1-Masse int N_i N_j
    // Asymmetrie von V0, Vrot vor der Symmetrisierung, ||A - A^T||_F / ||A||_F: Mass der Quadraturkonsistenz. V0, Vrot und
    // VrotFlat werden am Ende als (A + A^T)/2 gespeichert (die Galerkin-Matrizen sind exakt symmetrisch).
    double asym_V0 = 0.0, asym_Vrot = 0.0;
};

// nthreads = 0: hardware_concurrency
BoundaryOps assemble_boundary_ops(const TraceSurface& S, const QuadParams& qp = QuadParams(), int nthreads = 0);

// Einzelne Quadpaar-Integrale fuer die Tests der Quadratur (Selbstkonvergenz): 4x4-Block Vrot (lokale Kanten, ohne Vorzeichen)
// und Skalarwerte int int G, int int K N_j fuer das Quadpaar (p, q).
struct PairResult {
    double v0 = 0.0;
    double dt[4] = {0, 0, 0, 0};
    double vr[4][4] = {};
    double vrf[4][4] = {};   // koplanares Skalarprodukt, nur Gegenprobe
    double cm[4][4] = {};
};
PairResult integrate_quad_pair(const TraceSurface& S, int p, int q, const QuadParams& qp = QuadParams());

// Aussenfeld aus Spuren (Darstellungsformeln (R-A), (R-B) der Theorie, Abschnitt 2.1) an einem Punkt x ausserhalb:
//   A(x) = mu0 Psi(lambda) + curl Psi(n x A_t) - grad Psi(n.A),   B(x) = mu0 curl Psi(lambda) - grad Psi(n.B)
// lambda = sum_e lam[e] n x W_e,   n x A_t = sum_e at[e] n x W_e,   n.B = g_p0[q] (P0), n.A = nA_p0[q] (P0).
// sign_* skalieren die drei Terme (Gegenproben: -1).
struct ExteriorTerms { double sign_lambda = 1.0, sign_trace = 1.0, sign_normal = 1.0; };
struct FieldPoint { Vec3 A, B; };
FieldPoint evaluate_exterior(const TraceSurface& S, const std::vector<double>& lam, const std::vector<double>& at,
                             const std::vector<double>& g_p0, const std::vector<double>& nA_p0, const Vec3& x, double mu0,
                             const ExteriorTerms& terms = ExteriorTerms(), int ng = 6, int sub = 3);

// Projektionen exakter Felder auf die diskreten Raeume (Gauss-Quadratur auf den Quads)
// int_q f(x, n) . W_e(x) dS fuer jede globale Kante e (Testfunktional <f, W_e>)
std::vector<double> project_on_rooftop(const TraceSurface& S, const std::function<Vec3(const Vec3&, const Vec3&)>& f, int ng = 6);
// Kantenintegrale int_e A . dl, Richtung vmin -> vmax (gerade Sehne), also Koeffizienten der kovarianten Rooftop-Interpolierenden
std::vector<double> edge_line_integrals(const TraceSurface& S, const std::function<Vec3(const Vec3&)>& A, int ng = 8);
// P0-Mittelwerte (1/|q|) int_q f(x, n) dS
std::vector<double> quad_average(const TraceSurface& S, const std::function<double(const Vec3&, const Vec3&)>& f, int ng = 6);

} // namespace trace

#endif
