#ifndef NODAL_BEM_CUH
#define NODAL_BEM_CUH

// =============================================================================
// NodalBem.cuh -- vollbesetzte Randelementmethode mit Knotenbasis
// =============================================================================
//
// Teil des BASELINE-/REFERENZLOESERS (--reference). Bewusst KEINE ACA, keine
// Hierarchie, keine Clusterbaeume: V und K werden vollstaendig aufgebaut.
// Bei 4138 Randknoten sind das 2 * 4138^2 * 8 B = 0.26 GiB -- gegenueber dem
// 16-GiB-Budget vernachlaessigbar. Der Zweck dieser Datei ist
// Nachvollziehbarkeit, nicht Effizienz.
//
// -----------------------------------------------------------------------------
// Formulierung
// -----------------------------------------------------------------------------
// Aussenraum Omega_e (Luft) ist quellfrei, mu = mu_0 konstant, und mit
// Coulomb-Eichung ist jede kartesische Komponente von A dort harmonisch.
// Damit zerfaellt das Aussenproblem in drei identische skalare
// Laplace-Probleme -- V und K sind fuer alle drei Komponenten DIESELBE Matrix.
// Das ist der Grund, warum der Speicherbedarf nicht mit 3 skaliert.
//
//   G(x,y)      = 1 / (4*pi*|x-y|)                    (-Lap_y G = delta)
//   n           = Aussennormale von Omega, zeigt in die Luft
//   q           = dA/dn                               (Neumann-Spur)
//   (V q)(x)    = Int_Gamma G(x,y) q(y) dGamma_y      (Einfachschicht)
//   (K u)(x)    = PV Int_Gamma dG/dn_y (x,y) u(y) dGamma_y   (Doppelschicht)
//   dG/dn_y     = (x-y).n(y) / (4*pi*|x-y|^3)
//
// Green'sche Formel in D mit Aussennormale nu und Lap u = 0 liefert fuer x in D
//     u(x) = Int_dD [ G du/dnu - u dG/dnu ] dGamma
//
// INNEN  (D = Omega, nu = n),  Grenzwert von innen (Doppelschichtsprung -u/2):
//     (1/2 I + K) u = V q                                              (A)
// AUSSEN (D = Omega_e, nu = -n, Beitrag im Unendlichen faellt weg wie 1/R),
//        Grenzwert von aussen (Sprung +u/2):
//     V q = (K - 1/2 I) u                                              (B)
//
// (B) ist die hier benutzte Gleichung. Kontrolle an der Kugel mit Radius a und
// u = 1/r (aussen harmonisch): u = 1/a, q = -1/a^2, Int G dGamma = a, also
// V q = -1/a; und (K - I/2)u = -1/(2a) - 1/(2a) = -1/a. Stimmt.
// tests/test_reference_solver prueft (B) numerisch auf einem geschlossenen Netz.
//
// -----------------------------------------------------------------------------
// Diskretisierung: Kollokation in den Randknoten
// -----------------------------------------------------------------------------
// Kollokation statt Galerkin, weil fuer einen Referenzloeser Robustheit vor
// Symmetrie geht: es entfaellt das aeussere Integral und damit die gesamte
// Sauter-Schwab-Maschinerie fuer singulaere Flaechenpaare. Uebrig bleibt ein
// EINFACHES Flaechenintegral mit 1/r-Singularitaet, wenn der Kollokationspunkt
// auf dem Panel liegt -- und das ist ueber eine Dreieckszerlegung mit
// Polarkoordinaten (Duffy) exakt behandelbar, siehe unten.
//
// Ansatzfunktionen: bilineare Q1-Funktionen auf den Rand-Quads, stetig ueber
// Panelgrenzen hinweg. Sie sind die Spur der Volumen-Q1-Funktionen aus
// NodalHexahedron.cuh -- Dirichlet-Spur und FEM-Loesung leben damit exakt im
// selben Raum, was die Kopplung ohne Interpolationsfehler macht.
//
// -----------------------------------------------------------------------------
// Diagonale von K
// -----------------------------------------------------------------------------
// dG/dn_y ist stark singulaer (1/r^2); das Diagonalelement wird deshalb NICHT
// integriert, sondern aus der Konstantenbedingung von (A) gewonnen: u == 1 ist
// innen harmonisch mit q = 0, also (1/2 I + K)*1 = 0 und damit
//
//     K_ii = -1/2 - Summe_{j != i} K_ij
//
// Das ist exakt -- auch an Kanten und Ecken, wo der Raumwinkel nicht 2*pi ist;
// er steckt automatisch in der Zeilensumme. Und es gilt auch bei MEHREREN
// getrennten Randkomponenten: Gamma besteht hier aus drei geschlossenen
// Flaechen (Platte, Innenspule, Aussenspule). Fuer einen Kollokationspunkt auf
// der Platte liefert das Integral ueber eine Spulenflaeche exakt 0 (der Punkt
// liegt ausserhalb der Spule), das ueber die Plattenflaeche -1/2.
// =============================================================================

#include <cuda_runtime.h>

#include "Backend.cuh"
#include "BemIntegration.cuh"   // norm3, dot3, cross3, sub3, evaluate_quad_normal
#include "Mesh.cuh"

// -----------------------------------------------------------------------------
// Bilineare Flaechenbasis auf einem Quad
// -----------------------------------------------------------------------------
// Referenzquadrat (u,v) in [-1,1]^2, Knotenreihenfolge wie in QuadFace::v[4]
// (umlaufend). Die Spur der Volumen-Q1-Funktion auf einer Hexaederflaeche ist
// genau diese bilineare Funktion.
BF_HD inline void bem_quad_shape(double u, double v, double N[4]) {
    N[0] = 0.25 * (1.0 - u) * (1.0 - v);
    N[1] = 0.25 * (1.0 + u) * (1.0 - v);
    N[2] = 0.25 * (1.0 + u) * (1.0 + v);
    N[3] = 0.25 * (1.0 - u) * (1.0 + v);
}

/// Punkt, unnormierte Normale (= dx/du x dx/dv) und ihr Betrag (= Flaechen-
/// jacobian). Die Richtung folgt der Knotenumlaufrichtung von QuadFace; dass
/// sie nach AUSSEN zeigt, stellt der Aufrufer sicher (siehe
/// nodal_bem_face_orientation_sign()).
BF_HD inline void bem_quad_geometry(const double3 nd[4], double u, double v,
                                    double3& x, double3& normal_unnorm, double& jac)
{
    double N[4];
    bem_quad_shape(u, v, N);

    x = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < 4; ++i) {
        x.x += N[i] * nd[i].x;
        x.y += N[i] * nd[i].y;
        x.z += N[i] * nd[i].z;
    }

    // dN/du = ( -(1-v), (1-v), (1+v), -(1+v) ) / 4
    // dN/dv = ( -(1-u), -(1+u), (1+u), (1-u) ) / 4
    const double dNu[4] = { -(1.0 - v) * 0.25,  (1.0 - v) * 0.25,
                             (1.0 + v) * 0.25, -(1.0 + v) * 0.25 };
    const double dNv[4] = { -(1.0 - u) * 0.25, -(1.0 + u) * 0.25,
                             (1.0 + u) * 0.25,  (1.0 - u) * 0.25 };

    double3 xu = make_double3(0.0, 0.0, 0.0);
    double3 xv = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < 4; ++i) {
        xu.x += dNu[i] * nd[i].x;  xu.y += dNu[i] * nd[i].y;  xu.z += dNu[i] * nd[i].z;
        xv.x += dNv[i] * nd[i].x;  xv.y += dNv[i] * nd[i].y;  xv.z += dNv[i] * nd[i].z;
    }

    normal_unnorm = cross3(xu, xv);
    jac = norm3(normal_unnorm);
}

// -----------------------------------------------------------------------------
// Gauss-Legendre 1D auf [-1,1], N = 4 -- ausreichend fuer die regulaeren Paare
// -----------------------------------------------------------------------------
BF_HD inline int bem_gl_n() { return 4; }

BF_HD inline double bem_gl_x(int i) {
    const double x[4] = { -0.8611363115940526, -0.3399810435848563,
                           0.3399810435848563,  0.8611363115940526 };
    return x[i];
}
BF_HD inline double bem_gl_w(int i) {
    const double w[4] = { 0.3478548451374538, 0.6521451548625461,
                          0.6521451548625461, 0.3478548451374538 };
    return w[i];
}

// -----------------------------------------------------------------------------
// Kerne
// -----------------------------------------------------------------------------
BF_HD inline double bem_kernel_slp(const double3& x, const double3& y) {
    const double r = norm3(sub3(x, y));
    return (r > 0.0) ? (1.0 / (4.0 * 3.14159265358979323846 * r)) : 0.0;
}

/// dG/dn_y = (x-y).n_hat(y) / (4 pi |x-y|^3), mit n_hat der EINHEITSnormale.
/// Uebergeben wird die unnormierte Normale samt Jacobian, damit der Aufrufer
/// nicht doppelt normieren muss: n_hat = normal_unnorm / jac.
BF_HD inline double bem_kernel_dlp(const double3& x, const double3& y,
                                   const double3& normal_unnorm, double jac)
{
    const double3 d = sub3(x, y);          // x - y
    const double  r = norm3(d);
    if (r <= 0.0 || jac <= 0.0) return 0.0;
    const double r3 = r * r * r;
    return dot3(d, normal_unnorm) / (jac * 4.0 * 3.14159265358979323846 * r3);
}

// =============================================================================
// Regulaerer Beitrag eines Panels zu einer Kollokationszeile
// =============================================================================
//
// Berechnet fuer den Kollokationspunkt x und das Panel mit den Knoten nd[4]:
//     out_V[j] = Int_Panel G(x,y) N_j(y) dGamma_y
//     out_K[j] = Int_Panel dG/dn_y(x,y) N_j(y) dGamma_y
//
// 'subdiv' teilt das Panel in subdiv x subdiv Teilquadrate. Fuer Panels, die
// nahe an x liegen, ohne x zu enthalten, ist der Integrand glatt aber steil --
// eine feste 4x4-Regel unterschaetzt ihn dort deutlich. Der Aufrufer waehlt
// subdiv nach dem Abstands-zu-Groesse-Verhaeltnis
// (nodal_bem_subdivisions()).
BF_HD inline void nodal_bem_panel_regular(const double3& x, const double3 nd[4],
                                          double orient, int subdiv,
                                          double out_V[4], double out_K[4])
{
    for (int j = 0; j < 4; ++j) { out_V[j] = 0.0; out_K[j] = 0.0; }
    if (subdiv < 1) subdiv = 1;

    const double h = 2.0 / static_cast<double>(subdiv);   // Kantenlaenge im Referenzraum
    const int    n = bem_gl_n();

    for (int su = 0; su < subdiv; ++su) {
        const double u0 = -1.0 + su * h;
        for (int sv = 0; sv < subdiv; ++sv) {
            const double v0 = -1.0 + sv * h;

            for (int gu = 0; gu < n; ++gu) {
                // Abbildung [-1,1] -> [u0, u0+h]
                const double u = u0 + 0.5 * h * (bem_gl_x(gu) + 1.0);
                const double wu = bem_gl_w(gu) * 0.5 * h;

                for (int gv = 0; gv < n; ++gv) {
                    const double v = v0 + 0.5 * h * (bem_gl_x(gv) + 1.0);
                    const double wv = bem_gl_w(gv) * 0.5 * h;

                    double3 y, nrm; double jac;
                    bem_quad_geometry(nd, u, v, y, nrm, jac);
                    if (jac <= 0.0) continue;

                    double N[4];
                    bem_quad_shape(u, v, N);

                    const double w   = wu * wv * jac;
                    const double g   = bem_kernel_slp(x, y);
                    // orient dreht die Normale, falls die Knotenumlaufrichtung
                    // des Quads nach innen zeigt.
                    const double dg  = orient * bem_kernel_dlp(x, y, nrm, jac);

                    for (int j = 0; j < 4; ++j) {
                        out_V[j] += w * g  * N[j];
                        out_K[j] += w * dg * N[j];
                    }
                }
            }
        }
    }
}

// =============================================================================
// Singulaerer Beitrag: Kollokationspunkt IST ein Knoten des Panels
// =============================================================================
//
// Technik: Das Referenzquadrat wird von der singulaeren Ecke aus in zwei
// Dreiecke zerlegt und jedes Dreieck in Polarkoordinaten um diese Ecke
// integriert (Duffy). Das Flaechenelement rho*d(rho)*d(theta) hebt die
// 1/r-Singularitaet exakt auf, weil r ~ rho: der Integrand wird beschraenkt und
// eine gewoehnliche Gauss-Regel konvergiert. Verifiziert gegen den analytischen
// Wert Int_[0,1]^2 dA/r = 2*ln(1+sqrt(2)) fuer die Ecke des Einheitsquadrats.
//
// Behandelt werden BEIDE Operatoren, aber nicht alle Eintraege:
//
//   V (1/r):  alle vier Eintraege. Nach Duffy beschraenkt.
//
//   K (1/r^2): nur die drei Eintraege j != local_corner. Dort verschwindet
//     N_j an der Singularitaet linear, der Integrand geht also wie N_j/r^2
//     ~ rho/rho^2 = 1/rho und wird mit dem Flaechenelement rho beschraenkt --
//     schwach singulaer, Duffy genuegt. (Auf einem EBENEN Panel ist der
//     Beitrag ohnehin exakt null, weil dann (x-y) in der Panelebene liegt und
//     (x-y).n = 0; bei bilinear gekruemmten Quads ist er klein, aber nicht
//     null, und wird hier korrekt mitgenommen.)
//
//     Der Eintrag j == local_corner ist die Diagonale K_ii. Dort ist N_j -> 1
//     und 1/r^2 nicht integrierbar. Er wird hier auf 0 gesetzt und stattdessen
//     aus der Zeilensummenbedingung (1/2 I + K)*1 = 0 gewonnen, siehe
//     Dateikopf. Das ist der Grund, warum das stark singulaere Integral
//     nirgends ausgewertet werden muss.
//
// local_corner ist der lokale Index (0..3) des Knotens, der mit x
// zusammenfaellt.
BF_HD inline void nodal_bem_panel_singular(const double3 nd[4], int local_corner,
                                           double orient,
                                           double out_V[4], double out_K[4])
{
    for (int j = 0; j < 4; ++j) { out_V[j] = 0.0; out_K[j] = 0.0; }

    // Ecken des Referenzquadrats in der Reihenfolge von QuadFace::v
    const double cu[4] = { -1.0,  1.0,  1.0, -1.0 };
    const double cv[4] = { -1.0, -1.0,  1.0,  1.0 };

    const double su = cu[local_corner];
    const double sv = cv[local_corner];

    // Die beiden dem singulaeren Knoten gegenueberliegenden Kanten bilden die
    // zwei Dreiecke. Ecken relativ zur Singularitaet: die uebrigen drei Ecken
    // in Umlaufreihenfolge, das Dreieckspaar ist (c1,c2) und (c2,c3).
    const int c1 = (local_corner + 1) & 3;
    const int c2 = (local_corner + 2) & 3;
    const int c3 = (local_corner + 3) & 3;
    const int tri[2][2] = { { c1, c2 }, { c2, c3 } };

    const int n = bem_gl_n();

    // Kollokationspunkt in physikalischen Koordinaten (Panelknoten)
    const double3 x = nd[local_corner];

    for (int t = 0; t < 2; ++t) {
        const double au = cu[tri[t][0]] - su, av = cv[tri[t][0]] - sv;
        const double bu = cu[tri[t][1]] - su, bv = cv[tri[t][1]] - sv;

        // Dreieck in Referenzkoordinaten: (su,sv) + s*(au,av) + s*w*((bu,bv)-(au,av))
        // Standard-Duffy: s in [0,1] ist die radiale, w in [0,1] die
        // Winkelvariable; das Flaechenelement traegt den Faktor s, der die
        // 1/r-Singularitaet aufhebt.
        for (int gs = 0; gs < n; ++gs) {
            const double s  = 0.5 * (bem_gl_x(gs) + 1.0);      // [0,1]
            const double ws = bem_gl_w(gs) * 0.5;

            for (int gw = 0; gw < n; ++gw) {
                const double w_ = 0.5 * (bem_gl_x(gw) + 1.0);  // [0,1]
                const double ww = bem_gl_w(gw) * 0.5;

                const double du = au + w_ * (bu - au);
                const double dv = av + w_ * (bv - av);

                const double u = su + s * du;
                const double v = sv + s * dv;

                double3 y, nrm; double jac;
                bem_quad_geometry(nd, u, v, y, nrm, jac);
                if (jac <= 0.0) continue;

                double N[4];
                bem_quad_shape(u, v, N);

                // Jacobian der Duffy-Abbildung (u,v) <- (s,w):
                //   d(u,v)/d(s,w) = s * | (du,dv) x d/dw(du,dv) |
                //                 = s * |du*(bv-av) - dv*(bu-au)|
                const double duffy = s * fabs(du * (bv - av) - dv * (bu - au));

                const double r = norm3(sub3(x, y));
                if (r <= 0.0) continue;   // exakt in der Singularitaet: Mass 0

                const double weight = ws * ww * duffy * jac;
                const double g  = 1.0 / (4.0 * 3.14159265358979323846 * r);
                const double dg = orient * bem_kernel_dlp(x, y, nrm, jac);

                for (int j = 0; j < 4; ++j) {
                    out_V[j] += weight * g * N[j];
                    // j == local_corner bleibt ausgeschlossen: dort ist der
                    // Integrand nicht integrierbar, der Eintrag kommt aus der
                    // Zeilensumme (siehe Funktionskommentar).
                    if (j != local_corner) out_K[j] += weight * dg * N[j];
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Wahl der Unterteilung nach dem Abstands-zu-Groesse-Verhaeltnis
// -----------------------------------------------------------------------------
// Fuer weit entfernte Panels genuegt 1 (also 4x4 Gauss), fuer nahe wird
// unterteilt. Die Schwellen sind bewusst konservativ -- der Referenzloeser
// darf langsam sein, aber nicht ungenau.
BF_HD inline int nodal_bem_subdivisions(double distance, double panel_size) {
    if (panel_size <= 0.0) return 1;
    const double ratio = distance / panel_size;
    if (ratio > 6.0) return 1;
    if (ratio > 3.0) return 2;
    if (ratio > 1.5) return 3;
    return 5;
}

/// Charakteristische Panelgroesse: Wurzel der ueber die Ecken geschaetzten
/// Flaeche. Nur fuer die Heuristik oben, keine Genauigkeitsanforderung.
BF_HD inline double nodal_bem_panel_size(const double3 nd[4]) {
    const double3 d1 = sub3(nd[2], nd[0]);
    const double3 d2 = sub3(nd[3], nd[1]);
    const double area = 0.5 * norm3(cross3(d1, d2));
    return sqrt(area > 0.0 ? area : 0.0);
}

BF_HD inline double3 nodal_bem_panel_center(const double3 nd[4]) {
    return make_double3(0.25 * (nd[0].x + nd[1].x + nd[2].x + nd[3].x),
                        0.25 * (nd[0].y + nd[1].y + nd[2].y + nd[3].y),
                        0.25 * (nd[0].z + nd[1].z + nd[2].z + nd[3].z));
}

#endif // NODAL_BEM_CUH
