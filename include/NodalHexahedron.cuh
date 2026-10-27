#ifndef NODAL_HEXAHEDRON_CUH
#define NODAL_HEXAHEDRON_CUH

// =============================================================================
// NodalHexahedron.cuh -- trilineare Q1-Knotenformfunktionen fuer Hexaeder
// =============================================================================
//
// Teil des BASELINE-/REFERENZLOESERS (--reference), nicht des produktiven
// Nedelec-Pfads. Der produktive Loeser benutzt Kantenelemente
// (NedelecHexahedron.cuh); dieser hier Knotenelemente mit drei kartesischen
// Freiheitsgraden je Knoten:
//
//     A(x) = Summe_i N_i(x) * (Ax_i, Ay_i, Az_i)
//
// Beide Dateien existieren bewusst parallel und werden nicht vermischt.
//
// Knotennummerierung: dieselbe wie in Hexahedron::v[8] und in
// NedelecHexahedron.cuh, damit hex_nodes ohne Umsortierung nutzbar ist:
//
//        7---------6          zeta
//       /|        /|           |   eta
//      4---------5 |           |  /
//      | |       | |           | /
//      | 3-------|-2           |/
//      |/        |/            *------- xi
//      0---------1
//
//   Knoten 0..3: zeta = -1,  Knoten 4..7: zeta = +1
//
// Alles ist __host__ __device__, damit der Referenzloeser wie der uebrige
// Code sowohl auf der GPU als auch mit --cpu laeuft (siehe Backend.cuh).
// =============================================================================

#include <cuda_runtime.h>

#include "Backend.cuh"

// Vorzeichentabellen der acht Knoten im Referenzelement [-1,1]^3.
// Als Funktionen statt als __constant__-Arrays, damit sie auf beiden
// Ausfuehrungspfaden lesbar sind (siehe SauterSchwab.cuh, gleiche Begruendung).
BF_HD inline double nodal_sign_xi(int i) {
    const double s[8] = { -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0, -1.0 };
    return s[i];
}
BF_HD inline double nodal_sign_eta(int i) {
    const double s[8] = { -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0,  1.0 };
    return s[i];
}
BF_HD inline double nodal_sign_zeta(int i) {
    const double s[8] = { -1.0, -1.0, -1.0, -1.0,  1.0,  1.0,  1.0,  1.0 };
    return s[i];
}

// -----------------------------------------------------------------------------
// Formfunktionen und ihre Ableitungen im Referenzelement
// -----------------------------------------------------------------------------
//   N_i = 1/8 (1 + s_xi,i * xi)(1 + s_eta,i * eta)(1 + s_zeta,i * zeta)
BF_HD inline void evaluate_nodal_shape(double xi, double eta, double zeta,
                                       double N[8])
{
    for (int i = 0; i < 8; ++i) {
        N[i] = 0.125 * (1.0 + nodal_sign_xi(i)   * xi)
                     * (1.0 + nodal_sign_eta(i)  * eta)
                     * (1.0 + nodal_sign_zeta(i) * zeta);
    }
}

/// dN_i/d(xi,eta,zeta), Zeile i = Gradient der i-ten Formfunktion im
/// Referenzraum.
BF_HD inline void evaluate_nodal_shape_grad_ref(double xi, double eta, double zeta,
                                               double dN[8][3])
{
    for (int i = 0; i < 8; ++i) {
        const double sx = nodal_sign_xi(i);
        const double sy = nodal_sign_eta(i);
        const double sz = nodal_sign_zeta(i);
        dN[i][0] = 0.125 * sx * (1.0 + sy * eta) * (1.0 + sz * zeta);
        dN[i][1] = 0.125 * sy * (1.0 + sx * xi)  * (1.0 + sz * zeta);
        dN[i][2] = 0.125 * sz * (1.0 + sx * xi)  * (1.0 + sy * eta);
    }
}

// -----------------------------------------------------------------------------
// Jacobi-Matrix und physikalische Gradienten
// -----------------------------------------------------------------------------
//
// Bewusst eine eigene Funktion und nicht compute_jacobian() aus
// NedelecHexahedron.cuh: dort wird J aus denselben Q1-Formfunktionen gebildet,
// die Rueckgabe ist aber auf die Bedarfe der Kantenelemente zugeschnitten
// (invJ fuer die kovariante Piola-Transformation). Hier werden direkt die
// physikalischen Gradienten grad N_i gebraucht, und ein zweiter Durchlauf
// ueber dieselben Zahlen ist billiger als die Umrechnung an jeder Aufrufstelle.
//
// Rueckgabe:
//   gradN[i] = grad N_i im physikalischen Raum  (= invJ^T * dN_ref[i])
//   detJ     = Determinante, muss > 0 sein (Rechtssystem-Orientierung)
//
// Ergibt detJ <= 0, ist das Element entartet oder falsch orientiert; der
// Aufrufer muss das melden, nicht stillschweigend weiterrechnen.
BF_HD inline void evaluate_nodal_gradients(const double3 nodes[8],
                                          double xi, double eta, double zeta,
                                          double3 gradN[8], double& detJ)
{
    double dN[8][3];
    evaluate_nodal_shape_grad_ref(xi, eta, zeta, dN);

    // J[a][b] = d x_a / d r_b
    double J[3][3] = { {0,0,0}, {0,0,0}, {0,0,0} };
    for (int i = 0; i < 8; ++i) {
        J[0][0] += dN[i][0] * nodes[i].x;
        J[0][1] += dN[i][1] * nodes[i].x;
        J[0][2] += dN[i][2] * nodes[i].x;
        J[1][0] += dN[i][0] * nodes[i].y;
        J[1][1] += dN[i][1] * nodes[i].y;
        J[1][2] += dN[i][2] * nodes[i].y;
        J[2][0] += dN[i][0] * nodes[i].z;
        J[2][1] += dN[i][1] * nodes[i].z;
        J[2][2] += dN[i][2] * nodes[i].z;
    }

    detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
         - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
         + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

    const double inv_det = (detJ != 0.0) ? (1.0 / detJ) : 0.0;

    // invJ = adj(J)/detJ
    double invJ[3][3];
    invJ[0][0] =  (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * inv_det;
    invJ[0][1] = -(J[0][1] * J[2][2] - J[0][2] * J[2][1]) * inv_det;
    invJ[0][2] =  (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * inv_det;
    invJ[1][0] = -(J[1][0] * J[2][2] - J[1][2] * J[2][0]) * inv_det;
    invJ[1][1] =  (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * inv_det;
    invJ[1][2] = -(J[0][0] * J[1][2] - J[0][2] * J[1][0]) * inv_det;
    invJ[2][0] =  (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * inv_det;
    invJ[2][1] = -(J[0][0] * J[2][1] - J[0][1] * J[2][0]) * inv_det;
    invJ[2][2] =  (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * inv_det;

    // grad N_i = invJ^T * dN_ref[i]   (Kettenregel: dN/dx_a = dN/dr_b * dr_b/dx_a)
    for (int i = 0; i < 8; ++i) {
        gradN[i].x = invJ[0][0] * dN[i][0] + invJ[1][0] * dN[i][1] + invJ[2][0] * dN[i][2];
        gradN[i].y = invJ[0][1] * dN[i][0] + invJ[1][1] * dN[i][1] + invJ[2][1] * dN[i][2];
        gradN[i].z = invJ[0][2] * dN[i][0] + invJ[1][2] * dN[i][1] + invJ[2][2] * dN[i][2];
    }
}

/// Physikalische Lage eines Referenzpunkts.
BF_HD inline double3 nodal_map_to_physical(const double3 nodes[8],
                                           double xi, double eta, double zeta)
{
    double N[8];
    evaluate_nodal_shape(xi, eta, zeta, N);
    double3 p = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < 8; ++i) {
        p.x += N[i] * nodes[i].x;
        p.y += N[i] * nodes[i].y;
        p.z += N[i] * nodes[i].z;
    }
    return p;
}

// -----------------------------------------------------------------------------
// 2x2x2 Gauss-Legendre auf [-1,1]^3
// -----------------------------------------------------------------------------
// Fuer Q1-Elemente ist die 2-Punkt-Regel je Richtung exakt fuer die
// Steifigkeitsmatrix auf achsparallelen Elementen und ausreichend genau auf
// verzerrten. Dieselbe Regel benutzt auch der produktive Nedelec-Pfad
// (assemble_local_matrices_kernel), damit die Quadratur beim Vergleich der
// beiden Loeser nicht als Unterschied durchschlaegt.
// -----------------------------------------------------------------------------
// Waehlbare Ordnung: 2 oder 3 Punkte je Richtung
// -----------------------------------------------------------------------------
// Warum ueberhaupt waehlbar? Fuer die Steifigkeitsmatrix eines Q1-Elements ist
// die 2-Punkt-Regel auf achsparallelen Elementen exakt. Bei VERZERRTEN
// Elementen -- und die Platte ist ein mit Hexaedern vernetzter Zylinder --
// enthaelt der Integrand ueber invJ rationale Anteile, und die
// Lorentzkraftdichte J x B ist ein PRODUKT zweier aus A abgeleiteter Groessen.
// Deren Polynomgrad liegt hoeher als der von A selbst, die 2-Punkt-Regel ist
// dort also nur knapp ausreichend. Da die Kraft die empfindlichste
// Ausgabegroesse ist, wird sie mit beiden Ordnungen gerechnet: stimmen die
// Werte ueberein, ist die Quadratur nicht die Fehlerquelle, und man kann sich
// die Suche dort sparen.
BF_HD inline int nodal_gauss_count_n(int order) { return order * order * order; }

BF_HD inline void nodal_gauss_1d(int order, int i, double& x, double& w) {
    if (order <= 2) {
        const double p = 0.5773502691896257;      // 1/sqrt(3)
        x = (i == 0) ? -p : p;
        w = 1.0;
        return;
    }
    // 3-Punkt-Gauss-Legendre auf [-1,1]
    const double p = 0.7745966692414834;          // sqrt(3/5)
    const double xs[3] = { -p, 0.0, p };
    const double ws[3] = { 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0 };
    x = xs[i]; w = ws[i];
}

BF_HD inline void nodal_gauss_point_n(int order, int g,
                                      double& xi, double& eta, double& zeta, double& w)
{
    const int n = (order <= 2) ? 2 : 3;
    const int ix =  g % n;
    const int iy = (g / n) % n;
    const int iz =  g / (n * n);
    double wx, wy, wz;
    nodal_gauss_1d(n, ix, xi,   wx);
    nodal_gauss_1d(n, iy, eta,  wy);
    nodal_gauss_1d(n, iz, zeta, wz);
    w = wx * wy * wz;
}

BF_HD inline int  nodal_gauss_count() { return 8; }

BF_HD inline void nodal_gauss_point(int g, double& xi, double& eta, double& zeta,
                                    double& w)
{
    const double p = 0.5773502691896257;   // 1/sqrt(3)
    const double sx[8] = { -1,  1, -1,  1, -1,  1, -1,  1 };
    const double sy[8] = { -1, -1,  1,  1, -1, -1,  1,  1 };
    const double sz[8] = { -1, -1, -1, -1,  1,  1,  1,  1 };
    xi   = sx[g] * p;
    eta  = sy[g] * p;
    zeta = sz[g] * p;
    w    = 1.0;   // Produkt der drei 1D-Gewichte (je 1.0 auf [-1,1])
}

#endif // NODAL_HEXAHEDRON_CUH
