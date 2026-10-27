#ifndef NEDELEC_DISPATCH_CUH
#define NEDELEC_DISPATCH_CUH

#include "Mesh.cuh"
#include "NedelecHexahedron.cuh"
#include "NedelecTetrahedron.cuh"
#include "NedelecPrism.cuh"

// =============================================================================
// Typabhaengige Auswertung an EINER Stelle
// =============================================================================
//
// WOZU
//
// Mit dem FEM-direkt-Pfad gibt es drei Elementtypen. Die Auswertung von Basis,
// Rotation, Jacobi-Matrix, Formfunktionen und Quadratur unterscheidet sich je
// Typ, gebraucht wird sie aber an mindestens fuenf Stellen: Assemblierung
// (fem.cu), B-Feld-Ausgabe (VtkExporter), Lorentzkraft (HarmonicLorentz),
// Maxwell-Spannungstensor (TransientSolver) und die Referenzloeser.
//
// Wuerde die Verzweigung dort jeweils von Hand stehen, waere sie fuenfmal zu
// pflegen -- und ein vergessener Zweig faellt nicht als Absturz auf, sondern
// als falsche Kraft. Deshalb steht sie hier und nur hier.
//
// KONVENTIONEN
//
// Unveraendert die des Hexaeders: J[i][j] = d x_j / d xi_i, also J = J_std^T.
// Daraus folgt fuer alle Typen gleich:
//
//     W_phys    = invJ * W_ref                    (kovariant, ZEILENweise)
//     rot W_phys = (1/detJ) * J_std * rot W_ref   (kontravariant, SPALTENweise
//                                                  ueber das hier vorliegende J)
//
// Die Referenzkoordinaten heissen hier r, s, t. Beim Hexaeder sind das
// xi, eta, zeta in [-1,1]^3, beim Tetraeder die ersten drei baryzentrischen
// Koordinaten in [0,1], beim Prisma zwei Flaechenkoordinaten des Dreiecks und
// zeta in [-1,1].

// -----------------------------------------------------------------------------
// Basis und Rotation im Referenzelement
// -----------------------------------------------------------------------------
__host__ __device__ inline void evaluate_element_basis(
    int type, double r, double s, double t, double3 W[12])
{
    if (type == ELEM_TET)        evaluate_nedelec_basis_tet(r, s, t, W);
    else if (type == ELEM_PRISM) evaluate_nedelec_basis_prism(r, s, t, W);
    else                         evaluate_nedelec_basis(r, s, t, W);
}

__host__ __device__ inline void evaluate_element_curl(
    int type, double r, double s, double t, double3 curlW[12])
{
    if (type == ELEM_TET)        evaluate_nedelec_curl_tet(r, s, t, curlW);
    else if (type == ELEM_PRISM) evaluate_nedelec_curl_prism(r, s, t, curlW);
    else                         evaluate_nedelec_curl(r, s, t, curlW);
}

// -----------------------------------------------------------------------------
// Isoparametrische Abbildung
// -----------------------------------------------------------------------------
__host__ __device__ inline void compute_element_jacobian(
    int type, const double3 nodes[8], double r, double s, double t,
    double J[3][3], double invJ[3][3], double& detJ)
{
    if (type == ELEM_TET)        compute_jacobian_tet(nodes, r, s, t, J, invJ, detJ);
    else if (type == ELEM_PRISM) compute_jacobian_prism(nodes, r, s, t, J, invJ, detJ);
    else                         compute_jacobian(nodes, r, s, t, J, invJ, detJ);
}

// Skalare Formfunktionen, fuer die Lage des Auswertepunkts im physikalischen
// Raum (Stromrichtung in den Spulen, Ausgabekoordinate).
// N hat immer die Laenge 8; ungenutzte Plaetze werden auf null gesetzt.
__host__ __device__ inline void element_shape_functions(
    int type, double r, double s, double t, double N[8])
{
    for (int i = 0; i < 8; ++i) N[i] = 0.0;

    if (type == ELEM_TET) {
        tet_shape_functions(r, s, t, N);
    } else if (type == ELEM_PRISM) {
        prism_shape_functions(r, s, t, N);
    } else {
        const double sgX[8] = {-1, 1, 1,-1,-1, 1, 1,-1};
        const double sgY[8] = {-1,-1, 1, 1,-1,-1, 1, 1};
        const double sgZ[8] = {-1,-1,-1,-1, 1, 1, 1, 1};
        for (int i = 0; i < 8; ++i)
            N[i] = 0.125 * (1.0 + sgX[i]*r) * (1.0 + sgY[i]*s) * (1.0 + sgZ[i]*t);
    }
}

// -----------------------------------------------------------------------------
// Quadratur
// -----------------------------------------------------------------------------
// Hoechste Punktzahl ueber alle Typen -- fuer feste Feldgroessen im Aufrufer.
#define ELEMENT_MAX_QUAD_POINTS 8

__host__ __device__ inline int element_num_quad_points(int type) {
    if (type == ELEM_TET)   return TET_NUM_QUAD_POINTS;     // 4
    if (type == ELEM_PRISM) return PRISM_NUM_QUAD_POINTS;   // 6
    return 8;                                               // 2x2x2 Gauss
}

__host__ __device__ inline void element_quadrature_point(
    int type, int q, double& r, double& s, double& t, double& weight)
{
    if (type == ELEM_TET) {
        tet_quadrature_point(q, r, s, t, weight);
        return;
    }
    if (type == ELEM_PRISM) {
        prism_quadrature_point(q, r, s, t, weight);
        return;
    }
    const double gp = 0.5773502691896257;
    r = (q & 1) ? gp : -gp;
    s = (q & 2) ? gp : -gp;
    t = (q & 4) ? gp : -gp;
    weight = 1.0;   // Gauss-Gewicht 1*1*1
}

// -----------------------------------------------------------------------------
// Mittelpunkt des Referenzelements
// -----------------------------------------------------------------------------
// Fuer die zellweise Ausgabe eines Feldwerts (VtkExporter). Beim Hexaeder ist
// das (0,0,0), beim Tetraeder der Schwerpunkt (1/4,1/4,1/4), beim Prisma
// (1/3,1/3,0).
__host__ __device__ inline void element_reference_center(
    int type, double& r, double& s, double& t)
{
    if (type == ELEM_TET)        { r = s = 0.25; t = 0.25; return; }
    if (type == ELEM_PRISM)      { r = s = 1.0 / 3.0; t = 0.0; return; }
    r = s = t = 0.0;
}

// -----------------------------------------------------------------------------
// Piola-Transformationen, damit die Formeln nur einmal dastehen
// -----------------------------------------------------------------------------
__host__ __device__ inline double3 piola_covariant(
    const double invJ[3][3], const double3& w)
{
    return make_double3(invJ[0][0]*w.x + invJ[0][1]*w.y + invJ[0][2]*w.z,
                        invJ[1][0]*w.x + invJ[1][1]*w.y + invJ[1][2]*w.z,
                        invJ[2][0]*w.x + invJ[2][1]*w.y + invJ[2][2]*w.z);
}

__host__ __device__ inline double3 piola_contravariant(
    const double J[3][3], double detJ, const double3& c)
{
    return make_double3((J[0][0]*c.x + J[1][0]*c.y + J[2][0]*c.z) / detJ,
                        (J[0][1]*c.x + J[1][1]*c.y + J[2][1]*c.z) / detJ,
                        (J[0][2]*c.x + J[1][2]*c.y + J[2][2]*c.z) / detJ);
}

#endif // NEDELEC_DISPATCH_CUH
