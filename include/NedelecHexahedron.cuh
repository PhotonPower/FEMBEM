#ifndef NEDELEC_HEXAHEDRON_CUH
#define NEDELEC_HEXAHEDRON_CUH

#include <cuda_runtime.h>

// Da double3 in CUDA nativ definiert ist, verwenden wir es für Vektoren.
// Für Arrays von Knoten verwenden wir einfache Arrays konstanter Größe.

// =================================================================================
// Auswertung der 12 Nédélec-Basisvektoren (Whitney 1-Forms) 
// im Referenz-Hexaeder [-1, 1]^3
// =================================================================================
// HINWEIS ZUR ORIENTIERUNG (FIX):
// Die lokale Kantentabelle in TopologyExtractor::extract_edges lautet
//   {0,1},{1,2},{2,3},{3,0}, {4,5},{5,6},{6,7},{7,4}, {0,4},{1,5},{2,6},{3,7}
// Die Kanten 2 ({2,3}) und 6 ({6,7}) laufen damit in -xi-Richtung, die Kanten
// 3 ({3,0}) und 7 ({7,4}) in -eta-Richtung. Die Basisfunktion muss entlang der
// *lokalen Kantenrichtung* zeigen, weil hex_edge_signs nur die Relation
// lokale Kante <-> globale Kante (v_min -> v_max) korrigiert, nicht die Relation
// lokale Kante <-> Referenzachse. Vorher fehlte fuer 2,3,6,7 das Minuszeichen.
__host__ __device__ inline void evaluate_nedelec_basis(double xi, double eta, double zeta, double3 W[12]) {
    // Kanten in xi-Richtung (lokale Kanten 0, 2, 4, 6 nach VTK-Konvention für Hexaeder)
    W[0] = make_double3( 0.125 * (1.0 - eta) * (1.0 - zeta), 0.0, 0.0); // Kante {0,1}, +xi
    W[2] = make_double3(-0.125 * (1.0 + eta) * (1.0 - zeta), 0.0, 0.0); // Kante {2,3}, -xi
    W[4] = make_double3( 0.125 * (1.0 - eta) * (1.0 + zeta), 0.0, 0.0); // Kante {4,5}, +xi
    W[6] = make_double3(-0.125 * (1.0 + eta) * (1.0 + zeta), 0.0, 0.0); // Kante {6,7}, -xi
    
    // Kanten in eta-Richtung (lokale Kanten 1, 3, 5, 7)
    W[1] = make_double3(0.0,  0.125 * (1.0 + xi) * (1.0 - zeta), 0.0);  // Kante {1,2}, +eta
    W[3] = make_double3(0.0, -0.125 * (1.0 - xi) * (1.0 - zeta), 0.0);  // Kante {3,0}, -eta
    W[5] = make_double3(0.0,  0.125 * (1.0 + xi) * (1.0 + zeta), 0.0);  // Kante {5,6}, +eta
    W[7] = make_double3(0.0, -0.125 * (1.0 - xi) * (1.0 + zeta), 0.0);  // Kante {7,4}, -eta
    
    // Kanten in zeta-Richtung (lokale Kanten 8, 9, 10, 11)
    W[8]  = make_double3(0.0, 0.0,  0.125 * (1.0 - xi) * (1.0 - eta));  // Kante x=-1, y=-1
    W[9]  = make_double3(0.0, 0.0,  0.125 * (1.0 + xi) * (1.0 - eta));  // Kante x=+1, y=-1
    W[10] = make_double3(0.0, 0.0,  0.125 * (1.0 + xi) * (1.0 + eta));  // Kante x=+1, y=+1
    W[11] = make_double3(0.0, 0.0,  0.125 * (1.0 - xi) * (1.0 + eta));  // Kante x=-1, y=+1
}

// =================================================================================
// Auswertung der Rotation der Basisvektoren (curl W = nabla x W)
// im Referenz-Hexaeder [-1, 1]^3
// =================================================================================
// FIX: Diese Funktion war fuer 10 der 12 Kanten nicht die Rotation der oben
// definierten Basisfunktion (per zentraler Differenz nachgewiesen). Die Werte
// unten folgen direkt aus curl(W) fuer die korrigierte Basis:
//   nur W_x != 0 : curl = (0, dW_x/dzeta, -dW_x/deta)
//   nur W_y != 0 : curl = (-dW_y/dzeta, 0,  dW_y/dxi)
//   nur W_z != 0 : curl = ( dW_z/deta, -dW_z/dxi, 0)
__host__ __device__ inline void evaluate_nedelec_curl(double xi, double eta, double zeta, double3 curlW[12]) {
    // Rotation der xi-Kanten
    curlW[0] = make_double3(0.0, -0.125 * (1.0 - eta),  0.125 * (1.0 - zeta));
    curlW[2] = make_double3(0.0,  0.125 * (1.0 + eta),  0.125 * (1.0 - zeta));
    curlW[4] = make_double3(0.0,  0.125 * (1.0 - eta),  0.125 * (1.0 + zeta));
    curlW[6] = make_double3(0.0, -0.125 * (1.0 + eta),  0.125 * (1.0 + zeta));
    
    // Rotation der eta-Kanten
    curlW[1] = make_double3( 0.125 * (1.0 + xi), 0.0,  0.125 * (1.0 - zeta));
    curlW[3] = make_double3(-0.125 * (1.0 - xi), 0.0,  0.125 * (1.0 - zeta));
    curlW[5] = make_double3(-0.125 * (1.0 + xi), 0.0,  0.125 * (1.0 + zeta));
    curlW[7] = make_double3( 0.125 * (1.0 - xi), 0.0,  0.125 * (1.0 + zeta));
    
    // Rotation der zeta-Kanten
    curlW[8]  = make_double3(-0.125 * (1.0 - xi),  0.125 * (1.0 - eta), 0.0);
    curlW[9]  = make_double3(-0.125 * (1.0 + xi), -0.125 * (1.0 - eta), 0.0);
    curlW[10] = make_double3( 0.125 * (1.0 + xi), -0.125 * (1.0 + eta), 0.0);
    curlW[11] = make_double3( 0.125 * (1.0 - xi),  0.125 * (1.0 + eta), 0.0);
}

// =================================================================================
// Isoparametrisches Mapping: 
// Berechnung von Jacobi-Matrix (J), ihrer Determinante (detJ) und Inversen (invJ)
// =================================================================================
__host__ __device__ inline void compute_jacobian(
    const double3 nodes[8], 
    double xi, double eta, double zeta, 
    double J[3][3], double invJ[3][3], double& detJ) 
{
    // 1. Partielle Ableitungen der 8 trilinearen skalaren Formfunktionen N_i 
    // nach den lokalen Koordinaten (xi, eta, zeta).
    // Die VTK Knotenreihenfolge 0-7 lautet (im Referenzraum):
    // 0: (-1,-1,-1), 1: ( 1,-1,-1), 2: ( 1, 1,-1), 3: (-1, 1,-1)
    // 4: (-1,-1, 1), 5: ( 1,-1, 1), 6: ( 1, 1, 1), 7: (-1, 1, 1)
    
    double dN_dxi[8], dN_deta[8], dN_dzeta[8];

    // Ableitungen nach xi
    dN_dxi[0] = -0.125 * (1.0 - eta) * (1.0 - zeta);
    dN_dxi[1] =  0.125 * (1.0 - eta) * (1.0 - zeta);
    dN_dxi[2] =  0.125 * (1.0 + eta) * (1.0 - zeta);
    dN_dxi[3] = -0.125 * (1.0 + eta) * (1.0 - zeta);
    dN_dxi[4] = -0.125 * (1.0 - eta) * (1.0 + zeta);
    dN_dxi[5] =  0.125 * (1.0 - eta) * (1.0 + zeta);
    dN_dxi[6] =  0.125 * (1.0 + eta) * (1.0 + zeta);
    dN_dxi[7] = -0.125 * (1.0 + eta) * (1.0 + zeta);

    // Ableitungen nach eta
    dN_deta[0] = -0.125 * (1.0 - xi) * (1.0 - zeta);
    dN_deta[1] = -0.125 * (1.0 + xi) * (1.0 - zeta);
    dN_deta[2] =  0.125 * (1.0 + xi) * (1.0 - zeta);
    dN_deta[3] =  0.125 * (1.0 - xi) * (1.0 - zeta);
    dN_deta[4] = -0.125 * (1.0 - xi) * (1.0 + zeta);
    dN_deta[5] = -0.125 * (1.0 + xi) * (1.0 + zeta);
    dN_deta[6] =  0.125 * (1.0 + xi) * (1.0 + zeta);
    dN_deta[7] =  0.125 * (1.0 - xi) * (1.0 + zeta);

    // Ableitungen nach zeta
    dN_dzeta[0] = -0.125 * (1.0 - xi) * (1.0 - eta);
    dN_dzeta[1] = -0.125 * (1.0 + xi) * (1.0 - eta);
    dN_dzeta[2] = -0.125 * (1.0 + xi) * (1.0 + eta);
    dN_dzeta[3] = -0.125 * (1.0 - xi) * (1.0 + eta);
    dN_dzeta[4] =  0.125 * (1.0 - xi) * (1.0 - eta);
    dN_dzeta[5] =  0.125 * (1.0 + xi) * (1.0 - eta);
    dN_dzeta[6] =  0.125 * (1.0 + xi) * (1.0 + eta);
    dN_dzeta[7] =  0.125 * (1.0 - xi) * (1.0 + eta);

    // 2. Aufbau der 3x3 Jacobi-Matrix
    // J = [ dx/dxi   dy/dxi   dz/dxi ]
    //     [ dx/deta  dy/deta  dz/deta]
    //     [ dx/dzeta dy/dzeta dz/dzeta]
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            J[r][c] = 0.0;
        }
    }

    for (int i = 0; i < 8; ++i) {
        J[0][0] += dN_dxi[i]   * nodes[i].x;
        J[0][1] += dN_dxi[i]   * nodes[i].y;
        J[0][2] += dN_dxi[i]   * nodes[i].z;

        J[1][0] += dN_deta[i]  * nodes[i].x;
        J[1][1] += dN_deta[i]  * nodes[i].y;
        J[1][2] += dN_deta[i]  * nodes[i].z;

        J[2][0] += dN_dzeta[i] * nodes[i].x;
        J[2][1] += dN_dzeta[i] * nodes[i].y;
        J[2][2] += dN_dzeta[i] * nodes[i].z;
    }

    // 3. Berechnung der Determinante von J (detJ)
    detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
         - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
         + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

    // Wenn die Determinante Null oder negativ ist, ist das Hexaeder degeneriert oder invertiert.
    // In einer CUDA-__device__ Funktion können wir keine Exception werfen. Wir setzen detJ auf ein Epsilon,
    // um Division durch Null zu vermeiden, die Ergebnisse für dieses Element sind dann aber unphysikalisch.
    double invDet = 0.0;
    if (detJ > 1e-14 || detJ < -1e-14) {
        invDet = 1.0 / detJ;
    }

    // 4. Berechnung der Inversen von J (mittels der Adjunkten / Cramer'sche Regel)
    invJ[0][0] = (J[1][1] * J[2][2] - J[1][2] * J[2][1]) * invDet;
    invJ[0][1] = (J[0][2] * J[2][1] - J[0][1] * J[2][2]) * invDet;
    invJ[0][2] = (J[0][1] * J[1][2] - J[0][2] * J[1][1]) * invDet;

    invJ[1][0] = (J[1][2] * J[2][0] - J[1][0] * J[2][2]) * invDet;
    invJ[1][1] = (J[0][0] * J[2][2] - J[0][2] * J[2][0]) * invDet;
    invJ[1][2] = (J[0][2] * J[1][0] - J[0][0] * J[1][2]) * invDet;

    invJ[2][0] = (J[1][0] * J[2][1] - J[1][1] * J[2][0]) * invDet;
    invJ[2][1] = (J[0][1] * J[2][0] - J[0][0] * J[2][1]) * invDet;
    invJ[2][2] = (J[0][0] * J[1][1] - J[0][1] * J[1][0]) * invDet;
}

#endif // NEDELEC_HEXAHEDRON_CUH