#ifndef NEDELEC_PRISM_CUH
#define NEDELEC_PRISM_CUH

#include <cuda_runtime.h>

// =============================================================================
// Nédélec-Kantenelement erster Art auf dem Prisma (CPENTA mit 6 Knoten)
// =============================================================================
//
// WOZU
//
// Der Netzgenerator (HyperMesh) liefert für ein Gebiet mit schmalen Luftspalten
// keine reinen Hexaedernetze. Verfügbar sind penta6 und penta15; penta15 wird
// abgelehnt (der Löser ist erster Ordnung, und die Mittenknoten einfach
// wegzuwerfen wäre bei gekrümmten Rändern eine Geometrieänderung). Bleibt
// penta6, und das ist genau ein neues Element.
//
// Ein reines Prismennetz ist ein extrudiertes Netz: Dreiecksvernetzung in x-y,
// Schichtung in z. Für TEAM 28 passt das, die Geometrie ist in z geschichtet.
//
// DAS REFERENZELEMENT
//
// Dreieck in Flächenkoordinaten, Extrusion in [-1, 1]:
//
//     Lambda_0 = 1 - r - s,  Lambda_1 = r,  Lambda_2 = s,   zeta in [-1, 1]
//
//     Knoten 0: (0,0,-1)   3: (0,0,+1)
//            1: (1,0,-1)   4: (1,0,+1)
//            2: (0,1,-1)   5: (0,1,+1)
//
// Das Intervall [-1, 1] und nicht [0, 1], damit die Extrusionsrichtung dieselbe
// Konvention hat wie beim Hexaeder. Die Dreiecksrichtungen laufen dagegen von 0
// bis 1, wie bei Flächenkoordinaten üblich.
//
// DIE 9 BASISFUNKTIONEN
//
// Tensorprodukt aus der Whitney-1-Form des Dreiecks und dem linearen Ansatz in
// zeta:
//
//   * 6 horizontale Kanten (3 unten, 3 oben):  w_e(r,s) * (1 -/+ zeta)/2
//     mit der 2D-Whitney-Form  w_ab = L_a grad L_b - L_b grad L_a
//   * 3 vertikale Kanten:  (1/2) * Lambda_i * e_zeta
//
// Der Faktor 1/2 der vertikalen Kanten kommt aus der Kantenlänge 2 im
// Referenzraum: die Kantenzirkulation muss 1 sein, nicht 2.
//
// KANTENNUMMERIERUNG
//
// In Analogie zum Hexaeder (erst Bodenfläche, dann Deckfläche, dann die
// vertikalen Verbindungen), siehe PRISM_LOCAL_EDGE_TO_NODES weiter unten.
// TopologyExtractor MUSS dieselbe Tabelle verwenden. Sie steht deshalb hier und
// nicht dort: die Basisfunktion und die Kantentabelle sind ein Paar, und wenn
// sie auseinanderlaufen, entsteht kein Absturz, sondern eine falsche Kraft.
//
// ORIENTIERUNG
//
// Wie beim Hexaeder zeigt die Basisfunktion entlang der LOKALEN Kantenrichtung
// (von vA nach vB der Tabelle). Die Relation lokale Kante <-> globale Kante
// (v_min -> v_max) korrigiert allein das Vorzeichenarray beim Gather/Scatter.
// Die lokalen Elementmatrizen bleiben unorientiert.
//
// EINBETTUNG IN DIE BESTEHENDEN DATENSTRUKTUREN
//
// Es ändert sich keine Arrayform. Hexahedron::v[8] fasst die 6 Knoten,
// hex_edges und hex_edge_signs mit Schrittweite 12 fassen die 9 Kanten, und
// SOLVER_LOCAL_MATRIX_SIZE = 144 (12x12) fasst die 9x9-Elementmatrix. Die
// Plätze 9..11 bleiben mit -1 (Kantenindex) beziehungsweise 0.0
// (Matrixeintrag) belegt; die Assemblierung verarbeitet nur Einträge mit
// Kantenindex >= 0.

#define PRISM_NUM_NODES 6
#define PRISM_NUM_EDGES 9

// =============================================================================
// Lokale Kantentabelle
// =============================================================================
// Die Kante 'e' (0 bis 8) läuft von Knoten vA nach vB.
__host__ __device__ inline void prism_local_edge_nodes(int e, int& vA, int& vB) {
    // {0,1},{1,2},{2,0}  Bodenfläche  (zeta = -1)
    // {3,4},{4,5},{5,3}  Deckfläche   (zeta = +1)
    // {0,3},{1,4},{2,5}  vertikale Verbindungskanten
    const int tbl[PRISM_NUM_EDGES][2] = {
        {0, 1}, {1, 2}, {2, 0},
        {3, 4}, {4, 5}, {5, 3},
        {0, 3}, {1, 4}, {2, 5}
    };
    vA = tbl[e][0];
    vB = tbl[e][1];
}

// =============================================================================
// Skalare Formfunktionen und ihre Ableitungen
// =============================================================================
// N_i = Lambda_(i mod 3) * (1 -/+ zeta)/2
__host__ __device__ inline void prism_shape_functions(
    double r, double s, double zeta, double N[PRISM_NUM_NODES])
{
    const double L[3] = {1.0 - r - s, r, s};
    const double zm = 0.5 * (1.0 - zeta);
    const double zp = 0.5 * (1.0 + zeta);
    for (int i = 0; i < 3; ++i) {
        N[i]     = L[i] * zm;
        N[i + 3] = L[i] * zp;
    }
}

__host__ __device__ inline void prism_shape_derivatives(
    double r, double s, double zeta,
    double dN_dr[PRISM_NUM_NODES],
    double dN_ds[PRISM_NUM_NODES],
    double dN_dzeta[PRISM_NUM_NODES])
{
    const double L[3]  = {1.0 - r - s, r, s};
    const double dLr[3] = {-1.0, 1.0, 0.0};
    const double dLs[3] = {-1.0, 0.0, 1.0};
    const double zm = 0.5 * (1.0 - zeta);
    const double zp = 0.5 * (1.0 + zeta);

    for (int i = 0; i < 3; ++i) {
        dN_dr[i]     = dLr[i] * zm;
        dN_dr[i + 3] = dLr[i] * zp;

        dN_ds[i]     = dLs[i] * zm;
        dN_ds[i + 3] = dLs[i] * zp;

        dN_dzeta[i]     = -0.5 * L[i];
        dN_dzeta[i + 3] =  0.5 * L[i];
    }
}

// =============================================================================
// Auswertung der 9 Nédélec-Basisvektoren im Referenzprisma
// =============================================================================
// Ausgabe in W[0..8]; W[9..11] werden auf null gesetzt, damit der Aufrufer
// dieselbe Schleife über 12 Einträge fahren kann wie beim Hexaeder.
//
// Die 2D-Whitney-Formen des Dreiecks:
//   w_01 = L0 grad L1 - L1 grad L0 = ( 1 - s,  r     )
//   w_12 = L1 grad L2 - L2 grad L1 = (    -s,  r     )
//   w_20 = L2 grad L0 - L0 grad L2 = (    -s,  r - 1 )
// Jede hat Zirkulation 1 entlang ihrer eigenen Kante und 0 entlang der beiden
// anderen; die Rotation ist konstant 2.
__host__ __device__ inline void evaluate_nedelec_basis_prism(
    double r, double s, double zeta, double3 W[12])
{
    const double zm = 0.5 * (1.0 - zeta);
    const double zp = 0.5 * (1.0 + zeta);

    const double w[3][2] = {
        { 1.0 - s,  r       },   // w_01
        {     -s,   r       },   // w_12
        {     -s,   r - 1.0 }    // w_20
    };

    // Horizontale Kanten: Bodenfläche 0..2, Deckfläche 3..5
    for (int i = 0; i < 3; ++i) {
        W[i]     = make_double3(w[i][0] * zm, w[i][1] * zm, 0.0);
        W[i + 3] = make_double3(w[i][0] * zp, w[i][1] * zp, 0.0);
    }

    // Vertikale Kanten 6..8: (1/2) * Lambda_i in zeta-Richtung
    const double L[3] = {1.0 - r - s, r, s};
    for (int i = 0; i < 3; ++i) {
        W[i + 6] = make_double3(0.0, 0.0, 0.5 * L[i]);
    }

    // Polsterung auf die Schrittweite 12 des Hexaeders
    W[9] = W[10] = W[11] = make_double3(0.0, 0.0, 0.0);
}

// =============================================================================
// Auswertung der Rotation der Basisvektoren im Referenzprisma
// =============================================================================
// Für W = g(zeta) * f(r,s) mit f in der Ebene gilt
//     rot W = ( g' * (-f_s),  g' * f_r,  g * rot_2D f )
// und für W = h(r,s) * e_zeta
//     rot W = ( dh/ds,  -dh/dr,  0 ).
//
// Beim Hexaeder war genau diese Funktion für 10 von 12 Kanten NICHT die
// Rotation der Basis (per zentraler Differenz nachgewiesen). Deshalb prüft
// tests/test_nedelec_prism.cu die Rotation hier gegen zentrale Differenzen,
// bevor irgendetwas anderes geprüft wird.
__host__ __device__ inline void evaluate_nedelec_curl_prism(
    double r, double s, double zeta, double3 curlW[12])
{
    const double w[3][2] = {
        { 1.0 - s,  r       },
        {     -s,   r       },
        {     -s,   r - 1.0 }
    };

    // Bodenfläche: g = (1 - zeta)/2, g' = -1/2, rot_2D w = 2
    for (int i = 0; i < 3; ++i) {
        curlW[i] = make_double3( 0.5 * w[i][1],
                                -0.5 * w[i][0],
                                 1.0 - zeta);
    }
    // Deckfläche: g = (1 + zeta)/2, g' = +1/2
    for (int i = 0; i < 3; ++i) {
        curlW[i + 3] = make_double3(-0.5 * w[i][1],
                                     0.5 * w[i][0],
                                     1.0 + zeta);
    }

    // Vertikale Kanten: h = Lambda_i / 2, grad Lambda = (-1,-1), (1,0), (0,1)
    curlW[6] = make_double3(-0.5,  0.5, 0.0);
    curlW[7] = make_double3( 0.0, -0.5, 0.0);
    curlW[8] = make_double3( 0.5,  0.0, 0.0);

    curlW[9] = curlW[10] = curlW[11] = make_double3(0.0, 0.0, 0.0);
}

// =============================================================================
// Isoparametrisches Mapping
// =============================================================================
// Gleiche Konvention wie compute_jacobian() für das Hexaeder:
//
//     J[i][j] = d x_j / d xi_i        (also J = J_std^T)
//
// Damit gilt weiterhin  W_phys = invJ * W_ref  (zeilenweise) und
// rot W_phys = (1/detJ) * J_std * rot W_ref  (spaltenweise über J).
__host__ __device__ inline void compute_jacobian_prism(
    const double3 nodes[PRISM_NUM_NODES],
    double r, double s, double zeta,
    double J[3][3], double invJ[3][3], double& detJ)
{
    double dN_dr[PRISM_NUM_NODES], dN_ds[PRISM_NUM_NODES], dN_dzeta[PRISM_NUM_NODES];
    prism_shape_derivatives(r, s, zeta, dN_dr, dN_ds, dN_dzeta);

    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            J[a][b] = 0.0;

    for (int i = 0; i < PRISM_NUM_NODES; ++i) {
        J[0][0] += dN_dr[i]    * nodes[i].x;
        J[0][1] += dN_dr[i]    * nodes[i].y;
        J[0][2] += dN_dr[i]    * nodes[i].z;

        J[1][0] += dN_ds[i]    * nodes[i].x;
        J[1][1] += dN_ds[i]    * nodes[i].y;
        J[1][2] += dN_ds[i]    * nodes[i].z;

        J[2][0] += dN_dzeta[i] * nodes[i].x;
        J[2][1] += dN_dzeta[i] * nodes[i].y;
        J[2][2] += dN_dzeta[i] * nodes[i].z;
    }

    detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
         - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
         + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

    // Wie beim Hexaeder: in einer __device__-Funktion lässt sich keine
    // Ausnahme werfen. Bei entartetem oder invertiertem Element bleibt invJ
    // null, die Ergebnisse für dieses Element sind dann unphysikalisch. Der
    // Importer prüft die Determinante vorab, damit dieser Fall gar nicht erst
    // in den Löser gelangt.
    double invDet = 0.0;
    if (detJ > 1e-14 || detJ < -1e-14) {
        invDet = 1.0 / detJ;
    }

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

// =============================================================================
// Quadratur
// =============================================================================
// 3 Punkte auf dem Dreieck (exakt bis Grad 2) mal 2 Gausspunkte in zeta (exakt
// bis Grad 3). Das ist der Gegenwert zur 2x2x2-Regel des Hexaeders: dort ist
// W . W je Richtung vom Grad 2 und wird von der 2-Punkt-Regel exakt
// integriert, hier ebenso.
//
// Die Dreiecksgewichte summieren sich auf 1/2, die Fläche des
// Referenzdreiecks; die zeta-Gewichte auf 2, die Länge von [-1,1].
#define PRISM_NUM_QUAD_POINTS 6

__host__ __device__ inline void prism_quadrature_point(
    int q, double& r, double& s, double& zeta, double& weight)
{
    const double a = 1.0 / 6.0;
    const double b = 2.0 / 3.0;
    const double tri[3][2] = {{a, a}, {b, a}, {a, b}};
    const double gp = 0.5773502691896257;   // 1/sqrt(3)

    const int it = q % 3;
    const int iz = q / 3;

    r      = tri[it][0];
    s      = tri[it][1];
    zeta   = (iz == 0) ? -gp : gp;
    weight = (1.0 / 6.0) * 1.0;   // Dreiecksgewicht * Gaussgewicht
}

#endif // NEDELEC_PRISM_CUH
