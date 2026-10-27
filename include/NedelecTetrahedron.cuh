#ifndef NEDELEC_TETRAHEDRON_CUH
#define NEDELEC_TETRAHEDRON_CUH

#include <cuda_runtime.h>

// =============================================================================
// Nédélec-Kantenelement erster Art auf dem Tetraeder (CTETRA mit 4 Knoten)
// =============================================================================
//
// WOZU
//
// Das gelieferte Netz (team28_simple_air.bdf, HyperMesh) besteht
// ausschliesslich aus linearen Tetraedern. Fuer ein Gebiet mit Luft um die
// Leiter herum ist das die naheliegende Vernetzung: ein Tetraedernetz kommt
// ohne Extrusionsrichtung aus und fuellt beliebige Zwischenraeume.
//
// DAS ELEMENT
//
// Die Whitney-1-Form. Von allen vier Elementtypen der einfachste Fall:
//
//     W_ab = Lambda_a grad Lambda_b - Lambda_b grad Lambda_a
//     rot W_ab = 2 grad Lambda_a x grad Lambda_b
//
// Beide Gradienten sind auf dem Tetraeder konstant, die Rotation also
// ebenfalls. Es gibt keine gekruemmten Kanten und keine windschiefen Flaechen:
// die Abbildung ist IMMER affin. Damit entfaellt der Vorbehalt, der beim
// Prisma und beim Hexaeder gilt -- ein konstantes B wird auf jedem Tetraeder
// exakt reproduziert, unabhaengig von der Elementform. Ein schlecht geformter
// Tetraeder verschlechtert die Kondition, nicht die Konsistenz.
//
// Das Referenzelement ist der Einheitstetraeder
//
//     Lambda_0 = 1 - r - s - t,  Lambda_1 = r,  Lambda_2 = s,  Lambda_3 = t
//
//     Knoten 0: (0,0,0)  1: (1,0,0)  2: (0,1,0)  3: (0,0,1)
//
// Die Auswertung koennte direkt im physikalischen Raum erfolgen, weil grad
// Lambda dort geschlossen bekannt ist. Sie geht hier trotzdem ueber das
// Referenzelement und die Piola-Transformationen, damit der Assemblierungs-
// kernel fuer alle Elementtypen dieselbe Schleife fahren kann. Die
// Jacobi-Matrix ist konstant, der Aufwand also vernachlaessigbar.
//
// KANTENNUMMERIERUNG
//
// VTK-Konvention fuer den Tetraeder, identisch zur Knotenreihenfolge von
// CTETRA. TopologyExtractor MUSS dieselbe Tabelle verwenden.
//
// ORIENTIERUNG
//
// Wie bei den anderen Typen zeigt die Basisfunktion entlang der LOKALEN
// Kantenrichtung. Die Relation lokale Kante <-> globale Kante korrigiert
// allein das Vorzeichenarray beim Gather/Scatter; die lokalen Elementmatrizen
// bleiben unorientiert.
//
// EINBETTUNG
//
// Keine Datenstruktur aendert ihre Form: Hexahedron::v[8] fasst die 4 Knoten,
// hex_edges mit Schrittweite 12 die 6 Kanten, SOLVER_LOCAL_MATRIX_SIZE = 144
// die 6x6-Matrix. Plaetze 6..11 bleiben mit -1 beziehungsweise 0.0 belegt.

#define TET_NUM_NODES 4
#define TET_NUM_EDGES 6

// =============================================================================
// Lokale Kantentabelle (VTK-Konvention)
// =============================================================================
__host__ __device__ inline void tet_local_edge_nodes(int e, int& vA, int& vB) {
    const int tbl[TET_NUM_EDGES][2] = {
        {0, 1}, {1, 2}, {2, 0},   // Kanten der Grundflaeche
        {0, 3}, {1, 3}, {2, 3}    // Kanten zur Spitze
    };
    vA = tbl[e][0];
    vB = tbl[e][1];
}

// =============================================================================
// Skalare Formfunktionen und ihre Ableitungen
// =============================================================================
__host__ __device__ inline void tet_shape_functions(
    double r, double s, double t, double N[TET_NUM_NODES])
{
    N[0] = 1.0 - r - s - t;
    N[1] = r;
    N[2] = s;
    N[3] = t;
}

// Die Ableitungen sind konstant; die Signatur nimmt die Koordinaten trotzdem
// entgegen, damit sie zu den anderen Elementtypen passt.
__host__ __device__ inline void tet_shape_derivatives(
    double r, double s, double t,
    double dN_dr[TET_NUM_NODES],
    double dN_ds[TET_NUM_NODES],
    double dN_dt[TET_NUM_NODES])
{
    (void)r; (void)s; (void)t;
    dN_dr[0] = -1.0; dN_dr[1] = 1.0; dN_dr[2] = 0.0; dN_dr[3] = 0.0;
    dN_ds[0] = -1.0; dN_ds[1] = 0.0; dN_ds[2] = 1.0; dN_ds[3] = 0.0;
    dN_dt[0] = -1.0; dN_dt[1] = 0.0; dN_dt[2] = 0.0; dN_dt[3] = 1.0;
}

// =============================================================================
// Auswertung der 6 Basisvektoren im Referenztetraeder
// =============================================================================
// Im Referenzraum sind die Gradienten
//     grad L0 = (-1,-1,-1), grad L1 = (1,0,0), grad L2 = (0,1,0), grad L3 = (0,0,1)
// und damit W_ab = L_a grad L_b - L_b grad L_a.
__host__ __device__ inline void evaluate_nedelec_basis_tet(
    double r, double s, double t, double3 W[12])
{
    const double L[4] = {1.0 - r - s - t, r, s, t};
    const double g[4][3] = {
        {-1.0, -1.0, -1.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  1.0,  0.0},
        { 0.0,  0.0,  1.0}
    };

    for (int e = 0; e < TET_NUM_EDGES; ++e) {
        int a, b;
        tet_local_edge_nodes(e, a, b);
        W[e] = make_double3(L[a]*g[b][0] - L[b]*g[a][0],
                            L[a]*g[b][1] - L[b]*g[a][1],
                            L[a]*g[b][2] - L[b]*g[a][2]);
    }
    for (int e = TET_NUM_EDGES; e < 12; ++e)
        W[e] = make_double3(0.0, 0.0, 0.0);
}

// =============================================================================
// Auswertung der Rotation
// =============================================================================
// rot W_ab = 2 grad L_a x grad L_b, konstant ueber das Element.
__host__ __device__ inline void evaluate_nedelec_curl_tet(
    double r, double s, double t, double3 curlW[12])
{
    (void)r; (void)s; (void)t;
    const double g[4][3] = {
        {-1.0, -1.0, -1.0},
        { 1.0,  0.0,  0.0},
        { 0.0,  1.0,  0.0},
        { 0.0,  0.0,  1.0}
    };

    for (int e = 0; e < TET_NUM_EDGES; ++e) {
        int a, b;
        tet_local_edge_nodes(e, a, b);
        curlW[e] = make_double3(
            2.0 * (g[a][1]*g[b][2] - g[a][2]*g[b][1]),
            2.0 * (g[a][2]*g[b][0] - g[a][0]*g[b][2]),
            2.0 * (g[a][0]*g[b][1] - g[a][1]*g[b][0]));
    }
    for (int e = TET_NUM_EDGES; e < 12; ++e)
        curlW[e] = make_double3(0.0, 0.0, 0.0);
}

// =============================================================================
// Isoparametrisches Mapping
// =============================================================================
// Gleiche Konvention wie beim Hexaeder und beim Prisma: J[i][j] = d x_j / d xi_i.
// Auf dem Tetraeder ist J konstant, detJ ist das sechsfache Volumen.
__host__ __device__ inline void compute_jacobian_tet(
    const double3 nodes[TET_NUM_NODES],
    double r, double s, double t,
    double J[3][3], double invJ[3][3], double& detJ)
{
    double dN_dr[TET_NUM_NODES], dN_ds[TET_NUM_NODES], dN_dt[TET_NUM_NODES];
    tet_shape_derivatives(r, s, t, dN_dr, dN_ds, dN_dt);

    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            J[a][b] = 0.0;

    for (int i = 0; i < TET_NUM_NODES; ++i) {
        J[0][0] += dN_dr[i] * nodes[i].x;
        J[0][1] += dN_dr[i] * nodes[i].y;
        J[0][2] += dN_dr[i] * nodes[i].z;

        J[1][0] += dN_ds[i] * nodes[i].x;
        J[1][1] += dN_ds[i] * nodes[i].y;
        J[1][2] += dN_ds[i] * nodes[i].z;

        J[2][0] += dN_dt[i] * nodes[i].x;
        J[2][1] += dN_dt[i] * nodes[i].y;
        J[2][2] += dN_dt[i] * nodes[i].z;
    }

    detJ = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1])
         - J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0])
         + J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);

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
// 4 Punkte, exakt bis Grad 2. Fuer die Steifigkeitsmatrix waere ein Punkt
// genug (die Rotation ist konstant), fuer die Massenmatrix ist W . W vom
// Grad 2 -- und die Massenmatrix ist es, die den Wirbelstromterm traegt.
//
// Die Gewichte summieren sich auf 1/6, das Volumen des Referenztetraeders.
#define TET_NUM_QUAD_POINTS 4

__host__ __device__ inline void tet_quadrature_point(
    int q, double& r, double& s, double& t, double& weight)
{
    const double a = 0.5854101966249685;
    const double b = 0.1381966011250105;
    const double pts[4][3] = {
        {b, b, b}, {a, b, b}, {b, a, b}, {b, b, a}
    };
    r = pts[q][0];
    s = pts[q][1];
    t = pts[q][2];
    weight = 1.0 / 24.0;
}

#endif // NEDELEC_TETRAHEDRON_CUH
