#ifndef SAUTER_SCHWAB_CUH
#define SAUTER_SCHWAB_CUH

#include <cuda_runtime.h>
#include "Backend.cuh"        // BF_HD
#include "BemIntegration.cuh" // Für evaluate_rooftop_basis und norm3


// =================================================================================
// 1D-Gauss-Legendre-Quadratur auf [0,1], N = 4
// =================================================================================
//
// Frueher lagen diese Werte als "__constant__ double GL_PTS[4]" im
// Geraetespeicher, definiert in SauterSchwab.cu und hier per "extern"
// deklariert. Das hatte zwei Nachteile:
//
//   1. __constant__ ist ausschliesslich aus Device-Code lesbar. Der
//      CPU-Modus braucht dieselben Stuetzstellen aber im Host-Code -- die
//      Sauter-Schwab-Integration ist Teil des BEM-Nahfelds und laeuft dort
//      auf der CPU.
//   2. Ein "extern __constant__" ueber Uebersetzungseinheiten hinweg ist
//      genau einer der beiden Gruende, aus denen dieses Projekt
//      CUDA_SEPARABLE_COMPILATION (-rdc=true) braucht (siehe CMakeLists.txt).
//
// Beides loest eine inline-Zugriffsfunktion mit lokaler Tabelle: sie ist auf
// Host und Device gueltig, braucht kein eigenes Symbol und wird vom Compiler
// in beiden Faellen wegoptimiert. Die Zahlenwerte sind unveraendert.
BF_HD inline double GL_PT(int i) {
    const double pts[4] = {
        0.06943184420297371,
        0.33000947820757187,
        0.66999052179242813,
        0.93056815579702629
    };
    return pts[i];
}

BF_HD inline double GL_WG(int i) {
    const double wgt[4] = {
        0.17392742256872692,
        0.32607257743127305,
        0.32607257743127305,
        0.17392742256872692
    };
    return wgt[i];
}


// =================================================================================
// Mapping vom 4D Einheitswürfel in die 6 Coincident-Simplizes (Sauter/Schwab)
// =================================================================================
__host__ __device__ __forceinline__ void get_coincident_coords(
    int case_id, double xi, double eta1, double eta2, double eta3,
    double& x1, double& x2, double& y1, double& y2)
{
    // Die 6 Permutationen, die x2 <= x1 und y2 <= y1 im Bereich [0,1] garantieren
    if (case_id == 0) {
        x1 = xi; 
        x2 = xi * eta1; 
        y1 = xi * eta1 * eta2; 
        y2 = xi * eta1 * eta2 * eta3;
    } else if (case_id == 1) {
        x1 = xi; 
        y1 = xi * eta1; 
        x2 = xi * eta1 * eta2; 
        y2 = xi * eta1 * eta2 * eta3;
    } else if (case_id == 2) {
        x1 = xi; 
        y1 = xi * eta1; 
        y2 = xi * eta1 * eta2; 
        x2 = xi * eta1 * eta2 * eta3;
    } else if (case_id == 3) {
        y1 = xi; 
        x1 = xi * eta1; 
        x2 = xi * eta1 * eta2; 
        y2 = xi * eta1 * eta2 * eta3;
    } else if (case_id == 4) {
        y1 = xi; 
        x1 = xi * eta1; 
        y2 = xi * eta1 * eta2; 
        x2 = xi * eta1 * eta2 * eta3;
    } else { // case_id == 5
        y1 = xi; 
        y2 = xi * eta1; 
        x1 = xi * eta1 * eta2; 
        x2 = xi * eta1 * eta2 * eta3;
    }
}

// =================================================================================
// Mapping vom Dreieck (u,v) auf das Rooftop-Referenz-Quad [-1, 1]^2
// =================================================================================
__host__ __device__ __forceinline__ void map_tri_to_quad(double u, double v, int tri_id, double& u_R, double& v_R) {
    if (tri_id == 0) {
        // Dreieck A (Knoten 0, 1, 2)
        u_R = -1.0 + 2.0 * u + 2.0 * v;
        v_R = -1.0 + 2.0 * v;
    } else {
        // Dreieck B (Knoten 0, 2, 3)
        u_R = -1.0 + 2.0 * u;
        v_R = -1.0 + 2.0 * u + 2.0 * v;
    }
}

__host__ __device__ void integrate_triangle_coincident(
    const double3 nodes[4], int tri_p, int tri_q, double matrix_block[4][4]);

/*
// =================================================================================
// Wrapper für den Self-Patch (Identische Quads)
// =================================================================================
__host__ __device__ void integrate_singular_self_patch(
    const double3 nodes[4], double matrix_block[4][4]) 
{
    // Initialisieren
    for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) matrix_block[i][j] = 0.0;
    
    double temp_block[4][4];

    // 1. Identisches Dreieck A x A (Coincident)
    integrate_triangle_coincident(nodes, 0, 0, temp_block);
    for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) matrix_block[i][j] += temp_block[i][j];

    // 2. Identisches Dreieck B x B (Coincident)
    integrate_triangle_coincident(nodes, 1, 1, temp_block);
    for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) matrix_block[i][j] += temp_block[i][j];

    // 3. Gemeinsame Kante A x B (Common Edge)
    // integrate_triangle_common_edge(nodes, 0, 1, temp_block);
    // for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) matrix_block[i][j] += temp_block[i][j];

    // 4. Gemeinsame Kante B x A (Common Edge)
    // integrate_triangle_common_edge(nodes, 1, 0, temp_block);
    // for(int i=0; i<4; ++i) for(int j=0; j<4; ++j) matrix_block[i][j] += temp_block[i][j];
}
*/

// =================================================================================
// Mapping in die 5 Common-Edge Simplizes
// (Singularität liegt auf der Kante y=0 für beide Dreiecke)
// =================================================================================
__host__ __device__ __forceinline__ void get_common_edge_coords(
    int case_id, double xi, double eta1, double eta2, double eta3,
    double& x1, double& x2, double& y1, double& y2, double& jac)
{
    // FEHLERKORREKTUR. Bisher stand hier fuer ALLE fuenf Teilgebiete
    //     jac = xi^3 * eta1^2 * eta2
    // Das ist fuer das erste Teilgebiet falsch: dort fehlt der Faktor eta2.
    //
    // Nachweisbar ohne jeden Kern, allein ueber die Deckung des Integrations-
    // gebiets. Integriert man die konstante Funktion 1, muss die Flaeche des
    // Produkts zweier Referenzdreiecke herauskommen, also (1/2)*(1/2) = 1/4.
    // Mit dem alten Jacobi-Faktor liefert jedes der fuenf Teilgebiete
    // 1/24, in der Summe 5/24 = 0.2083333 statt 6/24 = 0.25 -- das Gebiet wird
    // also nur zu 5/6 ueberdeckt. Ohne den Faktor eta2 traegt Teilgebiet 0
    // 2/24, und die Summe ist exakt 0.25.
    //
    // Zur Kontrolle: dieselbe Rechnung fuer die beiden anderen Faelle stimmt
    // bereits. Koinzident 6 * (xi^3 eta1^2 eta2) = 6/24 = 1/4, Common Vertex
    // 2 * (xi^3 eta2) = 2 * 1/8 = 1/4.
    //
    // Betroffen war damit integrate_triangle_common_edge_self_patch und ueber
    // sie der Self-Patch: die beiden Dreiecke eines Quads teilen die Diagonale,
    // ihr Beitrag lief also durch genau diesen Zweig.
    // ZWEITE KORREKTUR. Neben dem Jacobi-Faktor waren auch die
    // Koordinatenformeln verstellt. Nachweisbar mit einem linearen Integranden:
    // ueber das Produkt zweier Referenzdreiecke gilt exakt
    //     Int (x1 + 2 x2 + 3 y1 + 4 y2) = 7/6
    // Die alten Formeln lieferten 1.0333 (11.4 % daneben), obwohl die
    // Gebietsflaeche nach der Jacobi-Korrektur bereits stimmte -- die
    // Teilgebiete ueberdeckten das Gebiet an der falschen Stelle. Zusaetzlich
    // verliessen 264 von 1280 Auswertepunkten das Referenzdreieck (x2 > x1
    // bzw. y2 > y1), was bei den Rooftop-Basisfunktionen eine Auswertung
    // ausserhalb ihres Definitionsbereichs bedeutet haette.
    //
    // Untenstehende Standardform nach Sauter und Schwab trifft beide Tests
    // exakt (0.25 und 7/6) und haelt alle Punkte im Referenzdreieck.
    //
    // Die Reihenfolge ist (x1, x2, y1, y2); die Teilgebiete 0 bis 4 entsprechen
    // den fuenf Faellen der Standardzerlegung.
    jac = (xi * xi * xi) * (eta1 * eta1) * ((case_id == 0) ? 1.0 : eta2);

    if (case_id == 0) {
        x1 = xi;
        x2 = xi * eta1 * eta3;
        y1 = xi * (1.0 - eta1 * eta2);
        y2 = xi * eta1 * (1.0 - eta2);
    } else if (case_id == 1) {
        x1 = xi;
        x2 = xi * eta1;
        y1 = xi * (1.0 - eta1 * eta2 * eta3);
        y2 = xi * eta1 * eta2 * (1.0 - eta3);
    } else if (case_id == 2) {
        x1 = xi * (1.0 - eta1 * eta2);
        x2 = xi * eta1 * (1.0 - eta2);
        y1 = xi;
        y2 = xi * eta1 * eta2 * eta3;
    } else if (case_id == 3) {
        x1 = xi * (1.0 - eta1 * eta2 * eta3);
        x2 = xi * eta1 * eta2 * (1.0 - eta3);
        y1 = xi;
        y2 = xi * eta1;
    } else { // case_id == 4
        x1 = xi * (1.0 - eta1 * eta2 * eta3);
        x2 = xi * eta1 * (1.0 - eta2 * eta3);
        y1 = xi;
        y2 = xi * eta1 * eta2;
    }
}

// =================================================================================
// Mapping in die 2 Common-Vertex Simplizes
// (Singularitaet liegt im Punkt x = y = 0 beider Dreiecke)
// =================================================================================
// Diese Funktion war bisher nur DEKLARIERT, nie definiert -- der
// Common-Vertex-Fall wurde im Loeser nie erreicht. Nachgetragen in der
// Standardform nach Sauter und Schwab: zwei Teilgebiete, Jacobi-Determinante
// xi^3 * eta2.
//
// Die Determinante hebt die 1/r-Singularitaet auf: am Beruehrpunkt gehen beide
// Punkte wie O(xi) gegen den Ursprung, der Kern waechst also wie O(1/xi),
// waehrend xi^3 dagegensteht. Der transformierte Integrand ist damit
// beschraenkt und glatt genug fuer Gauss-Legendre.
__host__ __device__ __forceinline__ void get_common_vertex_coords(
    int case_id, double xi, double eta1, double eta2, double eta3,
    double& x1, double& x2, double& y1, double& y2, double& jac)
{
    jac = (xi * xi * xi) * eta2;

    if (case_id == 0) {
        x1 = xi;            x2 = xi * eta1;
        y1 = xi * eta2;     y2 = xi * eta2 * eta3;
    } else {  // case_id == 1 -- die Rollen der beiden Dreiecke vertauscht
        x1 = xi * eta2;     x2 = xi * eta2 * eta3;
        y1 = xi;            y2 = xi * eta1;
    }
}

__host__ __device__ __forceinline__ void map_self_patch_edge_to_quad(
    double u, double v, int is_tri_B, double& u_R, double& v_R);

__host__ __device__ void integrate_triangle_common_edge_self_patch(
    const double3 nodes[4], int is_tri_B_p, int is_tri_B_q, double matrix_block[4][4]);

__host__ __device__ void integrate_singular_self_patch(
    const double3 nodes[4], double matrix_block[4][4]);

#endif // SAUTER_SCHWAB_CUH