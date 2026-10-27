#ifndef DISCRETE_GRADIENT_CUH
#define DISCRETE_GRADIENT_CUH

// =============================================================================
// DiscreteGradient.cuh -- Grundbausteine des Hiptmair-Vorkonditionierers
// =============================================================================
//
// Warum ueberhaupt. Der Operator curl-curl + sigma/dt hat einen riesigen
// NAHnullraum aus Gradientenfeldern: fuer A = grad phi ist curl A = 0, es
// bleibt nur der Massenterm sigma/dt, der in nichtleitenden Gebieten null ist.
// Punktglaetter -- Jacobi, Gauss-Seidel, ILU(0) -- daempfen genau diese Moden
// nicht, weil sie im Nahnullraum liegen. Das erklaert, warum der reine
// FEM-Fall dieses Loesers (nobem, magnetostatisch) 329 Iterationen braucht,
// wo ein gut vorkonditioniertes SPD-System mit 20 bis 50 auskaeme.
//
// Hiptmairs Beobachtung: die nicht gedaempften Moden sind Bilder des diskreten
// Gradienten. Man glaettet sie, indem man das Residuum mit G^T in den
// KNOTENraum holt, dort ein (skalares, gutartiges) Problem loest und die
// Korrektur mit G zurueckbildet:
//
//     1. glaetten im Kantenraum        x += M_edge^-1 r
//     2. Residuum                      r = b - A x
//     3. in den Knotenraum             r_n = G^T r
//     4. Hilfsproblem loesen           A_n e_n = r_n
//     5. zurueckbilden                 x += G e_n
//
// -----------------------------------------------------------------------------
// Der diskrete Gradient
// -----------------------------------------------------------------------------
// Fuer die Whitney-Kantenbasis gilt exakt
//     grad(phi) = sum_e (phi_{n2} - phi_{n1}) W_e
// mit GANZZAHLIGEN Koeffizienten. Voraussetzung ist die Normierung
// Int_e W_e . dl = 1, und die ist hier erfuellt: W[0] = 0.125(1-eta)(1-zeta)
// ergibt auf der Kante eta = zeta = -1 den Wert 0.5, integriert ueber
// xi in [-1,1] also 1. Die kovariante Piola-Transformation erhaelt
// Linienintegrale, das gilt also auch im verzerrten Element.
//
// G ist damit die Inzidenzmatrix: Zeile e hat -1 in Spalte n1 und +1 in
// Spalte n2, wobei (n1,n2) = mesh.edges[e] = (v_min, v_max).
//
// -----------------------------------------------------------------------------
// Die Hilfsraum-Matrix
// -----------------------------------------------------------------------------
//     A_n = G^T A G
//
// Weil curl(grad) = 0 ist, faellt der curl-curl-Anteil VOLLSTAENDIG heraus und
// es bleibt A_n = G^T (sigma/dt M) G, also der mit sigma/dt gewichtete
// Knoten-Laplace. Das ist die entscheidende Struktur -- und zugleich die
// Schwierigkeit dieses Problems:
//
//   * In der Platte (sigma > 0) ist A_n regulaer und gutartig.
//   * In den Spulen und im magnetostatischen Fall (sigma = 0) ist A_n EXAKT
//     NULL. Der Hilfsraum traegt dort nichts, und das Verfahren braucht eine
//     zusaetzliche Behandlung (Regularisierung oder Vektor-Hilfsraeume wie im
//     vollen AMS).
//
// Genau deshalb gibt build_auxiliary_nodal_matrix() die Zahl der praktisch
// verschwindenden Diagonaleintraege mit aus -- das ist keine Randnotiz,
// sondern die zentrale Entwurfsfrage.
//
// -----------------------------------------------------------------------------
// Eichung
// -----------------------------------------------------------------------------
// Mit Hiptmair wird ueblicherweise NICHT geeicht: man loest das singulaere,
// aber konsistente System und ueberlaesst den Nullraum dem Vorkonditionierer.
// Konsistenz heisst G^T b = 0, und das ist fuer dieses Problem nachgemessen
// (Verhaeltnis 2.1e-4, der Rest sitzt auf der facettierten
// Spulenmantelflaeche). Die Baum-Kobaum-Eichung ist dafuer nicht nur
// unnoetig, sie zerstoert die Struktur, die G ausnutzt: sie nullt Baumkanten
// und damit gerade die Eintraege, ueber die die Gradientenmoden dargestellt
// werden.
// =============================================================================

#include <cstddef>
#include <vector>

#include "Mesh.cuh"

/// Duennbesetzte Matrix im CSR-Format, host-seitig. Bewusst schlicht: diese
/// Groessen werden einmal je Zeitschritt aufgebaut und danach nur angewandt.
struct HostCsr {
    int num_rows = 0;
    int num_cols = 0;
    std::vector<int>    row_offsets;   // Laenge num_rows + 1
    std::vector<int>    col_indices;
    std::vector<double> values;

    int nnz() const { return static_cast<int>(col_indices.size()); }
};

/// Diskreter Gradient G (num_edges x num_nodes), Eintraege -1 und +1.
/// Zeile e: -1 in Spalte mesh.edges[e].x, +1 in Spalte mesh.edges[e].y.
HostCsr build_discrete_gradient(const Mesh& mesh);

/// Hilfsraum-Matrix A_n = G^T A G.
///
/// 'A' ist die assemblierte Kantenmatrix in CSR (num_edges x num_edges).
/// Berechnet wird direkt ueber die Nichtnullen von A: jeder Eintrag A[e][f]
/// traegt zu vier Positionen (i,j) bei, mit i aus den beiden Knoten von Kante e
/// und j aus denen von Kante f, mit den Vorzeichen von G. Bei 5e5 Nichtnullen
/// sind das 2e6 Operationen -- ein expliziter Dreifachproduktalgorithmus lohnt
/// hier nicht.
///
/// out_num_tiny_diagonal (optional) erhaelt die Zahl der Zeilen, deren
/// Diagonaleintrag betragsmaessig unter 'tiny_threshold' * reference_scale
/// liegt -- also der Anteil des Hilfsraums, der nichts traegt.
///
/// reference_scale MUSS von aussen kommen, ueblicherweise max|A|. Ein erster
/// Anlauf hat gegen das eigene Maximum von A_n normiert; ist A_n aber
/// vollstaendig null (reiner curl-curl-Fall), besteht dieses Maximum nur aus
/// Rundungsrauschen, und die Zaehlung wird sinnlos. Bei reference_scale <= 0
/// wird ersatzweise max|diag(A_n)| benutzt.
///
/// out_max_abs (optional) erhaelt max|A_n| -- die Groesse, an der sich
/// ablesen laesst, ob der Hilfsraum ueberhaupt etwas traegt.
HostCsr build_auxiliary_nodal_matrix(const HostCsr& A, const Mesh& mesh,
                                     int* out_num_tiny_diagonal = nullptr,
                                     double reference_scale = -1.0,
                                     double tiny_threshold = 1e-12,
                                     double* out_max_abs = nullptr);

// =============================================================================
// Vektorieller Hilfsraum (voller AMS nach Hiptmair und Xu)
// =============================================================================
//
// Der skalare Hilfsraum allein genuegt nicht. A_n = G^T A G enthaelt wegen
// curl(grad) = 0 nur den Massenterm, ist bei sigma = 0 also exakt null -- im
// magnetostatischen Fall und in den Spulen traegt er nichts. Der VEKTORIELLE
// Hilfsraum schliesst diese Luecke: A_Pi = Pi^T A Pi ist auch bei reinem
// curl-curl regulaer, denn ein konstantes Vektorfeld hat zwar verschwindende
// Rotation, ein allgemeines knotenbasiertes Vektorfeld aber nicht.
//
// Der Vorkonditionierer lautet dann
//     B = S  +  G A_G^-1 G^T  +  Pi A_Pi^-1 Pi^T
// mit S einem Punktglaetter im Kantenraum.
//
// -----------------------------------------------------------------------------
// Der Interpolationsoperator Pi
// -----------------------------------------------------------------------------
// Pi bildet ein knotenweise definiertes VEKTORfeld (drei Werte je Knoten) auf
// Kantenfreiheitsgrade ab. Der Kantenfreiheitsgrad ist das Linienintegral, und
// mit der Trapezregel entlang der geraden Kante:
//
//     (Pi v)_e = Int_e v . dl  ~  1/2 (v(n1) + v(n2)) . (x_n2 - x_n1)
//
// Zeile e hat damit sechs Eintraege: fuer n in {n1, n2} und c in {x,y,z} den
// Wert 1/2 (x_n2 - x_n1)_c.
//
// Die Trapezregel ist hier nicht Naeherung, sondern exakt fuer das, worauf es
// ankommt: fuer ein KONSTANTES Feld v liefert sie v . (x_n2 - x_n1), und das
// ist genau der Kantenfreiheitsgrad des Nedelec-Interpolanten dieses Feldes.
// Daraus folgt die Testeigenschaft S * Pi * (konstantes Feld) = 0 -- das
// Gegenstueck zu S * G * phi = 0 und ebenso scharf.
//
// Spaltenordnung: Freiheitsgrad (n, c) hat den Index 3*n + c.

/// Pi (num_edges x 3*num_nodes), sechs Eintraege je Zeile.
HostCsr build_nodal_vector_interpolation(const Mesh& mesh);

/// A_Pi = Pi^T A Pi   (3*num_nodes x 3*num_nodes).
///
/// Berechnet zeilenweise mit einem dichten Streuvektor statt ueber eine Map:
/// Pi hat sechs Eintraege je Zeile, jeder Nichtnulleintrag von A traegt also zu
/// 36 Positionen bei. Bei 5e5 Nichtnullen sind das 1.8e7 Beitraege -- mit
/// std::map waere das spuerbar langsam, mit Streuvektor und Trefferliste ist es
/// O(1) je Beitrag.
///
/// 'node_to_edges' muss die Inzidenzliste Knoten -> anliegende Kanten sein;
/// build_node_to_edges() liefert sie.
/// Knotenweiser Vektor-Poisson-Operator mit dem alpha-Koeffizienten nu = 1/mu,
/// blockdiagonal auf drei Komponenten aufgezogen. Ersatz fuer Pi^T A Pi als
/// Grundlage der AMG-Hierarchie im vektoriellen AMS-Hilfsraum; entspricht
/// HYPRE_AMSSetAlphaPoissonMatrix. nu_per_hex darf nullptr sein (konstant 1).
/// Ausfuehrliche Begruendung an der Definition in src/DiscreteGradient.cpp.
/// nu_per_hex = 1/mu je Hexaeder (Steifigkeit), beta_per_hex = sigma/dt je
/// Hexaeder (gelumpte Masse). nullptr bedeutet 1 bzw. 0. BEIDE werden
/// gebraucht -- ohne den Massenterm ist der Operator nicht spektral
/// aequivalent zu Pi^T A Pi, und der Hilfsraum wird unbrauchbar.
HostCsr build_alpha_poisson_matrix(const Mesh& mesh, const double* nu_per_hex,
                                   const double* beta_per_hex);

HostCsr build_auxiliary_vector_matrix(const HostCsr& A, const HostCsr& Pi,
                                      const Mesh& mesh,
                                      const std::vector<std::vector<int>>& node_to_edges,
                                      double* out_max_abs = nullptr);

/// Inzidenzliste Knoten -> anliegende Kanten.
std::vector<std::vector<int>> build_node_to_edges(const Mesh& mesh);

/// y = G x   (Knotenvektor -> Kantenvektor), also das Gradientenfeld zu x.
void apply_gradient(const HostCsr& G, const double* x_nodes, double* y_edges);

/// y = G^T x   (Kantenvektor -> Knotenvektor), die diskrete Divergenz.
void apply_gradient_transpose(const HostCsr& G, const double* x_edges,
                              double* y_nodes, int num_nodes);

/// y = A x fuer eine HostCsr.
void apply_host_csr(const HostCsr& A, const double* x, double* y);

#endif // DISCRETE_GRADIENT_CUH
