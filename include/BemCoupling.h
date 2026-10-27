#ifndef BEM_COUPLING_H
#define BEM_COUPLING_H

#include <vector>
#include "Mesh.cuh"
#include "ClusterTree.h"

// =============================================================================
// Kopplung des BEM-Blockoperators an die globalen FEM-Kanten-Freiheitsgrade
// =============================================================================
//
// PROBLEM
// -------
// Die BEM-Blöcke aus dem ACA-Aufbau sind in "Elementindizes" formuliert: Zeile
// bzw. Spalte eines Blocks ist (lokales Quad im Cluster) * 4 + (lokale Kante).
// Der gekoppelte Operator arbeitet dagegen auf total_dofs = mesh.num_edges,
// also globalen FEM-Kanten. Zwei Dinge fehlten dazwischen:
//
//   1. Die Übersetzung Randquad-Kante -> globale FEM-Kante. Sie liegt in
//      mesh.boundary_edges_to_fem_edges bereit, wurde aber nicht benutzt.
//      Ebenso mesh.boundary_edge_signs für die Orientierung.
//
//   2. Die Cluster eines ClusterTree enthalten *verstreute* Quad-Indizes, weil
//      build_recursive nach Schwerpunktkoordinate sortiert. Ein Block kann
//      daher nicht über einen zusammenhängenden Offset adressiert werden.
//
// LÖSUNG
// ------
// Wie in der H-Matrix-Literatur üblich induziert der Cluster-Baum eine
// Permutation: die Blätter werden in Baumreihenfolge durchlaufen und ihre
// Quad-Indizes hintereinandergehängt. In dieser permutierten Numerierung ist
// jedes Blatt per Konstruktion ein zusammenhängender Indexbereich.
//
// Der BEM-Operator wirkt damit auf einem eigenen Vektor der Länge
// num_bem_dofs = 4 * num_boundary_faces in permutierter Reihenfolge. Die
// Anbindung an den FEM-Raum geschieht durch
//
//     x_bem[i] = sign[i] * x_fem[map[i]]                        (Gather)
//     y_bem    = B * x_bem                                      (Blöcke)
//     y_fem[map[i]] += sign[i] * y_bem[i]                       (Scatter, atomar)
//
// Das ist genau die Assemblierung der Galerkin-Matrix im globalen Kantenraum:
//
//     B_glob[I][J] = sum_{i->I} sum_{j->J} s_i s_j B_lok[i][j]
//
// denn (B_glob x)_I = sum_{i->I} s_i * (B_lok x_bem)_i. Gather und Scatter
// laufen einmal pro apply(), nicht einmal pro Block -- bei 65536 Blöcken der
// wesentliche Unterschied zu einer blockweisen Indexliste.
//
// Jede Randkante gehört zu genau zwei Randquads, deshalb bildet der Scatter
// zwei BEM-Freiheitsgrade auf dieselbe FEM-Kante ab und muss addieren.
//
// ORIENTIERUNG
// ------------
// mesh.boundary_edge_signs ist +1, wenn die Kantenrichtung der Topologietabelle
// ({0,1},{1,2},{2,3},{3,0}) mit der globalen Richtung v_min -> v_max
// übereinstimmt. Es korrigiert also lokale gegen globale Orientierung, *nicht*
// lokale Kante gegen Referenzachse. Letzteres muss die Basisfunktion selbst
// leisten; siehe evaluate_rooftop_basis in BemIntegration.cuh.
// =============================================================================

struct BemCouplingMap {
    // Position im permutierten Raum -> Randquad-Index. Länge num_boundary_faces.
    std::vector<int> perm;

    // Baumknoten-Index -> Startposition des Blattes in perm, -1 für innere Knoten.
    // Der Zeilenoffset eines Blocks ist leaf_offset[node] * 4.
    std::vector<int> leaf_offset;

    // BEM-Freiheitsgrad (permutiert) -> globale FEM-Kanten-ID. Länge num_bem_dofs.
    std::vector<int> bem_to_fem;

    // BEM-Freiheitsgrad (permutiert) -> +1 oder -1. Länge num_bem_dofs.
    std::vector<int> bem_sign;

    // Startposition und Groesse JEDES Baumknotens im permutierten Raum, nicht nur
    // der Blaetter. Das gilt, weil die Permutation die Blaetter in Baumreihenfolge
    // aneinanderhaengt: die Blaetter eines Teilbaums sind darin zusammenhaengend,
    // also auch der von ihm belegte Indexbereich. Damit kann ein Block auch zu
    // einem inneren Knotenpaar gehoeren -- Voraussetzung fuer die hierarchische
    // Blockstruktur (siehe build_block_pairs).
    std::vector<int> node_offset;  // Laenge get_num_nodes()
    std::vector<int> node_size;    // Anzahl Quads im Teilbaum

    int num_bem_dofs = 0;
};

// -----------------------------------------------------------------------------
// Hierarchische Blockstruktur
// -----------------------------------------------------------------------------
// Ein Blockpaar der H-Matrix. row/col sind Baumknoten-Indizes; die zugehoerigen
// Indexbereiche stehen in BemCouplingMap::node_offset/node_size.
struct BemBlockPair {
    int row_node;
    int col_node;
    bool admissible;   // true -> ACA (Fernfeld), false -> dense (Nahfeld)
};

// Baut die Blockliste durch rekursiven Abstieg: ist ein Clusterpaar zulaessig,
// wird es als ein einziger Niedrigrangblock gefuehrt; sonst wird geteilt, bis
// beide Seiten Blaetter sind (dann dense).
//
// Der bisherige Code betrachtete ausschliesslich Blatt-Blatt-Paare. Das ist eine
// flache Zerlegung: zulaessige Paare auf hoeheren Ebenen werden nicht
// zusammengefasst, sondern in ihre Blaetter zersplittert. Da die ACA-Kosten
// O(k*(m+n)) und nicht O(m*n) sind, kostet ein zersplitterter Fernfeldbereich
// deutlich mehr als derselbe Bereich als ein grosser Block.
//
// Gemessen an team28.vtk mit eta = 1.0 und Blattgroesse 30:
//   flach        61628 ACA-Bloecke, sum(m+n) = 7.97e6
//   hierarchisch  6248 ACA-Bloecke, sum(m+n) = 1.67e6   -> Faktor 4.78
// Die abgedeckten Eintraege bleiben dabei praktisch gleich (2.574e8 zu 2.578e8),
// der dense-Anteil ebenfalls (1.632e7 zu 1.589e7). Es wird also nicht weniger
// Physik gerechnet, sondern dieselbe mit weniger Aufwand.
std::vector<BemBlockPair> build_block_pairs(const ClusterTree& tree, double eta);



// Baut die Abbildung. Die Blätter werden in aufsteigender Knotenindex-Reihenfolge
// durchlaufen -- identisch zur Blockschleife in mainloop(), damit leaf_offset und
// die dort erzeugten Blöcke zusammenpassen.
//
// Wirft std::runtime_error, wenn das Ergebnis inkonsistent ist (perm keine
// Bijektion, FEM-Kanten-ID ausserhalb des gueltigen Bereichs, Vorzeichen nicht
// +/-1). Diese Pruefungen sind billig und schlagen sonst erst als stille
// Falschrechnung zu.
BemCouplingMap build_bem_coupling_map(const Mesh& mesh, const ClusterTree& tree);

#endif // BEM_COUPLING_H
