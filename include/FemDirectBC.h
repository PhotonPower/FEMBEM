#ifndef FEM_DIRECT_BC_H
#define FEM_DIRECT_BC_H

#include <cstddef>
#include <vector>

#include "Mesh.cuh"
#include "MaterialTypes.cuh"

// =============================================================================
// FEM-direkt: Dirichlet-Rand und Regularisierung, materialisiert
// =============================================================================
//
// AP4 aus docs/design/fem-direct-plan.md.
//
// WARUM EIN EIGENES MODUL
//
// Der iterative Pfad behandelt Eichung und Randbedingung als Projektion,
// P*A*P + (I-P), angewendet im Operator. Fuer GMRES ist das richtig und
// billig. Fuer cuDSS ist es wertlos: dort wird eine MATRIX uebergeben, keine
// apply()-Funktion. Beides muss also in die CSR-Werte hinein.
//
// Die Signaturen nehmen absichtlich rohe Zeiger und nicht CSRMatrix. fem.cuh
// zieht ueber CoupledSolver.cuh cusparse.h herein; dieses Modul soll ohne
// CUDA-Bibliotheken uebersetzbar bleiben, damit tests/test_dirichlet_bc.cu es
// auf dem Host wirklich ausfuehren kann.
//
// =============================================================================

namespace fem_direct {

// -----------------------------------------------------------------------------
// Kantenmarkierung fuer A x n = 0 auf dem aeusseren Rand
// -----------------------------------------------------------------------------
// Rueckgabe: Feld der Laenge mesh.num_edges, 1 = festgehalten.
//
// Quelle ist mesh.boundary_edges_to_fem_edges, das TopologyExtractor bereits
// aufbaut (4 Eintraege je Randflaeche, bei einer Dreiecksflaeche steht der
// vierte auf -1). Damit wird kein Rand neu hergeleitet, sondern der bereits
// geprueften Topologie entnommen.
//
// ACHTUNG ZUR BEDEUTUNG: boundary_faces ist der aeussere Rand des GESAMTEN
// Netzes. Beim BEM-FEM-Pfad ist das die Kopplungsflaeche der Leiter, beim
// FEM-direkt-Pfad die Aussenwand des Luftkastens. Nur im zweiten Fall ist
// A x n = 0 dort physikalisch gemeint. Wer diese Funktion auf einem Netz ohne
// Luftgebiet aufruft, klemmt das Feld an der Leiteroberflaeche fest und
// bekommt Unsinn -- ohne Fehlermeldung.
std::vector<char> collect_dirichlet_edges(const Mesh& mesh);

// -----------------------------------------------------------------------------
// Homogene Dirichlet-Bedingung in die CSR-Werte einarbeiten
// -----------------------------------------------------------------------------
// Fuer jede festgehaltene Kante i:
//   * Zeile i wird genullt, A(i,i) = diag_value
//   * Spalte i wird in allen anderen Zeilen genullt
//   * rhs[i] = 0
//
// Weil der vorgeschriebene Wert NULL ist, entfaellt der Korrekturterm
// rhs[j] -= A(j,i) * wert fuer die freien Zeilen. Bei einem Wert ungleich null
// waere er zwingend; deshalb heisst die Funktion "homogeneous" und nicht
// einfach "apply".
//
// Die Symmetrie bleibt erhalten. Das ist die Voraussetzung dafuer, spaeter auf
// CUDSS_MTYPE_SYMMETRIC und ein einzelnes Dreieck umzustellen.
//
// ZUM DIAGONALWERT: diag_value <= 0 bedeutet "automatisch", dann wird der
// Mittelwert der Diagonale ueber die FREIEN Zeilen genommen. Eine 1.0 neben
// Eintraegen der Groessenordnung 1e6 waere nicht falsch -- ein Direktloeser
// kommt damit zurecht --, aber die Matrix ist dann schlecht skaliert, und
// jede Residuumsnorm ueber das Gesamtsystem verliert ihre Aussagekraft.
struct DirichletReport {
    size_t num_constrained = 0;
    size_t num_free = 0;
    size_t entries_zeroed = 0;    // genullte Eintraege ausserhalb der Diagonale
    double diag_value = 0.0;      // tatsaechlich gesetzter Diagonalwert
    double diag_min_free = 0.0;   // kleinster Diagonalwert der freien Zeilen
    double diag_max_free = 0.0;   // groesster, zur Beurteilung der Skalierung
};

DirichletReport apply_homogeneous_dirichlet(
    int num_rows,
    const int* row_offsets,
    const int* col_indices,
    double* values,
    double* rhs,                              // darf nullptr sein
    const std::vector<char>& is_constrained,
    double diag_value = 0.0);

// -----------------------------------------------------------------------------
// Regularisierung des Curl-Curl-Nullraums
// -----------------------------------------------------------------------------
// In den Gebieten mit sigma = 0 hat K einen riesigen Nullraum (alle
// Gradienten), dazu bei b1 != 0 die harmonischen Felder. Der iterative Pfad
// behandelt das mit Baum-Kobaum-Eichung und Kohomologie-Projektion. Fuer einen
// Direktloeser genuegt eine kleine Leitfaehigkeit in der Luft:
//
//     sigma_Luft = eps * sigma_Referenz,   eps ~ 1e-8
//
// Das entfernt Gradienten- UND Kohomologieanteil auf einen Schlag. Die
// Kondition verschlechtert sich dabei um Groessenordnungen -- fuer GMRES waere
// das untragbar, fuer eine Zerlegung ist es gleichgueltig. Genau darin liegt
// der Vorteil des Pfades.
//
// ABNAHME: das Ergebnis muss zwischen eps = 1e-6 und 1e-10 relativ
// stehenbleiben. Tut es das nicht, ist die Regularisierung keine
// Regularisierung, sondern Physik.
//
// Rueckgabe: Anzahl der geaenderten Elemente.
size_t regularize_nonconducting_regions(
    MaterialProperties* materials,
    const Mesh& mesh,
    double sigma_reference,
    double epsilon = 1e-8);

/// Ergebnis von project_out_gradient().
struct GradientProjectionReport {
    double rel_before = 0.0;   ///< ||G^T b|| / ||b|| vorher
    double rel_after  = 0.0;   ///< ||G^T b|| / ||b|| nachher
    double b_norm_before = 0.0;
    double b_norm_after  = 0.0;
    int    cg_iters   = 0;
    double cg_rel_res = 0.0;
    bool   converged  = false;
};

/// Projiziert den diskreten Gradientenanteil aus dem Quellterm heraus.
///
/// WARUM
///
/// Der Quellterm wird punktweise vorgegeben (J_s = J * phi_hat in den
/// Spulenelementen, null ausserhalb) und dann integriert: b_e = Int W_e . J_s.
/// Im Kontinuum ist div J_s = 0 und J_s . n = 0 auf der Spulenmantelflaeche,
/// diskret aber nicht: die Spule wird ELEMENTWEISE zugeordnet, ihre
/// Ersatzflaeche ist gezackt, und deren Normalen stehen nicht senkrecht auf
/// phi_hat. Der diskrete Spulenstrom ist damit nicht geschlossen. Gemessen
/// ||G^T b||/||b|| = 6,1e-03 gegen die 1e-4, die M2 verlangt -- und durch
/// Netzverfeinerung NICHT behebbar (test_patch_assembly, Pruefung 4 gegen 5).
///
/// Dieser Anteil ist nicht harmlos. Die Regularisierung sigma_Luft =
/// eps * sigma_Leiter verstaerkt ihn um 1/eps: auf Gradientenfeldern ist K
/// exakt null, die Blockform reduziert sich dort auf x_i = -b_g/(omega C).
/// Gemessen waechst max|A_i| exakt mit 1/eps ueber vier Dekaden. Und derselbe
/// Anteil ist das Einzige, woran GMRES mit ILU(0) scheitert -- das
/// Restresiduum ist dort 387-mal gradientenlastiger als die rechte Seite.
///
/// WAS
///
/// Gesucht ist das b' mit G^T b' = 0, das b am naechsten liegt. Das ist die
/// orthogonale Projektion auf den Kern von G^T:
///
///     (G^T G) phi = G^T b,     b' = b - G phi
///
/// G^T G ist der Knoten-Graph-Laplace: Diagonale = Knotengrad,
/// Nebendiagonale = -1 je Kante. Gut konditioniert, ein CG-Lauf.
///
/// G^T G ist singulaer -- konstantes phi liegt im Kern, je
/// Zusammenhangskomponente eines. Die rechte Seite ist dazu automatisch
/// orthogonal (jede Kante traegt +b_e und -b_e bei), das CG bleibt aber nur
/// stabil, wenn der konstante Anteil laufend herausgenommen wird. Genau das
/// tut die Umsetzung.
///
/// WAS SICH DADURCH AENDERT
///
/// Die Loesung aendert sich um genau den Gradientenanteil. Der ist ueber vier
/// eps-Dekaden gemessen ohne Wirkung auf B (rot grad = 0) und ohne Wirkung
/// auf die Kraft in der Platte. Die Aenderung ist damit die BESEITIGUNG eines
/// Artefakts, nicht ein Eingriff in die Physik.
/// `blocked_edge` markiert Kanten, auf denen b null bleiben MUSS (1 = gesperrt,
/// null = keine). Gedacht fuer die Dirichlet-Kanten.
///
/// WARUM DAS NOETIG IST -- ohne die Maske hebt die Randbehandlung einen Teil
/// der Projektion wieder auf.
///
/// b ist vor der Projektion nur im Spulenbereich besetzt, auf dem Rand des
/// Luftkastens exakt null. phi loest aber ein GLOBALES Poisson-Problem, also
/// ist G phi ueberall von null verschieden -- nach b <- b - G phi tragen die
/// Randkanten einen kleinen Wert. apply_homogeneous_dirichlet() setzt ihn
/// anschliessend zurueck auf null, und genau das bringt die Divergenz an den
/// Randknoten wieder ein: gemessen ||G^T b||/||b|| = 1,61e-05 in der
/// 2n-Blockform, wo die Projektion selbst 5,45e-16 gemeldet hatte.
///
/// Mit der Maske laeuft die Projektion nur ueber die FREIEN Kanten:
///
///     (G_F^T G_F) phi = G_F^T b,   b' = b - G_F phi
///
/// Auf gesperrten Kanten bleibt b' null per Konstruktion, und G^T b'
/// verschwindet exakt, weil die gesperrten Kanten nichts beitragen.
///
/// Knoten, die nur an gesperrten Kanten haengen, fallen aus dem Teilgraphen
/// heraus -- ihre Laplace-Zeile ist leer. Sie werden aus dem CG
/// ausgeschlossen, sonst brachte die Konstantenbereinigung ihnen einen Wert
/// ein, den keine Gleichung mehr korrigiert.
GradientProjectionReport project_out_gradient(const Mesh& mesh,
                                              double* b, size_t n,
                                              const char* blocked_edge,
                                              double tol, int max_iter);

}  // namespace fem_direct

#endif  // FEM_DIRECT_BC_H
