#ifndef REFERENCE_SOLVER_CUH
#define REFERENCE_SOLVER_CUH

// =============================================================================
// ReferenceSolver.cuh -- BASELINE-Loeser: Knotenelemente + vollbesetzte BEM
// =============================================================================
//
// Start mit:  ./fembem --reference [Schritte]
//
// Zweck: eine moeglichst einfach nachvollziehbare, unabhaengige Zweitrechnung
// fuer TEAM 28. Der produktive Pfad (Nedelec-Kantenelemente, Baum-Kobaum- und
// Kohomologie-Eichung, ACA-BEM, 2x2-Blocksystem) hat viele Stellen, an denen
// ein Fehler stecken kann. Dieser Loeser verzichtet auf alle davon:
//
//   produktiv                        Referenz (dieser Loeser)
//   -------------------------------  ----------------------------------------
//   Nedelec-Kantenelemente           Q1-Knotenelemente, 3 DOF je Knoten
//   Baum-Kobaum + Kohomologie        Strafterm-Eichung (div-div)
//   ACA-/H-Matrix-BEM                vollbesetzte BEM, keine Approximation
//   2x2-Blocksystem mit q            q eliminiert (Dirichlet-zu-Neumann)
//   Aequilibrierung, ILU-Gauge-Shift ILU(0) auf einem SPD-System
//
// -----------------------------------------------------------------------------
// Formulierung
// -----------------------------------------------------------------------------
// VORAUSSETZUNG: mu_r == 1 im gesamten Gebiet. Bei TEAM 28 ist das erfuellt
// (Aluminium, Kupfer, Luft), und der Loeser prueft es beim Start. Mit mu =
// mu_0 = const gilt
//
//     a(A,A') = Int (1/mu_0) [ curl A . curl A' + beta (div A)(div A') ]
//
// und fuer beta = 1 ist das bis auf Randterme der KOMPONENTENWEISE
// Vektor-Laplace. In genau dieser Form wird assembliert, denn nur dann ist der
// natuerliche Randterm eindeutig:
//
//     Int (1/mu_0) grad A : grad A'  -  Int_Gamma (1/mu_0) q . A' dGamma,
//     q = dA/dn
//
// Der div-div-Strafterm steckt darin (beta = 1) und beseitigt den
// Gradienten-Nullraum -- deshalb braucht dieser Loeser keine Eichung ueber
// Baum-Kobaum und keine Kohomologie-Projektoren. Ein abweichendes beta wird
// als ZUSATZterm (beta-1)*div-div addiert; das ist als Experimentierknopf
// gedacht, denn fuer beta != 1 passen die Randterme nicht mehr exakt zur
// BEM-Kopplung (siehe REFERENCE_PENALTY_BETA).
//
// Aussenraum: quellfrei, mu = mu_0, Coulomb-Eichung -> jede kartesische
// Komponente von A ist harmonisch. Das Aussenproblem zerfaellt in drei
// identische skalare Laplace-Probleme, V und K sind fuer alle drei Komponenten
// DIESELBE Matrix (siehe NodalBem.cuh). Deshalb kostet die vollbesetzte BEM
// nur 2 * N_bem^2 * 8 B und nicht das Dreifache.
//
// Kopplung: q = V^-1 (K - 1/2 I) A|_Gamma =: D A|_Gamma, eingesetzt in den
// Randterm. Das ergibt einen vollbesetzten Block G = T*D auf den Randknoten
// (T = Randmassenmatrix Int N_i N_j dGamma). Das Gesamtsystem ist damit
// "sparse FEM + ein dichter Block", ohne separate q-Unbekannten.
//
// -----------------------------------------------------------------------------
// Bekannte Grenze (gemessen, nicht vermutet)
// -----------------------------------------------------------------------------
// Die Neumann-Spur q wird mit STETIGER Knotenbasis dargestellt. An glatten
// Flaechen konvergiert die BEM damit mit O(h^2) (an der Kugel nachgemessen:
// 1.40e-2 -> 4.00e-3 -> 1.83e-3 bei 4/8/12 Panels je Seite). An GEOMETRISCHEN
// KANTEN ist q physikalisch unstetig, und eine stetige Basis kann das nicht
// darstellen -- am Wuerfel bleiben rund 10 % Restfehler und die Rate faellt auf
// etwa O(h^0.75). Platte und Spulen bei TEAM 28 sind Kaesten mit scharfen
// Kanten, dieser Fehler ist also vorhanden und wirkt auf die Kraft.
// Standardabhilfe waere eine unstetige q-Darstellung (Doppelknoten an Kanten);
// das ist bewusst NICHT Teil dieser ersten Fassung.
// tests/test_reference_solver misst beide Faelle.
//
// -----------------------------------------------------------------------------
// Laufzeit
// -----------------------------------------------------------------------------
// Assemblierung (FEM, V, K, T) und die LU von V laufen auf der GPU. Der
// Krylov-Teil laeuft noch host-seitig ueber bf:: auf Managed Memory: das
// Sparse-Mat-Vec benutzt bf::csr_spmv statt cusparseSpMV. Damit ist GMRES der
// Engpass -- grob 10..60 s je Zeitschritt bei diesem Netz. Fuer einen
// Vergleichslauf ueber wenige Schritte reicht das; fuer 1000 Schritte muss das
// Mat-Vec auf die GPU.
// =============================================================================

#include <cuda_runtime.h>

#include <vector>

#include "Backend.cuh"
#include "DeviceArray.cuh"
#include "Mesh.cuh"
#include "MaterialTypes.cuh"

// -----------------------------------------------------------------------------
// Einstellgroessen
// -----------------------------------------------------------------------------
/// Vorfaktor des div-div-Strafterms. 1.0 = konsistenter Fall (komponentenweiser
/// Vektor-Laplace, Randterme passen exakt zur BEM-Kopplung). Abweichende Werte
/// aendern die Formulierung; siehe Dateikopf.
constexpr double REFERENCE_PENALTY_BETA = 1.0;

/// GMRES: Restartlaenge und Schranken.
constexpr int    REFERENCE_GMRES_RESTART   = 40;
constexpr int    REFERENCE_GMRES_MAX_ITER  = 600;
constexpr double REFERENCE_GMRES_TOL       = 1e-9;

// -----------------------------------------------------------------------------
// Randtopologie fuer die BEM
// -----------------------------------------------------------------------------
struct ReferenceBoundary {
    /// Globale Knoten-IDs aller Randknoten, aufsteigend sortiert.
    std::vector<int> bnode_ids;
    /// Laenge num_nodes: Position in bnode_ids, oder -1 fuer innere Knoten.
    std::vector<int> node_to_bnode;
    /// Je Rand-Quad +1 oder -1, so dass cross(x_u, x_v) nach AUSSEN zeigt.
    /// Wird geometrisch bestimmt (Panelmitte gegen Mitte des Elternhexaeders),
    /// nicht aus einer Tabelle -- das ist unabhaengig von der
    /// Knotenumlaufkonvention in QuadFace und damit nicht falsch zu bekommen.
    std::vector<double> face_orientation;

    size_t num_bnodes() const { return bnode_ids.size(); }
};

ReferenceBoundary build_reference_boundary(const Mesh& mesh);

// -----------------------------------------------------------------------------
// Einsprungpunkt
// -----------------------------------------------------------------------------
/// Fuehrt den kompletten transienten Lauf mit dem Referenzloeser aus.
/// Erwartet ein bereits gelesenes Netz mit extrahierter Topologie und
/// zugewiesenen Materialien -- also genau den Zustand, den mainloop() nach
/// seiner Setup-Phase hat.
///
/// Gibt 0 zurueck bei Erfolg.
/// 'sources' sind die Objekte mit zeitabhaengigem Quellstrom (die beiden
/// Spulen). Sie muessen in jedem Zeitschritt fortgeschrieben UND in
/// d_materials zurueckgeschrieben werden -- dort liegen Kopien, nicht
/// Verweise.
int run_reference_solver(Mesh& mesh,
                         DeviceArray<MaterialProperties>& d_materials,
                         class PhysicalObject& plate,
                         const std::vector<class PhysicalObject*>& sources,
                         int max_steps,
                         double dt);

#endif // REFERENCE_SOLVER_CUH
