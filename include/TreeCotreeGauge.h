#ifndef TREE_COTREE_GAUGE_H
#define TREE_COTREE_GAUGE_H

#include <vector>
#include "Mesh.cuh"
#include "MaterialTypes.cuh"

// =============================================================================
// Baum-Kobaum-Eichung (tree-cotree gauging) für Kantenelemente
// =============================================================================
//
// PROBLEM
// -------
// A_sys = K + C/dt ist singulär. K ist der Curl-Curl-Operator und annihiliert
// jeden diskreten Gradienten: für a_e = φ(v_max) − φ(v_min) gilt K·a = 0 exakt
// (nachgewiesen in tests/test_curl_gradient_nullspace.cpp). Regularisiert wird
// das nur durch C = κ·∫W·W, und κ ist in den Spulen null.
//
// Der Nullraum von A_sys besteht daher genau aus den Gradienten Gφ, für die
// C·Gφ = 0 gilt, also aus jenen φ, die auf jeder zusammenhängenden leitenden
// Komponente konstant sind und auf den übrigen Knoten frei. Bei team28.vtk sind
// das 3905 von 7013 Knoten — eine Nullraumdimension in dieser Größenordnung.
//
// Folge: das Gleichungssystem ist nur lösbar, wenn die rechte Seite orthogonal
// zum Nullraum steht. Gemessen liegt die Nullraumkomponente von b bei 8.8e-3
// relativ (Quadraturfehler der 2x2x2-Regel und facettierte Spulenoberfläche),
// die Toleranz 1e-6 ist damit unerreichbar und ein Krylov-Verfahren stagniert
// oder divergiert.
//
// LÖSUNG
// ------
// Ein Spannbaum trifft den Gradientenanteil genau. Der diskrete Gradient
// G: Knotenwerte -> Kantenwerte ist auf den Kanten eines Spannbaums bijektiv:
// gibt man die Werte auf den Baumkanten vor, ist φ von einem geerdeten
// Startknoten aus eindeutig rekonstruierbar. Setzt man a_e = 0 auf allen
// Baumkanten, bleibt nur a = Gφ mit φ ≡ 0 übrig, also a = 0. Der Nullraum ist
// weg, ohne dass ein Materialparameter verändert wird.
//
// Wichtig sind zwei Einschränkungen:
//
//   1. Nur nichtleitende Kanten dürfen geeicht werden. Eine Kante, die zu
//      mindestens einem leitenden Element gehört, trägt echte physikalische
//      Information (dort ist A_sys regulär) und darf nicht fixiert werden.
//
//   2. Alle Knoten, die ein leitendes Element berühren, werden zu einem
//      einzigen Knoten kontrahiert. Sie sind gemeinsam "geerdet", weil φ auf
//      der leitenden Komponente konstant sein muss. Ohne diese Kontraktion
//      würde der Baum Kanten zwischen zwei Leiterknoten aufnehmen und damit
//      mehr fixieren als der Nullraum hergibt.
//
// UMSETZUNG IM OPERATOR
// ---------------------
// Angewendet wird die Eichung als Projektion. Mit P = "setze geeichte
// Einträge auf null" rechnet CoupledMVOperator::apply
//
//     y = P·A·P·x + (I − P)·x
//
// und die rechte Seite wird zu P·b. Für geeichte Zeilen lautet die Gleichung
// damit x_i = 0, für die übrigen bleibt sie unverändert. Die Form P·A·P erhält
// die Symmetrie der ursprünglichen Matrix, was einem Krylov-Verfahren
// entgegenkommt.
// =============================================================================

struct GaugeInfo {
    // Länge mesh.num_edges. 1 = Freiheitsgrad ist geeicht, also auf null fixiert.
    std::vector<char> is_gauged;

    int num_edges        = 0;
    int num_gauged       = 0;   // Baumkanten, entspricht der Nullraumdimension
    int num_conductor_nodes = 0;
    int num_conductor_edges = 0;
};

// Baut die Eichung. materials muss host-lesbar sein (Unified Memory) und
// mesh.num_hexes Einträge haben; als leitend gilt kappa > kappa_threshold.
//
// Die Kanten werden in aufsteigender Indexreihenfolge betrachtet, das Ergebnis
// ist damit plattformunabhängig reproduzierbar.
//
// Wirft std::runtime_error bei inkonsistenter Eingabe.
GaugeInfo build_tree_cotree_gauge(const Mesh& mesh,
                                  const MaterialProperties* materials,
                                  double kappa_threshold = 0.0);

#endif // TREE_COTREE_GAUGE_H
