#ifndef NODAL_DISPATCH_CUH
#define NODAL_DISPATCH_CUH

#include "Mesh.cuh"
#include "NedelecDispatch.cuh"     // compute_element_jacobian, piola_covariant,
                                   // element_shape_functions, Quadratur
#include "NodalHexahedron.cuh"     // evaluate_nodal_shape_grad_ref (Hexaeder)

// =============================================================================
// NodalDispatch.cuh -- SKALARE Knotenformfunktionen fuer alle Elementtypen
// =============================================================================
//
// WOZU
//
// NedelecDispatch.cuh deckt die KANTENbasis ab. Gebraucht werden aber auch die
// skalaren Knotenformfunktionen und vor allem ihre PHYSIKALISCHEN Gradienten,
// und zwar an zwei Stellen mit ganz verschiedener Herkunft:
//
//   * build_alpha_poisson_matrix() -- der knotenweise Vektor-Poisson-Operator,
//     auf dem der vektorielle AMS-Hilfsraum sein AMG aufbaut.
//   * NodalMaxwell.cuh / der Referenzloeser -- Knotenelemente statt
//     Kantenelemente.
//
// Beide Stellen waren bis hierher HEXAEDERFEST: sie lasen hex.v[0..7] und
// rechneten mit trilinearen Q1-Formfunktionen. Auf einem Tetraedernetz steht
// auf hex.v[4..7] eine -1, und die Formfunktionen sind ohnehin die falschen.
// Der FEM-direkt-Pfad vernetzt mit Tetraedern -- ohne diesen Header ist der
// vektorielle AMS-Hilfsraum dort nicht aufbaubar.
//
// WARUM DER HEXAEDER EINEN EIGENEN ZWEIG HAT
//
// Fuer Tetraeder und Prismen ruft element_nodal_gradients() die Jacobi-Matrix
// ueber compute_element_jacobian() aus NedelecDispatch.cuh, also GENAU die
// Abbildung, mit der auch die Kantenmatrix assembliert wird.
//
// Beim Hexaeder waere das ebenfalls moeglich -- und ist es nicht. Dort bleibt
// evaluate_nodal_gradients() aus NodalHexahedron.cuh stehen, weil der
// bestehende transiente Pfad denselben Operator benutzt und BITGLEICH weiter
// rechnen soll. Die beiden Wege sind algebraisch identisch und unterscheiden
// sich nur in der Reihenfolge von Transposition und Adjunkte:
// NodalHexahedron bildet J[a][b] = dx_a/dr_b und multipliziert mit invJ^T,
// NedelecDispatch bildet die transponierte Form und multipliziert mit invJ.
//
// Gemessen ueber 2000 zufaellig verzerrte Hexaeder und alle acht
// Quadraturpunkte betraegt der Unterschied hoechstens 6,4e-16 relativ im
// Gradienten und 2,2e-16 in detJ -- rund drei ulp, also reine Rundung.
//
// Wer die beiden Zweige spaeter zusammenlegen will, soll das mit dieser Zahl
// vor Augen tun: es ist sachlich folgenlos (ein Vorkonditionierer muss nicht
// exakt sein), macht aber jede frueher protokollierte Iterationszahl des
// transienten Laeufers unvergleichbar. Deshalb steht es hier so und nicht
// kuerzer.
//
// KONVENTIONEN (unveraendert die von NedelecDispatch.cuh)
//
//   J[a][b] = d x_b / d r_a          (also J_std TRANSPONIERT)
//   invJ    = J^-1
//
// Daraus folgt fuer den Gradienten einer skalaren Funktion
//
//   grad_x N = J_std^-T grad_r N = invJ * grad_r N = piola_covariant(invJ, .)
//
// Die kovariante Piola-Transformation IST also die Gradiententransformation.
// Das ist kein Zufall: die Whitney-Kantenbasis erfuellt
// grad(phi) = Summe_e (phi_n2 - phi_n1) W_e exakt, und beide Seiten muessen
// sich unter der Abbildung gleich verhalten.
//
// Referenzkoordinaten heissen r, s, t: beim Hexaeder xi, eta, zeta in
// [-1,1]^3, beim Tetraeder die ersten drei baryzentrischen Koordinaten in
// [0,1], beim Prisma zwei Flaechenkoordinaten des Dreiecks und zeta in [-1,1].
// =============================================================================

// -----------------------------------------------------------------------------
// Ableitungen der Formfunktionen im REFERENZelement
// -----------------------------------------------------------------------------
// dN[i][0..2] = dN_i/dr, dN_i/ds, dN_i/dt. Ungenutzte Plaetze werden auf null
// gesetzt -- der Aufrufer darf dann ueber alle 8 laufen, ohne den Typ zu
// kennen, und ein vergessener Eintrag traegt nichts bei statt Muell.
__host__ __device__ inline void element_nodal_shape_derivatives(
    int type, double r, double s, double t, double dN[8][3])
{
    for (int i = 0; i < 8; ++i) { dN[i][0] = dN[i][1] = dN[i][2] = 0.0; }

    if (type == ELEM_TET) {
        double dr[TET_NUM_NODES], ds[TET_NUM_NODES], dt[TET_NUM_NODES];
        tet_shape_derivatives(r, s, t, dr, ds, dt);
        for (int i = 0; i < TET_NUM_NODES; ++i) {
            dN[i][0] = dr[i]; dN[i][1] = ds[i]; dN[i][2] = dt[i];
        }
        return;
    }
    if (type == ELEM_PRISM) {
        double dr[PRISM_NUM_NODES], ds[PRISM_NUM_NODES], dz[PRISM_NUM_NODES];
        prism_shape_derivatives(r, s, t, dr, ds, dz);
        for (int i = 0; i < PRISM_NUM_NODES; ++i) {
            dN[i][0] = dr[i]; dN[i][1] = ds[i]; dN[i][2] = dz[i];
        }
        return;
    }
    // ELEM_HEX: dieselbe Q1-Basis wie bisher, aus NodalHexahedron.cuh. Der
    // Aufruf ist bewusst nicht durch eine Kopie ersetzt -- die Vorzeichen-
    // tabellen stehen dort und sollen an EINER Stelle stehen.
    evaluate_nodal_shape_grad_ref(r, s, t, dN);
}

// -----------------------------------------------------------------------------
// Physikalische Gradienten und detJ
// -----------------------------------------------------------------------------
// gradN[i] = grad N_i im physikalischen Raum; ungenutzte Plaetze sind null.
//
// detJ wird DURCHGEREICHT, nicht geprueft. Ein entartetes Element hat detJ
// nahe null, und was dann zu tun ist, haengt vom Aufrufer ab: die
// Assemblierung ueberspringt es, ein Test soll daran scheitern. Eine
// Entscheidung hier waere fuer eine der beiden Seiten falsch.
//
// Vorzeichen: beim Tetraeder ist detJ negativ, wenn die Knoten (0,1,2,3) kein
// Rechtssystem bilden. Der Gradient ist davon UNBERUEHRT -- invJ dreht das
// Vorzeichen mit. Nur das Volumenelement braucht |detJ|.
__host__ __device__ inline void element_nodal_gradients(
    int type, const double3 nodes[8], double r, double s, double t,
    double3 gradN[8], double& detJ)
{
    if (type != ELEM_TET && type != ELEM_PRISM) {
        // Hexaeder: unveraendert der bisherige Weg, siehe Kopf dieser Datei.
        evaluate_nodal_gradients(nodes, r, s, t, gradN, detJ);
        return;
    }

    double dN[8][3];
    element_nodal_shape_derivatives(type, r, s, t, dN);

    double J[3][3], invJ[3][3];
    compute_element_jacobian(type, nodes, r, s, t, J, invJ, detJ);

    for (int i = 0; i < 8; ++i) {
        const double3 g_ref = make_double3(dN[i][0], dN[i][1], dN[i][2]);
        gradN[i] = piola_covariant(invJ, g_ref);
    }
}

#endif // NODAL_DISPATCH_CUH
