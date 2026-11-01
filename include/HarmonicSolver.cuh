#ifndef HARMONIC_SOLVER_CUH
#define HARMONIC_SOLVER_CUH

// =============================================================================
// Harmonischer Loeser: eingeschwungener Zustand statt 1000 Zeitschritten
// =============================================================================
//
// WARUM DAS EXAKT IST, NICHT GENAEHERT
//
// TEAM 28 ist vollstaendig linear: mu_r = 1 in allen drei Koerpern, sigma
// konstant, keine Saettigung. Gesucht ist die Schwebehoehe, also die
// zeitgemittelte Kraft im eingeschwungenen Zustand bei RUHENDER Platte --
// kein Geschwindigkeitsterm, keine Anfahrdynamik. Der harmonische Ansatz ist
// deshalb keine Vereinfachung der Transiente, sondern derselbe Zustand,
// direkt gerechnet.
//
// DIE FORMULIERUNG
//
// Transient ist der FEM-Block K + C/dt (C ist die sigma-Massenmatrix).
// Harmonisch wird daraus K + j*omega*C. Der BEM-Block und die Kopplung sind
// frequenzunabhaengig und gehen unveraendert ein.
//
// Gerechnet wird REELL, in Blockform, mit x = [x_r; x_i] der Laenge 2*nd:
//
//     H = [  K    -omega*C ]
//         [ omega*C    K   ]
//
// wobei K der VOLLSTAENDIGE gekoppelte Operator ohne Massenterm ist (FEM-Block,
// BEM-Block, beide Kopplungsbloecke) und C nur im FEM-Block traegt.
//
// Reell und nicht komplex, weil damit die gesamte vorhandene Maschinerie
// unveraendert weiterverwendbar bleibt: CoupledMVOperator, ACA, ILU, AMS,
// GMRESSolver. Kein komplexer Loeser, keine komplexen Kernel, keine komplexe
// ACA.
//
// DER VORKONDITIONIERER
//
// C-to-R nach Axelsson/Kucherov: fuer H = [[W,-T],[T,W]] ist
//
//     P = [[W+T, 0], [T-W, W+T]]
//
// Eine Anwendung braucht ZWEI Loesungen mit (W+T) = K + omega*C. Das ist ein
// REELLER gekoppelter Operator von genau der Bauart, auf die ILU und AMS im
// Projekt ohnehin schon zielen -- kein neuer Vorkonditionierer noetig, nur ein
// anderer Aufruf des vorhandenen.
//
// WAS GEMESSEN IST (team28_coarse_12, --harmonic-study)
//
//   transient (dt = 1e-4):   kappa_2 = 1,07e+06, GMRES 4000 Iter, stagniert
//   harmonisch:              kappa_2 = 2,07e+06, GMRES 4000 Iter, stagniert
//   harmonisch + C-to-R:     38 Iterationen auf 7,49e-09
//
// kappa STEIGT, und die Iterationszahl faellt um Faktor 105. Nicht die
// Kondition entscheidet, sondern die Blockstruktur.
//
// Genauigkeit des inneren (W+T)-Loesers: 1e-1 und 1e-2 tragen nicht, ab 1e-3
// traegt es (81 aeussere Iterationen), bei 1e-4 sind es 41. Die Gesamtarbeit
// ist bei 1e-4 am kleinsten: 41 x 2914 = 2,39e+06 Matvecs gegen 4,00e+06 des
// transienten Laufs. Der Vorgabewert fuer inner_tol steht deshalb auf 1e-4.
//
// UND WAS NICHT ZU TUN IST
//
// Die Deckenmessung auf (W+T) hat gezeigt: eine blocktriangulare Behandlung
// des INNEREN Operators kostet 8 x 1205 x 639 = 6,16e+06 Matvecs je Loesung,
// schlichtes GMRES nur 2914 -- unvorkonditioniert ist 2100-mal billiger. Die
// Blockstruktur zahlt sich AUSSEN aus, in C-to-R, innen nicht. Der innere
// Loeser laeuft deshalb mit dem gewoehnlichen ILU-Pfad und ausdruecklich OHNE
// set_block_triangular_preconditioner().

#include "fem.cuh"
#include "CoupledSolver.cuh"
#include "HarmonicPhaseAverage.h"   // Zeitmittelung, reiner Host-Code

#include <vector>

namespace bf {

class DenseLU;

struct HarmonicSolveResult {
    int    iterations   = 0;
    double relative_res = -1.0;
    bool   converged    = false;
    long   inner_solves = 0;   ///< Zahl der (W+T)-Loesungen, fuer die Kostenrechnung
    long   inner_iters  = 0;   ///< deren Iterationen zusammen
};

/// Loest H [x_r; x_i] = [b_r; b_i] mit rechts-vorkonditioniertem GMRES(m).
///
/// op_W  muss mit der FEM-Matrix K + omega*C aufgesetzt sein und compute_ilu()
///       hinter sich haben. Das ist GENAU das, was
///       build_implicit_euler_system() bei dt = 1/omega liefert, denn dort ist
///       A_sys = K + C/dt = K + omega*C. Keine Differenzbildung noetig.
///
///       Ein zweiter Operator fuer K allein wird NICHT gebraucht: mit
///       K = (W+T) - omega*C laesst sich H vollstaendig durch (W+T) und
///       omega*C ausdruecken. Das erspart eine zweite vollstaendige
///       Konfiguration (BEM-Kopplung, Eichung, Kohomologie, AMS,
///       Aequilibrierung) und damit zwei Operatoren, die auseinanderlaufen
///       koennen, ohne dass es auffaellt.
/// csr_C ist die sigma-Massenmatrix im FEM-Raum (num_rows = total_dofs).
///
/// nd = total_dofs + num_bem_dofs. Die Vektoren d_x_* und d_b_* haben je
/// Laenge nd; C wird nur auf die ersten total_dofs Eintraege angewandt, der
/// BEM-Anteil bleibt unberuehrt (der BEM-Block ist frequenzunabhaengig).
///
/// inner_direct: wenn gesetzt, eine Zerlegung von (W+T) im Tilde-Raum (aus
///       sample_operator_dense() und DenseLU::factorize()). C-to-R loest dann
///       jede innere Gleichung exakt statt mit GMRES; inner_tol und
///       inner_max_iter sind ohne Wirkung.
///
/// Rechts-Vorkonditionierung und nicht links: nur so minimiert GMRES das
/// ECHTE Residuum ||b - Hx||, und nur so ist die Iterationszahl mit der des
/// transienten Loesers vergleichbar, der ebenfalls rechts vorkonditioniert.
HarmonicSolveResult solve_harmonic(CoupledMVOperator& op_W,
                                   const CSRMatrix& csr_C,
                                   int nd, int total_dofs, double omega,
                                   const double* d_b_r, const double* d_b_i,
                                   double* d_x_r, double* d_x_i,
                                   double tol, int max_iter, int restart,
                                   double inner_tol, int inner_max_iter,
                                   const DenseLU* inner_direct = nullptr);

/// Tastet op (den Operator im Tilde-Raum, wie ihn GMRES sieht) spaltenweise
/// in die dichte nd x nd Matrix d_A ab (column-major, Geraete- bzw.
/// Managed-Speicher). Fuer --harmonic-inner-direct: die Zerlegung davon ersetzt
/// das innere GMRES in C-to-R.
///
/// op.apply() addiert auf dem Eichnullraum (I - P) x, die abgetastete Matrix
/// ist dort also die Identitaet und damit regulaer, obwohl (W+T) auf den
/// geeichten Kanten und den Kohomologie-Richtungen singulaer ist.
void sample_operator_dense(CoupledMVOperator& op, int nd, double* d_A);

} // namespace bf

#endif // HARMONIC_SOLVER_CUH
