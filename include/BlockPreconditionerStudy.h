#ifndef BLOCK_PRECONDITIONER_STUDY_H
#define BLOCK_PRECONDITIONER_STUDY_H

// =============================================================================
// Die Deckenmessung fuer den BEM-Block
// =============================================================================
//
// ANLASS
//
// docs/reports/convergence-findings.tex nennt den BEM-Block den "Elefanten im Raum": G_BEM geht mit
// der Identitaet in den Loeser, und die blockweise Aufschluesselung des
// Residuums zeigt ihn mit 1,8e-04 gegen 1,3e-04 des FEM-Blocks als den
// GROESSEREN Anteil. Daraus wurde der Vorschlag abgeleitet, einen
// Calderon-Vorkonditionierer oder eine H-Matrix-Naeherungsinverse fuer G_BEM
// zu bauen.
//
// Beides ist teuer. Calderon braucht den hypersingulaeren Operator W auf einem
// dualen (barizentrischen) Netz mit Buffa-Christiansen-Basisfunktionen -- ein
// zweiter kompletter BEM-Assemblierungspfad. Die H-Matrix-Inverse braucht
// Addition, Multiplikation und Inversion auf der Blockstruktur des
// Clusterbaums.
//
// Bevor eine dieser beiden Wochen kostet, gehoert die Frage beantwortet, die
// der Bericht selbst als Regel aufstellt: JEDE VERMUTUNG ZIEHT EINE MESSUNG
// NACH SICH, DIE SIE WIDERLEGEN KANN.
//
// WAS HIER GEMESSEN WIRD
//
// Die blockweise Residuumsaufschluesselung sagt, WO das Residuum sitzt. Sie
// sagt NICHT, ob der Block schlecht konditioniert ist -- ein gut
// konditionierter Block kann den groesseren Residuumsanteil tragen, wenn die
// rechte Seite dort liegt. Die beiden Fragen sind verschieden, und nur die
// zweite entscheidet ueber den Vorkonditionierer.
//
// Gemessen wird deshalb die DECKE: was brachte der BESTMOEGLICHE
// Vorkonditionierer fuer den BEM-Block, naemlich die EXAKTE Inverse von
// A22 = mu_0*G_BEM?
//
// Das ist die entscheidende Zahl, denn:
//
//   Calderon, H-Matrix-Naeherungsinverse, ILU auf dem Nahfeld, Block-Jacobi --
//   sie alle sind NAEHERUNGEN an A22^-1. Keine davon kann besser sein als
//   A22^-1 selbst.
//
// Bringt die exakte Inverse nichts, ist der ganze Ansatz widerlegt, und zwar
// endgueltig, ohne dass eine einzige Zeile Calderon-Code geschrieben werden
// muss. Bringt sie viel, ist der Aufwand belegt gerechtfertigt und man kennt
// sogar den Gewinn, den man anstreben kann.
//
// Fuenf Varianten werden verglichen (n_A = FEM-Zeilen, n_K = BEM-Zeilen):
//
//   0. M = I                                  Bezugsgroesse
//   1. M = blockdiag(A11, I)                  exakter FEM-Block  -> Decke fuer AMS/ILU
//   2. M = blockdiag(I,  A22)                 exakter BEM-Block  -> DECKE FUER CALDERON
//   3. M = blockdiag(A11, A22)                beide exakt
//   4. M = [[A11, A12],[0, S]], S = A22 - A21 A11^-1 A12
//                                             exaktes Schur-Komplement
//
// Variante 4 ist zusaetzlich die Gegenprobe zu der im Bericht widerlegten
// Naeherung S~ = mu_0*G_BEM: sie zeigt, ob die Schur-Idee als solche falsch
// war oder nur die NAEHERUNG, die dafuer benutzt wurde. Das sind zwei sehr
// verschiedene Befunde. Der erste beendet den Ansatz, der zweite verlagert ihn
// auf die Frage, wie A21 A11^-1 A12 zu naehern ist.
//
// Ausgegeben wird je Variante kappa_2(M^-1 A) und die Zahl der
// GMRES-Iterationen bis zur Toleranz. Beide, weil sie verschiedene Fehler
// machen: kappa_2 ist fuer unsymmetrische Matrizen nur eine grobe Vorhersage
// der Krylov-Konvergenz (die haengt am Spektrum, nicht an den
// Singulaerwerten), und die Iterationszahl haengt an der rechten Seite. Wo
// beide dasselbe sagen, ist der Befund belastbar.
//
// BAUART
//
// Reiner Host-C++17-Code, keine CUDA-Abhaengigkeit, keine BLAS-Abhaengigkeit.
// Das ist Absicht: die Datei ist damit ohne CUDA-Toolkit uebersetzbar und
// direkt testbar (tests/test_block_preconditioner_study.cpp), und sie kann
// zugleich aus fem.cu heraus auf der Matrix aufgerufen werden, die --direct
// ohnehin schon spaltenweise abtastet. Es entsteht kein zweiter Abtastlauf.
//
// AUFWAND
//
// Kubisch in n. Gedacht fuer die groben Ersatznetze aus
// tools/make_coarse_team28.py (n = 1791 bis 9139), nicht fuer das Originalnetz.
// Fuer n = 4511 liegt eine Zerlegung bei rund 3e10 Flops; die Studie braucht
// mehrere davon.

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

namespace bf {

// -----------------------------------------------------------------------------
// Dichte LU mit Spaltenpivotisierung, spaltenweise Speicherung (column-major),
// wie bf::DenseLU -- aber ohne CUDA-Abhaengigkeit und MIT transponierter
// Loesung. Letztere braucht die Konditionsmessung: sigma_min(A) ist
// 1/sigma_max(A^-1), und die Potenzmethode auf A^-T A^-1 verlangt beide
// Richtungen. Ohne sie muesste A^T ein zweites Mal aufgestellt und zerlegt
// werden -- genau das tut --direct heute, und es kostet Speicher und Zeit
// doppelt.
// -----------------------------------------------------------------------------
class HostDenseLU {
public:
    /// Zerlegt die (n x n)-Matrix in place. Wirft bei singulaerem Pivot.
    ///
    /// Die Singularitaetspruefung ist RELATIV zur groessten Spaltennorm und
    /// schliesst Endlichkeit ausdruecklich ein -- die Lehre aus
    /// docs/design/nan-and-thresholds.md. "best == 0.0" laesst 1e-300 durch, und
    /// "v > best" sieht ein NaN nie.
    void factorize(std::vector<double>& A_colmajor, int n);

    /// Loest A x = b in place (x ueberschreibt b).
    void solve(double* b) const;

    /// Loest A^T x = b in place.
    void solve_transposed(double* b) const;

    /// Loest A X = B fuer nrhs Spalten (B column-major, n x nrhs), in place.
    void solve_many(double* B, int nrhs) const;

    int size() const { return m_n; }

private:
    int m_n = 0;
    std::vector<double> m_lu;    // L\U, column-major
    std::vector<int>    m_piv;   // Zeilentausche
};

// -----------------------------------------------------------------------------
// kappa_2 = sigma_max / sigma_min ueber Potenzmethoden.
// -----------------------------------------------------------------------------

struct ConditionEstimate {
    double sigma_max = 0.0;
    double sigma_min = 0.0;
    double kappa     = 0.0;   ///< -1.0, wenn nicht bestimmbar (singulaer)
    bool   converged = false; ///< beide Potenzmethoden unter der Toleranz
};

/// Misst kappa_2 der dichten (n x n)-Matrix. A wird NICHT veraendert
/// (intern wird eine Kopie zerlegt).
///
/// sigma_max: Potenzmethode auf A^T A.
/// sigma_min: 1 / sigma_max(A^-1), Potenzmethode auf A^-T A^-1 ueber die LU.
ConditionEstimate estimate_condition(const std::vector<double>& A_colmajor, int n,
                                     int max_power_iters = 5000,
                                     double power_tol = 1e-10);

/// Die k KLEINSTEN Singulaerwerte, ueber Block-Inverse-Iteration mit
/// Rayleigh-Ritz.
///
/// Wozu: sigma_min allein sagt nicht, WAS fast singulaer ist. Ein EINZELNER
/// abgesetzter Wert deutet auf einen Restmodus, den die Eichung nicht erwischt
/// hat -- etwas Behebbares. Ein ganzes BUENDEL kleiner Werte deutet auf einen
/// echten Fastnullraum des Operators -- dann ist es keine Eichfrage mehr,
/// sondern die Physik (nichtleitende Gebiete ohne sigma/dt-Term), und der
/// Vorkonditionierer muss diesen Raum behandeln statt ihn wegzuwuenschen.
///
/// Die beiden Faelle verlangen entgegengesetzte Massnahmen, und sigma_min
/// unterscheidet sie nicht. Deshalb diese Funktion.
///
/// Rueckgabe: aufsteigend sortiert, hoechstens k Werte. Leer, wenn A nicht
/// zerlegbar ist.
std::vector<double> smallest_singular_values(const std::vector<double>& A_colmajor, int n,
                                             int k, int max_iters = 200,
                                             double tol = 1e-10);
///
/// Gebraucht fuer die Frage, ob CG auf einer Matrix ueberhaupt zulaessig ist.
/// Das innere CG des blocktriangulaeren Zweigs setzt voraus, dass sein
/// Operator positiv definit ist. Fuer S~ = mu_0*G_BEM stimmt das (Single-Layer,
/// SPD). Fuer das ECHTE Schur-Komplement S = A22 - A21 A11^-1 A12 ist es eine
/// offene Frage -- und wenn lambda_min < 0 ist, war CG dort von vornherein das
/// falsche Verfahren.
void symmetric_part_extremes(const std::vector<double>& A_colmajor, int n,
                             double& lambda_min, double& lambda_max,
                             int max_power_iters = 5000, double power_tol = 1e-10);

// -----------------------------------------------------------------------------
// Dichtes GMRES(m) mit RECHTS-Vorkonditionierung
// -----------------------------------------------------------------------------
//
// Rechts und nicht links, weil der Loeser des Projekts (GMRESSolver in
// CoupledSolver.cuh) rechts vorkonditioniert. Bei Links-Vorkonditionierung
// minimiert GMRES das VORKONDITIONIERTE Residuum ||M^-1(b - Ax)||, bei
// Rechts-Vorkonditionierung das echte ||b - Ax||. Die Iterationszahlen sind
// deshalb nicht vergleichbar, und nur die rechte Variante misst dieselbe
// Groesse wie der echte Loeser.

struct DenseGmresResult {
    int    iterations   = 0;
    double relative_res = -1.0;
    bool   converged    = false;
};

/// Loest A x = b. apply_M_inverse(v) ueberschreibt v (Laenge n) mit M^-1 v.
/// restart <= 0 bedeutet volles GMRES (kein Neustart).
///
/// std::function und kein Template: die Kosten liegen bei den Matrix-Vektor-
/// Produkten (O(n^2) je Iteration), ein indirekter Aufruf je Iteration faellt
/// dagegen nicht ins Gewicht -- und die Definition bleibt in der .cpp, wo sie
/// getestet werden kann.
DenseGmresResult dense_gmres(const std::vector<double>& A_colmajor, int n,
                             const std::vector<double>& b,
                             const std::function<void(double*)>& apply_M_inverse,
                             double tol, int max_iter, int restart);

// -----------------------------------------------------------------------------
// Die Studie
// -----------------------------------------------------------------------------

/// Kosten der INNEREN Loesungen, in Krylov-Iterationen.
///
/// Die Reihen 7 bis 11 sagen, WIE GENAU die inneren Loesungen sein muessen.
/// Sie sagen nicht, WAS diese Genauigkeit kostet -- dort wird mit exakten
/// LU-Zerlegungen gerechnet, die es im Ernstfall nicht gibt. Ohne diese Zahl
/// ist die Empfehlung unvollstaendig: 127 aeussere Iterationen sind ein
/// Fortschritt gegenueber 4000 stagnierenden nur dann, wenn jede davon nicht
/// das Hundertfache kostet.
///
/// Gemessen wird deshalb, wieviele unvorkonditionierte Krylov-Schritte S_A
/// und A22 brauchen, um die geforderte Toleranz zu erreichen, und daraus die
/// Gesamtarbeit in BEM-Matrix-Vektor-Produkten -- der Waehrung, in der der
/// Produktivlauf bezahlt.
struct InnerCostEstimate {
    int    iters_S_A     = -1;   ///< Krylov-Schritte fuer S_A bis zur Toleranz
    int    iters_A22     = -1;   ///< Krylov-Schritte fuer A22 bis zur Toleranz
    double tol           = 0.0;
    int    outer_iters   = -1;   ///< aeussere Iterationen der Produktivlage
    double bem_matvecs   = -1.0; ///< geschaetzte BEM-Matvecs je Zeitschritt
    double baseline      = -1.0; ///< dieselbe Groesse fuer den heutigen Loeser
};

/// A_S_A und A_A22 sind die dichten Bloecke; tol die geforderte relative
/// Genauigkeit der inneren Loesung.
InnerCostEstimate estimate_inner_cost(const std::vector<double>& S_A_colmajor, int n_A,
                                      const std::vector<double>& A22_colmajor, int n_K,
                                      double tol, int outer_iters, int max_iter = 4000);

struct BlockPcVariant {
    std::string name;
    double kappa        = -1.0;
    double sigma_max    = 0.0;
    double sigma_min    = 0.0;
    int    gmres_iters  = -1;
    double gmres_relres = -1.0;
    bool   gmres_converged = false;
};

struct BlockPcStudyOptions {
    double gmres_tol      = 1e-8;
    int    gmres_max_iter = 3000;
    int    gmres_restart  = 0;     ///< 0 = volles GMRES
    bool   measure_kappa  = true;  ///< kappa_2(M^-1 A) je Variante (teuer)
    int    power_iters    = 5000;
    double power_tol      = 1e-10;
};

struct SchurDiagnostics {
    bool   available   = false;
    double kappa       = -1.0;   ///< kappa_2(S)
    double rel_coupling= -1.0;   ///< ||A21 A11^-1 A12||_F / ||S||_F
    double rel_asymmetry = -1.0; ///< ||S - S^T||_F / ||S||_F
    double sym_lambda_min = 0.0; ///< kleinster Eigenwert von (S+S^T)/2
    double sym_lambda_max = 0.0;
    bool   positive_definite = false;
};

struct BlockPcStudyResult {
    ConditionEstimate whole;   ///< kappa_2(A)
    ConditionEstimate block_A; ///< kappa_2(A11), FEM-Block
    ConditionEstimate block_K; ///< kappa_2(A22), BEM-Block  <- die Kernzahl
    SchurDiagnostics  schur;   ///< S   = A22 - A21 A11^-1 A12  (Elimination ueber A11)
    SchurDiagnostics  schur_A; ///< S_A = A11 - A12 A22^-1 A21  (Elimination ueber A22)
    std::vector<double> smallest_sv_A11;  ///< unteres Spektralende des FEM-Blocks
    InnerCostEstimate   inner_cost;       ///< Kosten der inneren Loesungen
    std::vector<BlockPcVariant> variants;

    /// Wieviel die EXAKTE Inverse des BEM-Blocks gegenueber der Identitaet an
    /// GMRES-Iterationen einspart, als Faktor. <= 1.0 heisst: sie bringt
    /// nichts, und damit kann auch keine Naeherung an sie etwas bringen.
    /// -1.0, wenn eine der beiden Varianten nicht konvergiert ist.
    double bem_block_speedup = -1.0;
};

/// A_colmajor: die (n x n)-Matrix des gekoppelten Systems, column-major, so
///             wie --direct sie spaltenweise abtastet (also im Tilde-Raum,
///             falls aequilibriert wird -- das ist genau der Raum, in dem der
///             Loeser rechnet, und damit der richtige).
/// n_A:        Zeilen des FEM-Blocks. n - n_A ist der BEM-Block.
/// b:          rechte Seite, Laenge n.
///
/// A und b werden nicht veraendert.
BlockPcStudyResult study_block_preconditioners(const std::vector<double>& A_colmajor,
                                               int n, int n_A,
                                               const std::vector<double>& b,
                                               const BlockPcStudyOptions& opt,
                                               std::ostream& out);

/// Vergleicht zwei Genauigkeitsreihen Stufe fuer Stufe und gibt zurueck,
/// wieviele Stufen die ZWEITE gewinnt (positiv) bzw. die erste (negativ).
///
/// Ausgelagert, weil hier schon einmal ein Fehler sass: die erste Fassung
/// verglich nur die groebste noch konvergierende Stufe. Landen beide Reihen
/// auf derselben Stufe, fiel der Vergleich unentschieden aus -- auch dann,
/// wenn die eine dort 1181 und die andere 29 Iterationen brauchte. Eine
/// einzelne Schwelle ist als Entscheidungsgrundlage zu grob, weil sie an der
/// willkuerlichen Iterationsobergrenze haengt.
///
/// Als eigene Funktion ist die Regel pruefbar, und tests/... prueft genau den
/// Gleichstand-Fall, der den Fehler verdeckt hatte.
int compare_accuracy_series(const std::vector<BlockPcVariant>& variants,
                            const std::string& prefix_a,
                            const std::string& prefix_b);




// =============================================================================
// Harmonischer Ansatz statt Transiente
// =============================================================================
//
// TEAM 28 ist VOLLSTAENDIG LINEAR: mu_r = 1 in allen drei Koerpern, sigma
// konstant, keine Saettigung. Die Benchmarkgroesse ist die Schwebehoehe, also
// die zeitgemittelte Kraft im eingeschwungenen Zustand bei ruhender Platte --
// kein Geschwindigkeitsterm, keine Anfahrdynamik.
//
// Ein harmonischer Ansatz ist deshalb keine Naeherung an die Transiente,
// sondern derselbe eingeschwungene Zustand, direkt gerechnet. Statt 1000
// Zeitschritten braucht es eine Loesung je Kandidatenhoehe und eine skalare
// Nullstellensuche fuer F = mg, also rund fuenf bis zehn.
//
// AUFSPALTUNG OHNE NEUE ASSEMBLIERUNG
//
// Der transiente Operator ist A(dt) = K + C/dt, und zwar blockweise: C traegt
// nur im FEM-Block (sigma-Massenmatrix), der BEM-Block und die Kopplung sind
// dt-unabhaengig. Zwei Abtastungen mit verschiedenem dt genuegen deshalb, um K
// und C exakt zu trennen:
//
//     C = (A1 - A2) / (1/dt1 - 1/dt2)
//     K =  A1 - C/dt1
//
// Voraussetzung: BEIDE Abtastungen im SELBEN Raum. Laeuft zwischen ihnen
// setup_equilibration() erneut, aendert sich D, und die Differenz zweier
// Matrizen aus verschiedenen Tilde-Raeumen ist bedeutungslos -- sie sieht
// trotzdem aus wie eine Matrix. Bleibt D gleich, gilt die Aufspaltung direkt
// im Tilde-Raum, denn D(K + C/dt)D = DKD + (DCD)/dt.
//
// DREI ABTASTUNGEN, NICHT ZWEI
//
// Zwei Matrizen und zwei Unbekannte je Eintrag (K und C) -- dieses System ist
// IMMER exakt loesbar. Ein Rueckbau von A1 und A2 aus dem eigenen Ergebnis
// kann deshalb gar nicht fehlschlagen, gleich welche zwei Matrizen man
// hineingibt. Eine solche Pruefung ist eine Tautologie und faengt insbesondere
// den Fall NICHT, dass die beiden Abtastungen aus verschiedenen Tilde-Raeumen
// stammen. (Die erste Fassung dieser Datei hatte genau diese wirkungslose
// Pruefung; der Selbsttest hat sie aufgedeckt.)
//
// Gebraucht wird deshalb eine DRITTE Abtastung bei dt3, die NICHT in die
// Aufspaltung eingeht und nur zur Kontrolle dient: laesst sich A3 aus K und C
// vorhersagen, stimmt das Modell A(dt) = K + C/dt und beide Raeume passen
// zusammen. Laesst es sich nicht, bricht die Studie ab, statt eine plausibel
// aussehende Zahl auszugeben.
//
// WAHL DER ZEITSCHRITTE
//
// K = A1 - C/dt1 ist eine Differenz zweier grosser Zahlen: im echten Problem
// ist sigma/dt = 3,4e+11 gegen nu = 8e+05, also Faktor 4,3e+05, und es gehen
// rund fuenf bis sechs Stellen durch Ausloeschung verloren.
//
// Abhilfe kostet nichts: dt2 GROSS waehlen (etwa 1 s). Dann ist C/dt2
// verschwindend und A2 praktisch schon K selbst -- die Ausloeschung
// verschwindet. split_conditioning meldet, wie gut die Wahl war.

struct KCSplit {
    std::vector<double> K;              ///< dt-unabhaengiger Anteil
    std::vector<double> C;              ///< sigma-Massenmatrix (nur FEM-Block)
    double rel_error_prediction = -1.0; ///< rel. Fehler der VORHERSAGE von A3
    double c_outside_fem_block  = -1.0; ///< ||C|| ausserhalb des FEM-Blocks, relativ
    double split_conditioning   = -1.0; ///< ||C/dt1|| / ||K||: Stellenverlust
    bool   valid                = false;
};

/// A1/dt1 und A2/dt2 bilden die Aufspaltung, A3/dt3 dient AUSSCHLIESSLICH der
/// Kontrolle. n_A ist die Groesse des FEM-Blocks: C MUSS ausserhalb davon
/// verschwinden, sonst ist die Annahme "nur der FEM-Block haengt an dt" falsch.
KCSplit split_K_and_C(const std::vector<double>& A1, double dt1,
                      const std::vector<double>& A2, double dt2,
                      const std::vector<double>& A3, double dt3,
                      int n, int n_A);

/// Baut das harmonische System in REELLER Blockform, Groesse 2n x 2n:
///
///     [  K    -omega*C ] [x_r]   [b_r]
///     [ omega*C    K   ] [x_i] = [b_i]
///
/// Reell und nicht komplex, weil damit die gesamte vorhandene Maschinerie
/// unveraendert weiterverwendbar bleibt -- kein komplexer Loeser, keine
/// komplexen Kernel, keine komplexe ACA.
std::vector<double> build_harmonic_system(const std::vector<double>& K,
                                          const std::vector<double>& C,
                                          double omega, int n);

struct HarmonicStudyResult {
    bool   valid            = false;
    double omega            = 0.0;
    ConditionEstimate cond;              ///< kappa_2 des harmonischen Systems
    int    iters_plain      = -1;        ///< GMRES ohne Vorkonditionierer
    double relres_plain     = -1.0;
    int    iters_ctor       = -1;        ///< GMRES mit C-to-R
    double relres_ctor      = -1.0;
    double kappa_transient  = -1.0;      ///< zum Vergleich
    int    iters_transient  = -1;

    // Genauigkeitsreihe fuer den INNEREN Loeser (W+T).
    //
    // Die 38 Iterationen von C-to-R gelten mit EXAKT geloestem (W+T) -- im
    // Produktivcode steht dort AMS oder ILU. Das ist dieselbe Frage, an der
    // der DtN-Zweig gescheitert ist. Der Unterschied: (W+T) = K + omega*C ist
    // ein REELLER transienter Operator mit dt = 1/omega, kein indefinites
    // Schur-Komplement mit kappa = 1e+08. Ob das reicht, sagt aber nicht die
    // Plausibilitaet, sondern diese Reihe.
    std::vector<double> wt_eps;          ///< geprueefte relative Fehler
    std::vector<int>    wt_iters;        ///< aeussere Iterationen dazu, -1 = keine Konvergenz
    double kappa_WpT        = -1.0;      ///< kappa_2(W+T)
    int    inner_iters_WpT  = -1;        ///< unvorkonditionierte Krylov-Schritte fuer (W+T)
    double best_eps         = -1.0;      ///< Toleranz, die die ARBEIT minimiert
    int    best_outer       = -1;        ///< aeussere Iterationen dazu
    int    reuse_per_setup  = -1;        ///< wie oft dasselbe (W+T) benutzt wird
    double work_harmonic    = -1.0;      ///< geschaetzte Gesamtarbeit, Matvecs
    double work_transient   = -1.0;
};

/// Vergleicht das harmonische System gegen das transiente bei dt_ref.
///
/// Der C-to-R-Vorkonditionierer (Axelsson/Kucherov) fuer
/// A = [[W, -T],[T, W]] lautet
///
///     P = [[W+T, 0], [T-W, W+T]]
///
/// und braucht ZWEI Loesungen mit (W+T) = K + omega*C. Das ist genau ein
/// transienter Operator mit dt = 1/omega -- also dieselbe Bauart, auf die AMS
/// und ILU im Projekt ohnehin schon zielen. Traegt C-to-R, ist der Umbau
/// billig: kein neuer Vorkonditionierer, nur ein anderer Aufruf des
/// vorhandenen.
HarmonicStudyResult study_harmonic_system(const std::vector<double>& A_transient,
                                          const KCSplit& split,
                                          double omega, double dt_ref,
                                          int n, int n_A,
                                          const BlockPcStudyOptions& opt,
                                          std::ostream& out);

/// Deutet das Ergebnis in Klartext -- ausdruecklich einschliesslich des
/// Falls, in dem der Befund gegen den eigenen Vorschlag spricht.
void report_block_pc_study(const BlockPcStudyResult& res, std::ostream& out);

} // namespace bf

#endif // BLOCK_PRECONDITIONER_STUDY_H
