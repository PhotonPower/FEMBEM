#ifndef FEMBEM_BACKEND_CUH
#define FEMBEM_BACKEND_CUH

// =============================================================================
// Backend.cuh -- Umschaltung zwischen GPU- und reinem CPU-Betrieb
// =============================================================================
//
// Ziel: das komplette Programm laesst sich zur Laufzeit ueber eine
// Kommandozeilenoption vollstaendig auf der CPU rechnen lassen, ohne dass
// eine GPU vorhanden sein muss:
//
//     ./fembem --cpu
//     ./fembem 3 --cpu --threads=8
//
// Warum ueberhaupt eine eigene Schicht und nicht einfach ein zweites
// Programm? Weil in diesem Loeser Physik und Parallelisierung verwoben sind:
// derselbe Integrand (Nedelec-Basis, Sauter-Schwab, BEM-Kerne) wird im
// GPU-Pfad in einem __global__-Kernel und im CPU-Pfad in einer Schleife
// gebraucht. Zwei getrennte Implementierungen desselben Integranden waeren
// die klassische Quelle fuer stille Divergenzen zwischen den Pfaden. Deshalb:
//
//   * Jeder Kernel besteht aus einem __host__ __device__ Rumpf
//     "<name>_body(i, args...)" -- EINE Quelle fuer die Physik -- und einem
//     duennen __global__ void <name>(args...), der nur den Index bildet.
//   * Die Aufrufstelle benutzt BF_LAUNCH_1D / BF_LAUNCH_2D. Das Makro
//     entscheidet zur Laufzeit, ob der Kernel gestartet oder der Rumpf in
//     einer Host-Schleife ausgefuehrt wird.
//
// Damit ist ausgeschlossen, dass GPU- und CPU-Pfad numerisch auseinander
// laufen: sie fuehren buchstaeblich denselben Code aus.
//
// Was der CPU-Modus NICHT tut: er entfernt die Link-Abhaengigkeit zu
// libcudart/libcublas/libcusparse nicht. Gebaut wird weiterhin mit nvcc. Es
// wird im CPU-Modus aber kein einziger CUDA-API-Aufruf mehr abgesetzt (keine
// Kontexterzeugung, keine Allokation, keine Kernelstarts), das Programm
// laeuft also auf einer Maschine ohne GPU bzw. ohne funktionsfaehigen
// Treiber.
// =============================================================================

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

// -----------------------------------------------------------------------------
// Qualifier-Kurzform fuer Code, der auf beiden Pfaden laufen muss
// -----------------------------------------------------------------------------
#define BF_HD __host__ __device__

// -----------------------------------------------------------------------------
// Fehlerpruefung. Bewusst ein eigenes Makro und nicht CUDA_CHECK_ERROR aus
// DeviceArray.cuh: dieser Header wird VON DeviceArray.cuh eingebunden, das
// Makro dort ist hier also noch nicht sichtbar.
// -----------------------------------------------------------------------------
#define BF_CUDA_CHECK(call)                                                     \
    do {                                                                        \
        cudaError_t bf_status_ = (call);                                        \
        if (bf_status_ != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("CUDA Error: ") +              \
                                     cudaGetErrorString(bf_status_) +           \
                                     " at " + __FILE__ + ":" +                  \
                                     std::to_string(__LINE__));                 \
        }                                                                       \
    } while (0)

namespace bf {

// -----------------------------------------------------------------------------
// Der globale Schalter
// -----------------------------------------------------------------------------
enum class Backend { GPU, CPU };

/// Definiert in src/Backend.cu. Standard ist GPU (bisheriges Verhalten).
extern Backend g_backend;

/// Anzahl der im CPU-Modus genutzten OpenMP-Threads (0 = Standard der
/// Laufzeitumgebung). Definiert in src/Backend.cu.
extern int g_cpu_threads;

/// true, wenn --reference angegeben wurde. Dann laeuft statt des
/// produktiven Nedelec-/ACA-Pfads der Baseline-Loeser aus
/// include/ReferenceSolver.cuh (Q1-Knotenelemente, Strafterm-Eichung,
/// vollbesetzte BEM). Beide Pfade bleiben vollstaendig erhalten.
extern bool g_use_reference_solver;

/// Anzahl uniformer Netzverfeinerungen (--refine=N), je Stufe 1 -> 8.
/// 0 = Originalnetz.
extern int g_refine_levels;

/// --verify-field. Rechnet EINEN magnetostatischen Schritt mit sigma = 0
/// (keine Wirbelstroeme) beim Strommaximum und vergleicht B_z auf der Achse mit
/// der analytischen Biot-Savart-Loesung. Prueft Quellterm, Assemblierung und
/// Aussenraumkopplung, ohne dass Wirbelstromphysik etwas verschleiern kann.
extern bool g_verify_field;

/// --no-conductor-penalty. Setzt den div-div-Strafterm in LEITENDEN
/// Elementen auf null. Diagnostisches Experiment: der Strafterm erzwingt dort
/// div A = 0, waehrend J = -sigma dA/dt gleichzeitig V = 0 annimmt. Beide
/// Eichungen gleichzeitig gelten nur bei exakt azimutaler Loesung. Aendert sich
/// die Wirbelstromstaerke stark, war der Strafterm die Ursache.
extern bool g_no_conductor_penalty;

/// --fixed-plate. Platte nicht bewegen. Voraussetzung fuer die saubere
/// Q-Messung: bei freier Bewegung legt die Platte in EINER Kraftperiode rund
/// 10 mm zurueck, in einem Feld, das darueber stark abfaellt. Ein
/// "Zyklusmittel" ist dann kein stationaerer Mittelwert, und aus der
/// Kraftschwankung laesst sich Q nicht ablesen -- die Geometrieaenderung
/// wuerde als Wirbelstromschwaeche fehlgedeutet.
extern bool g_fixed_plate;

/// --plate-z=<mm>. Platte auf diese Schwebehoehe setzen (Unterkante ueber
/// Spulenoberkante) statt auf die Netzlage von 3.8 mm. Negativ = unveraendert.
/// Zusammen mit --fixed-plate der strenge Test: bei der gemessenen Ruhelage
/// von 11.3 mm MUSS das Zyklusmittel der Vertikalkraft exakt m*g = 1.0497 N
/// sein, sonst schwebt die Platte dort nicht.
extern double g_plate_z_mm;

/// --nogauge. Baum-Kobaum-Eichung NICHT anwenden. Sinnvoll nur zusammen
/// mit --ams: ohne Eichung ist das curl-curl-System singulaer, aber konsistent
/// (nachgemessen: G^T b/||b|| = 2.1e-4), und ein AMS-Vorkonditionierer kommt
/// damit zurecht, ILU(0) nicht. Die Kohomologie-Projektoren bleiben in beiden
/// Faellen aktiv -- AMS deckt Gradienten ab, nicht die harmonischen Felder der
/// drei Volltori.
extern bool g_no_gauge;

/// --ams. Auxiliary-Space Maxwell Solver statt ILU(0) als
/// Vorkonditionierer des FEM-Blocks.
extern bool g_use_ams;

/// --nocohomology. Die Kohomologie-Projektoren NICHT anwenden.
///
/// Anlass ist eine Messung: ||A c||/||c|| betraegt fuer die beiden Generatoren
/// 4.9e-02 bzw. 7.2e-02 des Werts, den ein Zufallsvektor liefert. Fuer einen
/// echten Nullraumvektor waeren rund 1e-12 zu erwarten -- die Generatoren
/// liegen also NICHT im Nullraum von A. Die Projektion entfernt trotzdem 58 %
/// der rechten Seite entlang dieser Richtungen und wirft damit Loesung weg.
///
/// Physikalisch passt das zusammen: bei einer STROMgetriebenen Spule ist der
/// verkettete Fluss ein Ergebnis, keine Zwangsbedingung. Ihn auf null zu
/// projizieren ueberbestimmt das Problem. Bei Spannungstreibung waere der
/// Generator dagegen eine zusaetzliche Unbekannte mit der Spannung als
/// Gleichung -- dort gehoerte er ins System, nicht in einen Projektor.
extern bool g_no_cohomology;

/// --blocktri. Blocktriangulaerer Vorkonditionierer fuer das gekoppelte
/// 2x2-System statt des blockdiagonalen diag(A~, I).
///
/// Anlass ist eine Messung: die Kopplungsbloecke sind zu 81 % unsymmetrisch,
/// und zwar bauartbedingt -- oben rechts steht -T, unten links -0.5*T - D_BEM,
/// die koennen per Definition nicht zueinander transponiert sein. Ein
/// blockdiagonaler Vorkonditionierer ignoriert sie vollstaendig und entkoppelt
/// damit Leiter und Aussenraum; die Stetigkeit der Tangentialkomponente muss
/// das Krylov-Verfahren dann allein ueber Matrix-Vektor-Produkte aufbauen.
/// Der blocktriangulaere Ansatz nimmt den Kopplungsblock T mit auf und
/// approximiert das Schur-Komplement durch den Single-Layer-Operator,
/// S~ = mu_0*G_BEM (spektral aequivalent, siehe Hiptmair, Steinbach, Heuer).
extern bool g_use_block_tri_pc;

/// --pc-study. Deckenmessung fuer den BEM-Block. Setzt --direct voraus
/// (die dicht abgetastete Matrix wird wiederverwendet, es entsteht kein
/// zweiter Abtastlauf).
///
/// Beantwortet die Frage, die docs/reports/convergence-findings.tex offen laesst: der BEM-Block
/// geht mit der Identitaet in den Loeser, und die blockweise Residuums-
/// aufschluesselung zeigt ihn als den groesseren Anteil. Daraus folgt aber
/// NICHT, dass er schlecht konditioniert ist -- das eine sagt, wo das
/// Residuum sitzt, das andere, welcher Block die Kondition verursacht.
///
/// Gemessen wird deshalb die DECKE: was braechte die EXAKTE Inverse von
/// A22 = mu_0*G_BEM? Calderon-Vorkonditionierung und H-Matrix-Naeherungs-
/// inverse sind beide nur Naeherungen daran und koennen nicht mehr erreichen.
/// Bringt die exakte Inverse nichts, ist der Ansatz widerlegt -- bevor eine
/// Zeile davon geschrieben wird. Siehe include/BlockPreconditionerStudy.h.
extern bool g_pc_study;

/// --harmonic-study[=HZ]. Setzt --direct voraus.
///
/// TEAM 28 ist vollstaendig linear (mu_r = 1 ueberall, sigma konstant), und
/// gesucht ist die Schwebehoehe, also die zeitgemittelte Kraft im
/// eingeschwungenen Zustand bei ruhender Platte. Ein harmonischer Ansatz ist
/// deshalb keine Naeherung an die Transiente, sondern derselbe Zustand --
/// rund zehn Loesungen fuer die Nullstellensuche F = mg statt 1000
/// Zeitschritten.
///
/// Gemessen wird, ob das harmonische System besser konditioniert ist als das
/// transiente und ob der C-to-R-Vorkonditionierer greift. Siehe
/// include/BlockPreconditionerStudy.h.
/// --harmonic. Loest den eingeschwungenen Zustand statt 1000 Zeitschritte
/// zu rechnen, und gibt die zeitgemittelte Kraft aus.
///
/// Anders als --harmonic-study ist das kein Diagnoselauf, sondern der
/// Produktivweg: TEAM 28 ist linear, gesucht ist die Schwebehoehe aus der
/// zeitgemittelten Kraft bei ruhender Platte.
extern bool   g_harmonic_solve;
extern bool   g_fem_direct;   // --fem-direct (reines FEM + Direktloeser)

/// --fem-direct-iterative. Loest die 2n-Blockform mit GMRES(m) statt mit
/// cuDSS.
///
/// Zweck ist der VERGLEICH, nicht der Ersatz. Damit er etwas wert ist,
/// bekommt GMRES bitgenau dieselbe Matrix und dieselbe rechte Seite wie der
/// Direktloeser -- dieselben CSR-Arrays nach der Dirichlet-Behandlung. Die
/// einzige Variable ist der Loeser.
///
/// Nicht ueber solve_harmonic(), obwohl das die Blockform matrixfrei
/// behandeln koennte: dort saesse die Dirichlet-Bedingung an anderer Stelle,
/// und GMRES loeste ein anderes System als cuDSS. Ein Vergleich zweier
/// Loeser auf zwei verschiedenen Systemen sagt nichts.
///
/// Erwartung offen. Der Operator ist ungeeicht -- --fem-direct regularisiert
/// stattdessen ueber sigma_Luft --, und HarmonicSolver.cuh haelt fest, dass
/// GMRES auf den frueheren Problemen bei 4000 Iterationen stagnierte. Ob die
/// Regularisierung genuegt, wo die Eichung fehlt, ist genau die Frage.
extern bool   g_fem_direct_iterative;

/// --fem-direct-no-project. Schaltet die Gradientenprojektion des Quellterms
/// AB. Standard ist AN.
///
/// An, weil sie eine Anforderung erfuellt: der punktweise vorgegebene
/// Quellterm ist diskret nicht divergenzfrei (gemessen 6,1e-03 gegen die 1e-4
/// aus M2), weil die Spule elementweise zugeordnet wird und ihre gezackte
/// Ersatzflaeche J_s . n = 0 verletzt. Durch Netzverfeinerung ist das NICHT
/// behebbar -- siehe test_patch_assembly, Pruefung 4 gegen 5.
///
/// Abschaltbar, weil alle bisherigen Messreihen im Plan OHNE sie entstanden
/// sind. Ohne Schalter waere kein Vergleich mit ihnen mehr moeglich, und die
/// dreiteilige Vorhersage in docs/design/fem-direct-plan.md nicht pruefbar.
extern bool   g_fem_direct_no_project;

/// --fem-direct-gauge. Beseitigt den Curl-Curl-Nullraum durch die
/// BAUM-KOBAUM-EICHUNG statt durch die Regularisierung sigma_Luft. Die
/// Regularisierung wird dann NICHT angewandt.
///
/// Der Gedanke: die Regularisierung bezahlt die Beseitigung des Nullraums mit
/// einer Kondition, in der die geeichten Richtungen um 1/eps unter dem Rest
/// liegen -- gemessen max|omega*C|/max|K| = 2,85e-03 auf dem feinen Netz. Der
/// geeichte Curl-Curl auf den Kobaumkanten hat dagegen eine Kondition wie ein
/// Laplace-Operator. Genau das behauptet die [Gauge]-Warnung ("ILU(0)
/// konvergiert auf dem ungeeichten System nicht") seit Anfang an.
///
/// Anwendbar ist die Eichung hier, weil mit mitvernetzter Luft das
/// NICHTLEITENDE Gebiet Luft plus Spulen umfasst, also Kasten minus Platte.
/// Im BEM-Fall war es die Spule selbst, ein Volltorus mit einer nicht
/// zusammenziehbaren Schleife -- daher dort ein Kohomologie-Generator je
/// Spule. Ob das hier wirklich einfach zusammenhaengend ist, haengt daran, ob
/// das Plattennetz ein Loch auf der Achse hat; die Generatorensuche laeuft mit
/// diesem Schalter und beantwortet es als MESSUNG statt als Argument.
///
/// Wichtig: die Eichung ersetzt die Quellterm-Projektion NICHT. Sie beseitigt
/// den Nullraum des OPERATORS; die Projektion macht den QUELLTERM zu einem
/// geschlossenen Strom. Mit Eichung wird der Eintrag von b auf einer
/// Baumkante stillschweigend verworfen -- eine schiefe Projektion, nicht die
/// orthogonale auf ker(G^T). Beide bleiben unabhaengig voneinander aktiv.
///
/// Der Text oben beschreibt den Modus Gauge des Schalters darunter. Er stand
/// frueher an einem eigenen bool g_fem_direct_gauge; der ist im Tri-State
/// aufgegangen, die Begruendung bleibt.

// -----------------------------------------------------------------------------
// --fem-direct-nullspace=eps|gauge|ams
// -----------------------------------------------------------------------------
// Der Curl-Curl-Operator ist auf den Gradientenfeldern der nichtleitenden
// Gebiete singulaer -- Luft UND die stromgetriebenen Spulen, die kappa = 0
// tragen. Es gibt drei Wege, damit umzugehen, und sie schliessen einander aus:
//
//   Regularize  sigma_Luft = eps * sigma_Leiter (--fem-direct-eps). Macht die
//               Matrix regulaer und damit direkt zerlegbar. Der Preis steht in
//               doc/harmonische-fem.tex: die Regularisierung weist dem
//               Gradientenanteil einen um 1/eps VERSTAERKTEN Wert zu, sie
//               unterdrueckt ihn nicht.
//
//   Gauge       Baum-Kobaum-Eichung (frueher --fem-direct-gauge). Versucht und
//               verworfen: Residuum 4,9e-14, aber B in der Platte weicht um
//               2,4 % ab und F_z liegt bei 1,25 statt 1,02 N. B ist
//               eichinvariant -- das ist also ein anderes Feld und nicht ein
//               anderer Eichrepraesentant. Der Schalter bleibt, damit die
//               Messung wiederholbar ist.
//
//   Ams         Gar nichts tun und das singulaere, aber KONSISTENTE System
//               loesen. Das ist der Normalfall bei Hiptmair-Xu und in hypre der
//               singulaere Modus (HYPRE_AMSSetBetaPoissonMatrix(NULL)): der
//               skalare Hilfsraum wird ausserhalb des Leiters stillgelegt --
//               G^T A G ist dort bis auf Rundung null --, und die
//               Gradientenmoden traegt der VEKTORIELLE Hilfsraum ueber den
//               Alpha-Poisson-Operator mit nu > 0 ueberall.
//
//               Voraussetzung ist G^T b = 0. Die Quellterm-Projektion liefert
//               das (gemessen 6e-16); --fem-direct-no-project ist mit diesem
//               Modus deshalb unvertraeglich.
//
//               SCHLIESST DEN DIREKTLOESER AUS. Ohne Regularisierung und ohne
//               Eichung ist die 2n-Blockform echt singulaer; cuDSS pivotiert
//               darauf ins Nichts, und CUDSS_DATA_INFO meldet das nicht -- auf
//               diesem Pfad ist eine singulaere Matrix schon einmal mit
//               info = 0 zerlegt worden. --fem-direct-ams erzwingt deshalb den
//               iterativen Weg.
//
// Die drei sind bewusst EIN Schalter und nicht drei Flaggen. Eichung und
// Regularisierung zugleich waeren sinnlos -- die Eichung waehlt ihren Baum
// ueber kappa == 0, und nach der Regularisierung hat kein Element mehr
// kappa == 0, der Baum bliebe leer. Als drei unabhaengige bool laesst sich das
// hinschreiben, ohne dass es auffaellt.
enum class FemDirectNullspace { Regularize, Gauge, Ams };
extern FemDirectNullspace g_fem_direct_nullspace;

/// --harmonic-inner-tol=X und --harmonic-outer-cap=N.
///
/// Fuer den Linearitaetstest. Das innere GMRES stoppt bei RELATIVER Toleranz;
/// die Abbildung "rechte Seite -> Naeherungsloesung" haengt damit nichtlinear
/// von der Eingabe ab -- andere Eingabe, andere Iterationszahl, anderes
/// Polynom. Ein Nullstartwert allein macht den Vorkonditionierer NICHT zu
/// einem festen linearen Operator; das steht so falsch in einem frueheren
/// Kommentar in HarmonicSolver.cu.
///
/// Mit inner_tol = 1e-10 ist P praktisch exakt und damit praktisch linear.
/// Faellt die aeussere Kurve dann sauber (die Deckenmessung sagt 38
/// Iterationen bei exaktem (W+T)), ist die Nichtlinearitaet die Ursache des
/// steigenden Residuums. Steigt sie auch dann, liegt der Fehler im aeusseren
/// Loeser selbst.
/// --harmonic-checks. Schaltet die Diagnosen des harmonischen Loesers ein.
///
/// Drei Pruefungen sind nacheinander entstanden, jede um eine konkrete Frage
/// zu klaeren, und jede hat ihre Frage beantwortet:
///
///   Vorabmessung     -- traegt ILU auf (W+T)?        (ja, marginal: 2502
///                       Schritte gegen 2590 ohne)
///   Eichungs-Probe   -- braucht der innere Loeser
///                       projizierte Eingaben?         (ja, belegt: 8,3 gegen
///                       8,35e-05 am selben Vektor)
///   Linearitaet      -- ist P ein linearer Operator?  (offen)
///
/// Zusammen kosten sie rund 40.000 innere Schritte, BEVOR die erste aeussere
/// Iteration beginnt -- mehr als der eigentliche Loeser fuer mehrere aeussere
/// Iterationen braucht. Standardmaessig laufen sie deshalb nicht mehr mit.
///
/// Die Vorabmessung schuetzt allerdings davor, stundenlang auf einem
/// untauglichen inneren Loeser zu rechnen. Ohne sie faengt die Warnung in
/// CtoRPreconditioner den Fall ab -- sie meldet einmal und laesst weiterlaufen.
extern bool   g_harmonic_checks;

/// --harmonic-outer-restart=N (Standard 64).
///
/// Der Vergleich "echtes gegen geschaetztes Residuum" wird erst beim RESTART
/// sichtbar -- vorher gibt es nur die Schaetzung. Bei Restart 64 und rund 5400
/// inneren Schritten je aeusserer Iteration heisst das 350000 Schritte, bevor
/// die entscheidende Zahl erscheint. Mit N = 5 sind es 27000.
///
/// Fuer den Produktivlauf bleibt 64 richtig: der aeussere Loeser braucht laut
/// Deckenmessung 38 bis 41 Iterationen, ein kurzer Restart wuerfe die
/// Krylov-Information mitten darin weg.
extern int    g_harmonic_outer_restart;

extern double g_harmonic_inner_tol;
extern int    g_harmonic_outer_cap;

extern bool   g_harmonic_study;
extern double g_harmonic_hz;

/// --fem-direct-eps=X. Der Faktor in sigma_Luft = X * sigma_Leiter, mit dem
/// --fem-direct den Curl-Curl-Nullraum beseitigt (Abschnitt 1 des Plans).
/// Einstellbar, weil die Abnahme ausdruecklich verlangt, dass das Ergebnis
/// zwischen 1e-6 und 1e-10 stehenbleibt -- und weil der Scheingradient im
/// Imaginaerteil mit 1/X skaliert, ist das der einzige Weg, diese Skalierung
/// zu messen statt sie zu behaupten.
extern double g_fem_direct_eps;

/// --pc-inner-iters=N / --pc-inner-tol=X. Steuern das innere CG auf
/// S~ = mu_0*G_BEM. Ein Vorkonditionierer muss nur billig annaehern; die
/// Vorgaben (10 Iterationen, 1e-2) stammen aus einer Messung am Modellproblem.
/// --gmres-restart=N. Restart-Laenge des aeusseren GMRES.
///
/// Anlass ist eine Beobachtung, die sich durch alle Laeufe zieht: der
/// blocktriangulaere Vorkonditionierer senkt das Residuum in den ersten ~40
/// Iterationen deutlich schneller als der blockdiagonale, danach steht es
/// INNERHALB der Zyklen vollkommen still. Das ist die Signatur einer
/// restartbedingten Stagnation -- ein Vorkonditionierer, der die meisten
/// Eigenwerte gut buendelt, aber wenige Ausreisser stehen laesst, braucht
/// laengere Zyklen, weil GMRES(50) die dafuer noetigen Suchrichtungen bei
/// jedem Restart wieder wegwirft.
extern int    g_gmres_restart;

/// --ritz. Ritz-Spektrum nach dem ersten vollen Arnoldi-Zyklus ausgeben.
///
/// Der Bericht war schon implementiert (GMRESSolver::enable_ritz_report), aber
/// nur ueber --verify-field erreichbar und damit an eine ganz andere Pruefung
/// gekoppelt. Er liefert eine untere Schranke fuer die Konditionszahl des
/// VORKONDITIONIERTEN Operators -- also die Antwort auf die Frage, ob die
/// zugestandene Iterationszahl ueberhaupt reichen kann, statt sie zu erraten.
extern bool   g_want_ritz;

/// --max-iter=N. Iterationsobergrenze des aeusseren GMRES (Standard 1000).
/// War fest eincodiert. Gebraucht, um "langsam" von "steht" zu unterscheiden --
/// ein Residuum, das bei 1000 noch faellt, sagt nichts darueber, ob es bei
/// 5000 ankommt oder auf einem Sockel liegt.
extern int    g_max_iter;

/// --mesh=DATEI. Netzdatei statt des fest eincodierten "team28.vtk".
///
/// Gebraucht fuer die Netzstudie: der Loeser stagniert am Originalnetz bei
/// einem relativen Residuum um 0.015. Ob das an der Netzgroesse liegt oder an
/// der Formulierung, entscheidet nur ein Vergleich ueber mehrere Netze --
/// faellt das Residuum auf groben Netzen sauber auf 1e-6, ist es die
/// Konditionierung; stagniert es auch dort, sitzt das Problem nicht im Loeser.
/// tools/make_coarse_team28.py erzeugt die passenden Netze.
extern std::string g_mesh_file;

/// --direct. Das gekoppelte System dicht zerlegen und das Residuum der
/// EXAKTEN Loesung melden. Beantwortet, ob das System ueberhaupt loesbar ist --
/// die Frage, die kein Vorkonditionierer beantworten kann. Nur fuer kleine
/// Netze (Speicher waechst quadratisch, Aufwand kubisch).
extern bool   g_direct_solve;

/// --dt=X. Zeitschrittweite in Sekunden (Standard 1e-4).
///
/// Die vierte fest eincodierte Zahl in dieser Arbeit nach Restart-Laenge,
/// Iterationsobergrenze und Netzdatei -- und wie jene stand sie einer Messung
/// im Weg.
///
/// Gemessen am Netz n=12; sigma_max bleibt durchweg bei 155, nur sigma_min
/// bewegt sich:
///
///   dt=1e-6  Kontrast 4.3e+07  sigma_min=7.56e-05  kappa_2=2.05e+06
///   dt=1e-5  Kontrast 4.3e+06  sigma_min=1.50e-04  kappa_2=1.03e+06   <- Minimum
///   dt=1e-4  Kontrast 4.3e+05  sigma_min=1.45e-04  kappa_2=1.07e+06   (Vorgabe)
///   dt=1e-3  Kontrast 4.3e+04  sigma_min=5.89e-05  kappa_2=2.63e+06
///   dt=1e-2  Kontrast 4.3e+03  sigma_min=7.25e-06  kappa_2=2.14e+07
///
/// Das ist eine U-KURVE, kein Trend -- und damit sind BEIDE naheliegenden
/// Erklaerungen richtig, jede in einem Ast:
///
///   * Grosses dt: der Massenterm sigma/dt wird klein, der Operator naehert
///     sich dem reinen Curl-Curl, und der ist singulaer. Der Massenterm ist
///     hier also das Einzige, was die Konditionierung begrenzt.
///   * Kleines dt: der Massenterm wird gross, und der KONTRAST zwischen Leiter
///     (sigma/dt riesig) und Luft (null) begrenzt.
///
/// Dazwischen liegt ein Minimum bei kappa ~ 1e+06, und die Vorgabe dt = 1e-4
/// liegt praktisch darauf. Am Zeitschritt ist also nichts zu gewinnen: kappa
/// ~ 1e+06 ist ein Sockel, der aus der Struktur des Problems kommt --
/// Curl-Curl-Nullraum plus Leiter/Luft-Uebergang -- und nicht aus der
/// Parametrierung.
///
/// Anders gesagt: nur ein Vorkonditionierer, der BEIDE Regime behandelt, kann
/// hier helfen. Genau dafuer ist AMS gebaut.
extern double g_dt;

/// --amg-theta=X. Schwelle der starken Kopplung im AMG-Staerkegraphen
/// (Standard 0.08). Siehe die Messtabelle an AlgebraicMultigrid::Options.
/// Wer sie anfasst, muss die Grobgittergroessen in der [AMS]-Ausgabe mitlesen:
/// ein traumhafter Konvergenzfaktor bei unveraendertem Grobgitter ist keine
/// Vergroeberung, sondern eine direkte Loesung.
extern double g_amg_theta;

extern int    g_pc_inner_iters;
extern double g_pc_inner_tol;

inline bool use_reference_solver() { return g_use_reference_solver; }
inline int  refine_levels()         { return g_refine_levels; }
inline bool verify_field()          { return g_verify_field; }
inline bool no_conductor_penalty()  { return g_no_conductor_penalty; }
inline bool   fixed_plate()         { return g_fixed_plate; }
inline double plate_z_mm()          { return g_plate_z_mm; }
// --fem-direct zaehlt hier mit: dieser Pfad entfernt den Curl-Curl-Nullraum
// durch eine kleine Leitfaehigkeit in der Luft, nicht durch Eichung. Die
// Antwort an EINER Stelle zu geben ist sicherer, als die Abfrage an jeder der
// fuenf Stellen zu wiederholen, die Eichung, Kohomologie-Projektor und AMS
// aufsetzen -- eine vergessene waere ein Zugriff auf eine leere Eichmaske.
inline bool   no_gauge()            { return g_no_gauge || g_fem_direct; }
inline bool   use_ams()             { return g_use_ams; }
inline bool   no_cohomology()       { return g_no_cohomology; }
inline bool   use_block_tri_pc()    { return g_use_block_tri_pc; }
inline int    pc_inner_iters()      { return g_pc_inner_iters; }
inline int    gmres_restart()        { return g_gmres_restart; }
inline bool   want_ritz()            { return g_want_ritz; }
inline int    max_iter()              { return g_max_iter; }
inline const std::string& mesh_file() { return g_mesh_file; }
inline bool   use_direct_solve()     { return g_direct_solve; }
inline bool   want_pc_study()        { return g_pc_study; }
inline bool   want_harmonic_study()  { return g_harmonic_study; }
inline bool   want_harmonic_solve()  { return g_harmonic_solve; }
// FEM-direkt: reines FEM mit Luftgebiet und Sparse-Direktloeser (cuDSS).
// Setzt want_harmonic_solve() voraus und schliesst BEM sowie --cpu aus.
inline bool   want_fem_direct()      { return g_fem_direct; }
inline double fem_direct_eps()       { return g_fem_direct_eps; }
inline bool   fem_direct_iterative() { return g_fem_direct_iterative; }
inline bool   fem_direct_project()   { return !g_fem_direct_no_project; }
inline FemDirectNullspace fem_direct_nullspace() { return g_fem_direct_nullspace; }
inline bool   fem_direct_gauge()     { return g_fem_direct_nullspace == FemDirectNullspace::Gauge; }
inline bool   fem_direct_ams()       { return g_fem_direct_nullspace == FemDirectNullspace::Ams; }
/// Wird der Nullraum ANDERS als durch Regularisierung beseitigt? Genau das
/// fragt preflight_fem_direct() ab: nur im regularisierten Fall ist kappa = 0
/// eine Beanstandung, sonst ist es der Normalfall.
inline bool   fem_direct_nullspace_handled() {
    return g_fem_direct_nullspace != FemDirectNullspace::Regularize;
}
inline double harmonic_frequency()   { return g_harmonic_hz; }
inline double harmonic_inner_tol()   { return g_harmonic_inner_tol; }
inline int    harmonic_outer_cap()   { return g_harmonic_outer_cap; }
inline bool   want_harmonic_checks() { return g_harmonic_checks; }
inline int    harmonic_outer_restart() { return g_harmonic_outer_restart; }
inline double time_step()            { return g_dt; }
inline double amg_theta()            { return g_amg_theta; }
inline double pc_inner_tol()        { return g_pc_inner_tol; }

inline bool on_cpu() { return g_backend == Backend::CPU; }
inline bool on_gpu() { return g_backend == Backend::GPU; }

const char* backend_name();

// -----------------------------------------------------------------------------
// Kommandozeile
// -----------------------------------------------------------------------------
//
// Erkannte Optionen:
//   --cpu                 alles auf der CPU rechnen (keine GPU noetig)
//   --gpu                 GPU erzwingen (Standard)
//   --threads=N           OpenMP-Threads im CPU-Modus (nur mit OpenMP-Build)
//   --help / -h           Kurzhilfe ausgeben
//
// Zusaetzlich wird die Umgebungsvariable FEMBEM_BACKEND ausgewertet
// (Werte "cpu" oder "gpu"), damit sich Batch-Laeufe umstellen lassen, ohne
// die Aufrufzeile anzufassen. Eine explizite Option auf der Kommandozeile
// hat Vorrang vor der Umgebungsvariablen.
//
// MUSS aufgerufen werden, bevor irgendein DeviceArray angelegt wird -- also
// als erstes in main(). Danach darf g_backend nicht mehr wechseln, sonst
// wuerden Puffer aus dem einen Modus im anderen freigegeben.
//
// Rueckgabe: false, wenn --help angefordert wurde und das Programm sich
// beenden soll; sonst true.
// -----------------------------------------------------------------------------
bool select_backend_from_args(int argc, char** argv);

/// Kurzhilfe auf stdout.
void print_usage(const char* exe_name);

/// Gibt aus, welches Backend aktiv ist (und im GPU-Fall welche Karte).
void print_backend_banner();

/// true, wenn das Argument eine Option (beginnt mit '-') ist. Die bestehende
/// positionale Auswertung (Schrittzahl, "nobem", "noilu") kann damit die
/// neuen Optionen sauber ueberspringen.
bool is_option(const char* arg);

// -----------------------------------------------------------------------------
// Speicher
// -----------------------------------------------------------------------------
//
// Die _d-Funktionen ersetzen 1:1 die frueheren cudaMalloc/cudaFree/
// cudaMemcpy/cudaMemset/cudaDeviceSynchronize-Aufrufe. Im GPU-Modus rufen sie
// genau diese auf (inklusive Fehlerpruefung), im CPU-Modus arbeiten sie auf
// gewoehnlichem Host-Speicher. cudaMemcpyKind wird im CPU-Modus ignoriert --
// es gibt dort nur einen Adressraum, jede Richtung ist ein memcpy.
// -----------------------------------------------------------------------------
void  malloc_d_raw(void** ptr, size_t bytes);
void  free_d(void* ptr);
void  memcpy_d(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
void  memset_d(void* ptr, int value, size_t bytes);
void  sync();

/// Typisierte Fassung, damit die Aufrufstellen wie bei cudaMalloc
/// "bf::malloc_d(&d_ptr, bytes)" schreiben koennen.
template <typename T>
inline void malloc_d(T** ptr, size_t bytes) {
    void* raw = nullptr;
    malloc_d_raw(&raw, bytes);
    *ptr = static_cast<T*>(raw);
}

/// Unified Memory (GPU) bzw. schlichter Host-Speicher (CPU). Wird von
/// DeviceArray benutzt.
void  malloc_managed_raw(void** ptr, size_t bytes);
void  free_managed(void* ptr);

/// Prefetch-Hinweis. Im CPU-Modus ein No-Op -- es gibt nichts zu migrieren.
void  prefetch_to_device_hint(const void* ptr, size_t bytes, int device_id, cudaStream_t stream);
void  prefetch_to_host_hint(const void* ptr, size_t bytes, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Atomare Addition
// -----------------------------------------------------------------------------
//
// Auf der GPU das echte atomicAdd, auf der CPU eine gewoehnliche Addition:
// die CPU-Pfade der betroffenen Kernel laufen bewusst SERIELL (siehe
// BF_LAUNCH_1D unten -- die Streu-Kernel benutzen die nicht parallelisierte
// Variante), es gibt dort also kein Datenrennen.
//
// Angenehmer Nebeneffekt: die serielle Summation ist reproduzierbar. Die
// Reihenfolge, in der atomicAdd auf der GPU Beitraege aufsummiert, ist es
// nicht -- der CPU-Modus liefert damit bitidentische Ergebnisse ueber
// mehrere Laeufe hinweg und eignet sich als Referenz fuer
// tests/test_determinism_fingerprint.
// -----------------------------------------------------------------------------
BF_HD inline void atomic_add(double* address, double value) {
#ifdef __CUDA_ARCH__
    atomicAdd(address, value);
#else
    *address += value;
#endif
}

} // namespace bf

// -----------------------------------------------------------------------------
// OpenMP: nur fuer nachweislich rennfreie Kernel (kein atomic_add, jeder
// Index schreibt ausschliesslich in seine eigenen Ausgabeplaetze).
// -----------------------------------------------------------------------------
#if defined(_OPENMP)
    #if defined(_MSC_VER)
        #define BF_OMP_PARALLEL_FOR __pragma(omp parallel for)
    #else
        #define BF_OMP_PARALLEL_FOR _Pragma("omp parallel for")
    #endif
#else
    #define BF_OMP_PARALLEL_FOR
#endif

// -----------------------------------------------------------------------------
// Start-Makros
// -----------------------------------------------------------------------------
//
// Voraussetzung an der Definitionsstelle des Kernels:
//
//     BF_HD inline void meinKernel_body(int i, ARGS...) { ... }
//     __global__ void meinKernel(ARGS...) {
//         int i = blockIdx.x * blockDim.x + threadIdx.x;
//         if (i < n) meinKernel_body(i, ARGS...);
//     }
//
// Aufrufstelle:
//
//     BF_LAUNCH_1D(meinKernel, n, ARGS...);        // seriell auf der CPU
//     BF_LAUNCH_1D_PAR(meinKernel, n, ARGS...);    // OpenMP auf der CPU
//
// Die _PAR-Variante ist ausschliesslich fuer Kernel ohne bf::atomic_add und
// ohne ueberlappende Schreibziele erlaubt. Im Zweifel die serielle Variante
// nehmen: sie ist langsamer, aber nie falsch.
// -----------------------------------------------------------------------------

#define BF_LAUNCH_1D(KERNEL, N, ...)                                            \
    do {                                                                        \
        const int bf_n_ = static_cast<int>(N);                                  \
        if (bf_n_ > 0) {                                                        \
            if (::bf::on_cpu()) {                                               \
                for (int bf_i_ = 0; bf_i_ < bf_n_; ++bf_i_) {                   \
                    KERNEL##_body(bf_i_, __VA_ARGS__);                          \
                }                                                               \
            } else {                                                            \
                KERNEL<<<(bf_n_ + 255) / 256, 256>>>(__VA_ARGS__);              \
                BF_CUDA_CHECK(cudaGetLastError());                              \
            }                                                                   \
        }                                                                       \
    } while (0)

#define BF_LAUNCH_1D_PAR(KERNEL, N, ...)                                        \
    do {                                                                        \
        const int bf_n_ = static_cast<int>(N);                                  \
        if (bf_n_ > 0) {                                                        \
            if (::bf::on_cpu()) {                                               \
                BF_OMP_PARALLEL_FOR                                             \
                for (int bf_i_ = 0; bf_i_ < bf_n_; ++bf_i_) {                   \
                    KERNEL##_body(bf_i_, __VA_ARGS__);                          \
                }                                                               \
            } else {                                                            \
                KERNEL<<<(bf_n_ + 255) / 256, 256>>>(__VA_ARGS__);              \
                BF_CUDA_CHECK(cudaGetLastError());                              \
            }                                                                   \
        }                                                                       \
    } while (0)

// 2D-Variante fuer die BEM-Blockkernel (Zeile = Quad p, Spalte = Quad q).
// Rumpfsignatur: <name>_body(int ix, int iy, ARGS...)
#define BF_LAUNCH_2D(KERNEL, NX, NY, ...)                                       \
    do {                                                                        \
        const int bf_nx_ = static_cast<int>(NX);                                \
        const int bf_ny_ = static_cast<int>(NY);                                \
        if (bf_nx_ > 0 && bf_ny_ > 0) {                                         \
            if (::bf::on_cpu()) {                                               \
                BF_OMP_PARALLEL_FOR                                             \
                for (int bf_ix_ = 0; bf_ix_ < bf_nx_; ++bf_ix_) {               \
                    for (int bf_iy_ = 0; bf_iy_ < bf_ny_; ++bf_iy_) {           \
                        KERNEL##_body(bf_ix_, bf_iy_, __VA_ARGS__);             \
                    }                                                           \
                }                                                               \
            } else {                                                            \
                const dim3 bf_threads_(16, 16);                                 \
                const dim3 bf_blocks_((bf_nx_ + 15) / 16, (bf_ny_ + 15) / 16);  \
                KERNEL<<<bf_blocks_, bf_threads_>>>(__VA_ARGS__);               \
                BF_CUDA_CHECK(cudaGetLastError());                              \
            }                                                                   \
        }                                                                       \
    } while (0)

#endif // FEMBEM_BACKEND_CUH
