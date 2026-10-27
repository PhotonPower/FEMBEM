// =============================================================================
// Backend.cu -- Implementierung der GPU/CPU-Umschaltung
// =============================================================================

#include "Backend.cuh"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bf {

// GPU bleibt der Standard: ein Aufruf ohne neue Optionen verhaelt sich exakt
// wie vor dieser Erweiterung.
Backend g_backend   = Backend::GPU;
int     g_cpu_threads = 0;
bool    g_use_reference_solver = false;
int     g_refine_levels = 0;
bool    g_verify_field = false;
bool    g_no_conductor_penalty = false;
bool    g_fixed_plate = false;
double  g_plate_z_mm = -1.0;
bool    g_no_gauge = false;
bool    g_use_ams = false;
bool    g_no_cohomology = false;
bool    g_use_block_tri_pc = false;
int     g_pc_inner_iters = 10;
int     g_gmres_restart = 50;
bool    g_want_ritz = false;
int     g_max_iter = 1000;
std::string g_mesh_file = "team28.vtk";
bool    g_direct_solve = false;
bool    g_pc_study     = false;
bool    g_harmonic_study = false;
bool    g_harmonic_solve = false;
bool    g_fem_direct     = false;   // --fem-direct
double  g_fem_direct_eps = 1e-8;    // sigma_Luft = eps * sigma_Leiter
bool    g_fem_direct_iterative = false;  // --fem-direct-iterative
bool    g_fem_direct_no_project = false; // --fem-direct-no-project
bool    g_fem_direct_gauge      = false; // --fem-direct-gauge
bool    g_harmonic_checks    = false;  // --harmonic-checks
int     g_harmonic_outer_restart = 64;
double  g_harmonic_inner_tol = 1e-4;   // Stufe mit der kleinsten Gesamtarbeit
int     g_harmonic_outer_cap = 0;      // 0 = bf::max_iter()
double  g_harmonic_hz    = 50.0;   // TEAM 28: Netzfrequenz
double  g_dt = 1e-4;
double  g_amg_theta = 0.60;
double  g_pc_inner_tol = 1e-2;

const char* backend_name() {
    return on_cpu() ? "CPU" : "GPU";
}

bool is_option(const char* arg) {
    return arg != nullptr && arg[0] == '-';
}

void print_usage(const char* exe_name) {
    const char* name = (exe_name != nullptr) ? exe_name : "fembem";
    std::cout
        << "\nFEMBEM -- BEM-FEM Transient Solver (TEAM 28)\n\n"
        << "Aufruf:\n"
        << "  " << name << " [Schritte] [Diagnoseschalter] [Optionen]\n\n"
        << "Positionale Argumente:\n"
        << "  Schritte       Anzahl Zeitschritte fuer einen Kurztest (Standard: 1000).\n"
        << "  nobem          BEM-Kopplung abschalten (reines FEM-System).\n"
        << "  noilu          ILU(0)-Vorkonditionierer umgehen (Identitaet).\n\n"
        << "Loeserwahl:\n"
        << "  --reference    Baseline-/Referenzloeser statt des produktiven Pfads:\n"
        << "                 Q1-KNOTENelemente (3 DOF je Knoten) statt Nedelec-\n"
        << "                 Kantenelemente, Eichung ueber einen div-div-STRAFTERM\n"
        << "                 statt Baum-Kobaum plus Kohomologie, und eine\n"
        << "                 VOLLBESETZTE BEM statt ACA. Bewusst einfach gehalten,\n"
        << "                 damit er als unabhaengige Zweitrechnung taugt.\n"
        << "                 Voraussetzung: mu_r == 1 ueberall (wird geprueft).\n"
        << "                 Details in include/ReferenceSolver.cuh.\n\n"
        << "Vorkonditionierung:\n"
        << "  --ams          Auxiliary-Space Maxwell Solver (Hiptmair und Xu) statt\n"
        << "                 ILU(0) fuer den FEM-Block. Die beiden Hilfsprobleme\n"
        << "                 werden mit Aggregations-AMG geloest. Am Modellproblem\n"
        << "                 gemessen: 23 bis 26 CG-Iterationen gegen 298 bis 694 mit\n"
        << "                 Jacobi, und praktisch unabhaengig von sigma.\n"
        << "  --nocohomology Kohomologie-Projektoren NICHT anwenden. Gemessen liegt\n"
        << "                 ||A c||/||c|| fuer die Generatoren bei 4.9e-02 bzw.\n"
        << "                 7.2e-02 des Werts eines Zufallsvektors -- fuer einen\n"
        << "                 Nullraumvektor waeren 1e-12 zu erwarten. Die Projektion\n"
        << "                 entfernt damit 58 %% der rechten Seite entlang Richtungen,\n"
        << "                 auf die der Operator wirkt, und wirft Loesung weg.\n"
        << "  --blocktri     Blocktriangulaerer Vorkonditionierer fuer das gekoppelte\n"
        << "                 2x2-System statt des blockdiagonalen diag(A~, I). Der\n"
        << "                 Kopplungsblock T wird mitgenommen, das Schur-Komplement\n"
        << "                 durch den Single-Layer-Operator S~ = mu_0*G_BEM genaehert\n"
        << "                 (spektral aequivalent; Hiptmair, Steinbach, Heuer) und mit\n"
        << "                 einem inneren CG angewendet. Anlass: die Kopplungsbloecke\n"
        << "                 sind bauartbedingt zu 81 %% unsymmetrisch (oben rechts -T,\n"
        << "                 unten links -0.5*T - D_BEM), ein blockdiagonaler\n"
        << "                 Vorkonditionierer ignoriert sie vollstaendig.\n"
        << "  --ritz         Ritz-Spektrum nach dem ersten Arnoldi-Zyklus ausgeben.\n"
        << "                 Liefert eine untere Schranke fuer die Konditionszahl des\n"
        << "                 VORKONDITIONIERTEN Operators und damit die Antwort darauf,\n"
        << "                 ob die zugestandene Iterationszahl reichen kann. Am\n"
        << "                 aussagekraeftigsten mit grossem --gmres-restart.\n"
        << "  --amg-theta=X  Schwelle der starken Kopplung im AMG (Standard 0.60).\n"
        << "                 Bezug ist die groesste Kopplung DER ZEILE (zeilenrelativer\n"
        << "                 Maszstab), nicht das Diagonalelement. Gemessen am\n"
        << "                 Poisson-Problem, Faktor je V-Zyklus (Grobgitter in Klammern,\n"
        << "                 fein 1156):\n"
        << "                    Zell-Verh.  alt: diagonal 0.08   neu: zeilenrel. 0.60\n"
        << "                       1.0      0.237 (42)          0.273 (21)\n"
        << "                       4.5      0.774 (40)          0.439 (36)\n"
        << "                      10.0      0.854 (40)          0.478 (36)\n"
        << "                      30.0      0.873 (41)          0.490 (36)\n"
        << "                 FALLE: ein zu grosses theta liefert scheinbar traumhafte\n"
        << "                 Faktoren, aber dann ist das Grobgitter so gross wie das\n"
        << "                 feine -- eine direkte Loesung, kein Mehrgitter. Immer die\n"
        << "                 Grobgittergroessen in der [AMS]-Ausgabe mitlesen.\n"
        << "  --dt=X         Zeitschrittweite in Sekunden (Standard 1e-4).\n"
        << "                 Die Konditionszahl haengt daran als U-KURVE, nicht als\n"
        << "                 Trend. Gemessen am Netz n=12 (sigma_max konstant 155):\n"
        << "                    dt=1e-6  kappa_2=2.05e+06\n"
        << "                    dt=1e-5  kappa_2=1.03e+06   <- Minimum\n"
        << "                    dt=1e-4  kappa_2=1.07e+06   (Vorgabe, praktisch optimal)\n"
        << "                    dt=1e-3  kappa_2=2.63e+06\n"
        << "                    dt=1e-2  kappa_2=2.14e+07\n"
        << "                 Grosses dt: der Massenterm wird klein, der Operator naehert\n"
        << "                 sich dem singulaeren Curl-Curl. Kleines dt: der Kontrast\n"
        << "                 Leiter gegen Luft begrenzt. Am Zeitschritt ist also nichts\n"
        << "                 zu gewinnen -- kappa ~ 1e+06 ist ein Sockel aus der Struktur\n"
        << "                 des Problems.\n"
        << "                 ACHTUNG: aendert die Physik, nicht nur die Numerik.\n"
        << "  --direct       Das gekoppelte System dicht zerlegen und das Residuum der\n"
        << "                 EXAKTEN Loesung melden. Bleibt es bei ~1e-2, ist das System\n"
        << "                 inkonsistent und kein Vorkonditionierer hilft; faellt es auf\n"
        << "                 ~1e-15, ist es loesbar und GMRES scheitert aus einem anderen\n"
        << "                 Grund. Nur fuer kleine Netze -- Speicher quadratisch.\n"
        << "  --harmonic[=HZ]  Eingeschwungenen Zustand loesen statt 1000 Zeitschritte zu\n"
        << "                 rechnen (Standard 50 Hz). TEAM 28 ist linear, der harmonische\n"
        << "                 Ansatz ist also exakt. Gibt die ZEITGEMITTELTE Kraft aus.\n"
        << "  --harmonic-outer-restart=N  Restart des AEUSSEREN Loesers (Standard 64).\n"
        << "                 Klein setzen, um den Vergleich echtes/geschaetztes Residuum\n"
        << "                 frueh zu sehen -- der erscheint erst beim Restart.\n"
        << "  --harmonic-checks  Diagnosen des harmonischen Loesers: Vorabmessung von\n"
        << "                 (W+T) mit und ohne ILU, Eichungs-Gegenprobe, Linearitaet des\n"
        << "                 Vorkonditionierers. Kostet rund 40000 innere Schritte VOR der\n"
        << "                 ersten aeusseren Iteration -- deshalb nicht mehr Standard.\n"
        << "  --harmonic-inner-tol=X  Toleranz des inneren (W+T)-Loesers (Standard 1e-4).\n"
        << "                 Fuer den Linearitaetstest: mit 1e-10 ist der Vorkonditionierer\n"
        << "                 praktisch exakt und damit praktisch linear.\n"
        << "  --harmonic-outer-cap=N  Deckel fuer die aeusseren Iterationen.\n"
        << "  --harmonic-study[=HZ]  Harmonischer Ansatz statt Transiente (schaltet --direct\n"
        << "                 mit ein, Standard 50 Hz). Spaltet A(dt) = K + C/dt ueber drei\n"
        << "                 Abtastungen auf, baut daraus das reelle Blocksystem\n"
        << "                 [[K, -w*C],[w*C, K]] und vergleicht es mit dem transienten --\n"
        << "                 mit und ohne C-to-R-Vorkonditionierer. TEAM 28 ist linear, der\n"
        << "                 harmonische Ansatz ist also exakt und ersetzt 1000 Zeitschritte\n"
        << "                 durch rund zehn Loesungen der Hoehen-Nullstellensuche.\n"
        << "                 Zusammen mit --pc-study laeuft die Deckenmessung auf\n"
        << "                 (W+T) = K + omega*C statt auf dem transienten Operator --\n"
        << "                 also auf dem, den der C-to-R-Zweig innen loesen muss.\n"
        << "  --pc-study     Deckenmessung fuer den BEM-Block (schaltet --direct mit ein).\n"
        << "                 Miszt kappa_2 der beiden Diagonalbloecke GETRENNT und\n"
        << "                 vergleicht GMRES unter fuenf idealisierten Vorkonditionierern,\n"
        << "                 darunter der EXAKTEN Inverse von A22 = mu_0*G_BEM.\n"
        << "                 Diese ist die obere Schranke fuer jeden BEM-Block-\n"
        << "                 Vorkonditionierer -- Calderon und H-Matrix-Naeherungsinverse\n"
        << "                 sind beide nur Naeherungen daran. Bringt sie nichts, ist der\n"
        << "                 Ansatz widerlegt, bevor er gebaut wird. Kubischer Aufwand,\n"
        << "                 also nur fuer die groben Ersatznetze.\n"
        << "  --mesh=DATEI        Netzdatei (Standard team28.vtk). Fuer die Netzstudie:\n"
        << "                 tools/make_coarse_team28.py erzeugt grobe Ersatznetze\n"
        << "                 derselben Geometrie. Faellt das Residuum dort sauber auf\n"
        << "                 1e-6 und wird mit feinerem Netz systematisch schlechter,\n"
        << "                 ist es ein Konditionierungsproblem; stagniert es auch auf\n"
        << "                 dem groebsten Netz, sitzt das Problem nicht im Loeser.\n"
        << "  --fem-direct-gauge  beseitigt den Curl-Curl-Nullraum durch die\n"
        << "                 Baum-Kobaum-Eichung statt durch sigma_Luft. Die\n"
        << "                 Regularisierung entfaellt dann. Erwartung: bessere\n"
        << "                 Kondition, weil die geeichten Richtungen nicht mehr um\n"
        << "                 1/eps unter dem Rest liegen. Die Generatorensuche laeuft\n"
        << "                 dabei und sagt, ob das nichtleitende Gebiet wirklich\n"
        << "                 einfach zusammenhaengend ist.\n"
        << "  --fem-direct-no-project  schaltet die Gradientenprojektion des Quellterms\n"
        << "                 ab (Standard: an). Der punktweise vorgegebene Quellterm ist\n"
        << "                 diskret nicht divergenzfrei, weil die Spule elementweise\n"
        << "                 zugeordnet wird -- gemessen 6,1e-03 gegen die 1e-4 aus M2,\n"
        << "                 und durch Verfeinerung nicht behebbar. Abschaltbar, weil\n"
        << "                 alle Messreihen in docs/design/fem-direct-plan.md ohne sie\n"
        << "                 entstanden sind.\n"
        << "  --fem-direct-iterative  loest die 2n-Blockform mit GMRES(m) statt cuDSS.\n"
        << "                 Zum VERGLEICH: GMRES bekommt bitgenau dieselbe Matrix und\n"
        << "                 rechte Seite wie der Direktloeser, die einzige Variable ist\n"
        << "                 der Loeser. Restart ueber --gmres-restart, Obergrenze ueber\n"
        << "                 --max-iter. Der Operator ist ungeeicht (--fem-direct\n"
        << "                 regularisiert stattdessen), ILU(0) kann daran scheitern --\n"
        << "                 genau das ist die Frage.\n"
        << "  --fem-direct-eps=X  sigma_Luft = X * sigma_Leiter, der Faktor, mit dem\n"
        << "                 --fem-direct den Curl-Curl-Nullraum beseitigt (Standard 1e-8).\n"
        << "                 Die Abnahme verlangt, dass das Ergebnis zwischen 1e-6 und\n"
        << "                 1e-10 stehenbleibt -- tut es das nicht, ist die\n"
        << "                 Regularisierung keine Regularisierung, sondern Physik.\n"
        << "                 Achtung: der Scheingradient im Imaginaerteil von A skaliert\n"
        << "                 mit 1/X, max|A_i| ist also KEINE stehenbleibende Groesse.\n"
        << "                 Verglichen wird B beziehungsweise die Kraft.\n"
        << "  --max-iter=N        Iterationsobergrenze des aeusseren GMRES (Standard 1000).\n"
        << "                 Gebraucht, um \"langsam\" von \"steht\" zu unterscheiden: ein\n"
        << "                 Residuum, das bei 1000 noch faellt, sagt nichts darueber, ob\n"
        << "                 es bei 5000 ankommt oder auf einem Sockel liegt.\n"
        << "  --gmres-restart=N   Restart-Laenge des aeusseren GMRES (Standard 50).\n"
        << "                 Der blocktriangulaere Vorkonditionierer senkt das Residuum\n"
        << "                 frueh schnell und stagniert dann INNERHALB der Zyklen --\n"
        << "                 die Signatur restartbedingter Stagnation. Laengere Zyklen\n"
        << "                 kosten Speicher (N Vektoren der Laenge total+bem) und\n"
        << "                 Rechenzeit pro Iteration, koennen aber die Ausreisser im\n"
        << "                 Spektrum aufloesen, die GMRES(50) bei jedem Restart wieder\n"
        << "                 wegwirft.\n"
        << "  --pc-inner-iters=N  Maximale Iterationszahl des inneren CG (Standard 10).\n"
        << "  --pc-inner-tol=X    Relative Toleranz des inneren CG (Standard 1e-2).\n"
        << "                 Nur mit --blocktri wirksam. Mehr Genauigkeit kostet innere\n"
        << "                 Iterationen, ohne die aeussere Iterationszahl weiter zu\n"
        << "                 senken (Modellproblem: 5 -> 22, 10 -> 20, 20 -> 20).\n"
        << "  --nogauge      Baum-Kobaum-Eichung weglassen. Nur mit --ams sinnvoll:\n"
        << "                 ohne Eichung ist das System singulaer aber konsistent,\n"
        << "                 und ILU(0) konvergiert darauf ueberhaupt nicht, AMS schon.\n"
        << "                 Die Kohomologie-Projektoren bleiben aktiv -- AMS deckt\n"
        << "                 Gradienten ab, nicht die harmonischen Felder der zwei\n"
        << "                 Volltori (die Ringspulen; die Platte ist eine Vollscheibe).\n\n"
        << "Pruefungen:\n"
        << "  --verify-field Magnetostatische Probe: Spulenstrom KONSTANT auf 20 A,\n"
        << "                 300 Schritte, damit die L/R-Transiente abklingt (tau =\n"
        << "                 1.92 ms). Danach ist J = 0 und B_z auf der Achse wird\n"
        << "                 gegen die analytische Biot-Savart-Loesung\n"
        << "                 stellen. Prueft Quellstromdichte, Assemblierung und\n"
        << "                 Aussenraumkopplung in einem Zug, ohne dass\n"
        << "                 Wirbelstromphysik etwas verschleiert.\n"
        << "                 Gilt fuer BEIDE Pfade und druckt dieselbe Tabelle:\n"
        << "                   --verify-field               produktiv (Nedelec + ACA)\n"
        << "                   --verify-field --reference    Knoten-FEM + dichte BEM\n\n"
        << "  --plate-z=<mm> Platte auf diese Schwebehoehe setzen (Unterkante ueber\n"
        << "                 Spulenoberkante) statt auf die Netzlage von 3.8 mm.\n"
        << "                 Zusammen mit --fixed-plate der strenge Test: bei der\n"
        << "                 gemessenen Ruhelage 11.3 mm MUSS das Zyklusmittel der\n"
        << "                 Vertikalkraft exakt m*g = 1.0497 N sein. Das ist der\n"
        << "                 einzige streng exakte Referenzwert aus der Messung.\n"
        << "  --fixed-plate  Platte festhalten (keine Bewegung). Voraussetzung fuer\n"
        << "                 die Q-Messung: bei freier Bewegung legt die Platte in\n"
        << "                 EINER Kraftperiode rund 10 mm zurueck, in einem Feld das\n"
        << "                 darueber stark abfaellt -- ein Zyklusmittel ist dann kein\n"
        << "                 stationaerer Mittelwert. Nebeneffekt: die BEM-Matrizen\n"
        << "                 aendern sich nicht und werden nur EINMAL gebaut, der Lauf\n"
        << "                 ist damit um ein Vielfaches schneller.\n"
        << "  --no-conductor-penalty\n"
        << "                 div-div-Strafterm in leitenden Elementen abschalten.\n"
        << "                 Diagnostisches Experiment: dort erzwingt er div A = 0,\n"
        << "                 waehrend J = -sigma dA/dt gleichzeitig V = 0 annimmt --\n"
        << "                 beides gleichzeitig gilt nur bei exakt azimutaler\n"
        << "                 Loesung. Aendert sich die Wirbelstromstaerke stark, war\n"
        << "                 der Strafterm die Ursache. ACHTUNG: die Randterme der\n"
        << "                 BEM-Kopplung passen dann nicht mehr exakt, das Ergebnis\n"
        << "                 ist als DIAGNOSE zu lesen, nicht als Loesung.\n\n"
        << "Netz:\n"
        << "  --refine=N     Netz N mal uniform verfeinern, je Stufe 1 -> 8 Kinder\n"
        << "                 (2x2x2, also auch ueber die Plattendicke). 0..4.\n"
        << "                 Stufe 1:  4805 ->  38440 Elemente,  7013 ->  46991 Knoten\n"
        << "                 Stufe 2: 38440 -> 307520 Elemente\n"
        << "                 ACHTUNG bei --reference: die vollbesetzte BEM waechst\n"
        << "                 mit N_bem^2. Stufe 1 braucht 6.1 GiB, Stufe 2 rund\n"
        << "                 100 GiB und passt nicht in 16 GiB.\n"
        << "                 Die Geometrie wird nicht runder, nur das Feld feiner.\n\n"
        << "Optionen:\n"
        << "  --cpu          Komplett auf der CPU rechnen. Es wird keine GPU und kein\n"
        << "                 CUDA-Treiber benoetigt; es werden keinerlei Kernel\n"
        << "                 gestartet. Numerisch identischer Code wie im GPU-Pfad\n"
        << "                 (dieselben __host__ __device__ Kernelruempfe), lediglich\n"
        << "                 seriell bzw. per OpenMP statt per CUDA ausgefuehrt.\n"
        << "  --gpu          GPU-Pfad erzwingen (Standard).\n"
        << "  --threads=N    Anzahl OpenMP-Threads im CPU-Modus. Nur wirksam, wenn mit\n"
        << "                 OpenMP gebaut wurde (-DFEMBEM_ENABLE_OPENMP=ON, Standard).\n"
        << "  -h, --help     Diese Hilfe.\n\n"
        << "Umgebungsvariable:\n"
        << "  FEMBEM_BACKEND=cpu|gpu   Wie --cpu/--gpu, wird von einer expliziten\n"
        << "                           Kommandozeilenoption ueberstimmt.\n\n"
        << "Beispiele:\n"
        << "  " << name << " 3 --cpu               drei Zeitschritte, reine CPU\n"
        << "  " << name << " 5 nobem --cpu --threads=8\n"
        << "  " << name << "                       Vollauf auf der GPU\n\n";
}

bool select_backend_from_args(int argc, char** argv) {
    // 1. Umgebungsvariable als Vorbelegung.
    if (const char* env = std::getenv("FEMBEM_BACKEND")) {
        if (std::strcmp(env, "cpu") == 0 || std::strcmp(env, "CPU") == 0) {
            g_backend = Backend::CPU;
        } else if (std::strcmp(env, "gpu") == 0 || std::strcmp(env, "GPU") == 0) {
            g_backend = Backend::GPU;
        } else {
            std::cerr << "[Backend] Warnung: FEMBEM_BACKEND=\"" << env
                      << "\" ist unbekannt, erwartet wird \"cpu\" oder \"gpu\". "
                      << "Wird ignoriert.\n";
        }
    }

    // 2. Kommandozeile ueberstimmt die Umgebungsvariable.
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--nocohomology") {
            g_no_cohomology = true;
        } else if (arg == "--ams") {
            g_use_ams = true;
        } else if (arg == "--blocktri") {
            g_use_block_tri_pc = true;
        } else if (arg == "--ritz") {
            g_want_ritz = true;
        } else if (arg == "--fem-direct-gauge") {
            g_fem_direct_gauge = true;
            g_fem_direct       = true;
            g_harmonic_solve   = true;
        } else if (arg == "--fem-direct-no-project") {
            g_fem_direct_no_project = true;
        } else if (arg == "--fem-direct-iterative") {
            // Impliziert --fem-direct: ohne die 2n-Blockform gibt es nichts
            // iterativ zu loesen.
            g_fem_direct_iterative = true;
            g_fem_direct           = true;
            g_harmonic_solve       = true;
        } else if (arg.rfind("--fem-direct-eps=", 0) == 0) {
            // Grenzen wie in regularize_nonconducting_regions: (0, 1). Die
            // Abnahme aus Abschnitt 1 des Plans laeuft ueber 1e-6 bis 1e-10;
            // ausserhalb dieser Spanne ist der Wert zulaessig, aber dann
            // steht keine Aussage mehr dahinter, dass die Regularisierung
            // eine Regularisierung ist und keine Physik.
            const double v = std::atof(arg.c_str() + 17);
            if (v > 0.0 && v < 1.0) {
                g_fem_direct_eps = v;
            } else {
                std::cerr << "[Backend] Warnung: " << arg << " erwartet einen Wert "
                             "in (0, 1). Bleibe bei "
                          << g_fem_direct_eps << ".\n";
            }
        } else if (arg.rfind("--amg-theta=", 0) == 0) {
            const double v = std::atof(arg.c_str() + 12);
            if (v > 0.0 && v < 1.0) {
                g_amg_theta = v;
            } else {
                std::cerr << "[Backend] Warnung: " << arg << " erwartet einen Wert "
                             "zwischen 0 und 1. Wird ignoriert.\n";
            }
        } else if (arg.rfind("--dt=", 0) == 0) {
            const double v = std::atof(arg.c_str() + 5);
            if (v > 0.0) {
                g_dt = v;
            } else {
                std::cerr << "[Backend] Warnung: " << arg << " erwartet einen positiven "
                             "Wert. Wird ignoriert.\n";
            }
        } else if (arg == "--direct") {
            g_direct_solve = true;
        } else if (arg.rfind("--harmonic-outer-restart=", 0) == 0) {
            const int n = std::atoi(arg.c_str() + 25);
            if (n > 0) g_harmonic_outer_restart = n;
            else std::cerr << "[Backend] Warnung: --harmonic-outer-restart ungueltig, "
                              "bleibe bei " << g_harmonic_outer_restart << ".\n";
        } else if (arg == "--harmonic-checks") {
            g_harmonic_checks = true;
        } else if (arg.rfind("--harmonic-inner-tol=", 0) == 0) {
            const double t = std::atof(arg.c_str() + 21);
            if (std::isfinite(t) && t > 0.0) g_harmonic_inner_tol = t;
            else std::cerr << "[Backend] Warnung: --harmonic-inner-tol ungueltig, bleibe bei "
                           << g_harmonic_inner_tol << ".\n";
        } else if (arg.rfind("--harmonic-outer-cap=", 0) == 0) {
            const int n = std::atoi(arg.c_str() + 21);
            if (n > 0) g_harmonic_outer_cap = n;
            else std::cerr << "[Backend] Warnung: --harmonic-outer-cap ungueltig, ignoriert.\n";
        } else if (arg == "--fem-direct") {
            // Reines FEM mit mitvernetztem Luftgebiet und Sparse-Direktloeser.
            //
            // Impliziert --harmonic: der Pfad ist fuer den harmonischen Fall
            // gebaut, und ohne Frequenz gibt es keine 2n-Blockform. Wer nur
            // --fem-direct angibt, meint das.
            //
            // Unvertraeglichkeiten werden NICHT hier geprueft, sondern in
            // fem.cu, wo auch das Netz bekannt ist -- eine Meldung "BEM geht
            // nicht" ist erst dann vollstaendig, wenn man sagen kann, warum
            // (Dreiecksflaechen im Netz).
            g_fem_direct     = true;
            g_harmonic_solve = true;
        } else if (arg == "--harmonic" || arg.rfind("--harmonic=", 0) == 0) {
            g_harmonic_solve = true;
            if (arg.size() > 11) {
                const double hz = std::atof(arg.c_str() + 11);
                if (std::isfinite(hz) && hz > 0.0) {
                    g_harmonic_hz = hz;
                } else {
                    std::cerr << "[Backend] Warnung: --harmonic=\"" << (arg.c_str() + 11)
                              << "\" ist keine gueltige Frequenz. Bleibe bei "
                              << g_harmonic_hz << " Hz.\n";
                }
            }
        } else if (arg == "--harmonic-study" || arg.rfind("--harmonic-study=", 0) == 0) {
            // Setzt --direct voraus und schaltet es mit ein: die Studie
            // arbeitet auf der dicht abgetasteten Matrix.
            g_harmonic_study = true;
            g_direct_solve   = true;
            if (arg.size() > 17) {
                const double hz = std::atof(arg.c_str() + 17);
                // Relativ pruefen und Endlichkeit einschliessen -- "> 0.0"
                // laesst NaN durch, und NaN als Frequenz wuerde stillschweigend
                // ein System aus lauter NaN erzeugen.
                if (std::isfinite(hz) && hz > 0.0) {
                    g_harmonic_hz = hz;
                } else {
                    std::cerr << "[Backend] Warnung: --harmonic-study=\"" << (arg.c_str() + 17)
                              << "\" ist keine gueltige Frequenz. Bleibe bei "
                              << g_harmonic_hz << " Hz.\n";
                }
            }
        } else if (arg == "--pc-study") {
            // Setzt --direct voraus und schaltet es deshalb gleich mit ein --
            // die Studie arbeitet auf der Matrix, die --direct ohnehin schon
            // spaltenweise abtastet. Ein zweiter Abtastlauf waere reine
            // Verschwendung, und ein "--pc-study ohne Wirkung, weil --direct
            // fehlt" waere genau die Sorte stiller Fehlbedienung, die dieses
            // Projekt schon Messtage gekostet hat.
            g_pc_study     = true;
            g_direct_solve = true;
        } else if (arg.rfind("--mesh=", 0) == 0) {
            g_mesh_file = arg.substr(7);
            if (g_mesh_file.empty()) {
                std::cerr << "[Backend] Warnung: --mesh= ohne Dateinamen. "
                             "Wird ignoriert.\n";
                g_mesh_file = "team28.vtk";
            }
        } else if (arg.rfind("--max-iter=", 0) == 0) {
            const int v = std::atoi(arg.c_str() + 11);
            if (v > 0) {
                g_max_iter = v;
            } else {
                std::cerr << "[Backend] Warnung: " << arg << " erwartet eine positive "
                             "Zahl. Wird ignoriert.\n";
            }
        } else if (arg.rfind("--gmres-restart=", 0) == 0) {
            const int v = std::atoi(arg.c_str() + 16);
            if (v > 0) {
                g_gmres_restart = v;
            } else {
                std::cerr << "[Backend] Warnung: " << arg << " erwartet eine positive "
                             "Zahl. Wird ignoriert.\n";
            }
        } else if (arg.rfind("--pc-inner-iters=", 0) == 0) {
            const int v = std::atoi(arg.c_str() + 17);
            if (v > 0) {
                g_pc_inner_iters = v;
            } else {
                std::cerr << "[Backend] Warnung: " << arg << " erwartet eine positive "
                             "Zahl. Wird ignoriert.\n";
            }
        } else if (arg.rfind("--pc-inner-tol=", 0) == 0) {
            const double v = std::atof(arg.c_str() + 15);
            if (v > 0.0) {
                g_pc_inner_tol = v;
            } else {
                std::cerr << "[Backend] Warnung: " << arg << " erwartet einen positiven "
                             "Wert. Wird ignoriert.\n";
            }
        } else if (arg == "--nogauge") {
            g_no_gauge = true;
        } else if (arg == "--fixed-plate") {
            g_fixed_plate = true;
        } else if (arg == "--no-conductor-penalty") {
            g_no_conductor_penalty = true;
        } else if (arg == "--verify-field") {
            g_verify_field = true;
            // Laeuft jetzt in BEIDEN Pfaden -- --reference wird NICHT mehr
            // automatisch gesetzt, sonst waere der produktive Pfad nicht
            // pruefbar, und genau der ist die offene Frage.
        } else if (arg == "--reference" || arg == "--ref") {
            g_use_reference_solver = true;
        } else if (arg == "--cpu") {
            g_backend = Backend::CPU;
        } else if (arg == "--gpu") {
            g_backend = Backend::GPU;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return false;
        } else if (arg.rfind("--plate-z=", 0) == 0) {
            const double v = std::atof(arg.c_str() + 10);
            if (v > 0.0) {
                g_plate_z_mm = v;
            } else {
                std::cerr << "[Backend] Warnung: \"" << arg
                          << "\" ergibt keine positive Hoehe, wird ignoriert.\n";
            }
        } else if (arg.rfind("--refine=", 0) == 0) {
            const int n = std::atoi(arg.c_str() + 9);
            if (n >= 0 && n <= 4) {
                g_refine_levels = n;
            } else {
                std::cerr << "[Backend] Warnung: \"" << arg
                          << "\" liegt ausserhalb 0..4 und wird ignoriert. "
                             "Jede Stufe vervielfacht die Elementzahl mit 8.\n";
            }
        } else if (arg.rfind("--threads=", 0) == 0) {
            const int n = std::atoi(arg.c_str() + 10);
            if (n > 0) {
                g_cpu_threads = n;
            } else {
                std::cerr << "[Backend] Warnung: \"" << arg
                          << "\" ergibt keine positive Threadzahl, wird ignoriert.\n";
            }
        } else if (is_option(argv[i])) {
            std::cerr << "[Backend] Warnung: unbekannte Option \"" << arg
                      << "\" wird ignoriert. \"--help\" zeigt die Liste.\n";
        }
        // Alles andere ist positional (Schrittzahl, nobem, noilu) und wird
        // weiterhin in mainloop() ausgewertet.
    }

#ifdef _OPENMP
    if (on_cpu() && g_cpu_threads > 0) {
        omp_set_num_threads(g_cpu_threads);
    }
#else
    if (on_cpu() && g_cpu_threads > 0) {
        std::cerr << "[Backend] Hinweis: --threads=" << g_cpu_threads
                  << " hat keine Wirkung, dieser Build enthaelt kein OpenMP "
                     "(cmake -DFEMBEM_ENABLE_OPENMP=ON).\n";
    }
#endif

    return true;
}

void print_backend_banner() {
    if (on_cpu()) {
        std::cout << "[Backend] CPU-Modus aktiv -- es werden keine CUDA-Kernel "
                     "gestartet und kein Geraetespeicher belegt.\n";
#ifdef _OPENMP
        std::cout << "[Backend] OpenMP: bis zu " << omp_get_max_threads()
                  << " Threads fuer die parallelisierbaren Kernel.\n";
#else
        std::cout << "[Backend] Ohne OpenMP gebaut: alle Kernel laufen seriell. "
                     "Fuer spuerbar mehr Tempo mit "
                     "-DFEMBEM_ENABLE_OPENMP=ON neu konfigurieren.\n";
#endif
        std::cout << "[Backend] Hinweis: die Streu-Kernel (RHS, CSR, BEM-Scatter) "
                     "laufen bewusst seriell. Das ist langsamer als auf der GPU, "
                     "dafuer ist die Summationsreihenfolge fest und das Ergebnis "
                     "damit von Lauf zu Lauf bitidentisch.\n";
        return;
    }

    int device = -1;
    cudaError_t st = cudaGetDevice(&device);
    if (st != cudaSuccess) {
        std::cerr << "[Backend] GPU-Modus gewaehlt, aber kein CUDA-Geraet ansprechbar: "
                  << cudaGetErrorString(st)
                  << "\n           Mit \"--cpu\" laeuft die Rechnung vollstaendig "
                     "auf der CPU.\n";
        return;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        std::cout << "[Backend] GPU-Modus aktiv -- Geraet " << device << ": "
                  << prop.name << " (sm_" << prop.major << prop.minor << ")\n";
    } else {
        std::cout << "[Backend] GPU-Modus aktiv -- Geraet " << device << "\n";
    }
}

// -----------------------------------------------------------------------------
// Speicher
// -----------------------------------------------------------------------------

void malloc_d_raw(void** ptr, size_t bytes) {
    if (bytes == 0) { *ptr = nullptr; return; }

    if (on_cpu()) {
        void* p = std::malloc(bytes);
        if (p == nullptr) {
            throw std::runtime_error("bf::malloc_d: Host-Allokation von " +
                                     std::to_string(bytes) + " Bytes fehlgeschlagen.");
        }
        *ptr = p;
        return;
    }
    BF_CUDA_CHECK(cudaMalloc(ptr, bytes));
}

void free_d(void* ptr) {
    if (ptr == nullptr) return;
    if (on_cpu()) { std::free(ptr); return; }
    // Im Aufraeumpfad (auch aus Destruktoren) bewusst ohne Ausnahme.
    cudaFree(ptr);
}

void memcpy_d(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    if (bytes == 0 || dst == src) return;

    if (on_cpu()) {
        // Ein einziger Adressraum -- die Richtung (Host/Device) ist bedeutungslos.
        std::memcpy(dst, src, bytes);
        return;
    }
    BF_CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
}

void memset_d(void* ptr, int value, size_t bytes) {
    if (ptr == nullptr || bytes == 0) return;
    if (on_cpu()) { std::memset(ptr, value, bytes); return; }
    BF_CUDA_CHECK(cudaMemset(ptr, value, bytes));
}

void sync() {
    if (on_cpu()) return;   // Host-Schleifen sind bereits abgeschlossen.
    BF_CUDA_CHECK(cudaDeviceSynchronize());
}

void malloc_managed_raw(void** ptr, size_t bytes) {
    if (bytes == 0) { *ptr = nullptr; return; }

    if (on_cpu()) {
        void* p = std::malloc(bytes);
        if (p == nullptr) {
            throw std::runtime_error("bf::malloc_managed: Host-Allokation von " +
                                     std::to_string(bytes) + " Bytes fehlgeschlagen.");
        }
        *ptr = p;
        return;
    }
    BF_CUDA_CHECK(cudaMallocManaged(ptr, bytes));
}

void free_managed(void* ptr) {
    if (ptr == nullptr) return;
    if (on_cpu()) { std::free(ptr); return; }
    cudaFree(ptr);
}

void prefetch_to_device_hint(const void* ptr, size_t bytes, int device_id, cudaStream_t stream) {
    if (on_cpu() || ptr == nullptr || bytes == 0) return;
#ifndef _WIN32
    // Unter Windows (WDDM) ist cudaMemPrefetchAsync nicht nutzbar, siehe
    // urspruenglicher Kommentar in DeviceArray.cuh.
#if CUDART_VERSION >= 13000
    // CUDA 13 hat die Signatur geaendert: statt der Geraetenummer ein
    // cudaMemLocation, dazu ein Flags-Argument (die alte Form gibt es nicht mehr).
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeDevice;
    loc.id   = device_id;
    BF_CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0u, stream));
#else
    BF_CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, device_id, stream));
#endif
#else
    (void)device_id; (void)stream;
#endif
}

void prefetch_to_host_hint(const void* ptr, size_t bytes, cudaStream_t stream) {
    if (on_cpu() || ptr == nullptr || bytes == 0) return;
#ifndef _WIN32
#if CUDART_VERSION >= 13000
    cudaMemLocation loc{};
    loc.type = cudaMemLocationTypeHost;   // id wird bei Host ignoriert
    loc.id   = 0;
    BF_CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, loc, 0u, stream));
#else
    BF_CUDA_CHECK(cudaMemPrefetchAsync(ptr, bytes, cudaCpuDeviceId, stream));
#endif
#else
    (void)stream;
#endif
}

} // namespace bf
