#include "HarmonicSolver.cuh"
#include "Backend.cuh"
#include "HostLinalg.cuh"

#include <algorithm>
#include <cmath>
#include <string>   // std::string / std::to_string: MSVC zieht das
                    // NICHT ueber andere Header mit, GCC schon.
#include <iostream>
#include <limits>
#include <random>
#include <vector>

namespace bf {

namespace {

// -----------------------------------------------------------------------------
// Der harmonische Blockoperator
// -----------------------------------------------------------------------------
//
//     y_r = K x_r - omega*C x_i
//     y_i = omega*C x_r + K x_i
//
// K ist der VOLLSTAENDIGE gekoppelte Operator ohne Massenterm (also mit
// FEM-Block, BEM-Block und beiden Kopplungsbloecken), C traegt nur im
// FEM-Block. Der BEM-Anteil von C ist null, weil der Single-Layer-Operator
// frequenzunabhaengig ist -- deshalb laeuft die C-Anwendung nur ueber die
// ersten total_dofs Eintraege und laesst den Rest unberuehrt.
// EIN Operator genuegt. Mit K = (W+T) - omega*C laesst sich H allein durch
// (W+T) und omega*C ausdruecken:
//
//     y_r = K x_r - wC x_i = (W+T) x_r - wC (x_r + x_i)
//     y_i = wC x_r + K x_i = (W+T) x_i + wC (x_r - x_i)
//
// Das erspart einen ZWEITEN vollstaendig aufgesetzten CoupledMVOperator --
// mit setup_bem_coupling, setup_bem_mesh, setup_gauge, den
// Kohomologie-Projektoren, setup_ams und setup_equilibration. Jede dieser
// Konfigurationen doppelt zu fuehren waere nicht nur Aufwand, sondern eine
// dauerhafte Fehlerquelle: zwei Operatoren, die auseinanderlaufen koennen,
// ohne dass es auffaellt.
struct HarmonicOperator {
    CoupledMVOperator& op_W;
    const CSRMatrix&   csr_C;
    int    nd;
    int    total_dofs;
    double omega;

    // Streicher fuer die vier Zwischenergebnisse. Einmal angelegt statt je
    // Anwendung -- bei einigen hundert Iterationen waere das sonst ein
    // Allokationssturm.
    mutable DeviceArray<double> t1, t2, t3, t4, t_scale;

    cublasHandle_t cublas;

    HarmonicOperator(CoupledMVOperator& W, const CSRMatrix& C,
                     int nd_, int td_, double w)
        : op_W(W), csr_C(C), nd(nd_), total_dofs(td_), omega(w),
          cublas(W.get_cublas_handle()),
          t1(static_cast<size_t>(nd_)), t2(static_cast<size_t>(nd_)),
          t3(static_cast<size_t>(nd_)), t4(static_cast<size_t>(nd_)),
          t_scale(static_cast<size_t>(nd_)) {}

    /// y = omega * C * x, nur auf dem FEM-Anteil; der BEM-Anteil wird genullt.
    /// y = omega * C~ * x im TILDE-RAUM, also y = omega * D C D x.
    ///
    /// HIER LAG DER FEHLER. op_W.apply() rechnet im Tilde-Raum -- es skaliert
    /// intern mit D. csr_C ist dagegen die ROHE Matrix im physikalischen Raum.
    /// Beide Anteile von H = [[W+T, ...],[..., W+T]] lagen damit in
    /// VERSCHIEDENEN Raeumen, H war keine sinnvolle Matrix, und das aeussere
    /// GMRES stand exakt auf Residuum 1 -- ohne Absturz, ohne NaN, ohne
    /// irgendein Anzeichen ausser der Zahl selbst.
    ///
    /// Im Tilde-Raum gilt A~ = D A D, also C~ = D C D. Deshalb: vor der
    /// Anwendung mit D skalieren, danach noch einmal.
    /// Ergebnis ist eichprojiziert. omega*C traegt auf allen FEM-Kanten,
    /// auch den 478 geeichten; ohne Projektion verunreinigt jeder
    /// omega*C-Beitrag den Vektor, in den er addiert wird. Zentral hier statt
    /// an den vier Aufrufstellen -- eine vergessene Stelle faellt sonst nur
    /// als seltsame Iterationszahl auf, und das hat in diesem Projekt schon
    /// mehrere Laeufe gekostet.
    void apply_omega_C(const double* d_x, double* d_y) const {
        bf::dcopy(cublas, nd, d_x, t_scale.data());
        op_W.scale_by_D(t_scale.data());          // D x
        bf::csr_spmv(csr_C.num_rows, csr_C.row_offsets.data(), csr_C.col_indices.data(),
                     csr_C.values.data(), t_scale.data(), d_y);
        // Der BEM-Anteil muss VOR der zweiten D-Skalierung genullt sein: C
        // traegt dort nicht, und D_K ist gross (8,6e+03 bis 2e+05). Ein
        // stehengebliebener Rest wuerde dort massiv verstaerkt.
        if (nd > total_dofs)
            bf::memset_d(d_y + total_dofs, 0,
                         static_cast<size_t>(nd - total_dofs) * sizeof(double));
        op_W.scale_by_D(d_y);                     // D (C D x)
        bf::dscal(cublas, total_dofs, omega, d_y);
        op_W.project_gauge_tilde_orthogonal(d_y);
    }

    /// [y_r; y_i] = H [x_r; x_i]
    void apply(const double* d_x_r, const double* d_x_i,
               double* d_y_r, double* d_y_i) const {
        // t3 = x_r + x_i,  t4 = x_r - x_i
        bf::dcopy(cublas, nd, d_x_r, t3.data());
        bf::daxpy(cublas, nd, +1.0, d_x_i, t3.data());
        bf::dcopy(cublas, nd, d_x_r, t4.data());
        bf::daxpy(cublas, nd, -1.0, d_x_i, t4.data());

        op_W.apply(d_x_r, d_y_r);              // (W+T) x_r
        apply_omega_C(t3.data(), t1.data());   // wC (x_r + x_i)
        bf::daxpy(cublas, nd, -1.0, t1.data(), d_y_r);

        op_W.apply(d_x_i, d_y_i);              // (W+T) x_i
        apply_omega_C(t4.data(), t2.data());   // wC (x_r - x_i)
        bf::daxpy(cublas, nd, +1.0, t2.data(), d_y_i);

        // Ergebnis ebenfalls projizieren. op_W.apply() tut das fuer seinen
        // Anteil bereits selbst, die omega*C-Beitraege aber nicht -- und ohne
        // sie baut der aeussere Krylov-Raum genau die geeichten Richtungen
        // auf, die der Vorkonditionierer anschliessend jedes Mal verwirft.
        // Der aeussere Prozess arbeitete dann gegen sich selbst.
        op_W.project_gauge_tilde_orthogonal(d_y_r);
        op_W.project_gauge_tilde_orthogonal(d_y_i);
    }
};

// -----------------------------------------------------------------------------
// C-to-R (Axelsson/Kucherov)
// -----------------------------------------------------------------------------
//
//     P = [[W+T, 0], [T-W, W+T]],   W = K,  T = omega*C
//
//     z_r = (W+T)^-1 v_r
//     z_i = (W+T)^-1 ( v_i - (T-W) z_r )
//         = (W+T)^-1 ( v_i - omega*C z_r + K z_r )
//
// Zwei innere Loesungen mit (W+T) = K + omega*C. Der innere Loeser laeuft
// ausdruecklich mit dem GEWOEHNLICHEN ILU-Pfad und ohne
// set_block_triangular_preconditioner(): die Deckenmessung auf (W+T) hat
// gezeigt, dass die blocktriangulare Behandlung dort 6,16e+06 Matvecs je
// Loesung kostet gegen 2914 des schlichten GMRES. Die Blockstruktur zahlt sich
// aussen aus, innen nicht.
struct CtoRPreconditioner {
    CoupledMVOperator&      op_W;
    const HarmonicOperator& H;
    int    nd;
    double inner_tol;
    int    inner_max_iter;
    int    restart;

    cublasHandle_t              cublas;
    mutable GMRESSolver         inner;
    mutable DeviceArray<double> s1, s2;
    mutable long solves = 0;
    mutable long iters  = 0;

    // Der Restart des INNEREN Loesers ist bewusst NICHT der des aeusseren.
    //
    // Die Deckenmessung auf (W+T) hat 2914 Schritte bis 1e-4 gebraucht -- mit
    // VOLLEM GMRES ohne Neustart. Mit einem kurzen Restart erreicht der innere
    // Loeser die geforderte Toleranz gar nicht, unabhaengig von der Guete des
    // ILU; er stagniert dann und verbraucht trotzdem seinen ganzen
    // Iterationsdeckel. Genau das ist beim ersten Lauf passiert: Restart 50,
    // Deckel 2000, Stagnation bei 0,0448.
    //
    // 400 ist ein Kompromiss zwischen Speicher (400 Basisvektoren der Laenge
    // nd) und Konvergenz. Der Deckel liegt bei 4000, also oberhalb der
    // gemessenen 2914.
    CtoRPreconditioner(CoupledMVOperator& W, const HarmonicOperator& H_,
                       int nd_, double tol, int max_it, int m)
        : op_W(W), H(H_), nd(nd_), inner_tol(tol), inner_max_iter(max_it),
          restart(m), cublas(H_.cublas), inner(nd_, m),
          s1(static_cast<size_t>(nd_)), s2(static_cast<size_t>(nd_)) {}

    /// Meldet EINMAL, wenn der innere Loeser seine Toleranz verfehlt.
    ///
    /// Ohne diese Meldung laeuft der aeussere Loeser stundenlang auf einem
    /// Vorkonditionierer, der seine Aufgabe nicht erfuellt -- und die
    /// Ausgabe zeigt nur eine aeussere Iteration, die nicht vorankommt, ohne
    /// den Grund zu nennen. Genau das ist beim ersten Lauf passiert.
    void warn_if_inner_failed(double rel) const {
        // rel <= ... statt !(rel > ...): ein NaN-Residuum ist ein
        // Fehlschlag und muss die Warnung ausloesen, nicht unterdruecken.
        if (m_warned || rel <= inner_tol * 10.0) return;
        m_warned = true;
        std::cout << "[Harmonisch] WARNUNG: der innere (W+T)-Loeser erreicht seine Toleranz "
                  << inner_tol << " nicht (rel. Residuum " << rel << ").\n"
                     "[Harmonisch]   C-to-R kann damit nicht wirken, und der aeussere Loeser\n"
                     "[Harmonisch]   wird nicht konvergieren. Ursache ist der Vorkonditionierer\n"
                     "[Harmonisch]   des inneren Systems, nicht die harmonische Formulierung:\n"
                     "[Harmonisch]   die Deckenmessung braucht fuer (W+T) rund 2900 Schritte bei\n"
                     "[Harmonisch]   1e-4, und zwar mit vollem GMRES. Mehr Restart und ein\n"
                     "[Harmonisch]   hoeherer Deckel sind Voraussetzung; ob ILU dort ueberhaupt\n"
                     "[Harmonisch]   traegt, ist eine offene Frage.\n";
    }

    mutable bool m_warned = false;

    void apply(double* d_v_r, double* d_v_i) const {
        // -------------------------------------------------------------------
        // Eingabe auf den Bildraum projizieren -- BELEGT, nicht vermutet.
        // -------------------------------------------------------------------
        //
        // (W+T) ist auf dem geeichten Anteil singulaer: 478 von 1791
        // FEM-Freiheitsgraden sind festgehalten, dazu zwei
        // Kohomologie-Generatoren. Die physikalische rechte Seite liegt im
        // Bildraum (der Lauf prueft das: ||G^T b||/||b|| = 2,4e-08). Die
        // Krylov-Vektoren des AEUSSEREN Prozesses tun das nicht.
        //
        // Ohne Projektion minimiert das innere GMRES ueber einen Unterraum, in
        // dem keine Loesung existiert: das rekursiv verfolgte Residuum sinkt,
        // das echte nicht, und beim Restart springt es. Gemessen an ein und
        // demselben Zufallsvektor:
        //
        //   nicht projiziert : 4000 Schritte, rel. Residuum 10,6
        //   projiziert       : 2742 Schritte, rel. Residuum 8,35e-05
        //
        // Zum Vergleich die physikalische rechte Seite: 2502 Schritte.
        //
        // Die Projektion ist ein linearer Operator und aendert nichts daran,
        // dass P als Ganzes linear bleibt -- Voraussetzung dafuer, dass die
        // Arnoldi-Rekursion des aeusseren GMRES ihre Grundlage behaelt.
        op_W.project_gauge_tilde_orthogonal(d_v_r);
        op_W.project_gauge_tilde_orthogonal(d_v_i);

        // z_r = (W+T)^-1 v_r  -- Startwert null, damit die Abbildung ein
        // FESTER linearer Operator bleibt. Ein Warmstart aus dem vorigen
        // Aufruf machte sie zustandsabhaengig, und dann verliert die
        // Arnoldi-Rekursion des aeusseren GMRES ihre Grundlage: die
        // Iterationszahlen saehen weiterhin plausibel aus, waeren aber
        // bedeutungslos.
        bf::memset_d(s1.data(), 0, static_cast<size_t>(nd) * sizeof(double));
        inner.solve(op_W, d_v_r, s1.data(), inner_tol, inner_max_iter);
        ++solves; iters += inner.last_iterations();
        warn_if_inner_failed(inner.last_relative_residual());

        // v_i <- v_i - (T-W) z_r,  und (T-W) z = 2*omega*C z - (W+T) z.
        //
        // (W+T) z_r ist bei EXAKTER innerer Loesung genau v_r. Der innere
        // Loeser ist aber nicht exakt, deshalb wird hier ausdruecklich
        // angewandt statt v_r eingesetzt: nur so haengt der Vorkonditionierer
        // allein von seiner Eingabe ab und bleibt ein FESTER linearer
        // Operator. Setzte man v_r ein, waere P von der Guete der inneren
        // Loesung abhaengig -- ein Unterschied, den die Iterationszahlen nicht
        // zeigen wuerden.
        H.apply_omega_C(s1.data(), s2.data());
        bf::daxpy(cublas, nd, -2.0, s2.data(), d_v_i);
        H.op_W.apply(s1.data(), s2.data());
        bf::daxpy(cublas, nd, +1.0, s2.data(), d_v_i);

        bf::dcopy(cublas, nd, s1.data(), d_v_r);

        // ERNEUT projizieren -- die Projektion am Eingang genuegt NICHT.
        //
        // Die Aktualisierung oben addiert omega*C*z_r, und omega*C ist nicht
        // eichprojiziert: die Massenmatrix traegt auf allen FEM-Kanten,
        // einschliesslich der 478 geeichten. Damit liegt d_v_i danach wieder
        // ausserhalb des Bildraums, und die ZWEITE innere Loesung bekommt
        // genau das, was die Gegenprobe als untauglich belegt hat
        // (Zufallsvektor unprojiziert: Residuum 8,3 statt 8,4e-05).
        //
        // Der (W+T)-Anteil daneben ist unkritisch, op_W.apply() projiziert
        // seinen Ausgang selbst. Es ist allein der omega*C-Term.
        op_W.project_gauge_tilde_orthogonal(d_v_i);

        bf::memset_d(s1.data(), 0, static_cast<size_t>(nd) * sizeof(double));
        inner.solve(op_W, d_v_i, s1.data(), inner_tol, inner_max_iter);
        ++solves; iters += inner.last_iterations();
        warn_if_inner_failed(inner.last_relative_residual());
        bf::dcopy(cublas, nd, s1.data(), d_v_i);
    }
};

// Reproduzierbarer Zufallsvektor. Fester Seed: eine Linearitaetspruefung, die
// von Lauf zu Lauf andere Zahlen liefert, ist als Befund wertlos.
//
// Lokal definiert -- der gleichnamige Helfer in BlockPreconditionerStudy.cpp
// liegt in einem anonymen Namensraum und ist hier nicht sichtbar. Ich hatte
// ihn zunaechst einfach aufgerufen; das waere ein Uebersetzungsfehler
// gewesen.
void fill_random(double* v, int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(-1.0, 1.0);
    for (int i = 0; i < n; ++i) v[i] = uni(rng);
}

double norm2_pair(const double* a, const double* b, int n) {
    // Endlichkeit einschliessen und ZAEHLEN statt vergleichen -- die Lehre aus
    // docs/design/nan-and-thresholds.md. Ein Vektor aus lauter NaN haette sonst die
    // Norm 0.0 und saehe wie perfekte Konvergenz aus.
    double s = 0.0;
    int non_finite = 0;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++non_finite; continue; }
        s += a[i] * a[i] + b[i] * b[i];
    }
    if (non_finite > 0) return std::numeric_limits<double>::quiet_NaN();
    return std::sqrt(s);
}

} // namespace

HarmonicSolveResult solve_harmonic(CoupledMVOperator& op_W,
                                   const CSRMatrix& csr_C,
                                   int nd, int total_dofs, double omega,
                                   const double* d_b_r, const double* d_b_i,
                                   double* d_x_r, double* d_x_i,
                                   double tol, int max_iter, int restart,
                                   double inner_tol, int inner_max_iter) {
    HarmonicSolveResult res;
    const int m = std::max(1, restart);
    const size_t N = static_cast<size_t>(nd);

    const cublasHandle_t cublas = op_W.get_cublas_handle();

    // Eigene Kopie der rechten Seite: d_b_r/d_b_i gehoeren dem Aufrufer und
    // duerfen nicht veraendert werden.
    DeviceArray<double> d_b_r_proj(static_cast<size_t>(nd));
    DeviceArray<double> d_b_i_proj(static_cast<size_t>(nd));
    bf::memcpy_d(d_b_r_proj.data(), d_b_r,
                 static_cast<size_t>(nd) * sizeof(double), cudaMemcpyDeviceToDevice);
    bf::memcpy_d(d_b_i_proj.data(), d_b_i,
                 static_cast<size_t>(nd) * sizeof(double), cudaMemcpyDeviceToDevice);
    HarmonicOperator   H(op_W, csr_C, nd, total_dofs, omega);

    // Der Restart des INNEREN Loesers ist NICHT der des aeusseren.
    //
    // m ist der aeussere Restart (64) -- der aeussere Loeser braucht laut
    // Studie 38 bis 41 Iterationen, da genuegt das. Der innere Loeser braucht
    // rund 2500 Schritte bis 1e-4, und mit Restart 64 stagniert er bei 0,027
    // statt zu konvergieren. Genau das ist im GPU-Lauf passiert: die
    // Vorabmessung (Restart 400 bzw. 3000) hat bestanden, der eigentliche
    // innere Loeser dann nicht -- weil er einen anderen, viel kuerzeren
    // Restart bekam.
    //
    // Beide Groessen an denselben Parameter zu haengen war der Fehler. Der
    // innere Restart ist deshalb so lang wie sein eigener Iterationsdeckel,
    // also praktisch volles GMRES. Bei nd = 4511 und 4000 Basisvektoren sind
    // das rund 145 MB je Basis.
    const int inner_restart = std::max(1, inner_max_iter);
    CtoRPreconditioner P(op_W, H, nd, inner_tol, inner_max_iter, inner_restart);
    std::cout << "[Harmonisch] aeusserer Restart " << m
              << ", innerer Restart " << inner_restart
              << ", innere Toleranz " << inner_tol << "\n";

    // Die RECHTE SEITE muss in den Bildraum, nicht nur der Startwert.
    //
    // Hier stand der Fehler, der das aeussere Residuum bei 0,707 festhielt:
    // der Kommentar sprach von der rechten Seite, projiziert wurden aber
    // d_x_r und d_x_i -- die Loesungsvektoren, die zu diesem Zeitpunkt null
    // sind. Ein No-Op. b wurde nie angefasst.
    //
    // Und b hat einen grossen Anteil ausserhalb des Bildraums. Der Lauf misst
    // ihn selbst:
    //
    //   ||P b||     = 486
    //   ||(I-P) b|| = 503
    //   ||b||       = 699,4     ->  503/699,4 = 0,719
    //
    // H bildet in den projizierten Raum ab (op_W.apply projiziert, und
    // apply_omega_C jetzt auch). Der unprojizierbare Anteil von b ist damit
    // fuer jedes x unerreichbar, und das relative Residuum kann nie unter
    // 0,719 fallen. Beobachtet wurden 0,707 -- das war kein Konvergenzwert,
    // sondern der Boden.
    //
    // Die Projektion von b ist keine Willkuer: der transiente Loeser tut
    // dasselbe. Das Log weist den entfernten Anteil sogar aus ("durch die
    // Eichung entfernt"). Nur der harmonische Pfad hat ihn stehen lassen.
    op_W.project_gauge_tilde_orthogonal(d_b_r_proj.data());
    op_W.project_gauge_tilde_orthogonal(d_b_i_proj.data());
    op_W.project_gauge_tilde_orthogonal(d_x_r);
    op_W.project_gauge_tilde_orthogonal(d_x_i);

    // -----------------------------------------------------------------------
    // Linearitaetspruefung des VORKONDITIONIERERS
    // -----------------------------------------------------------------------
    //
    // Das aeussere Residuum ist im Lauf GESTIEGEN (0,707 bei iter 0 auf 0,986
    // bei iter 145). Innerhalb eines GMRES-Zyklus ist das unmoeglich -- das
    // Residuum ist per Konstruktion das Minimum ueber einen wachsenden
    // Unterraum. Steigt es ueber einen Restart hinweg, stimmen das rekursiv
    // verfolgte und das neu berechnete Residuum nicht ueberein.
    //
    // Erster Verdaechtiger ist P selbst: das innere GMRES stoppt bei
    // RELATIVER Toleranz, die Abbildung "rechte Seite -> Naeherungsloesung"
    // haengt also nichtlinear von der Eingabe ab. Ein Nullstartwert allein
    // genuegt NICHT, um P linear zu machen -- das steht so falsch in einem
    // frueheren Kommentar weiter oben, und ich habe mich darauf verlassen.
    //
    // Gemessen wird direkt: P(a*u + b*v) gegen a*P(u) + b*P(v). Das ist
    // dieselbe Pruefung, die --direct fuer apply() macht (dort: 2,6e-16), nur
    // fuer den Vorkonditionierer. Sie kostet fuenf P-Anwendungen, also rund
    // zehn innere Loesungen -- gegenueber einem Lauf, der auf falscher
    // Grundlage stundenlang rechnet, ist das nichts.
    if (bf::want_harmonic_checks()) {
        std::vector<double> hu(N), hv(N);
        fill_random(hu.data(), nd, 11111u);
        fill_random(hv.data(), nd, 22222u);

        DeviceArray<double> u_r(N), u_i(N), v_r(N), v_i(N), c_r(N), c_i(N);
        auto to_dev = [&](const std::vector<double>& h, DeviceArray<double>& d) {
            bf::memcpy_d(d.data(), h.data(), N * sizeof(double), cudaMemcpyHostToDevice);
        };
        const double aa = 0.7, bb = -1.3;

        // comb = a*u + b*v, bevor irgendetwas angewandt wird
        std::vector<double> hc(N);
        for (size_t i = 0; i < N; ++i) hc[i] = aa * hu[i] + bb * hv[i];
        to_dev(hc, c_r); to_dev(hc, c_i);
        to_dev(hu, u_r); to_dev(hu, u_i);
        to_dev(hv, v_r); to_dev(hv, v_i);

        P.apply(u_r.data(), u_i.data());
        P.apply(v_r.data(), v_i.data());
        P.apply(c_r.data(), c_i.data());

        // c sollte jetzt a*P(u) + b*P(v) sein
        bf::daxpy(cublas, nd, -aa, u_r.data(), c_r.data());
        bf::daxpy(cublas, nd, -bb, v_r.data(), c_r.data());
        bf::daxpy(cublas, nd, -aa, u_i.data(), c_i.data());
        bf::daxpy(cublas, nd, -bb, v_i.data(), c_i.data());

        std::vector<double> hdr(N), hdi(N), hpr(N), hpi(N);
        bf::memcpy_d(hdr.data(), c_r.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
        bf::memcpy_d(hdi.data(), c_i.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
        bf::memcpy_d(hpr.data(), u_r.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
        bf::memcpy_d(hpi.data(), u_i.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
        const double dev = norm2_pair(hdr.data(), hdi.data(), nd);
        const double ref = norm2_pair(hpr.data(), hpi.data(), nd);
        const double rel_lin = (ref > 0.0) ? dev / (std::fabs(aa) * ref) : -1.0;

        std::cout << "[Harmonisch] Linearitaet des Vorkonditionierers: "
                  << "|P(au+bv) - aP(u) - bP(v)| / |aP(u)| = " << rel_lin << "\n";
        if (!(rel_lin < 1e-8)) {
            std::cout << "[Harmonisch]   -> P ist NICHT linear. Damit ist die "
                         "Arnoldi-Rekursion des aeusseren\n"
                         "[Harmonisch]      GMRES ohne Grundlage, und ein steigendes Residuum "
                         "ist die Folge,\n"
                         "[Harmonisch]      kein Widerspruch. Ursache ist das relative "
                         "Abbruchkriterium des\n"
                         "[Harmonisch]      inneren Loesers: andere Eingabe, andere "
                         "Iterationszahl, anderes\n"
                         "[Harmonisch]      Polynom. Abhilfe: --harmonic-inner-tol=1e-10 (macht "
                         "P praktisch\n"
                         "[Harmonisch]      exakt) oder ein richtiges FGMRES aussen.\n";
        } else {
            std::cout << "[Harmonisch]   -> P ist linear. Ein steigendes aeusseres Residuum "
                         "hat dann eine\n"
                         "[Harmonisch]      andere Ursache -- dann liegt der Fehler im "
                         "aeusseren Loeser selbst.\n";
        }
    }

    std::vector<double> h_br(N), h_bi(N);
    bf::memcpy_d(h_br.data(), d_b_r_proj.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
    bf::memcpy_d(h_bi.data(), d_b_i_proj.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
    const double b_norm = norm2_pair(h_br.data(), h_bi.data(), nd);

    // Wieviel hat die Projektion von der rechten Seite entfernt? Das ist
    // genau die Schranke, unter die das relative Residuum ohne sie nie haette
    // fallen koennen -- und im Lauf zuvor war es exakt dieser Wert (0,707).
    // Die Zahl steht hier, damit ein Rueckfall sofort sichtbar waere.
    {
        std::vector<double> h0r(N), h0i(N);
        bf::memcpy_d(h0r.data(), d_b_r, N * sizeof(double), cudaMemcpyDeviceToHost);
        bf::memcpy_d(h0i.data(), d_b_i, N * sizeof(double), cudaMemcpyDeviceToHost);
        const double b_raw = norm2_pair(h0r.data(), h0i.data(), nd);
        if (b_raw > 0.0) {
            const double removed = std::sqrt(std::max(0.0, b_raw * b_raw - b_norm * b_norm));
            std::cout << "[Harmonisch] rechte Seite: ||b|| = " << b_raw
                      << ", davon im Bildraum " << b_norm
                      << ", von der Eichung entfernt " << removed
                      << "  (relativ " << (removed / b_raw) << ")\n";
        }
    }
    if (!(b_norm > 0.0) || !std::isfinite(b_norm)) {
        std::cout << "[Harmonisch] Rechte Seite ist null oder nicht endlich -- "
                     "nichts zu loesen.\n";
        res.converged = std::isfinite(b_norm);
        res.relative_res = 0.0;
        return res;
    }

    // Arnoldi-Basis und die vorkonditionierten Richtungen. Beide als
    // Real/Imaginaer-Paar, weil der Operator auf [x_r; x_i] arbeitet.
    std::vector<DeviceArray<double>> V_r, V_i, Z_r, Z_i;
    V_r.reserve(m + 1); V_i.reserve(m + 1); Z_r.reserve(m); Z_i.reserve(m);
    for (int i = 0; i <= m; ++i) { V_r.emplace_back(N); V_i.emplace_back(N); }
    for (int i = 0; i < m; ++i)  { Z_r.emplace_back(N); Z_i.emplace_back(N); }

    DeviceArray<double> r_r(N), r_i(N), w_r(N), w_i(N);
    std::vector<double> Hm(static_cast<size_t>(m + 1) * m, 0.0);
    std::vector<double> g(static_cast<size_t>(m + 1), 0.0);
    std::vector<double> cs(static_cast<size_t>(m), 0.0), sn(static_cast<size_t>(m), 0.0);
    std::vector<double> y(static_cast<size_t>(m), 0.0);
    std::vector<double> hv_r(N), hv_i(N);

    int total = 0;
    double rel = 1.0;
    int    cycle = 0;      // Zaehler fuer die Restart-Zyklen
    bool   nonfinite = false;  // NaN/Inf im Arnoldi-Schritt, siehe unten
    double g_est = 0.0;    // Residuenschaetzung am Ende des letzten Zyklus

    while (total < max_iter) {
        H.apply(d_x_r, d_x_i, r_r.data(), r_i.data());
        bf::dscal(cublas, nd, -1.0, r_r.data());  bf::daxpy(cublas, nd, 1.0, d_b_r_proj.data(), r_r.data());
        bf::dscal(cublas, nd, -1.0, r_i.data());  bf::daxpy(cublas, nd, 1.0, d_b_i_proj.data(), r_i.data());

        bf::memcpy_d(hv_r.data(), r_r.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
        bf::memcpy_d(hv_i.data(), r_i.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
        const double beta = norm2_pair(hv_r.data(), hv_i.data(), nd);
        if (!std::isfinite(beta)) {
            std::cout << "[Harmonisch] Residuum nicht endlich -- Abbruch.\n";
            break;
        }
        rel = beta / b_norm;

        // Echtes gegen geschaetztes Residuum. beta ist hier NEU BERECHNET als
        // ||b - H x||; g_est ist der Wert, den die Givens-Drehungen des
        // vorigen Zyklus vorhergesagt haben. Weichen sie ab, ist die
        // Arnoldi-Buchfuehrung defekt -- und man weiss, in welchem Zyklus.
        //
        // Genau diese Diskrepanz erklaert einen STEIGENDEN Residuenverlauf:
        // die Kurve, die der Lauf ausgibt, ist die Schaetzung; das echte
        // Residuum wird nur beim Restart sichtbar.
        if (bf::want_harmonic_checks() && cycle > 0) {
            const double dev = (g_est > 0.0) ? std::fabs(beta - g_est) / beta : -1.0;
            std::cout << "[Harmonisch]   Zyklus " << cycle
                      << ": geschaetzt " << (g_est / b_norm)
                      << ", ECHT " << rel
                      << "   Abweichung " << dev
                      << (dev > 1e-6 ? "   <- BUCHFUEHRUNG DEFEKT" : "") << "\n";
        }
        ++cycle;

        if (rel <= tol) break;

        bf::dcopy(cublas, nd, r_r.data(), V_r[0].data());  bf::dscal(cublas, nd, 1.0 / beta, V_r[0].data());
        bf::dcopy(cublas, nd, r_i.data(), V_i[0].data());  bf::dscal(cublas, nd, 1.0 / beta, V_i[0].data());
        std::fill(g.begin(), g.end(), 0.0);  g[0] = beta;
        std::fill(Hm.begin(), Hm.end(), 0.0);

        int k = 0;
        for (; k < m && total < max_iter; ++k, ++total) {
            bf::dcopy(cublas, nd, V_r[k].data(), Z_r[k].data());
            bf::dcopy(cublas, nd, V_i[k].data(), Z_i[k].data());
            P.apply(Z_r[k].data(), Z_i[k].data());
            H.apply(Z_r[k].data(), Z_i[k].data(), w_r.data(), w_i.data());

            // Modifiziertes Gram-Schmidt auf dem Paar
            for (int i = 0; i <= k; ++i) {
                double h_a = 0.0, h_b = 0.0;
                bf::ddot(cublas, nd, V_r[i].data(), w_r.data(), &h_a);
                bf::ddot(cublas, nd, V_i[i].data(), w_i.data(), &h_b);
                const double h = h_a + h_b;
                bf::daxpy(cublas, nd, -h, V_r[i].data(), w_r.data());
                bf::daxpy(cublas, nd, -h, V_i[i].data(), w_i.data());
                Hm[static_cast<size_t>(i) * m + k] = h;
            }
            bf::memcpy_d(hv_r.data(), w_r.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
            bf::memcpy_d(hv_i.data(), w_i.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
            const double h_next = norm2_pair(hv_r.data(), hv_i.data(), nd);
            Hm[static_cast<size_t>(k + 1) * m + k] = h_next;

            // NaN/Inf aus H.apply() oder P.apply() landen in w und damit in
            // h_next. Ohne Abbruch hier fiel der Fall erst in "!(den > 0)"
            // auf, und das fuehrte mit ++k in die Aktualisierung von x --
            // x wurde NaN. So bleibt x der letzte endliche Stand.
            if (!std::isfinite(h_next)) { nonfinite = true; break; }

            // ---------------------------------------------------------------
            // Arnoldi-Relation pruefen:  H*Z_k  ==  sum_i V_i * Hm(i,k)
            // ---------------------------------------------------------------
            //
            // Das ist die Gleichung, auf der die ganze Residuenschaetzung
            // ruht. Ist sie verletzt, liegt der Fehler in MEINER Buchfuehrung
            // (Hm-Indizierung, Gram-Schmidt, Givens). Ist sie erfuellt und die
            // Kurve steigt trotzdem, liegt er im Operator oder in der
            // Loesungsaktualisierung.
            //
            // Vier Erklaerungen fuer den Anstieg habe ich erschlossen und alle
            // vier waren falsch: Nichtlinearitaet von P (widerlegt, 3,4e-10),
            // unprojizierte rechte Seite (widerlegt, entfernter Anteil 0), ein
            // Zuordnungsfehler wegen der 1/sqrt(2) (die Zahl war unauffaellig),
            // Mutation der Eingabe durch apply() (nein, const). Deshalb hier
            // eine Messung statt einer fuenften Vermutung.
            //
            // Es wird w REKONSTRUIERT: w_orig = w_rest + sum_i h_i * V_i, wobei
            // w_rest der nach Gram-Schmidt verbliebene Rest ist. Verglichen
            // wird gegen H*Z_k, das dafuer ein zweites Mal berechnet wird --
            // teuer, aber nur unter --harmonic-checks.
            if (bf::want_harmonic_checks() && k < 3) {
                DeviceArray<double> a_r(N), a_i(N);
                H.apply(Z_r[k].data(), Z_i[k].data(), a_r.data(), a_i.data());
                // a -= w_rest  und  a -= sum_i h_i V_i   ->  muss null sein
                bf::daxpy(cublas, nd, -1.0, w_r.data(), a_r.data());
                bf::daxpy(cublas, nd, -1.0, w_i.data(), a_i.data());
                for (int i = 0; i <= k; ++i) {
                    const double hi = Hm[static_cast<size_t>(i) * m + k];
                    bf::daxpy(cublas, nd, -hi, V_r[i].data(), a_r.data());
                    bf::daxpy(cublas, nd, -hi, V_i[i].data(), a_i.data());
                }
                std::vector<double> ar(N), ai(N);
                bf::memcpy_d(ar.data(), a_r.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(ai.data(), a_i.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
                const double defect = norm2_pair(ar.data(), ai.data(), nd);
                double hnorm = h_next * h_next;
                for (int i = 0; i <= k; ++i) {
                    const double hi = Hm[static_cast<size_t>(i) * m + k];
                    hnorm += hi * hi;
                }
                hnorm = std::sqrt(hnorm);
                const double relq = (hnorm > 0.0) ? defect / hnorm : -1.0;
                std::cout << "[Harmonisch]   Arnoldi k=" << k
                          << ": |H Z_k - sum V_i h_ik| / |h| = " << relq
                          << (relq > 1e-10 ? "   <- RELATION VERLETZT" : "") << "\n";
            }

            for (int i = 0; i < k; ++i) {
                const double h1 = Hm[static_cast<size_t>(i) * m + k];
                const double h2 = Hm[static_cast<size_t>(i + 1) * m + k];
                Hm[static_cast<size_t>(i) * m + k]     =  cs[i] * h1 + sn[i] * h2;
                Hm[static_cast<size_t>(i + 1) * m + k] = -sn[i] * h1 + cs[i] * h2;
            }
            const double hk  = Hm[static_cast<size_t>(k) * m + k];
            const double hk1 = Hm[static_cast<size_t>(k + 1) * m + k];
            const double den = std::hypot(hk, hk1);
            if (!(den > 0.0)) { ++k; ++total; break; }
            cs[k] = hk / den;  sn[k] = hk1 / den;
            Hm[static_cast<size_t>(k) * m + k]     = den;
            Hm[static_cast<size_t>(k + 1) * m + k] = 0.0;
            g[k + 1] = -sn[k] * g[k];
            g[k]     =  cs[k] * g[k];
            rel = std::fabs(g[k + 1]) / b_norm;

            if (h_next > 1e-14 * std::max(1.0, den)) {
                bf::dcopy(cublas, nd, w_r.data(), V_r[k + 1].data());
                bf::dscal(cublas, nd, 1.0 / h_next, V_r[k + 1].data());
                bf::dcopy(cublas, nd, w_i.data(), V_i[k + 1].data());
                bf::dscal(cublas, nd, 1.0 / h_next, V_i[k + 1].data());
            } else { ++k; ++total; break; }

            if ((total % 5) == 0 || total < 5)
                std::cout << "  [Harmonisch] iter " << total << " | Rel. Res: " << rel << "\n";
            if (rel <= tol) { ++k; ++total; break; }
        }

        if (nonfinite) {
            std::cout << "[Harmonisch] ABBRUCH: Arnoldi-Vektor nicht endlich (NaN/Inf) bei "
                         "Iteration " << total << "; x bleibt beim letzten endlichen Stand.\n";
            break;
        }

        // Schaetzung fuer den Vergleich beim naechsten Restart merken.
        g_est = (k > 0) ? std::fabs(g[k]) : beta;

        // Rueckwaertssubstitution mit RELATIVER Pivotschranke. Ohne sie wird
        // y[i] bei Stagnation unendlich und x komplett NaN -- ohne Meldung.
        double h_scale = 0.0;
        for (int i = 0; i < k; ++i)
            h_scale = std::max(h_scale, std::fabs(Hm[static_cast<size_t>(i) * m + i]));
        for (int i = k - 1; i >= 0; --i) {
            double acc = g[i];
            for (int j = i + 1; j < k; ++j) acc -= Hm[static_cast<size_t>(i) * m + j] * y[j];
            const double d = Hm[static_cast<size_t>(i) * m + i];
            y[i] = (std::isfinite(d) && std::fabs(d) > 1e-14 * h_scale) ? acc / d : 0.0;
        }
        for (int j = 0; j < k; ++j) {
            if (y[j] == 0.0) continue;
            bf::daxpy(cublas, nd, y[j], Z_r[j].data(), d_x_r);
            bf::daxpy(cublas, nd, y[j], Z_i[j].data(), d_x_i);
        }

        if (rel <= tol) break;
        if (k == 0) break;
    }

    // Das ECHTE Residuum, nicht das aus den Givens-Drehungen mitgefuehrte.
    H.apply(d_x_r, d_x_i, r_r.data(), r_i.data());
    bf::dscal(cublas, nd, -1.0, r_r.data());  bf::daxpy(cublas, nd, 1.0, d_b_r_proj.data(), r_r.data());
    bf::dscal(cublas, nd, -1.0, r_i.data());  bf::daxpy(cublas, nd, 1.0, d_b_i_proj.data(), r_i.data());
    bf::memcpy_d(hv_r.data(), r_r.data(), N * sizeof(double), cudaMemcpyDeviceToHost);
    bf::memcpy_d(hv_i.data(), r_i.data(), N * sizeof(double), cudaMemcpyDeviceToHost);

    res.iterations   = total;
    res.relative_res = norm2_pair(hv_r.data(), hv_i.data(), nd) / b_norm;
    res.converged    = std::isfinite(res.relative_res) && res.relative_res <= tol * 10.0;
    res.inner_solves = P.solves;
    res.inner_iters  = P.iters;

    std::cout << "[Harmonisch] " << res.iterations << " aeussere Iterationen, rel. Residuum "
              << res.relative_res << (res.converged ? "" : "   (NICHT konvergiert)") << "\n"
              << "[Harmonisch] innere (W+T)-Loesungen: " << res.inner_solves
              << " mit zusammen " << res.inner_iters << " Iterationen"
              << (res.inner_solves > 0
                      ? "  (im Mittel " + std::to_string(res.inner_iters / res.inner_solves) + ")"
                      : std::string())
              << "\n";
    return res;
}

} // namespace bf
