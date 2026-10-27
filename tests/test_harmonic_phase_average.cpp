// =============================================================================
// Selbstpruefung der Zeitmittelung des harmonischen Loesers
// =============================================================================
//
// Der ganze harmonische Weg steht und faellt damit, dass die zeitgemittelte
// Kraft stimmt -- daraus folgt die Schwebehoehe, und die ist die
// Benchmarkgroesse von TEAM 28.
//
// Gemittelt wird ueber N Phasen des reellen Kraftpfads statt den
// Maxwell-Tensor komplex neu zu schreiben. Das ist exakt, WENN zwei Dinge
// stimmen: das Vorzeichen in x(phi) = x_r cos(phi) - x_i sin(phi), und dass N
// Stuetzstellen eine quadratische Groesse tatsaechlich exakt mitteln.
//
// Beides faellt bei einem Fehler NICHT auf: es gibt keinen Absturz und kein
// NaN, sondern eine Kraft, die um einen systematischen Betrag danebenliegt.
// Genau deshalb steht dieser Test hier.
//
// Baubar ohne CUDA:
//   g++ -std=c++17 -O2 -Iinclude tests/test_harmonic_phase_average.cpp
//       src/HarmonicPhaseAverage.cpp -o test_harmonic_phase_average

#include "HarmonicPhaseAverage.h"

#include <cmath>
#include <string>   // std::string / std::to_string: MSVC zieht das
                    // NICHT ueber andere Header mit, GCC schon.
#include <iostream>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
    std::cout << (ok ? "  [ok]   " : "  [FEHL] ") << what << "\n";
    if (!ok) ++g_failures;
}

void test_sign_convention() {
    std::cout << "\nTest 1: Vorzeichenkonvention x(phi) = x_r cos(phi) - x_i sin(phi)\n";
    const std::vector<double> xr{1.0, 0.0, 2.0};
    const std::vector<double> xi{0.0, 1.0, -3.0};
    std::vector<double> out;

    bf::harmonic_phase_sample(xr, xi, 0.0, out);
    check(std::fabs(out[0] - 1.0) < 1e-15 && std::fabs(out[1] - 0.0) < 1e-15,
          "bei phi = 0 kommt der Realteil heraus");

    // Bei phi = pi/2 ist cos = 0, sin = 1, also x = -x_i.
    // Ein Pluszeichen im Code gaebe hier +x_i -- die konjugierte Loesung,
    // physikalisch eine Phasenumkehr.
    bf::harmonic_phase_sample(xr, xi, std::acos(-1.0) / 2.0, out);
    std::cout << "    phi = pi/2:  erwartet " << -xi[1] << ", gemessen " << out[1] << "\n";
    check(std::fabs(out[1] - (-xi[1])) < 1e-15,
          "bei phi = pi/2 kommt MINUS der Imaginaerteil heraus");
    check(std::fabs(out[2] - (-xi[2])) < 1e-15,
          "auch fuer negative Imaginaerteile stimmt das Vorzeichen");

    bool threw = false;
    try { bf::harmonic_phase_sample(xr, std::vector<double>{1.0}, 0.0, out); }
    catch (const std::exception&) { threw = true; }
    check(threw, "verschieden lange Vektoren werden abgelehnt");
}

void test_quadratic_average_is_exact() {
    std::cout << "\nTest 2: das Mittel ueber N Phasen ist fuer QUADRATISCHE Groessen exakt\n";
    std::cout << "        (die Kraft ist quadratisch in den Feldern -- daran haengt alles)\n";

    std::mt19937 rng(20260821u);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int n = 40;
    std::vector<double> ar(n), ai(n), br(n), bi(n);
    for (int i = 0; i < n; ++i) { ar[i] = nd(rng); ai[i] = nd(rng); br[i] = nd(rng); bi[i] = nd(rng); }

    // Analytischer Zeitmittelwert von Re(a e^{jwt}) * Re(b e^{jwt}) ueber eine
    // Periode: 1/2 * Re(a * conj(b)) = 1/2 (a_r b_r + a_i b_i).
    double exact = 0.0;
    for (int i = 0; i < n; ++i) exact += 0.5 * (ar[i] * br[i] + ai[i] * bi[i]);

    for (int N : {3, 4, 8, 16}) {
        const std::vector<double> phases = bf::harmonic_phases(N);
        std::vector<double> a_t, b_t;
        double avg = 0.0;
        for (double phi : phases) {
            bf::harmonic_phase_sample(ar, ai, phi, a_t);
            bf::harmonic_phase_sample(br, bi, phi, b_t);
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += a_t[i] * b_t[i];
            avg += s;
        }
        avg /= static_cast<double>(N);
        const double rel = std::fabs(avg - exact) / std::fabs(exact);
        std::cout << "    N = " << N << ":  analytisch " << exact
                  << ", gemittelt " << avg << ",  Abweichung " << rel << "\n";
        check(rel < 1e-13, "N = " + std::to_string(N) + " mittelt exakt");
    }

    // Die Gegenprobe: mit ZWEI Phasen darf es NICHT exakt sein. Ohne sie
    // bestuende auch eine Mittelung, die den Zeitverlauf gar nicht ansieht.
    // (harmonic_phases() lehnt N < 3 ab, deshalb hier von Hand.)
    {
        const double pi = std::acos(-1.0);
        double avg = 0.0;
        std::vector<double> a_t, b_t;
        for (double phi : {0.0, pi}) {
            bf::harmonic_phase_sample(ar, ai, phi, a_t);
            bf::harmonic_phase_sample(br, bi, phi, b_t);
            double s = 0.0;
            for (int i = 0; i < n; ++i) s += a_t[i] * b_t[i];
            avg += s;
        }
        avg *= 0.5;
        const double rel = std::fabs(avg - exact) / std::fabs(exact);
        std::cout << "    N = 2 (Gegenprobe): Abweichung " << rel << "\n";
        check(rel > 1e-3, "zwei Phasen reichen NICHT -- die Probe ist also wirksam");
    }

    bool threw = false;
    try { bf::harmonic_phases(2); } catch (const std::exception&) { threw = true; }
    check(threw, "harmonic_phases lehnt N < 3 ab");
}

void test_phases_are_uniform() {
    std::cout << "\nTest 3: die Phasen ueberdecken genau eine Periode\n";
    const std::vector<double> p = bf::harmonic_phases(8);
    const double two_pi = 6.283185307179586476925286766559;
    bool ok = (p.size() == 8) && std::fabs(p[0]) < 1e-15;
    for (size_t i = 1; i < p.size() && ok; ++i)
        if (std::fabs((p[i] - p[i-1]) - two_pi / 8.0) > 1e-14) ok = false;
    check(ok, "8 gleichverteilte Phasen von 0 bis vor 2*pi");
}

void test_derivative() {
    std::cout << "\nTest 4: analytische Zeitableitung gegen Differenzenquotient\n";
    std::cout << "        (die Formel ist gegenueber dem Momentanwert VERTAUSCHT --\n";
    std::cout << "         sin gehoert zum Realteil, cos zum Imaginaerteil)\n";

    std::mt19937 rng(97531u);
    std::normal_distribution<double> nd(0.0, 1.0);
    const int n = 25;
    std::vector<double> xr(n), xi(n);
    for (int i = 0; i < n; ++i) { xr[i] = nd(rng); xi[i] = nd(rng); }

    const double omega = 2.0 * std::acos(-1.0) * 50.0;
    const double phi   = 0.937;
    std::vector<double> d_ana, a_plus, a_minus;
    bf::harmonic_phase_derivative(xr, xi, phi, omega, d_ana);

    // Zentraler Differenzenquotient in der ZEIT: t = phi/omega, also
    // dphi = omega*dt.
    const double dt = 1e-7;
    bf::harmonic_phase_sample(xr, xi, phi + omega * dt, a_plus);
    bf::harmonic_phase_sample(xr, xi, phi - omega * dt, a_minus);

    double err = 0.0, ref = 0.0;
    for (int i = 0; i < n; ++i) {
        const double fd = (a_plus[i] - a_minus[i]) / (2.0 * dt);
        err += (fd - d_ana[i]) * (fd - d_ana[i]);
        ref += d_ana[i] * d_ana[i];
    }
    const double rel = std::sqrt(err / ref);
    std::cout << "    rel. Abweichung = " << rel << "\n";
    check(rel < 1e-6, "analytische Ableitung stimmt mit dem Differenzenquotienten");

    // Gegenprobe: die FALSCHE Formel (wie harmonic_phase_sample) muss deutlich
    // danebenliegen. Ohne sie bestuende auch eine Ableitung, die zufaellig
    // aehnlich aussieht.
    double err2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double falsch = -omega * (xr[i] * std::cos(phi) - xi[i] * std::sin(phi));
        err2 += (falsch - d_ana[i]) * (falsch - d_ana[i]);
    }
    std::cout << "    mit der Formel des Momentanwerts: " << std::sqrt(err2 / ref) << "\n";
    check(std::sqrt(err2 / ref) > 0.1,
          "die vertauschte Formel liegt deutlich daneben -- die Probe ist wirksam");
}

} // namespace

int main() {
    std::cout << "=== Selbstpruefung: Zeitmittelung des harmonischen Loesers ===\n";
    try {
        test_sign_convention();
        test_quadratic_average_is_exact();
        test_phases_are_uniform();
        test_derivative();
    } catch (const std::exception& ex) {
        std::cout << "\nABBRUCH mit Ausnahme: " << ex.what() << "\n";
        return 1;
    }
    std::cout << "\n";
    if (g_failures == 0) { std::cout << "Alle Pruefungen bestanden.\n"; return 0; }
    std::cout << g_failures << " Pruefung(en) fehlgeschlagen.\n";
    return 1;
}
