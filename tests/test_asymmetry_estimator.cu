// =============================================================================
// test_asymmetry_estimator.cu -- Schaetzer fuer ||A-A^T||_F / ||A+A^T||_F
// =============================================================================
//
// Der Loeser misst die Unsymmetrie seines Operators, ohne A^T zu besitzen. Das
// geht ueber die Identitaet: fuer x, y mit unabhaengigen Eintraegen der
// Varianz 1 gilt
//
//     y^T A x - x^T A y = y^T (A - A^T) x        und   E[(y^T M x)^2] = ||M||_F^2
//
// Der Mittelwert der Quadrate ueber mehrere Paare liefert also die
// Frobeniusnormen von A - A^T und A + A^T.
//
// Warum dieser Test noetig ist. Eine erste Fassung im Loeser verglich EIN
// Zufallspaar und meldete ein Verhaeltnis von 11.5. Diese Zahl war wertlos: der
// symmetrische Anteil ist selbst eine Zufallsgroesse und kann nahe null liegen.
// Der Test unten zeigt beides -- die Streuung des Einzelpaars und dass die
// Mittelung sie beseitigt.
//
// Braucht keine GPU.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
void ck(bool ok, const char* w) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", w);
    if (!ok) ++g_fail;
}

unsigned g_seed = 12345u;
double rnd() {
    g_seed ^= g_seed << 13; g_seed ^= g_seed >> 17; g_seed ^= g_seed << 5;
    return ((g_seed >> 8) & 0xffff) / 32767.5 - 1.0;
}

/// Baut A = S + alpha*K mit S symmetrisch und K antisymmetrisch, beide aus
/// derselben Zufallsmatrix, und gibt die WAHRE Kennzahl zurueck.
double build(int n, double alpha, std::vector<double>& A) {
    std::vector<double> R(static_cast<size_t>(n) * n);
    for (auto& v : R) v = rnd();
    A.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double sym  = 0.5 * (R[i*n+j] + R[j*n+i]);
            const double anti = 0.5 * (R[i*n+j] - R[j*n+i]);
            A[i*n+j] = sym + alpha * anti;
        }
    double d2 = 0.0, s2 = 0.0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            const double d = A[i*n+j] - A[j*n+i];
            const double s = A[i*n+j] + A[j*n+i];
            d2 += d*d; s2 += s*s;
        }
    return std::sqrt(d2) / std::sqrt(s2);
}

/// Der Schaetzer, wie im Loeser: nur ueber Anwendungen von A.
double estimate(const std::vector<double>& A, int n, int n_pairs) {
    double sd = 0.0, ss = 0.0;
    std::vector<double> x(n), y(n), Ax(n), Ay(n);
    for (int r = 0; r < n_pairs; ++r) {
        for (int i = 0; i < n; ++i) { x[i] = rnd(); y[i] = rnd(); }
        for (int i = 0; i < n; ++i) {
            double a = 0.0, b = 0.0;
            for (int j = 0; j < n; ++j) { a += A[i*n+j]*x[j]; b += A[i*n+j]*y[j]; }
            Ax[i] = a; Ay[i] = b;
        }
        double p = 0.0, q = 0.0;
        for (int i = 0; i < n; ++i) { p += y[i]*Ax[i]; q += x[i]*Ay[i]; }
        sd += (p-q)*(p-q); ss += (p+q)*(p+q);
    }
    return std::sqrt(sd / n_pairs) / std::sqrt(ss / n_pairs);
}

} // namespace

int main() {
    std::puts("========================================================");
    std::puts(" Schaetzer fuer die Unsymmetrie eines Operators          ");
    std::puts("========================================================\n");
    const int n = 200;

    // Zur Genauigkeit: der Schaetzer mittelt Quadrate von Zufallsgroessen, sein
    // relativer Standardfehler faellt daher nur mit 1/sqrt(n_pairs). Bei 64
    // Paaren sind rund 20 bis 30 % zu erwarten, einzelne Ziehungen liegen auch
    // bei 50 %. Das ist fuer den Zweck ausreichend: gefragt ist, ob die
    // Unsymmetrie bei null oder bei 0.1 bis 1 liegt, nicht ihre dritte Stelle.
    // Die Schranke unten ist entsprechend gesetzt und nicht enger, als der
    // Schaetzer hergibt.
    std::printf("%-10s%16s%16s%14s\n", "alpha", "wahr", "geschaetzt(64)", "Abweichung");
    for (double alpha : {0.0, 0.01, 0.1, 0.5, 1.0}) {
        std::vector<double> A;
        const double wahr = build(n, alpha, A);
        const double est  = estimate(A, n, 64);
        const double dev  = (wahr > 1e-14) ? std::fabs(est/wahr - 1.0) : est;
        std::printf("%-10.2f%16.6f%16.6f%13.1f %%\n", alpha, wahr, est, dev*100);
        if (alpha == 0.0) ck(est < 1e-12, "symmetrische Matrix -> Schaetzer null");
        else              ck(dev < 0.60,  "Schaetzer trifft die Groessenordnung");
    }

    std::puts("\n  Zur Streuung des EINZELPAARS (das war der urspruengliche Fehler):");
    {
        std::vector<double> A;
        const double wahr = build(n, 0.1, A);
        double mn = 1e300, mx = 0.0;
        for (int r = 0; r < 200; ++r) {
            const double e = estimate(A, n, 1);
            mn = std::fmin(mn, e); mx = std::fmax(mx, e);
        }
        std::printf("      wahr = %.4f,  Einzelpaar streut von %.4f bis %.2f\n", wahr, mn, mx);
        ck(mx > 5.0 * wahr, "Einzelpaar streut tatsaechlich um Groessenordnungen");
        ck(std::fabs(estimate(A, n, 64)/wahr - 1.0) < 0.60,
           "die Mittelung ueber 64 Paare beseitigt die Streuung");
    }

    std::puts("\n========================================================");
    if (g_fail == 0) { std::puts(" ALLE PRUEFUNGEN BESTANDEN");
                       std::puts("========================================================"); return 0; }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
