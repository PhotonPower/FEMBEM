// Volumenstrom-Biot-Savart in C++ (include/TraceReference.h) gegen die Python-Fassung tools/theory/check_volume_biot_savart.py
// (Werte dort mit rect_ring_fields erzeugt, Strom 1 A) und gegen eine unabhaengige Fadensumme.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "TraceMeshes.h"
#include "TraceReference.h"

using namespace trace;
static int g_fail = 0;
static void check(bool ok, const char* what, double value, double limit)
{
    std::printf("  [%s] %s  (%.3e %s %.3e)\n", ok ? "OK " : "FEHLER", what, value, ok ? "<" : ">=", limit);
    if (!ok) ++g_fail;
}

int main()
{
    // {rho, z, A_phi, B_rho, B_z}, Python rect_ring_fields (q = 0.3, 11 Stufen, 8x8)
    const double inner[][5] = {
        {0.0400, 0.0200, 7.201549002177664e-05, 2.574000868282773e-03, 2.320287592624677e-03},
        {0.0400, -0.0200, 2.332971803790789e-04, 1.173324864608758e-03, 6.297206678774809e-03},
        {0.0550, -0.0200, 1.861522188619043e-04, 8.844682790231434e-04, -3.379762661229786e-03},
        {0.0700, -0.0300, 1.144938427445472e-04, -3.027891599884931e-04, -1.762107672330182e-03},
        {0.1000, 0.1000, 1.223522568311862e-05, 1.784691000724629e-04, 1.109317366951344e-04},
        {0.0300, 0.0128, 8.224541702431885e-05, 3.188809419152207e-03, 4.537616913714105e-03},
        {0.0270, -0.0520, 1.257488563011089e-04, -4.927285167334339e-03, 9.501488539348145e-03},
        {0.0620, 0.0123, 8.227569292990903e-05, 2.166589434489076e-03, 7.086121150279146e-05}};
    const double outer[][5] = {
        {0.0400, 0.0200, 5.737571866762680e-05, 8.705785493551403e-04, 2.850855256544976e-03},
        {0.0400, -0.0200, 8.468651992857550e-05, 2.138493533194968e-04, 4.549788871115292e-03},
        {0.0550, -0.0200, 1.249535316527884e-04, 3.895681317690276e-04, 5.266196439532570e-03},
        {0.0700, -0.0300, 1.780515517185690e-04, -4.574021016809354e-04, 6.619627006400251e-03},
        {0.1000, 0.1000, 2.677819444698770e-05, 3.728977465400565e-04, 3.265180162239349e-04},
        {0.0300, 0.0128, 4.748547848441712e-05, 6.029241243917483e-04, 3.192923084729362e-03},
        {0.0270, -0.0520, 4.900857513454771e-05, -4.432384463881490e-04, 3.702109049060471e-03},
        {0.0620, 0.0123, 1.000454328595828e-04, 1.803571530165379e-03, 3.175248340423471e-03}};
    double worst = 0;
    for (int c = 0; c < 2; ++c) {
        const RingCoil coil = c == 0 ? RingCoil{27e-3, 55e-3, -52e-3, 0.0, 960.0, 1.0} : RingCoil{80e-3, 95e-3, -52e-3, 0.0, 576.0, 1.0};
        for (const auto& p : (c == 0 ? inner : outer)) {
            double A, Br, Bz;
            ring_fields(coil, p[0], p[1], A, Br, Bz);
            const double nrm = std::sqrt(p[3] * p[3] + p[4] * p[4]);
            worst = std::max({worst, std::fabs(Br - p[3]) / nrm, std::fabs(Bz - p[4]) / nrm, std::fabs(A - p[2]) / std::fabs(p[2])});
        }
    }
    std::printf("[R] C++ gegen Python (16 Punkte, innen/aussen, im Querschnitt, auf dem Rand, Luft, weit): max. rel. Abweichung %.3e\n", worst);
    check(worst <= 1e-9, "C++ = Python (gleiches Verfahren)", worst, 1e-9);
    // Unabhaengige Fadensumme: 300 x 300 Faeden im Querschnitt der Innenspule, Schleifenformel per direkter Summe (Linienintegral, 2000 Segmente)
    {
        const RingCoil coil{27e-3, 55e-3, -52e-3, 0.0, 960.0, 1.0};
        const double rho = 0.040, z = 0.020;
        double Br = 0, Bz = 0;
        const int n = 120, ns = 600;
        for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) {
            const double a = coil.r0 + (coil.r1 - coil.r0) * (i + 0.5) / n, zp = coil.z0 + (coil.z1 - coil.z0) * (j + 0.5) / n;
            const double w = coil.turns / (static_cast<double>(n) * n);
            for (int k = 0; k < ns; ++k) {
                const double ph = 2 * 3.14159265358979323846 * (k + 0.5) / ns;
                const double dl = a * 2 * 3.14159265358979323846 / ns;
                // dl = dl (-sin, cos, 0), R = x - src, x = (rho, 0, z)
                const double sx = a * std::cos(ph), sy = a * std::sin(ph);
                const double rx = rho - sx, ry = -sy, rz = z - zp;
                const double r3 = std::pow(rx * rx + ry * ry + rz * rz, 1.5);
                const double dx = -std::sin(ph) * dl, dy = std::cos(ph) * dl;
                const double bx = (dy * rz) , bz = (dx * ry - dy * rx);
                Br += w * 4e-7 * 3.14159265358979323846 / (4 * 3.14159265358979323846) * bx / r3;
                Bz += w * 4e-7 * 3.14159265358979323846 / (4 * 3.14159265358979323846) * bz / r3;
            }
        }
        double A, br, bz;
        ring_fields(coil, rho, z, A, br, bz);
        const double e = std::hypot(br - Br, bz - Bz) / std::hypot(br, bz);
        std::printf("     unabhaengige Fadensumme (120 x 120 Faeden, 600 Segmente) bei (rho, z) = (40, 20) mm: |B - B_Faden|/|B| = %.3e\n", e);
        check(e <= 2e-3, "Fadensumme (Selbsttest der Referenz, Diskretisierung der Summe ~1e-3)", e, 2e-3);
        // Gegenprobe: Vorzeichen von B_rho
        check(std::hypot(-br - Br, bz - Bz) / std::hypot(br, bz) > 1e-2, "Gegenprobe: B_rho mit falschem Vorzeichen faellt durch", std::hypot(-br - Br, bz - Bz) / std::hypot(br, bz), 1e-2);
    }
    std::printf(g_fail == 0 ? "ALLE TESTS BESTANDEN\n" : "FEHLGESCHLAGEN: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
