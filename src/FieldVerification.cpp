#include "FieldVerification.cuh"

#include "BiotSavart.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>

double report_field_verification(const char* label,
                                 std::vector<FieldSample> samples,
                                 double current,
                                 double r_ring,
                                 double residual_power)
{
    std::sort(samples.begin(), samples.end(),
              [](const FieldSample& a, const FieldSample& b) { return a.z < b.z; });

    std::printf("\n========================================================\n");
    std::printf(" PRUEFUNG B_z AUF DER ACHSE GEGEN BIOT-SAVART            \n");
    std::printf(" Pfad: %s,  Spulenstrom %.3f A\n", label, current);
    std::printf("========================================================\n");
    std::printf(" Innerster Elementring bei r = %.3f mm, %zu Auswertepunkte\n\n",
                r_ring * 1000.0, samples.size());
    std::printf("   z [mm]   B_z Loeser [T]    B_z analytisch [T]   rel. Abweichung\n");

    double worst = 0.0;
    std::vector<double> num, ana;
    for (const FieldSample& s : samples) {
        const double ref = biot_savart::team28_bz_on_axis(s.z, current);
        const double rel = (std::fabs(ref) > 0.0) ? (s.bz - ref) / std::fabs(ref) : 0.0;
        worst = std::fmax(worst, std::fabs(rel));
        num.push_back(s.bz);
        ana.push_back(ref);
        std::printf("   %6.3f   %+.6e     %+.6e        %+7.2f %%\n",
                    s.z * 1000.0, s.bz, ref, rel * 100.0);
    }

    // Das Abklingverhaeltnis ist die empfindlichere Groesse: eine falsche
    // Kopplung an den Aussenraum zeigt sich zuerst im GRADIENTEN, nicht im
    // Absolutwert.
    if (num.size() >= 2 && std::fabs(num.front()) > 0.0 && std::fabs(ana.front()) > 0.0) {
        const double g_num = num.back() / num.front();
        const double g_ana = ana.back() / ana.front();
        std::printf("\n   Abklingverhaeltnis B_z(oben)/B_z(unten):  Loeser %.4f,"
                    "  analytisch %.4f   Abweichung %+.2f %%\n",
                    g_num, g_ana, (g_num / g_ana - 1.0) * 100.0);
    }

    if (residual_power >= 0.0) {
        std::printf("\n   Restliche Verlustleistung: %.3e W\n", residual_power);
        if (residual_power > 1e-3) {
            std::printf("   -> WARNUNG: die L/R-Transiente ist NICHT abgeklungen. Der\n"
                        "      Vergleich setzt J = 0 voraus; mit Restwirbelstroemen ist\n"
                        "      er bedeutungslos. Mehr Schritte rechnen.\n");
        }
    }

    std::printf("\n   Groesste Abweichung im Betrag: %.2f %%\n", worst * 100.0);
    if (worst < 0.05) {
        std::printf("   -> Quellfeld und Aussenraumkopplung sind sauber. Ein\n"
                    "      Kraftfehler dieses Pfads sitzt im Wirbelstrom- oder\n"
                    "      Kraftteil, nicht im Feldloeser.\n");
    } else {
        std::printf("   -> Das QUELLFELD stimmt schon nicht. Alles Weitere\n"
                    "      (Formulierung, Elementordnung, Kraftschaetzer) ist damit\n"
                    "      zweitrangig -- hier zuerst ansetzen.\n");
    }
    std::printf("========================================================\n\n");
    return worst;
}
