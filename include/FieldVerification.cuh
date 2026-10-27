#ifndef FIELD_VERIFICATION_CUH
#define FIELD_VERIFICATION_CUH

// =============================================================================
// FieldVerification.cuh -- gemeinsamer Vergleich B_z(Achse) gegen Biot-Savart
// =============================================================================
//
// Wird von BEIDEN Loeserpfaden benutzt (--verify-field):
//   * Referenzpfad: B aus Q1-Knotengraden, B = sum grad N_i x A_i
//   * produktiver Pfad: B aus Nedelec-Kantengraden, mit Piola-Transformation
//     B = sum a_e (J * curl W_e) / detJ
//
// Der Vergleich selbst und die Ausgabe liegen hier, damit beide Pfade
// buchstaeblich dieselbe Tabelle drucken und dieselben Auswertepunkte benutzen.
// Waeren es zwei Fassungen, wuerde ein Unterschied in der Auswertung als
// Unterschied in der Loesung erscheinen -- genau der Fehler, den diese Pruefung
// ausschliessen soll.
//
// -----------------------------------------------------------------------------
// Was die Probe prueft, und warum sigma = 0
// -----------------------------------------------------------------------------
// Mit sigma = 0 in allen Elementen gibt es keine Wirbelstroeme. Uebrig bleibt
// das reine Quellfeld der Spulen, und dafuer gibt es eine analytische Loesung
// (BiotSavart.cuh). Geprueft werden damit in einem Zug: Quellstromdichte samt
// azimutaler Richtung, die Assemblierung des Operators und die
// Aussenraumkopplung -- ohne dass Wirbelstromphysik etwas verschleiern kann.
//
// WICHTIG fuer den produktiven Pfad: sigma muss VOR der Baum-Kobaum-Eichung
// genullt werden. Die Eichung eicht nur NICHTleitende Kanten und laesst
// leitende frei, weil dort der Term sigma dA/dt die Eichung fixiert (siehe
// TreeCotreeGauge.h). Mit sigma = 0 sind alle Kanten nichtleitend und werden
// geeicht -- das System ist dann vollstaendig geeicht und besser gestellt als im
// transienten Fall. Wuerde sigma erst nach der Eichung genullt, blieben die
// 8375 "leitenden" Kanten ungeeicht und das System waere singulaer.
//
// -----------------------------------------------------------------------------
// Auswertestelle
// -----------------------------------------------------------------------------
// Die Platte ist eine Vollscheibe ohne Bohrung (TEAM 28 hat kein Loch). Im
// Standardnetz ueberdeckt je Lage eine Mittelzelle die Achse; es liegt aber
// kein Knoten genau auf ihr (der naechste bei r = 1.8436 mm). Ausgewertet wird
// in den achsennaechsten Plattenelementen, deren Schwerpunkte bei r = 1.615 mm
// liegen; dort weicht B_z um O((r/L)^2), also etwa 0.2 %, vom Achsenwert ab.
// Gegenueber den gesuchten Abweichungen von mehreren Prozent ist das
// unerheblich. (Fruehere Fassungen nannten hier eine "zentrale Bohrung" --
// das war eine Fehldeutung des kleinsten Knotenradius.)
// =============================================================================

#include <cstddef>
#include <vector>

/// Ein Auswertepunkt: Lage und der vom Loeser gerechnete B_z.
struct FieldSample {
    double r;
    double z;
    double bz;
};

/// Stellt die Werte gegen biot_savart::team28_bz_on_axis(z, current) und gibt
/// die Tabelle samt Bewertung aus. 'label' benennt den Pfad ("Knoten-FEM" bzw.
/// "Nedelec"), 'current' ist der Momentanwert des Spulenstroms in Ampere.
///
/// Rueckgabe: groesste relative Abweichung im Betrag.
/// 'residual_power' ist die noch verbleibende ohmsche Verlustleistung in W
/// (negativ = nicht verfuegbar). Sie ist die GUELTIGKEITSPRUEFUNG des Tests:
/// die Probe setzt voraus, dass die L/R-Transiente abgeklungen und J = 0 ist.
/// Ist die Restleistung nicht verschwindend klein, vergleicht man nicht mit dem
/// statischen Feld und das Ergebnis ist bedeutungslos.
double report_field_verification(const char* label,
                                 std::vector<FieldSample> samples,
                                 double current,
                                 double r_ring,
                                 double residual_power = -1.0);

#endif // FIELD_VERIFICATION_CUH
