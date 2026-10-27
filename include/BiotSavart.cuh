#ifndef BIOT_SAVART_CUH
#define BIOT_SAVART_CUH

// =============================================================================
// BiotSavart.cuh -- analytische Referenz fuer das Quellfeld der Spulen
// =============================================================================
//
// Zweck: eine von FEM, BEM und Wirbelstromphysik VOLLSTAENDIG unabhaengige
// Vergleichsgroesse. Wenn der Loeser diese Zahlen nicht trifft, ist jede
// Diskussion ueber Formulierungen und Elementordnungen Kaffeesatz -- dann sitzt
// der Fehler im Quellterm, in der Assemblierung oder in der Aussenraumkopplung,
// und zwar ohne dass Wirbelstroeme irgendetwas verschleiern.
//
// -----------------------------------------------------------------------------
// Herleitung
// -----------------------------------------------------------------------------
// Auf der Achse einer Kreisschleife mit Radius r, Strom I, axialem Abstand h:
//
//     B_z = mu_0 I r^2 / (2 (r^2 + h^2)^(3/2))
//
// Eine Spule mit Rechteckquerschnitt (r in [r1,r2], z in [z1,z2]) und
// homogener azimutaler Stromdichte J ergibt daraus
//
//     B_z(z) = (mu_0 J / 2) Int_{z1}^{z2} Int_{r1}^{r2}
//                  r^2 / (r^2 + (z-zeta)^2)^(3/2)  dr  dzeta
//
// Das innere Integral ist geschlossen loesbar:
//
//     Int r^2/(r^2+h^2)^(3/2) dr = ln(r + sqrt(r^2+h^2)) - r/sqrt(r^2+h^2)
//
// Nachrechnen: d/dr ln(r+sqrt(r^2+h^2)) = 1/sqrt(r^2+h^2), und
// d/dr [r/sqrt(r^2+h^2)] = h^2/(r^2+h^2)^(3/2). Die Differenz ist
// (r^2+h^2-h^2)/(r^2+h^2)^(3/2) = r^2/(r^2+h^2)^(3/2). Stimmt.
//
// Uebrig bleibt ein eindimensionales Integral ueber zeta. Der Integrand ist
// glatt -- auch bei h = 0, denn er haengt nur ueber h^2 von zeta ab -- und wird
// mit Gauss-Legendre ausgewertet. Bei r1 > 0 (hier 27 bzw. 80 mm) ist er
// zusaetzlich ueberall beschraenkt; nur fuer r1 = 0 wuerde ln(|h|) divergieren.
//
// Damit ist die Referenz auf Maschinengenauigkeit exakt, ohne Netz und ohne
// Loeser.
// =============================================================================

namespace biot_savart {

/// B_z auf der Achse (r = 0) bei Hoehe z, erzeugt von EINER Ringspule mit
/// Rechteckquerschnitt und homogener azimutaler Stromdichte.
///
///   r1, r2   innerer und aeusserer Radius [m]
///   z1, z2   untere und obere Kante [m]
///   J        Stromdichte [A/m^2], Vorzeichen = Umlaufrichtung
///   n_gauss  Stuetzstellen fuer das zeta-Integral (Standard reicht weit aus)
///
/// Rueckgabe in Tesla.
double coil_bz_on_axis(double r1, double r2, double z1, double z2,
                       double J, double z, int n_gauss = 200);

/// Summe ueber die beiden TEAM-28-Spulen, mit den Werten aus der
/// Aufgabenstellung (Fig. 2, am Netz bestaetigt):
///   Innenspule w1: r 27..55 mm, 960 Windungen
///   Aussenspule w2: r 80..95 mm, 576 Windungen, GEGENSINNIG
///   beide z -52..0 mm
/// 'current' ist der Momentanwert des Spulenstroms [A].
double team28_bz_on_axis(double z, double current);

} // namespace biot_savart

#endif // BIOT_SAVART_CUH
