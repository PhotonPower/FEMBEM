#ifndef TRACE_REFERENCE_H
#define TRACE_REFERENCE_H

// Volumenstrom-Biot-Savart fuer Ringspulen mit Rechteckquerschnitt (Referenz der Phase 2). C++-Port des Verfahrens aus
// tools/theory/check_volume_biot_savart.py (Phase 1a): Kreisschleifenformel mit vollstaendigen elliptischen Integralen (K ueber das AGM
// mit p = 1 - m, auch bei m -> 1 genau), Querschnittsintegral als Produkt-Gauss (8 x 8) auf zum Aufpunkt hin geometrisch verdichteten Panels
// (Faktor 0.3, 11 Stufen). Gewaehlt statt Nachauswertung in Python: gleiche Punkte (Elementmitten, Luftpunkte) und Einheiten ohne Dateiumweg,
// und die Referenz laeuft im selben Treiber wie die Rechnung. Genauigkeit: Test gegen die Python-Fassung und deren Pruefungen (B1 bis B5, ~1e-7).
#include "TraceSurface.h"

namespace trace {

struct RingCoil {
    double r0, r1, z0, z1;   // Querschnitt [r0,r1] x [z0,z1] in Metern
    double turns;            // Windungszahl
    double current;          // Strom je Windung in A (Vorzeichen: + = e_phi)
};

// (A_phi, B_rho, B_z) der Spule am Aufpunkt (rho > 0, z)
void ring_fields(const RingCoil& c, double rho, double z, double& Aphi, double& Brho, double& Bz);

// B und A in kartesischen Komponenten (Aufpunkt x) fuer eine Liste von Spulen (Achse = z-Achse durch den Ursprung)
Vec3 ring_B(const std::vector<RingCoil>& coils, const Vec3& x);
Vec3 ring_A(const std::vector<RingCoil>& coils, const Vec3& x);

} // namespace trace

#endif
