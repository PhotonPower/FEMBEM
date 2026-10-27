#ifndef HARMONIC_PHASE_AVERAGE_H
#define HARMONIC_PHASE_AVERAGE_H

// =============================================================================
// Zeitmittelung ueber Phasen -- REINER HOST-HEADER, ohne CUDA
// =============================================================================
//
// Bewusst von HarmonicSolver.cuh getrennt. Jener zieht ueber fem.cuh das
// cuda_runtime.h herein, und damit laesst sich der Host-Test nicht mehr als
// CXX-Ziel uebersetzen -- genau daran ist der erste Bauversuch gescheitert.
//
// Die Trennung ist kein Notbehelf: die Zeitmittelung ist der Teil, an dem ein
// Vorzeichenfehler still bliebe (kein Absturz, kein NaN, nur eine
// systematisch falsche Kraft und damit eine falsche Schwebehoehe). Genau
// dieser Teil soll ohne GPU-Umgebung uebersetzbar und pruefbar sein.
//
// ZUR SACHE
//
// Die Kraft ist QUADRATISCH in den Feldern. Aus dem Phasor A^ folgt der
// Momentanwert A(t) = Re(A^ e^{j w t}) = A_r cos(wt) - A_i sin(wt), und der
// Zeitmittelwert einer quadratischen Groesse ueber eine Periode ist exakt das
// arithmetische Mittel ueber N >= 3 gleichverteilte Phasen.
//
// Deshalb wird der Maxwell-Tensor NICHT komplex neu geschrieben: der
// vorhandene, reelle Kraftpfad wird N-mal mit verschiedenen Phasen aufgerufen
// und gemittelt. Exakt, und ohne eine einzige Zeile neuer Physik.
//
// N = 8 statt der theoretisch genuegenden 3, weil der Kraftpfad ueber
// Flaechenintegration und Interpolation laeuft und dabei Anteile hoeherer
// Ordnung entstehen koennen, die von 3 Stuetzstellen nicht mehr exakt
// gemittelt wuerden.

#include <vector>

namespace bf {

/// Momentanwert bei Phase phi: x(phi) = x_r cos(phi) - x_i sin(phi).
void harmonic_phase_sample(const std::vector<double>& x_r,
                           const std::vector<double>& x_i,
                           double phi, std::vector<double>& out);

/// Die N Phasen, ueber die gemittelt wird: phi_k = 2*pi*k/N.
std::vector<double> harmonic_phases(int n_phases);

/// Zeitableitung bei Phase phi, ANALYTISCH:
///
///     x(t)   = x_r cos(wt) - x_i sin(wt)
///     dx/dt  = -w ( x_r sin(wt) + x_i cos(wt) )
///
/// Gebraucht fuer das Lorentz-Volumenintegral: dessen Kernel bildet die
/// Stromdichte als J = -sigma (A - A_prev)/dt, also ueber einen
/// Differenzenquotienten. Im harmonischen Fall ist die Ableitung exakt
/// bekannt -- ein Differenzenquotient waere hier nur eine Naeherung an
/// etwas, das man geschlossen hinschreiben kann.
void harmonic_phase_derivative(const std::vector<double>& x_r,
                               const std::vector<double>& x_i,
                               double phi, double omega,
                               std::vector<double>& out);

} // namespace bf

#endif // HARMONIC_PHASE_AVERAGE_H
