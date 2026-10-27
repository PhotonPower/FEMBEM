// =============================================================================
// Zeitmittelung ueber Phasen -- der Host-Anteil des harmonischen Loesers
// =============================================================================
//
// Bewusst eine eigene .cpp ohne CUDA-Abhaengigkeit: das ist der Teil, an dem
// ein Vorzeichenfehler still bliebe. Ein falsches Vorzeichen in
//
//     x(t) = x_r cos(wt) - x_i sin(wt)
//
// erzeugt keinen Absturz und keine NaN, sondern eine Kraft, die um einen
// systematischen Betrag danebenliegt -- und die Schwebehoehe mit ihr. Hier ist
// er uebersetzbar und pruefbar (tests/test_harmonic_phase_average.cpp).

#include "HarmonicPhaseAverage.h"

#include <cmath>
#include <string>
#include <vector>   // std::string / std::to_string: MSVC zieht das
                    // NICHT ueber andere Header mit, GCC schon.
#include <stdexcept>

namespace bf {

void harmonic_phase_sample(const std::vector<double>& x_r,
                           const std::vector<double>& x_i,
                           double phi, std::vector<double>& out) {
    if (x_r.size() != x_i.size())
        throw std::runtime_error("harmonic_phase_sample: Real- und Imaginaerteil "
                                 "verschieden lang (" + std::to_string(x_r.size())
                                 + " gegen " + std::to_string(x_i.size()) + ").");
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    out.resize(x_r.size());
    // MINUS beim Sinus. Das folgt aus Re(x^ e^{j phi}) mit x^ = x_r + j x_i:
    //   Re((x_r + j x_i)(cos + j sin)) = x_r cos - x_i sin
    // Ein Plus hier ergaebe die konjugierte Loesung -- physikalisch eine
    // Phasenumkehr, und die Kraft laege systematisch daneben, ohne dass
    // irgendetwas auffiele.
    for (size_t k = 0; k < x_r.size(); ++k) out[k] = x_r[k] * c - x_i[k] * s;
}

void harmonic_phase_derivative(const std::vector<double>& x_r,
                               const std::vector<double>& x_i,
                               double phi, double omega,
                               std::vector<double>& out) {
    if (x_r.size() != x_i.size())
        throw std::runtime_error("harmonic_phase_derivative: Real- und Imaginaerteil "
                                 "verschieden lang.");
    // d/dt [ x_r cos(wt) - x_i sin(wt) ] = -w [ x_r sin(wt) + x_i cos(wt) ]
    //
    // Das MINUS und die Zuordnung sin<->x_r, cos<->x_i sind gegenueber
    // harmonic_phase_sample() VERTAUSCHT -- genau so, wie es die Ableitung
    // verlangt. Wer hier aus Gewohnheit dieselbe Formel schreibt, bekommt eine
    // um 90 Grad verschobene Stromdichte und damit eine Kraft, die um denselben
    // Winkel falsch liegt. Test 4 prueft das gegen einen Differenzenquotienten.
    const double c = std::cos(phi);
    const double s = std::sin(phi);
    out.resize(x_r.size());
    for (size_t k = 0; k < x_r.size(); ++k)
        out[k] = -omega * (x_r[k] * s + x_i[k] * c);
}

std::vector<double> harmonic_phases(int n_phases) {
    if (n_phases < 3)
        throw std::runtime_error("harmonic_phases: mindestens 3 Phasen noetig, um eine "
                                 "quadratische Groesse exakt zu mitteln (angefordert: "
                                 + std::to_string(n_phases) + ").");
    std::vector<double> phi(static_cast<size_t>(n_phases));
    const double two_pi = 6.283185307179586476925286766559;
    for (int k = 0; k < n_phases; ++k)
        phi[static_cast<size_t>(k)] = two_pi * static_cast<double>(k)
                                    / static_cast<double>(n_phases);
    return phi;
}

} // namespace bf
