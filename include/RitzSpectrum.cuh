#ifndef RITZ_SPECTRUM_CUH
#define RITZ_SPECTRUM_CUH

// =============================================================================
// RitzSpectrum.cuh -- Konditionsschaetzung aus dem Arnoldi-Prozess
// =============================================================================
//
// Wozu. Nach mehreren behobenen Fehlern faellt das Residuum des produktiven
// Pfads von 0.112 auf 0.0401, konvergiert aber weiterhin nicht. Die Kurve
// faellt dabei stetig ohne Plateau -- das Bild eines schlecht konditionierten,
// nicht eines singulaeren Systems. Bevor weitere Einzelverdaechtige abgearbeitet
// werden, ist die Frage zu klaeren, ob 1000 Iterationen bei dieser
// Konditionierung ueberhaupt reichen KOENNEN.
//
// GMRES baut dafuer bereits alles auf. Die Arnoldi-Rekursion liefert eine
// Hessenberg-Matrix H, deren Eigenwerte -- die Ritz-Werte -- das Spektrum des
// vorkonditionierten Operators auf dem Krylov-Unterraum annaehern. Aus ihnen
// folgt eine Schaetzung von
//
//     kappa ~ max|lambda| / min|lambda|
//
// die fuer normale Matrizen die Konditionszahl ist und sonst eine untere
// Schranke fuer deren Groessenordnung liefert.
//
// -----------------------------------------------------------------------------
// Was die Zahl aussagt, und was nicht
// -----------------------------------------------------------------------------
// Der Krylov-Unterraum nach m Schritten sieht nur einen Teil des Spektrums, und
// zwar bevorzugt die aeusseren Eigenwerte. Die Schaetzung ist deshalb eine
// UNTERE Schranke: die wahre Konditionszahl ist mindestens so gross. Fuer die
// Frage "reichen 1000 Iterationen" genuegt das -- eine untere Schranke von 1e8
// beantwortet sie bereits mit nein.
//
// Zusaetzlich ausgegeben wird, wie viele Ritz-Werte betragsmaessig nahe null
// liegen. Ein Cluster dort bedeutet einen Nahnullraum, und der ist die
// haeufigste Ursache fuer genau dieses Konvergenzbild.
//
// -----------------------------------------------------------------------------
// Verfahren
// -----------------------------------------------------------------------------
// QR-Iteration auf der Hessenberg-Matrix mit Wilkinson-Shift und Deflation,
// 2x2-Bloecke fuer komplex konjugierte Paare analytisch geloest. Bei m = 50 ist
// das eine Sache von Millisekunden und faellt gegenueber einer einzigen
// Operatoranwendung nicht ins Gewicht.
//
// WICHTIG: GMRES wendet die Givens-Rotationen IN PLACE auf H an und zerstoert
// dabei die Hessenberg-Struktur. Die Eigenwerte muessen daher aus einer Kopie
// berechnet werden, die vor der Rotation gezogen wurde.
// =============================================================================

#include <complex>
#include <vector>

/// Eigenwerte einer oberen Hessenberg-Matrix.
///
/// 'H' ist zeilenweise abgelegt mit Zeilenabstand 'ld' und wird ueberschrieben.
/// 'n' ist die tatsaechlich belegte Dimension (kann kleiner als ld sein, wenn
/// GMRES vor dem Restart konvergiert oder abbricht).
///
/// Rueckgabe: false, wenn die Iteration nicht in allen Bloecken konvergierte;
/// die bis dahin gefundenen Werte stehen trotzdem in 'out'.
bool hessenberg_eigenvalues(std::vector<double>& H, int n, int ld,
                            std::vector<std::complex<double>>& out);

/// Wertet die Ritz-Werte aus und gibt eine Einordnung aus.
/// 'label' benennt die Konfiguration, damit sich mehrere Laeufe vergleichen
/// lassen.
void report_ritz_spectrum(const char* label,
                          const std::vector<std::complex<double>>& ritz);

#endif // RITZ_SPECTRUM_CUH
