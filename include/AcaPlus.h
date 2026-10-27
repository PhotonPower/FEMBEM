#ifndef ACA_PLUS_H
#define ACA_PLUS_H

#include <vector>
#include <functional>

// Struktur zur Speicherung der komprimierten Block-Matrix M_approx = U * V^T
struct CompressedBlock {
    int num_rows; // m
    int num_cols; // n
    int rank;     // k

    // Flache Arrays im Column-Major oder Row-Major Format.
    // U hat die Dimension (m x k), V hat die Dimension (n x k).
    // Wir speichern sie als 1D-Vektoren für einfachen GPU-Transfer später.
    std::vector<double> U; 
    std::vector<double> V; 
};

class AcaPlus {
public:
    /**
     * @brief Führt die Adaptive Cross Approximation mit Partial Pivoting aus.
     * 
     * @param num_rows Anzahl der Zeilen des Matrixblocks (m)
     * @param num_cols Anzahl der Spalten des Matrixblocks (n)
     * @param evaluator Eine Funktion, die auf Zuruf den Matrixeintrag M(row, col) berechnet.
     * @param tolerance Das Abbruchkriterium epsilon (z.B. 1e-4)
     * @param max_rank Maximal erlaubter Rang (als Notausstieg, falls Block nicht low-rank ist)
     * @return CompressedBlock Das Resultat U und V
     */
    static CompressedBlock compress(
        int num_rows, 
        int num_cols, 
        const std::function<double(int, int)>& evaluator, 
        double tolerance = 1e-4, 
        int max_rank = -1);
};

#endif // ACA_PLUS_H