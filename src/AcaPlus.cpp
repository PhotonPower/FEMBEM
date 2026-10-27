#include "AcaPlus.h"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

// Prueft einen Kernelwert und bricht mit benannter Fundstelle ab, wenn er nicht
// endlich ist.
//
// Warum das noetig ist: die Pivotsuche unten waehlt ueber
//     if (!used_cols[j] && std::abs(val) > max_val_v)
// Ist val ein NaN, ist dieser Vergleich FALSCH -- das NaN wird also nie als
// Pivot gewaehlt, wandert aber trotzdem in current_v bzw. current_u und damit
// in die Basisvektoren U/V. Die Kompression laeuft ohne jede Meldung durch und
// liefert einen Block, dessen Anwendung jeden Vektor in NaN verwandelt. Genau
// so ist es passiert: die Audit-Kette in apply_physical zeigte einen endlichen
// Eingangsvektor und danach y_K zu 16544/16544 NaN.
//
// Ein NaN im Kernel ist nie zulaessig, und stillschweigend 0 einzusetzen waere
// falsche Physik statt eines Fehlers. Deshalb hier ein Abbruch, der die
// Matrixposition nennt -- damit laesst sich das verantwortliche Panel-Paar
// direkt nachrechnen.
inline void require_finite_kernel_value(double val, int i, int j, const char* where) {
    if (std::isfinite(val)) return;
    throw std::runtime_error(
        std::string("AcaPlus::compress: der BEM-Kernel liefert bei (") + std::to_string(i)
        + ", " + std::to_string(j) + ") den Wert "
        + (std::isnan(val) ? "NaN" : "Inf") + " (" + where + "). "
        "Die Kreuzapproximation kann das nicht abfangen -- ihre Pivotsuche vergleicht "
        "mit std::abs(val) > max, und jeder Vergleich mit NaN ist falsch, das NaN "
        "wuerde also unbemerkt in U/V uebernommen. Ursache ist typischerweise eine "
        "singuläre oder fast singuläre Panel-Paarung (beruehrende bzw. zusammenfallende "
        "Randelemente), die die Quadratur des Kerns nicht abdeckt.");
}

} // namespace

CompressedBlock AcaPlus::compress(
    int num_rows, 
    int num_cols, 
    const std::function<double(int, int)>& evaluator, 
    double tolerance, 
    int max_rank) 
{
    CompressedBlock block;
    block.num_rows = num_rows;
    block.num_cols = num_cols;
    block.rank = 0;

    if (num_rows == 0 || num_cols == 0) return block;

    // Default: Max Rang ist die volle Dimension (limitiert auf die kleinere)
    if (max_rank <= 0) {
        max_rank = std::min(num_rows, num_cols);
    }

    // Speicher für U und V vorallozieren (dynamisch wachsend)
    std::vector<std::vector<double>> U_cols;
    std::vector<std::vector<double>> V_cols;

    std::vector<bool> used_rows(num_rows, false);
    std::vector<bool> used_cols(num_cols, false);

    double norm_sq_approx = 0.0; // ||Z_k||^2_F
    int k = 0;                   // Aktueller Rang

    // 1. Pivot-Strategie: Starte mit Zeile 0 (oder einer beliebigen)
    int pivot_row = 0;

    while (k < max_rank) {
        // ---------------------------------------------------------------------
        // Schritt A: Berechne das Residuum für die gewählte Zeile (pivot_row)
        // R(pivot_row, j) = M(pivot_row, j) - sum_{l=1}^k U(pivot_row, l) * V(j, l)
        // ---------------------------------------------------------------------
        std::vector<double> current_v(num_cols, 0.0);
        double max_val_v = -1.0;
        int pivot_col = -1;

        for (int j = 0; j < num_cols; ++j) {
            double val = evaluator(pivot_row, j);
            require_finite_kernel_value(val, pivot_row, j, "Zeilendurchlauf");

            // Abziehen der bereits approximierten Ränge
            for (int l = 0; l < k; ++l) {
                val -= U_cols[l][pivot_row] * V_cols[l][j];
            }
            current_v[j] = val;

            // Suche nach dem größten Element in dieser Zeile für die nächste Spalte
            if (!used_cols[j] && std::abs(val) > max_val_v) {
                max_val_v = std::abs(val);
                pivot_col = j;
            }
        }

        // ---------------------------------------------------------------------
        // Schritt B: Partial Pivoting Fallback (Das "+" in ACA+)
        // ---------------------------------------------------------------------
        // Wenn die gefundene Zeile komplett Null ist (max_val_v ~ 0) oder alle 
        // Spalten schon benutzt wurden, suchen wir uns eine neue Zeile.
        if (pivot_col == -1 || max_val_v < 1e-14) {
            bool found_new_row = false;
            for (int i = 0; i < num_rows; ++i) {
                if (!used_rows[i]) {
                    pivot_row = i;
                    used_rows[pivot_row] = true;
                    found_new_row = true;
                    break;
                }
            }
            if (found_new_row) {
                continue; // Nächster Versuch mit neuer Zeile
            } else {
                break; // Alle Zeilen sind Null -> Matrix ist exakt approximiert
            }
        }

        // ---------------------------------------------------------------------
        // Schritt C: Berechne das Residuum für die Spalte (pivot_col)
        // R(i, pivot_col) = M(i, pivot_col) - sum_{l=1}^k U(i, l) * V(pivot_col, l)
        // ---------------------------------------------------------------------
        std::vector<double> current_u(num_rows, 0.0);
        double pivot_value = current_v[pivot_col]; // R(pivot_row, pivot_col)
        
        double max_val_u = -1.0;
        int next_pivot_row = -1;

        for (int i = 0; i < num_rows; ++i) {
            double val = evaluator(i, pivot_col);
            require_finite_kernel_value(val, i, pivot_col, "Spaltendurchlauf");

            for (int l = 0; l < k; ++l) {
                val -= U_cols[l][i] * V_cols[l][pivot_col];
            }
            
            // Normalisierung von U mit dem Pivot-Element
            val /= pivot_value;
            current_u[i] = val;

            // Suche nach dem Maximum in U für den nächsten Iterationsschritt
            if (!used_rows[i] && std::abs(val) > max_val_u) {
                max_val_u = std::abs(val);
                next_pivot_row = i;
            }
        }

        // Markiere aktuelle Pivots als benutzt
        used_rows[pivot_row] = true;
        used_cols[pivot_col] = true;

        // Vektoren zur Basis hinzufügen
        U_cols.push_back(current_u);
        V_cols.push_back(current_v);

        // ---------------------------------------------------------------------
        // Schritt D: Abbruchkriterium (Norm des Updates)
        // ---------------------------------------------------------------------
        // Berechne Skalarprodukte: ||u||^2 und ||v||^2
        double norm_u_sq = 0.0;
        for (double val : current_u) norm_u_sq += val * val;

        double norm_v_sq = 0.0;
        for (double val : current_v) norm_v_sq += val * val;

        // Update der quadrierten Frobenius-Norm von Z_k (rekursive Formel)
        double cross_term = 0.0;
        for (int l = 0; l < k; ++l) {
            double dot_u = 0.0, dot_v = 0.0;
            for (int i = 0; i < num_rows; ++i) dot_u += current_u[i] * U_cols[l][i];
            for (int j = 0; j < num_cols; ++j) dot_v += current_v[j] * V_cols[l][j];
            cross_term += dot_u * dot_v;
        }
        
        norm_sq_approx += 2.0 * cross_term + norm_u_sq * norm_v_sq;

        // Das eigentliche Kriterium: ||U_new * V_new^T||_F^2 <= tol^2 * ||Z_k||_F^2
        double update_norm_sq = norm_u_sq * norm_v_sq;
        
        k++; // Rang erhöhen

        if (update_norm_sq <= tolerance * tolerance * norm_sq_approx) {
            break; // Konvergenz erreicht!
        }

        // Setze die nächste Pivot-Zeile
        if (next_pivot_row != -1) {
            pivot_row = next_pivot_row;
        } else {
            // Fallback, falls kein ungenutztes Maximum gefunden wurde
            for (int i = 0; i < num_rows; ++i) {
                if (!used_rows[i]) {
                    pivot_row = i;
                    break;
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Schritt E: Formatierung in flache 1D-Arrays (Column-Major)
    // ---------------------------------------------------------------------
    block.rank = k;
    block.U.resize(num_rows * k);
    block.V.resize(num_cols * k);

    for (int l = 0; l < k; ++l) {
        for (int i = 0; i < num_rows; ++i) {
            block.U[l * num_rows + i] = U_cols[l][i];
        }
        for (int j = 0; j < num_cols; ++j) {
            block.V[l * num_cols + j] = V_cols[l][j];
        }
    }

    return block;
}