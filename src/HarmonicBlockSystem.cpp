#include "HarmonicBlockSystem.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace fem_direct {

// -----------------------------------------------------------------------------
void build_block_structure(
    int n,
    const int* row_offsets,
    const int* col_indices,
    std::vector<int>& out_row_offsets,
    std::vector<int>& out_col_indices)
{
    if (n <= 0) throw std::runtime_error("build_block_structure: n muss positiv sein.");
    if (!row_offsets || !col_indices)
        throw std::runtime_error("build_block_structure: Muster ist null.");

    const int nnz = row_offsets[n];

    out_row_offsets.assign(static_cast<size_t>(2) * n + 1, 0);
    out_col_indices.clear();
    out_col_indices.reserve(static_cast<size_t>(4) * nnz);

    // Obere Blockzeile: [K | -wC], also erst die Spalten unverschoben,
    // dann dieselben Spalten um n verschoben.
    for (int r = 0; r < n; ++r) {
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_col_indices.push_back(col_indices[k]);
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_col_indices.push_back(col_indices[k] + n);
        out_row_offsets[r + 1] = static_cast<int>(out_col_indices.size());
    }

    // Untere Blockzeile: dieselbe Aufteilung.
    for (int r = 0; r < n; ++r) {
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_col_indices.push_back(col_indices[k]);
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_col_indices.push_back(col_indices[k] + n);
        out_row_offsets[n + r + 1] = static_cast<int>(out_col_indices.size());
    }

    if (static_cast<int>(out_col_indices.size()) != 4 * nnz) {
        throw std::runtime_error(
            "build_block_structure: nnz ist " + std::to_string(out_col_indices.size()) +
            ", erwartet 4 * " + std::to_string(nnz) + ". Das Eingabemuster ist "
            "inkonsistent (row_offsets passt nicht zu col_indices).");
    }
}

// -----------------------------------------------------------------------------
void fill_block_values(
    int n,
    const int* row_offsets,
    const int* col_indices,
    const double* K,
    const double* C,
    double omega,
    BlockForm form,
    double* out_values)
{
    if (!K || !C || !out_values)
        throw std::runtime_error("fill_block_values: Eingabe oder Ausgabe ist null.");
    (void)col_indices;

    const int nnz = row_offsets[n];
    const double wc_sign_upper = -omega;    // rechter Block der oberen Zeile
    const double wc_sign_lower = (form == BlockForm::Symmetric) ? -omega : omega;
    const double k_sign_lower  = (form == BlockForm::Symmetric) ? -1.0   : 1.0;

    size_t out = 0;

    // Obere Blockzeile: [ K | -wC ]
    for (int r = 0; r < n; ++r) {
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_values[out++] = K[k];
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_values[out++] = wc_sign_upper * C[k];
    }

    // Untere Blockzeile: General [ wC | K ], Symmetric [ -wC | -K ]
    for (int r = 0; r < n; ++r) {
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_values[out++] = wc_sign_lower * C[k];
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            out_values[out++] = k_sign_lower * K[k];
    }

    if (out != static_cast<size_t>(4) * nnz) {
        throw std::runtime_error("fill_block_values: geschrieben " + std::to_string(out) +
                                 " von erwarteten " + std::to_string(4 * nnz) + " Werten.");
    }
}

// -----------------------------------------------------------------------------
void fill_block_rhs(
    int n,
    const double* b_r,
    const double* b_i,
    BlockForm form,
    double* out_rhs)
{
    if (!b_r || !out_rhs)
        throw std::runtime_error("fill_block_rhs: b_r oder Ausgabe ist null.");

    // Symmetric entsteht aus der NEGIERTEN zweiten Gleichung, also gehoert das
    // Minuszeichen auf die untere Haelfte der rechten Seite. Wer das vergisst,
    // bekommt eine Loesung, deren Imaginaerteil das falsche Vorzeichen hat --
    // und damit eine Kraft mit falschem Vorzeichen, ohne jede Fehlermeldung.
    const double sign_lower = (form == BlockForm::Symmetric) ? -1.0 : 1.0;

    for (int i = 0; i < n; ++i) out_rhs[i] = b_r[i];
    for (int i = 0; i < n; ++i)
        out_rhs[n + i] = sign_lower * (b_i ? b_i[i] : 0.0);
}

// -----------------------------------------------------------------------------
void split_block_solution(int n, const double* x_2n, double* out_xr, double* out_xi) {
    if (!x_2n || !out_xr || !out_xi)
        throw std::runtime_error("split_block_solution: Zeiger ist null.");
    for (int i = 0; i < n; ++i) out_xr[i] = x_2n[i];
    for (int i = 0; i < n; ++i) out_xi[i] = x_2n[n + i];
}

// -----------------------------------------------------------------------------
size_t extract_upper_triangle(
    int num_rows,
    const int* row_offsets,
    const int* col_indices,
    const double* values,
    std::vector<int>& out_row_offsets,
    std::vector<int>& out_col_indices,
    std::vector<double>& out_values,
    double symmetry_tolerance)
{
    if (num_rows <= 0) throw std::runtime_error("extract_upper_triangle: num_rows <= 0.");

    // Erst pruefen, ob die Matrix ueberhaupt symmetrisch ist. Ein Dreieck aus
    // einer unsymmetrischen Matrix zu ziehen ist keine Vereinfachung, sondern
    // eine stille Verfaelschung -- cuDSS wuerde daraus die falsche Matrix
    // rekonstruieren und ohne Fehler ein falsches Ergebnis liefern.
    std::unordered_map<long long, double> entry;
    entry.reserve(static_cast<size_t>(row_offsets[num_rows]));
    for (int r = 0; r < num_rows; ++r)
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k)
            entry[static_cast<long long>(r) * num_rows + col_indices[k]] = values[k];

    double scale = 0.0;
    for (const auto& kv : entry) scale = std::fmax(scale, std::fabs(kv.second));
    if (scale == 0.0) scale = 1.0;

    for (const auto& kv : entry) {
        const int r = static_cast<int>(kv.first / num_rows);
        const int c = static_cast<int>(kv.first % num_rows);
        if (r == c) continue;
        const auto it = entry.find(static_cast<long long>(c) * num_rows + r);
        const double mirrored = (it == entry.end()) ? 0.0 : it->second;
        if (std::fabs(kv.second - mirrored) > symmetry_tolerance * scale) {
            throw std::runtime_error(
                "extract_upper_triangle: Matrix ist nicht symmetrisch bei (" +
                std::to_string(r) + "," + std::to_string(c) + "): " +
                std::to_string(kv.second) + " gegen " + std::to_string(mirrored) +
                ". Mit BlockForm::General ist das zu erwarten -- dort wird das "
                "volle Muster gebraucht (CUDSS_MTYPE_GENERAL).");
        }
    }

    out_row_offsets.assign(static_cast<size_t>(num_rows) + 1, 0);
    out_col_indices.clear();
    out_values.clear();
    out_col_indices.reserve(static_cast<size_t>(row_offsets[num_rows]) / 2 + num_rows);
    out_values.reserve(out_col_indices.capacity());

    for (int r = 0; r < num_rows; ++r) {
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
            if (col_indices[k] < r) continue;      // unteres Dreieck weglassen
            out_col_indices.push_back(col_indices[k]);
            out_values.push_back(values[k]);
        }
        out_row_offsets[r + 1] = static_cast<int>(out_col_indices.size());
    }

    return out_col_indices.size();
}

}  // namespace fem_direct
