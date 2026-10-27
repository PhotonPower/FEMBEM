#ifndef HARMONIC_BLOCK_SYSTEM_H
#define HARMONIC_BLOCK_SYSTEM_H

#include <cstddef>
#include <vector>

// =============================================================================
// Reelle 2n-Blockform des harmonischen Systems
// =============================================================================
//
// AP5 aus docs/design/fem-direct-plan.md.
//
// DAS SYSTEM
//
// Zu loesen ist das komplexe System
//
//     (K + j*omega*C) (x_r + j x_i) = b_r + j b_i
//
// Real- und Imaginaerteil getrennt:
//
//     K x_r - omega C x_i = b_r        (1)
//     omega C x_r + K x_i = b_i        (2)
//
// ZWEI FORMEN, UND WARUM ES ZWEI SIND
//
// General -- (1) und (2) direkt untereinander:
//
//     [  K    -wC ] [x_r]   [b_r]
//     [  wC    K  ] [x_i] = [b_i]
//
//   Nicht symmetrisch (die Nebenblocke sind -wC und +wC), also LU. Dafuer
//   braucht es keine neue Assemblierung: K und C liegen schon als zwei
//   Wertefelder ueber demselben CSR-Muster vor. Das ist die Form fuer die
//   Inbetriebnahme.
//
// Symmetric -- Gleichung (2) negiert und die Blockzeilen getauscht:
//
//     [  K    -wC ] [x_r]   [ b_r]
//     [ -wC   -K  ] [x_i] = [-b_i]
//
//   Real symmetrisch (beide Nebenblocke sind -wC, und K wie C sind selbst
//   symmetrisch), aber indefinit. Damit sind eine symmetrische Ordnung, LDLt
//   und ein einzelnes Dreieck moeglich: etwa halbe Faktorisierungsarbeit.
//
//   Der LOESUNGSVEKTOR ist in beiden Formen derselbe [x_r; x_i] -- nur die
//   rechte Seite unterscheidet sich. Deshalb gibt es kein form-abhaengiges
//   Zurueckuebersetzen.
//
// WARUM ROHE ZEIGER
//
// Wie bei FemDirectBC: fem.cuh zieht ueber CoupledSolver.cuh cusparse.h
// herein. Dieses Modul soll ohne CUDA-Bibliotheken uebersetzbar bleiben, damit
// tests/test_block_system.cu es auf dem Host wirklich ausfuehren kann.
//
// VORAUSSETZUNG AN DAS EINGABEMUSTER
//
// K und C teilen sich EIN CSR-Muster (so entstehen sie in fem.cu: zwei
// Assemblierungsdurchgaenge in dieselbe Struktur), die Spaltenindizes je Zeile
// sind aufsteigend sortiert, und die Diagonale ist im Muster enthalten
// (build_csr_structure fuegt die Selbstkopplung immer ein).

namespace fem_direct {

enum class BlockForm {
    General,     ///< [[K, -wC], [wC, K]] -- unsymmetrisch, LU
    Symmetric    ///< [[K, -wC], [-wC, -K]] -- symmetrisch indefinit, LDLt
};

// -----------------------------------------------------------------------------
// Muster des 2n-Systems
// -----------------------------------------------------------------------------
// Haengt weder an omega noch an der Form: nur die Werte unterscheiden sich.
// Also EINMAL aufbauen und ueber alle Frequenzen und beide Formen benutzen.
//
// Die Spaltensortierung ergibt sich von selbst und muss nicht nachsortiert
// werden: in der oberen Blockzeile kommen erst die Spalten des linken Blocks
// (alle < n, aufsteigend) und dann die des rechten (alle >= n, aufsteigend);
// in der unteren ebenso. Aufsteigend sortierte Eingabe bleibt damit
// aufsteigend sortiert.
//
// nnz des Ergebnisses ist genau 4 * nnz der Eingabe.
void build_block_structure(
    int n,
    const int* row_offsets,
    const int* col_indices,
    std::vector<int>& out_row_offsets,
    std::vector<int>& out_col_indices);

// -----------------------------------------------------------------------------
// Werte fuellen
// -----------------------------------------------------------------------------
// K und C sind Wertefelder ueber dem EINGABEmuster (Laenge nnz), out_values
// hat die Laenge 4*nnz und muss zu einem build_block_structure() mit demselben
// Eingabemuster passen.
void fill_block_values(
    int n,
    const int* row_offsets,
    const int* col_indices,
    const double* K,
    const double* C,
    double omega,
    BlockForm form,
    double* out_values);

// -----------------------------------------------------------------------------
// Rechte Seite und Zurueckuebersetzen
// -----------------------------------------------------------------------------
// b_i darf nullptr sein (dann als null behandelt -- der haeufige Fall, weil
// der Spulenstrom als reine Referenzphase angesetzt wird).
void fill_block_rhs(
    int n,
    const double* b_r,
    const double* b_i,
    BlockForm form,
    double* out_rhs);

// Der Loesungsvektor ist in beiden Formen [x_r; x_i]; die Funktion ist
// deshalb absichtlich form-unabhaengig.
void split_block_solution(
    int n,
    const double* x_2n,
    double* out_xr,
    double* out_xi);

// -----------------------------------------------------------------------------
// Oberes Dreieck herausziehen
// -----------------------------------------------------------------------------
// Fuer CUDSS_MTYPE_SYMMETRIC mit CUDSS_MVIEW_UPPER wird nur ein Dreieck
// uebergeben. Nur mit BlockForm::Symmetric sinnvoll; bei General waere das
// Ergebnis schlicht falsch, weil die Matrix dort nicht symmetrisch ist.
// Deshalb pruefen wir es und brechen ab, statt es zuzulassen.
//
// Rueckgabe: nnz des Dreiecks.
size_t extract_upper_triangle(
    int num_rows,
    const int* row_offsets,
    const int* col_indices,
    const double* values,
    std::vector<int>& out_row_offsets,
    std::vector<int>& out_col_indices,
    std::vector<double>& out_values,
    double symmetry_tolerance = 1e-12);

}  // namespace fem_direct

#endif  // HARMONIC_BLOCK_SYSTEM_H
