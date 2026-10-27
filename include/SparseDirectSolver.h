#ifndef SPARSE_DIRECT_SOLVER_H
#define SPARSE_DIRECT_SOLVER_H

#include <cstdint>
#include <string>
#include <vector>

// =============================================================================
// Dünn besetzter Direktlöser (cuDSS) für den FEM-direkt-Pfad
// =============================================================================
//
// AP6 aus docs/design/fem-direct-plan.md. Entwickelt gegen cuDSS 0.8.0.
//
// WARUM DIESER HEADER KEIN cudss.h EINBINDET
//
// cuDSS ist eine optionale Abhängigkeit (-DFEMBEM_WITH_CUDSS=ON). Wer nur am
// transienten oder am BEM-FEM-Pfad arbeitet, braucht sie nicht. Damit das
// Projekt ohne cuDSS unverändert baut, liegt der ganze Bibliothekskram hinter
// einem opaken Zeiger; ohne cuDSS übersetzt dieselbe Datei und wirft zur
// Laufzeit mit Begründung.
//
// DER ABLAUF
//
//     analyze()      Ordnung und symbolische Faktorisierung. Hängt nur am
//                    MUSTER, nicht an den Werten. Einmal je Netz.
//     factorize()    numerische Faktorisierung. Je Frequenz beziehungsweise
//                    je Zeitschritt, wenn sich Werte geändert haben.
//     refactorize()  dito, aber unter Wiederverwendung der Pivotfolge.
//                    Deutlich billiger, aber nur zulässig, wenn sich die
//                    Werte nicht so stark geändert haben, dass die Pivots
//                    unbrauchbar werden.
//     solve()        Vorwärts-/Rückwärtseinsetzen. Beliebig oft je
//                    Faktorisierung -- genau darin liegt der Hebel für den
//                    transienten Pfad: bei ruhender Platte und konstantem dt
//                    ist K + C/dt über alle Zeitschritte DIESELBE Matrix.
//
// SPEICHER
//
// Die Zeiger werden NICHT kopiert. Sie müssen von analyze() bis zum letzten
// solve() gültig bleiben, und die Werte müssen auf dem Gerät liegen oder in
// Unified Memory -- was auf DeviceArray zutrifft.

namespace fem_direct {

// -----------------------------------------------------------------------------
// Prüfung des Eingabemusters -- läuft AUCH OHNE cuDSS
// -----------------------------------------------------------------------------
// cuDSS prüft das Muster nicht vollständig, und ein Löser auf fehlerhaftem CSR
// liefert keine Fehlermeldung, sondern Zahlen. Deshalb vorher selbst prüfen:
//
//   * row_offsets[0] == 0 und monoton steigend
//   * row_offsets[num_rows] == nnz
//   * Spaltenindizes in [0, num_rows)
//   * je Zeile streng aufsteigend (CSR verlangt das; cuDSS toleriert es
//     teilweise, aber nicht verlässlich)
//   * Diagonale in jeder Zeile vorhanden -- eine fehlende Diagonale ist bei
//     einer FEM-Matrix immer ein Fehler im Musteraufbau
//
// Rückgabe: leerer String, wenn alles stimmt, sonst die Beanstandung.
std::string validate_csr(
    int num_rows,
    int nnz,
    const int* row_offsets,
    const int* col_indices,
    bool require_diagonal = true);

// -----------------------------------------------------------------------------
enum class MatrixKind {
    General,            ///< volles Muster, LU        -> BlockForm::General
    SymmetricIndefinite ///< oberes Dreieck, LDLt     -> BlockForm::Symmetric
};

enum class Reordering {
    Default,
    Amd,
    NestedDissection
};

struct SolverOptions {
    Reordering reordering = Reordering::NestedDissection;

    /// Reproduzierbarkeit vor Geschwindigkeit. Dieses Projekt hat einen
    /// eigenen Test dafür (test_determinism_fingerprint), und ein Löser, der
    /// je Lauf um Rundungsfehler abweicht, macht jeden Bitvergleich wertlos.
    /// cuDSS bietet dafür CUDSS_CONFIG_DETERMINISTIC_MODE.
    bool deterministic = true;

    /// Faktor teilweise im Hostspeicher halten. Bei einem ernsthaften
    /// Luftkasten ist das der Normalfall, nicht die Ausnahme.
    bool hybrid_memory = false;

    /// Obergrenze des Gerätespeichers in Bytes für den Hybridmodus. 0 =
    /// cuDSS entscheiden lassen.
    std::int64_t hybrid_device_memory_limit = 0;

    /// Schritte iterativer Nachbesserung. 0 = aus. Ein oder zwei Schritte
    /// kosten wenig und fangen den Genauigkeitsverlust einer stark
    /// skalierten Matrix auf -- und stark skaliert ist sie hier, weil die
    /// Regularisierung sigma_Luft = 1e-8 * sigma_Al genau das bewirkt.
    int refinement_steps = 2;
};

struct FactorizationReport {
    std::int64_t lu_nnz = 0;        ///< Nichtnullen im Faktor (Füllung)
    int npivots = 0;                ///< Zahl der Pivotierungen
    int info = 0;                   ///< CUDSS_DATA_INFO, 0 = in Ordnung
    std::int64_t inertia_pos = 0;   ///< nur bei SymmetricIndefinite belegt
    std::int64_t inertia_neg = 0;
    double fill_ratio = 0.0;        ///< lu_nnz / nnz der Matrix
};

// -----------------------------------------------------------------------------
class SparseDirectSolver {
public:
    SparseDirectSolver();
    ~SparseDirectSolver();

    SparseDirectSolver(const SparseDirectSolver&) = delete;
    SparseDirectSolver& operator=(const SparseDirectSolver&) = delete;

    /// Wurde mit cuDSS gebaut? Ohne cuDSS werfen alle anderen Methoden.
    static bool available();

    /// Version der eingebundenen cuDSS, z. B. "0.8.0". Ohne cuDSS leer.
    static std::string version();

    void set_options(const SolverOptions& opt);

    /// Ordnung und symbolische Faktorisierung. Die Zeiger müssen bis zum
    /// letzten solve() gültig bleiben und auf dem Gerät oder in Unified
    /// Memory liegen.
    ///
    /// Bei SymmetricIndefinite wird nur das OBERE DREIECK erwartet
    /// (fem_direct::extract_upper_triangle). Das volle Muster mit
    /// CUDSS_MTYPE_SYMMETRIC zu übergeben ergibt eine andere Matrix als
    /// gemeint -- ohne Fehlermeldung.
    void analyze(int num_rows, int nnz,
                 int* row_offsets, int* col_indices, double* values,
                 MatrixKind kind);

    FactorizationReport factorize();
    FactorizationReport refactorize();

    /// rhs und x müssen num_rows Einträge haben und dürfen dieselbe Adresse
    /// haben. Beide auf dem Gerät oder in Unified Memory.
    void solve(double* rhs, double* x);

    int num_rows() const;

private:
    struct Impl;
    Impl* m_impl;
};

}  // namespace fem_direct

#endif  // SPARSE_DIRECT_SOLVER_H
