#include "SparseDirectSolver.h"

#include <cstdlib>   // getenv, atoi fuer die FEMBEM_CUDSS_*-Variablen
#include <iostream>  // Konfigurationszeile im Protokoll
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef FEMBEM_HAVE_CUDSS
#include <cudss.h>
#endif

namespace fem_direct {

// =============================================================================
// Musterprüfung -- läuft auch ohne cuDSS
// =============================================================================
std::string validate_csr(
    int num_rows, int nnz,
    const int* row_offsets, const int* col_indices,
    bool require_diagonal)
{
    std::ostringstream e;

    if (num_rows <= 0) { e << "num_rows = " << num_rows << " ist nicht positiv."; return e.str(); }
    if (nnz <= 0)      { e << "nnz = " << nnz << " ist nicht positiv."; return e.str(); }
    if (!row_offsets || !col_indices) return "row_offsets oder col_indices ist null.";

    if (row_offsets[0] != 0) {
        e << "row_offsets[0] = " << row_offsets[0] << ", erwartet 0 "
          << "(nullbasiert, CUDSS_BASE_ZERO).";
        return e.str();
    }
    if (row_offsets[num_rows] != nnz) {
        e << "row_offsets[" << num_rows << "] = " << row_offsets[num_rows]
          << ", erwartet nnz = " << nnz << ".";
        return e.str();
    }

    for (int r = 0; r < num_rows; ++r) {
        if (row_offsets[r + 1] < row_offsets[r]) {
            e << "row_offsets ist bei Zeile " << r << " nicht monoton: "
              << row_offsets[r] << " -> " << row_offsets[r + 1] << ".";
            return e.str();
        }
        if (row_offsets[r + 1] == row_offsets[r]) {
            e << "Zeile " << r << " ist leer. Bei einer FEM-Matrix ist das immer "
              << "ein Fehler im Musteraufbau -- jede Kante koppelt mindestens "
              << "mit sich selbst.";
            return e.str();
        }

        bool has_diag = false;
        for (int k = row_offsets[r]; k < row_offsets[r + 1]; ++k) {
            const int c = col_indices[k];
            if (c < 0 || c >= num_rows) {
                e << "Spaltenindex " << c << " in Zeile " << r
                  << " liegt ausserhalb [0, " << num_rows << ").";
                return e.str();
            }
            if (k > row_offsets[r] && c <= col_indices[k - 1]) {
                e << "Spaltenindizes in Zeile " << r << " sind nicht streng "
                  << "aufsteigend (" << col_indices[k - 1] << " dann " << c
                  << "). CSR verlangt das; ein Loeser darauf liefert keine "
                  << "Fehlermeldung, sondern Zahlen.";
                return e.str();
            }
            if (c == r) has_diag = true;
        }
        if (require_diagonal && !has_diag) {
            e << "Zeile " << r << " hat keinen Diagonaleintrag. Bei einer "
              << "FEM-Matrix ist das ein Fehler im Musteraufbau.";
            return e.str();
        }
    }
    return std::string();
}

// =============================================================================
#ifdef FEMBEM_HAVE_CUDSS

namespace {

const char* status_name(cudssStatus_t s) {
    switch (s) {
        case CUDSS_STATUS_SUCCESS:          return "SUCCESS";
        case CUDSS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
        case CUDSS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
        case CUDSS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
        case CUDSS_STATUS_NOT_SUPPORTED:    return "NOT_SUPPORTED";
        case CUDSS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUDSS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
        case CUDSS_STATUS_IR_FAILED:        return "IR_FAILED";
        default:                            return "UNBEKANNT";
    }
}

// Die zuletzt gesetzte Konfiguration, damit check() sie in die Fehlermeldung
// aufnehmen kann. NOT_SUPPORTED nennt die beanstandete Einstellung nicht, und
// eine Meldung, die stattdessen die haeufigste Ursache RAET, schickt den Leser
// in die falsche Richtung -- genau das ist beim ersten Anlauf passiert: die
// Meldung nannte CUDSS_PIVOT_DIAGONAL, gesetzt war bei MTYPE_GENERAL aber
// PIVOT_AUTO. Besser die Tatsachen nennen als eine Vermutung.
std::string g_last_config;

// Jeder cuDSS-Aufruf wird geprüft. Ein stillschweigend ignorierter Status
// führt hier nicht zu einem Absturz, sondern zu einem Ergebnisvektor mit
// Datenmüll, der wie eine Lösung aussieht.
void check(cudssStatus_t s, const char* what) {
    if (s == CUDSS_STATUS_SUCCESS) return;
    std::ostringstream e;
    e << "cuDSS: " << what << " gab " << status_name(s)
      << " (" << static_cast<int>(s) << ").";
    if (s == CUDSS_STATUS_ALLOC_FAILED) {
        e << " Der Faktor passt nicht in den Geraetespeicher. Die Fuellung "
             "waechst deutlich schneller als das System, eine Verfeinerungsstufe "
             "kann den Bedarf also vervielfachen. Abhilfe, in dieser "
             "Reihenfolge: FEMBEM_CUDSS_HYBRID=1 verlagert Teile in den "
             "Hauptspeicher (langsamer, aber der Faktor muss nicht "
             "hineinpassen), notfalls zusaetzlich "
             "FEMBEM_CUDSS_HYBRID_LIMIT=<MiB> als Obergrenze fuer den "
             "Geraeteanteil.";
    }
    if (s == CUDSS_STATUS_NOT_SUPPORTED) {
        e << " Diese Fassung von cuDSS traegt eine der gesetzten Einstellungen "
             "nicht. Gesetzt war: "
          << (g_last_config.empty() ? std::string("(nicht aufgezeichnet)")
                                    : g_last_config)
          << ". Welche davon beanstandet wird, sagt cuDSS nicht -- einzeln "
             "durchprobieren ueber FEMBEM_CUDSS_REORDER (default|amd|nd), "
             "FEMBEM_CUDSS_DETERMINISTIC (0|1) und FEMBEM_CUDSS_IR (Zahl).";
    }
    throw std::runtime_error(e.str());
}

}  // namespace

struct SparseDirectSolver::Impl {
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t   data   = nullptr;

    cudssMatrix_t A = nullptr;
    cudssMatrix_t b = nullptr;
    cudssMatrix_t x = nullptr;

    SolverOptions opt;
    MatrixKind kind = MatrixKind::General;
    int n = 0;
    int nnz = 0;
    bool analyzed = false;
    bool factorized = false;

    Impl() {
        check(cudssCreate(&handle), "cudssCreate");
        check(cudssConfigCreate(&config), "cudssConfigCreate");
        check(cudssDataCreate(handle, &data), "cudssDataCreate");
    }

    ~Impl() {
        if (x) cudssMatrixDestroy(x);
        if (b) cudssMatrixDestroy(b);
        if (A) cudssMatrixDestroy(A);
        if (data)   cudssDataDestroy(handle, data);
        if (config) cudssConfigDestroy(config);
        if (handle) cudssDestroy(handle);
    }

    void apply_config() {
        // FEMBEM_CUDSS_MINIMAL=1 setzt GAR NICHTS und laesst cuDSS bei seinen
        // eigenen Vorgaben. Das ist der eine Lauf, der Konfiguration von
        // Matrix trennt: geht ANALYSIS damit durch, liegt es an einem der
        // ConfigSet-Aufrufe unten; scheitert es weiter, liegt es am Muster,
        // am Matrixtyp oder an der Groesse -- und dann ist jede weitere
        // Bisektion ueber die Knoepfe verschwendet.
        //
        // Notwendig, weil ein Parameter auf "aus" NICHT dasselbe ist wie ein
        // nicht gesetzter Parameter: cudssConfigSet nimmt auch entgegen, was
        // die Bibliothek nicht traegt, und beanstandet wird erst beim
        // Ausfuehren.
        if (const char* v = std::getenv("FEMBEM_CUDSS_MINIMAL")) {
            if (std::atoi(v)) {
                g_last_config = std::string(
                    "MINIMAL -- kein einziges cudssConfigSet, Matrixtyp ")
                    + (kind == MatrixKind::SymmetricIndefinite
                           ? "SYMMETRIC/UPPER" : "GENERAL/FULL");
                std::cout << "[cuDSS] Konfiguration: " << g_last_config << "\n";
                return;
            }
        }

        // Die drei ANALYSIS-relevanten Knoepfe sind zur Laufzeit
        // ueberschreibbar. Grund: cudssExecute ANALYSIS gibt NOT_SUPPORTED
        // zurueck, ohne zu sagen WELCHE Einstellung es nicht anbietet, und
        // welche Kombinationen eine cuDSS-Fassung traegt, steht nirgends
        // vollstaendig. Ohne Umgebungsvariablen kostet jede Bisektion eine
        // Neuuebersetzung. Muster wie FEMBEM_BACKEND (siehe BUILD.md).
        //
        //   FEMBEM_CUDSS_MINIMAL       = 1  (nichts setzen, siehe oben)
        //   FEMBEM_CUDSS_REORDER       = default | amd | nd | skip
        //   FEMBEM_CUDSS_DETERMINISTIC = 0 | 1 | skip
        //   FEMBEM_CUDSS_IR            = Zahl | skip
        //
        // "skip" laesst den jeweiligen ConfigSet-Aufruf weg, statt ihn mit
        // einem harmlosen Wert auszufuehren.
        const auto skipped = [](const char* name) {
            const char* v = std::getenv(name);
            return v && std::string(v) == "skip";
        };

        // --- Ordnung -----------------------------------------------------
        Reordering reordering = opt.reordering;
        if (const char* v = std::getenv("FEMBEM_CUDSS_REORDER")) {
            const std::string s(v);
            if      (s == "default") reordering = Reordering::Default;
            else if (s == "amd")     reordering = Reordering::Amd;
            else if (s == "nd")      reordering = Reordering::NestedDissection;
            else if (s != "skip") throw std::runtime_error(
                "FEMBEM_CUDSS_REORDER: erwartet default, amd, nd oder skip, "
                "bekommen '" + s + "'.");
        }

        const char* alg_name = "(nicht gesetzt)";
        if (!skipped("FEMBEM_CUDSS_REORDER")) {
            cudssReorderingAlg_t alg = CUDSS_REORDERING_ALG_NESTED_DISSECTION;
            alg_name = "NESTED_DISSECTION";
            if (reordering == Reordering::Amd) {
                alg = CUDSS_REORDERING_ALG_AMD;      alg_name = "AMD";
            }
            if (reordering == Reordering::Default) {
                alg = CUDSS_REORDERING_ALG_DEFAULT;  alg_name = "DEFAULT";
            }
            check(cudssConfigSet(config, CUDSS_CONFIG_REORDERING_ALG, &alg, sizeof(alg)),
                  "ConfigSet REORDERING_ALG");
        }

        // --- Pivotierung -------------------------------------------------
        // CUDSS_PIVOT_DIAGONAL ist laut Header genau fuer symmetrisch
        // indefinite Matrizen mit AMD oder NESTED_DISSECTION gedacht -- also
        // fuer unsere 2n-Blockform. Bei General bleibt AUTO, das dort
        // LOCAL_BLOCK waehlt.
        const char* pivot_name = "(nicht gesetzt)";
        if (!skipped("FEMBEM_CUDSS_PIVOT")) {
            cudssPivotType_t pivot = CUDSS_PIVOT_AUTO;
            pivot_name = "AUTO";
            if (kind == MatrixKind::SymmetricIndefinite &&
                reordering != Reordering::Default) {
                pivot = CUDSS_PIVOT_DIAGONAL;  pivot_name = "DIAGONAL";
            }
            check(cudssConfigSet(config, CUDSS_CONFIG_PIVOT_TYPE, &pivot, sizeof(pivot)),
                  "ConfigSet PIVOT_TYPE");
        }

        // --- Die zwei Unvertraeglichkeiten von cuDSS 0.8 ------------------
        // GEMESSEN durch Bisektion ueber alle Einzel- und Paarkombinationen,
        // auf RTX 5060 (sm_120) und RTX 2080 (sm_75):
        //
        //   DETERMINISTIC_MODE + IR_N_STEPS > 0   -> ANALYSIS NOT_SUPPORTED
        //   DETERMINISTIC_MODE + HYBRID_MEMORY    -> ANALYSIS NOT_SUPPORTED
        //
        // Ordnung und Pivotierung sind an beidem unbeteiligt. Beide Male ist
        // es der Determinismus: er schraenkt die interne Ablaufplanung ein,
        // und sowohl die Nachbesserung als auch die Auslagerung in den
        // Hauptspeicher brauchen dort Freiheit.
        //
        // Die Rangfolge ergibt sich aus dem, was verzichtbar ist:
        //
        //   1. HYBRID gewinnt immer. Ohne ihn passt der Faktor eines
        //      verfeinerten Netzes nicht in den Geraetespeicher, und das
        //      System laeuft dann gar nicht. Ein nicht gerechnetes Ergebnis
        //      ist nicht reproduzierbar, sondern abwesend.
        //   2. Sonst gewinnt der DETERMINISMUS ueber die Nachbesserung. Das
        //      Projekt vergleicht bitweise (test_determinism_fingerprint),
        //      und das Residuum wird auf diesem Pfad ohnehin explizit
        //      nachgerechnet -- die Guete der Loesung ist also unabhaengig
        //      messbar, die Reproduzierbarkeit nicht.
        //
        // Nuetzliche Folge: ist der Determinismus wegen HYBRID ohnehin aus,
        // faellt auch die Sperre fuer die Nachbesserung weg. Sie wird dann
        // NICHT stillschweigend weggelassen, sondern wieder eingeschaltet --
        // gerade auf einem verfeinerten Netz, wo omega*C gegen K um zwei
        // Groessenordnungen faellt und die Kondition entsprechend schlechter
        // wird, ist sie dort am nuetzlichsten.
        bool hybrid = opt.hybrid_memory;
        if (const char* v = std::getenv("FEMBEM_CUDSS_HYBRID")) hybrid = (std::atoi(v) != 0);

        int det = opt.deterministic ? 1 : 0;
        if (const char* v = std::getenv("FEMBEM_CUDSS_DETERMINISTIC")) {
            det = std::atoi(v) ? 1 : 0;
        }
        int ir = opt.refinement_steps;
        if (const char* v = std::getenv("FEMBEM_CUDSS_IR")) {
            ir = std::atoi(v);
            if (ir < 0) throw std::runtime_error(
                "FEMBEM_CUDSS_IR darf nicht negativ sein.");
        }

        if (hybrid && det) {
            std::cout << "[cuDSS] Determinismus abgeschaltet: cuDSS 0.8.0 traegt "
                         "DETERMINISTIC_MODE und HYBRID_MEMORY_MODE nicht "
                         "gemeinsam (ANALYSIS gibt NOT_SUPPORTED). Der "
                         "Hybridmodus hat Vorrang, weil das System sonst gar "
                         "nicht rechnet. ACHTUNG: die Ergebnisse dieses Laufs "
                         "sind nicht bitweise reproduzierbar, "
                         "test_determinism_fingerprint gilt fuer ihn nicht.\n";
            det = 0;
        }
        if (det && ir > 0) {
            std::cout << "[cuDSS] Nachbesserung von " << ir << " auf 0 gesetzt: "
                         "cuDSS 0.8.0 traegt DETERMINISTIC_MODE und IR_N_STEPS "
                         "nicht gemeinsam (ANALYSIS gibt NOT_SUPPORTED). "
                         "Determinismus hat Vorrang; mit "
                         "FEMBEM_CUDSS_DETERMINISTIC=0 gilt die Umkehrung.\n";
            ir = 0;
        } else if (!det && ir > 0) {
            std::cout << "[cuDSS] Nachbesserung aktiv (" << ir << " Schritte) -- "
                         "ohne Determinismus ist sie erlaubt.\n";
        }

        std::string det_name = "(nicht gesetzt)";
        if (!skipped("FEMBEM_CUDSS_DETERMINISTIC")) {
            det_name = det ? "ja" : "nein";
            check(cudssConfigSet(config, CUDSS_CONFIG_DETERMINISTIC_MODE,
                                 &det, sizeof(det)),
                  "ConfigSet DETERMINISTIC_MODE");
        }

        std::string ir_name = "(nicht gesetzt)";
        if (!skipped("FEMBEM_CUDSS_IR")) {
            ir_name = std::to_string(ir);
            check(cudssConfigSet(config, CUDSS_CONFIG_IR_N_STEPS, &ir, sizeof(ir)),
                  "ConfigSet IR_N_STEPS");
        }

        // Die tatsaechlich gesetzte Kombination protokollieren. Ohne diese
        // Zeile steht in einem gescheiterten Lauf nirgends, was probiert
        // wurde -- und bei einer Bisektion ueber Umgebungsvariablen ist das
        // die einzige Aufzeichnung davon.
        g_last_config = std::string("Ordnung ") + alg_name
                      + ", Pivot " + pivot_name
                      + ", deterministisch " + det_name
                      + ", Nachbesserung " + ir_name
                      + ", Matrixtyp "
                      + (kind == MatrixKind::SymmetricIndefinite
                             ? "SYMMETRIC/UPPER" : "GENERAL/FULL");
        std::cout << "[cuDSS] Konfiguration: " << g_last_config << "\n";

        // --- Hybridspeicher ----------------------------------------------
        // Verlagert Teile des Faktors in den Hauptspeicher. Kostet Bandbreite,
        // rettet aber Systeme, deren Faktor nicht in den VRAM passt -- und das
        // ist der haeufigste Grund, an dem eine Verfeinerungsstufe scheitert:
        // die Fuellung waechst deutlich schneller als das System.
        //
        // Der Schalter wird oben gelesen, weil er ueber den Determinismus
        // entscheidet. Gesetzt wird er hier -- vor ANALYSIS, das ist
        // Bedingung.
        //
        //   FEMBEM_CUDSS_HYBRID       = 0 | 1
        //   FEMBEM_CUDSS_HYBRID_LIMIT = Obergrenze des Geraetespeichers in MiB
        if (hybrid) {
            int on = 1;
            check(cudssConfigSet(config, CUDSS_CONFIG_HYBRID_MEMORY_MODE,
                                 &on, sizeof(on)),
                  "ConfigSet HYBRID_MEMORY_MODE");

            std::int64_t limit = opt.hybrid_device_memory_limit;
            if (const char* v = std::getenv("FEMBEM_CUDSS_HYBRID_LIMIT"))
                limit = static_cast<std::int64_t>(std::atol(v)) * 1024 * 1024;
            if (limit > 0) {
                check(cudssConfigSet(config, CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT,
                                     &limit, sizeof(limit)),
                      "ConfigSet HYBRID_DEVICE_MEMORY_LIMIT");
            }
            std::cout << "[cuDSS] Hybridspeicher aktiv"
                      << (limit > 0
                            ? (", Geraeteanteil hoechstens "
                               + std::to_string(limit / (1024 * 1024)) + " MiB")
                            : std::string(", ohne Obergrenze"))
                      << ". Langsamer als reiner Geraetespeicher, aber der Faktor "
                         "muss nicht hineinpassen.\n";
        }
    }

    FactorizationReport run_factorization(int phase, const char* what) {
        if (!analyzed) throw std::runtime_error(
            "SparseDirectSolver: analyze() muss vor der Faktorisierung laufen.");

        check(cudssExecute(handle, phase, config, data, A, x, b), what);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error(
                std::string("SparseDirectSolver: cudaDeviceSynchronize nach ") + what
                + " fehlgeschlagen.");
        }

        FactorizationReport rep;
        size_t written = 0;

        // CUDSS_DATA_INFO ist die Stelle, an der eine singulaere Matrix
        // gemeldet wird. Wer sie nicht liest, bekommt eine "Loesung" aus einer
        // Zerlegung, die es nicht gibt. Genau der Fall, der eintritt, wenn die
        // Regularisierung (FemDirectBC) vergessen wurde: K ist in der Luft
        // singulaer.
        cudssDataGet(handle, data, CUDSS_DATA_INFO, &rep.info,
                     sizeof(rep.info), &written);
        if (rep.info != 0) {
            std::ostringstream e;
            e << "cuDSS meldet CUDSS_DATA_INFO = " << rep.info
              << " nach der Faktorisierung. Bei einer positiven Zahl ist das "
                 "der Index der ersten singulaeren Zeile. Haeufigste Ursache "
                 "auf diesem Pfad: die Regularisierung des Curl-Curl-Nullraums "
                 "fehlt (fem_direct::regularize_nonconducting_regions) oder der "
                 "Dirichlet-Rand wurde nicht eingearbeitet "
                 "(apply_homogeneous_dirichlet).";
            throw std::runtime_error(e.str());
        }

        cudssDataGet(handle, data, CUDSS_DATA_LU_NNZ, &rep.lu_nnz,
                     sizeof(rep.lu_nnz), &written);
        cudssDataGet(handle, data, CUDSS_DATA_NPIVOTS, &rep.npivots,
                     sizeof(rep.npivots), &written);

        if (kind == MatrixKind::SymmetricIndefinite) {
            // Traegheitsindex: Zahl der positiven und negativen Eigenwerte.
            //
            // ERWARTUNG, nicht Beweis: in der Grenze C -> 0 ist die 2n-Form
            // diag(K, -K), also n positive und n negative Eigenwerte. Eine
            // deutlich andere Aufteilung ist ein Hinweis darauf, dass etwas
            // nicht stimmt -- ein Vorzeichen in der Blockform etwa. Der
            // Uebergang von C = 0 zu C > 0 laesst sich hier nicht ohne
            // Weiteres als stetig nachweisen, weil K in der Luft singulaer
            // ist; die Zahl ist deshalb eine Diagnose, keine Zusicherung.
            std::int64_t inertia[2] = {0, 0};
            if (cudssDataGet(handle, data, CUDSS_DATA_INERTIA, inertia,
                             sizeof(inertia), &written) == CUDSS_STATUS_SUCCESS) {
                rep.inertia_pos = inertia[0];
                rep.inertia_neg = inertia[1];
            }
        }

        if (nnz > 0) {
            rep.fill_ratio = static_cast<double>(rep.lu_nnz) / static_cast<double>(nnz);
        }
        factorized = true;
        return rep;
    }
};

SparseDirectSolver::SparseDirectSolver() : m_impl(new Impl()) {}
SparseDirectSolver::~SparseDirectSolver() { delete m_impl; }

bool SparseDirectSolver::available() { return true; }

std::string SparseDirectSolver::version() {
    std::ostringstream s;
    s << CUDSS_VERSION_MAJOR << "." << CUDSS_VERSION_MINOR << "." << CUDSS_VERSION_PATCH;
    return s.str();
}

void SparseDirectSolver::set_options(const SolverOptions& opt) { m_impl->opt = opt; }

void SparseDirectSolver::analyze(int num_rows, int nnz,
                                 int* row_offsets, int* col_indices, double* values,
                                 MatrixKind kind)
{
    // Bei SymmetricIndefinite wird nur ein Dreieck uebergeben, dort fehlen also
    // Eintraege -- die Diagonale muss aber trotzdem da sein.
    const std::string problem = validate_csr(num_rows, nnz, row_offsets, col_indices, true);
    if (!problem.empty()) {
        throw std::runtime_error("SparseDirectSolver::analyze: ungueltiges CSR-Muster: "
                                 + problem);
    }

    m_impl->kind = kind;
    m_impl->n = num_rows;
    m_impl->nnz = nnz;
    m_impl->apply_config();

    const cudssMatrixType_t mtype =
        (kind == MatrixKind::SymmetricIndefinite) ? CUDSS_MTYPE_SYMMETRIC
                                                  : CUDSS_MTYPE_GENERAL;
    const cudssMatrixViewType_t mview =
        (kind == MatrixKind::SymmetricIndefinite) ? CUDSS_MVIEW_UPPER
                                                  : CUDSS_MVIEW_FULL;

    if (m_impl->A) { cudssMatrixDestroy(m_impl->A); m_impl->A = nullptr; }

    // rowEnd = nullptr bedeutet Standard-CSR (row_offsets[r+1] ist das Ende
    // von Zeile r). Offsets und Indizes sind int, also CUDSS_R_32I.
    check(cudssMatrixCreateCsr(&m_impl->A, num_rows, num_rows, nnz,
                               row_offsets, nullptr, col_indices, values,
                               CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                               mtype, mview, CUDSS_BASE_ZERO),
          "cudssMatrixCreateCsr");

    // Platzhalter fuer rechte Seite und Loesung. Die Zeiger werden in solve()
    // gesetzt; fuer die Analysephase braucht cuDSS gueltige Wrapper, aber die
    // Werte spielen dort keine Rolle.
    if (m_impl->b) { cudssMatrixDestroy(m_impl->b); m_impl->b = nullptr; }
    if (m_impl->x) { cudssMatrixDestroy(m_impl->x); m_impl->x = nullptr; }
    check(cudssMatrixCreateDn(&m_impl->b, num_rows, 1, num_rows, nullptr,
                              CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
          "cudssMatrixCreateDn (rhs)");
    check(cudssMatrixCreateDn(&m_impl->x, num_rows, 1, num_rows, nullptr,
                              CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
          "cudssMatrixCreateDn (solution)");

    check(cudssExecute(m_impl->handle, CUDSS_PHASE_ANALYSIS,
                       m_impl->config, m_impl->data,
                       m_impl->A, m_impl->x, m_impl->b),
          "cudssExecute ANALYSIS");
    if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("SparseDirectSolver::analyze: cudaDeviceSynchronize fehlgeschlagen.");
    }

    m_impl->analyzed = true;
    m_impl->factorized = false;
}

FactorizationReport SparseDirectSolver::factorize() {
    return m_impl->run_factorization(CUDSS_PHASE_FACTORIZATION,
                                     "cudssExecute FACTORIZATION");
}

FactorizationReport SparseDirectSolver::refactorize() {
    if (!m_impl->factorized) {
        throw std::runtime_error(
            "SparseDirectSolver::refactorize: es gibt noch keine Faktorisierung, "
            "deren Pivotfolge wiederverwendet werden koennte. Erst factorize() "
            "aufrufen.");
    }
    return m_impl->run_factorization(CUDSS_PHASE_REFACTORIZATION,
                                     "cudssExecute REFACTORIZATION");
}

void SparseDirectSolver::solve(double* rhs, double* x) {
    if (!m_impl->factorized) {
        throw std::runtime_error(
            "SparseDirectSolver::solve: keine Faktorisierung vorhanden. "
            "factorize() muss vorher laufen.");
    }
    if (!rhs || !x) throw std::runtime_error("SparseDirectSolver::solve: Zeiger ist null.");

    check(cudssMatrixSetValues(m_impl->b, rhs), "cudssMatrixSetValues (rhs)");
    check(cudssMatrixSetValues(m_impl->x, x),   "cudssMatrixSetValues (solution)");

    check(cudssExecute(m_impl->handle, CUDSS_PHASE_SOLVE,
                       m_impl->config, m_impl->data,
                       m_impl->A, m_impl->x, m_impl->b),
          "cudssExecute SOLVE");
    if (cudaDeviceSynchronize() != cudaSuccess) {
        throw std::runtime_error("SparseDirectSolver::solve: cudaDeviceSynchronize fehlgeschlagen.");
    }
}

int SparseDirectSolver::num_rows() const { return m_impl->n; }

// =============================================================================
#else  // ohne cuDSS

namespace {
const char* kNoCudss =
    "Dieser Aufruf braucht cuDSS, es wurde aber ohne gebaut. Neu konfigurieren "
    "mit -DFEMBEM_WITH_CUDSS=ON und -DCMAKE_PREFIX_PATH=<cuDSS-Verzeichnis>. "
    "Siehe BUILD.md, Abschnitt cuDSS. Alle anderen Pfade des Loesers "
    "(transient, BEM-FEM, harmonisch iterativ) funktionieren ohne cuDSS.";
}

struct SparseDirectSolver::Impl { SolverOptions opt; int n = 0; };

SparseDirectSolver::SparseDirectSolver() : m_impl(new Impl()) {}
SparseDirectSolver::~SparseDirectSolver() { delete m_impl; }

bool SparseDirectSolver::available() { return false; }
std::string SparseDirectSolver::version() { return std::string(); }

void SparseDirectSolver::set_options(const SolverOptions& opt) { m_impl->opt = opt; }

void SparseDirectSolver::analyze(int, int, int*, int*, double*, MatrixKind) {
    throw std::runtime_error(std::string("SparseDirectSolver::analyze: ") + kNoCudss);
}
FactorizationReport SparseDirectSolver::factorize() {
    throw std::runtime_error(std::string("SparseDirectSolver::factorize: ") + kNoCudss);
}
FactorizationReport SparseDirectSolver::refactorize() {
    throw std::runtime_error(std::string("SparseDirectSolver::refactorize: ") + kNoCudss);
}
void SparseDirectSolver::solve(double*, double*) {
    throw std::runtime_error(std::string("SparseDirectSolver::solve: ") + kNoCudss);
}
int SparseDirectSolver::num_rows() const { return m_impl->n; }

#endif  // FEMBEM_HAVE_CUDSS

}  // namespace fem_direct
