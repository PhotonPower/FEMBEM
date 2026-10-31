#include "CoupledSolver.cuh"
#include "AmsPreconditioner.cuh"
#include "RitzSpectrum.cuh"
#include "Backend.cuh"
#include "HostLinalg.cuh"
#include "BemIntegration.cuh"
#include "CouplingKernels.cuh"
#include <iostream>
#include <cmath>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <limits>

// =================================================================================
// Hilfs-Kernel fuer die Symmetrische Diagonale Aequilibrierung (SDE)
// =================================================================================

// Ab hier gilt durchgehend das Muster aus include/Backend.cuh: eine
// __host__ __device__ Rumpffunktion <name>_body(i, ...) traegt die Rechnung,
// der __global__ Kernel bildet nur den Index. BF_LAUNCH_1D(_PAR) an der
// Aufrufstelle entscheidet zur Laufzeit zwischen Kernelstart und Host-Schleife.
BF_HD inline void k_vec_mult_body(int i, int n, const double* a, const double* b, double* out) {
    (void)n; out[i] = a[i] * b[i];
}
__global__ void k_vec_mult(int n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) k_vec_mult_body(i, n, a, b, out);
}

BF_HD inline void k_vec_div_body(int i, int n, const double* a, const double* b, double* out) {
    (void)n; out[i] = a[i] / b[i];
}
__global__ void k_vec_div(int n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) k_vec_div_body(i, n, a, b, out);
}

BF_HD inline void k_extract_fem_diag_and_build_D_body(
    int row, int num_rows, const int* row_offsets, const int* col_indices,
    const double* values, double* D_scale)
{
    (void)num_rows;

    double diag_val = 0.0;
    for (int j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
        if (col_indices[j] == row) {
            diag_val = values[j];
            break;
        }
    }
    // D_ii = 1 / sqrt(|A_ii|)
    D_scale[row] = 1.0 / sqrt(fabs(diag_val) + 1e-14);
}
__global__ void k_extract_fem_diag_and_build_D(
    int num_rows, const int* __restrict__ row_offsets, const int* __restrict__ col_indices,
    const double* __restrict__ values, double* __restrict__ D_scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    k_extract_fem_diag_and_build_D_body(row, num_rows, row_offsets, col_indices, values, D_scale);
}

BF_HD inline void k_build_D_bem_body(int i, int num_bem_dofs, int offset,
                                     const double* bem_inv_diag, double* D_scale) {
    (void)num_bem_dofs;
    // bem_inv_diag enthaelt bereits 1 / (-mu0 * G_ii).
    // Wir wollen D = 1 / sqrt(|-mu0 * G_ii|) = sqrt(|bem_inv_diag|).
    D_scale[offset + i] = sqrt(fabs(bem_inv_diag[i]));
}
__global__ void k_build_D_bem(int num_bem_dofs, int offset, const double* __restrict__ bem_inv_diag, double* __restrict__ D_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bem_dofs) return;
    k_build_D_bem_body(i, num_bem_dofs, offset, bem_inv_diag, D_scale);
}

BF_HD inline void k_scale_csr_matrix_body(
    int row, int num_rows, const int* row_offsets, const int* col_indices,
    double* values, const double* D_scale)
{
    (void)num_rows;

    double D_row = D_scale[row];
    for (int j = row_offsets[row]; j < row_offsets[row + 1]; ++j) {
        int col = col_indices[j];
        // \tilde{A}_{ij} = D_i * A_{ij} * D_j
        values[j] = D_row * values[j] * D_scale[col];
    }
}
__global__ void k_scale_csr_matrix(
    int num_rows, const int* __restrict__ row_offsets, const int* __restrict__ col_indices,
    double* __restrict__ values, const double* __restrict__ D_scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    k_scale_csr_matrix_body(row, num_rows, row_offsets, col_indices, values, D_scale);
}

// der k_negate_vector-Kernel
// (BEM-Diagonale nach SDE-Skalierung war -1.0, siehe Fix 1) wird nicht mehr
// gebraucht -- apply_physical() negiert jetzt die gesamte K-Gleichung, die
// skalierte Diagonale ist dadurch +1.0, der Preconditioner braucht dafuer
// nur noch die reine Identitaets-Kopie (siehe apply_preconditioner()).

// ein k_apply_bgs_coupling-Kernel
// (Block-Gauss-Seidel ueber die lokale Randmassenmatrix T) wurde hier
// versucht, aber empirisch schlechter als die einfache Negation (siehe
// Kommentar in apply_preconditioner()) -- entfernt statt nur inaktiv
// stehengelassen, um keinen toten Code als Referenz zu suggerieren.

// der Nullraum-Penalty-Term (I-P)*x, ausgefuehrt im
// Tilde-Raum statt im physikalischen Raum (siehe apply()). x_proj muss
// P_tilde(x) sein (via project_gauge_tilde() erzeugt); fuer den BEM-Anteil
// (i >= total_dofs) muss der Aufrufer x_proj[i] == x[i] setzen, damit der
// Term dort automatisch 0 beitraegt.
BF_HD inline void k_add_null_space_residual_tilde_body(int i, int n, const double* x,
                                                       const double* x_proj, double* y) {
    (void)n; y[i] += x[i] - x_proj[i];
}
__global__ void k_add_null_space_residual_tilde(int n, const double* __restrict__ x, const double* __restrict__ x_proj, double* __restrict__ y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) k_add_null_space_residual_tilde_body(i, n, x, x_proj, y);
}

// =================================================================================
// Implementierung: CoupledMVOperator (Aufgabe 6.1)
// =================================================================================

// Initialisierungsreihenfolge entspricht der Deklarationsreihenfolge in der
// Klasse (cusparse_buffer_size, d_cusparse_buffer, total_dofs) -- sonst warnt
// GCC mit -Wreorder, weil die Liste nicht die tatsaechliche Reihenfolge abbildet.
CoupledMVOperator::CoupledMVOperator(int N)
    : cusparse_buffer_size(0), d_cusparse_buffer(nullptr), total_dofs(N) {
    // keine Handles anlegen. cublasCreate/cusparseCreate
    // erzeugen einen CUDA-Kontext und schlagen auf einer Maschine ohne
    // Geraet/Treiber fehl -- genau das soll der CPU-Modus vermeiden. Die
    // Handles bleiben nullptr; die bf::-Wrapper in HostLinalg ignorieren sie
    // in diesem Fall.
    if (bf::on_gpu()) {
        CUBLAS_CHECK(cublasCreate(&cublas));
        CUSPARSE_CHECK(cusparseCreate(&cusparse));
    } else {
        cublas  = nullptr;
        cusparse = nullptr;
    }

    // Allokiere einen moderaten Puffer für ACA (z = V^T * x).
    // Ein Rang > 2000 ist bei ACA höchst unüblich, dies deckt alles ab.
    d_temp_z.allocate(2000); 
    d_ilu_temp.allocate(static_cast<size_t>(N)); // Puffer fuer ILU-Vorwaertseinsetzen
    d_ilu_probe.allocate(static_cast<size_t>(N)); // getrennter Ausgabepuffer fuer SpSV-Analyse
}

CoupledMVOperator::~CoupledMVOperator() {
    // im CPU-Pfad wurde nichts davon angelegt -- keine
    // Deskriptoren, keine Puffer, keine Handles. Die Aufraeumarbeit besteht
    // hier ausschliesslich aus den std::vector-Membern, die sich selbst
    // abbauen. Ein cusparseDestroy* auf einem nullptr-Handle waere undefiniert.
    if (bf::on_cpu()) {
        return;
    }

    // ILU(0)-Ressourcen freigeben (umgekehrte Reihenfolge zur Anlage)
    if (d_spsv_buffer_L) cudaFree(d_spsv_buffer_L);
    if (d_spsv_buffer_U) cudaFree(d_spsv_buffer_U);
    if (d_ilu_buffer) cudaFree(d_ilu_buffer);
    if (spsv_info_L) cusparseSpSV_destroyDescr(spsv_info_L);
    if (spsv_info_U) cusparseSpSV_destroyDescr(spsv_info_U);
    if (mat_L) cusparseDestroySpMat(mat_L);
    if (mat_U) cusparseDestroySpMat(mat_U);
    if (ilu_info) cusparseDestroyCsrilu02Info(ilu_info);
    if (ilu_descr) cusparseDestroyMatDescr(ilu_descr);
    if (d_ilu_values) cudaFree(d_ilu_values);

    if (d_cusparse_buffer) {
        cudaFree(d_cusparse_buffer);
    }
    cusparseDestroySpMat(mat_FEM);
    cusparseDestroy(cusparse);
    cublasDestroy(cublas);
}

// =================================================================================
// Endlichkeits-Audit
// =================================================================================
//
// Warum das noetig ist, obwohl es ueberall Diagnoseausgaben gibt: min/max-
// Schleifen koennen NaN prinzipiell nicht sehen. std::min(a, NaN) liefert a
// (jeder Vergleich mit NaN ist falsch), NaN faellt also lautlos durch jede
// min/max-Statistik. Genau daran ist die Suche nach den NaN-Kraeften lange
// vorbeigelaufen: "D_scale FEM-Anteil: min = 1.14e-05, max = 0.000131" schliesst
// NICHT aus, dass in D_scale NaN stehen. Dieselbe Falle steckte in bf::dnrm2.
//
// Ein Audit muss deshalb explizit zaehlen statt zu vergleichen.
// Liefert einen HOST-lesbaren Zeiger auf n Doubles, die irgendwo liegen
// koennen: in Managed Memory (DeviceArray), in rohem cudaMalloc-Speicher
// (d_ilu_values, die BEM-Bloecke) oder im gewoehnlichen Host-Speicher
// (host_D.data()). Liegt v auf dem Geraet, wird nach scratch kopiert und
// dessen Zeiger zurueckgegeben; sonst kommt v unveraendert zurueck.
//
// Zentral, weil der Zeigertyp an der Aufrufstelle nicht abzulesen ist -- und
// weil genau dieses Muster im GPU-Modus schon zwei Abstuerze verursacht hat
// (audit_finite auf d_ilu_values, audit_bem_blocks auf den Blockdaten). Im
// CPU-Modus ist es ein No-Op.
static const double* host_readable(const double* v, size_t n,
                                   std::vector<double>& scratch) {
    if (v == nullptr || n == 0 || bf::on_cpu()) return v;
    cudaPointerAttributes attr{};
    const cudaError_t st = cudaPointerGetAttributes(&attr, v);
    // Bei nicht registriertem Host-Speicher setzt die Abfrage den
    // CUDA-Fehlerzustand. Wird er nicht abgeraeumt, schlaegt er beim naechsten
    // CUDA_CHECK an einer unbeteiligten Stelle zu -- und die Fehlersuche
    // beginnt an der falschen Stelle von vorn.
    cudaGetLastError();
    const bool on_device = (st == cudaSuccess)
                        && (attr.type == cudaMemoryTypeDevice
                            || attr.type == cudaMemoryTypeManaged);
    if (!on_device) return v;
    scratch.resize(n);
    bf::memcpy_d(scratch.data(), v, n * sizeof(double), cudaMemcpyDeviceToHost);
    return scratch.data();
}

static void audit_finite(const char* label, const double* v, int n) {
    if (v == nullptr || n <= 0) return;
    bf::sync();

    // v kann GERAETESPEICHER sein, und zwar nicht nur Managed Memory:
    // d_ilu_values kommt aus einem einfachen cudaMalloc. Ein Host-Zugriff
    // darauf ist ein Segfault -- und der schlaegt erst HIER zu, also weit weg
    // von der Ursache. Im CPU-Modus zeigt derselbe Zeiger auf
    // m_cpu_ilu_values.data(), also auf echten Host-Speicher; deshalb ist das
    // nie aufgefallen, solange nur der CPU-Pfad benutzt wurde.
    //
    // Kosten: eine Kopie von n Doubles je Aufruf auf Geraetespeicher.
    // Gegenueber einer gekoppelten apply()-Anwendung mit ACA faellt das nicht
    // ins Gewicht, und die haeufigen Aufrufe im apply()-Pfad sind ohnehin
    // durch audit_this_call gedrosselt.
    std::vector<double> host_copy;
    const double* p = host_readable(v, static_cast<size_t>(n), host_copy);

    int n_nan = 0, n_inf = 0, first = -1;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(p[i]))      { ++n_nan; if (first < 0) first = i; }
        else if (std::isinf(p[i])) { ++n_inf; if (first < 0) first = i; }
    }
    if (n_nan == 0 && n_inf == 0) return;   // Normalfall: schweigen
    std::cout << "  [AUDIT] " << label << ": " << n_nan << " NaN und " << n_inf
              << " Inf von " << n << " Eintraegen";
    if (first >= 0) std::cout << ", erster bei Index " << first;
    std::cout << "\n";
}

void CoupledMVOperator::setup_fem_matrix(int num_rows, int nnz, int* d_row_offsets, int* d_col_indices, double* d_values) {
    // Rohe CSR-Zeiger merken -- compute_ilu() braucht sie, cusparseCreateCsr
    // haelt selbst keine Kopie. Eine neu assemblierte Matrix (naechster
    // Zeitschritt) invalidiert eine evtl. vorher berechnete Faktorisierung.
    m_ilu_num_rows = num_rows;
    m_ilu_nnz      = nnz;
    m_row_offsets  = d_row_offsets;
    m_col_indices  = d_col_indices;
    m_values       = d_values;
    ilu_ready = false;

    // Erste Station der Kette: ist die assemblierte FEM-Matrix selbst schon
    // defekt? Wichtig, weil ein NaN hier ueber A*x auch bei x == 0 in jede
    // spaetere Groesse durchschlaegt (0 * NaN = NaN).
    audit_finite("FEM-Matrixwerte (assembliert)", d_values, nnz);

    // kein Deskriptor. apply_physical() rechnet das
    // Matrix-Vektor-Produkt dort direkt aus m_row_offsets/m_col_indices/
    // m_values (bf::csr_spmv), die oben ohnehin schon gemerkt werden.
    if (bf::on_cpu()) {
        return;
    }

    CUSPARSE_CHECK(cusparseCreateCsr(&mat_FEM, num_rows, num_rows, nnz,
                                     d_row_offsets, d_col_indices, d_values,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
}

// =================================================================================
// Gather / Scatter zwischen FEM-Kantenraum und permutiertem BEM-Elementraum
// Siehe include/BemCoupling.h fuer die Herleitung.
// =================================================================================
BF_HD inline void k_gather_bem_body(int i, const double* x_fem, const int* map,
                                    const int* sign, double* x_bem, int num_bem_dofs)
{
    (void)num_bem_dofs;
    x_bem[i] = static_cast<double>(sign[i]) * x_fem[map[i]];
}
__global__ void k_gather_bem(const double* __restrict__ x_fem,
                             const int* __restrict__ map,
                             const int* __restrict__ sign,
                             double* __restrict__ x_bem,
                             int num_bem_dofs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bem_dofs) return;
    k_gather_bem_body(i, x_fem, map, sign, x_bem, num_bem_dofs);
}

__global__ void k_scatter_add_bem(const double* __restrict__ y_bem,
                                  const int* __restrict__ map,
                                  const int* __restrict__ sign,
                                  double* __restrict__ y_fem,
                                  int num_bem_dofs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bem_dofs) return;
    // Jede Randkante gehoert zu genau zwei Randquads, zwei BEM-Freiheitsgrade
    // treffen also dieselbe FEM-Kante. Daher atomar addieren.
    atomicAdd(&y_fem[map[i]], static_cast<double>(sign[i]) * y_bem[i]);
}

// Projektion P: kopiert x und nullt die geeichten Eintraege.
BF_HD inline void k_project_gauge_body(int i, const double* x, const char* is_gauged,
                                       double* x_out, int n)
{
    (void)n;
    x_out[i] = is_gauged[i] ? 0.0 : x[i];
}
__global__ void k_project_gauge(const double* __restrict__ x,
                                const char* __restrict__ is_gauged,
                                double* __restrict__ x_out,
                                int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    k_project_gauge_body(i, x, is_gauged, x_out, n);
}

// Ergaenzt (I - P)*x, wobei P jetzt die VOLLE Projektion ist (Baum-Koordinaten-
// maske UND alle registrierten Kohomologie-Projektoren, siehe apply()). x_proj
// enthaelt bereits P(x), berechnet in apply() vor dem eigentlichen
// Matrix-Vektor-Produkt. Diese Fassung ersetzt das fruehere k_finish_gauge
// (das nur die Koordinatenmaske kannte): x - x_proj ist fuer eine reine
// Koordinatenmaske exakt x_i auf geeichten Zeilen und 0 sonst -- identisches
// Verhalten wie vorher --, erfasst zusaetzlich aber auch den Anteil, den die
// Kohomologie-Projektoren herausprojiziert haben, ohne dass diese Funktion
// wissen muss, WAS P im Detail tut.
BF_HD inline void k_add_null_space_residual_body(int i, const double* x,
                                                 const double* x_proj, double* y, int n)
{
    (void)n;
    y[i] += x[i] - x_proj[i];
}
__global__ void k_add_null_space_residual(const double* __restrict__ x,
                                          const double* __restrict__ x_proj,
                                          double* __restrict__ y,
                                          int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    k_add_null_space_residual_body(i, x, x_proj, y, n);
}

void CoupledMVOperator::setup_gauge(const char* d_flags) {
    if (d_flags == nullptr) {
        throw std::runtime_error("setup_gauge: d_flags ist nullptr.");
    }
    d_is_gauged = d_flags;
    d_x_proj.allocate(static_cast<size_t>(total_dofs));
}

void CoupledMVOperator::add_cohomology_projector(const double* d_vector, double norm_sq) {
    if (d_vector == nullptr) {
        throw std::runtime_error("add_cohomology_projector: d_vector ist nullptr.");
    }
    if (!(norm_sq > 0.0)) {
        throw std::runtime_error("add_cohomology_projector: norm_sq muss positiv sein "
            "(erhalten: " + std::to_string(norm_sq) + "). Ein (nahezu) verschwindender "
            "Generator deutet auf einen Fehler in find_cohomology_generators hin.");
    }
    if (d_is_gauged == nullptr) {
        throw std::runtime_error("add_cohomology_projector: setup_gauge() muss zuerst "
            "aufgerufen werden -- die Kohomologie-Projektoren setzen auf der "
            "Baum-Koordinatenmaske auf.");
    }
    CohomologyProjector p;
    p.d_vector = d_vector;
    p.norm_sq = norm_sq;
    cohomology_projectors.push_back(p);
}

// Volle Eichungs-Projektion P: erst die Baum-Koordinatenmaske
// (k_project_gauge), danach alle registrierten Kohomologie-Projektoren
// (x -= c*(c^T x)/(c^T c)) -- exakt in dieser Reihenfolge, siehe Kommentar
// bei cohomology_projectors fuer die Idempotenz-Begruendung. d_in und d_out
// duerfen auf denselben Puffer zeigen (in-place), die Kernel-/BLAS-Aufrufe
// unten sind dafuer sicher (jede Operation liest, was sie zuletzt selbst
// geschrieben hat, keine ueberlappenden Lese-/Schreibkonflikte).
void CoupledMVOperator::project_gauge(const double* d_in, double* d_out) {
    if (d_is_gauged == nullptr) {
        // Keine Eichung aktiv: P = Identitaet.
        if (d_out != d_in) {
            bf::memcpy_d(d_out, d_in, static_cast<size_t>(total_dofs) * sizeof(double),
                                        cudaMemcpyDeviceToDevice);
        }
        return;
    }

    BF_LAUNCH_1D_PAR(k_project_gauge, total_dofs, d_in, d_is_gauged, d_out, total_dofs);

    for (const CohomologyProjector& proj : cohomology_projectors) {
        double dot = 0.0;
        bf::ddot(cublas, total_dofs, proj.d_vector, d_out, &dot);
        const double scale = -(dot / proj.norm_sq);
        bf::daxpy(cublas, total_dofs, scale, proj.d_vector, d_out);
    }
}

// project_gauge_tilde() fuer Vektoren im aequilibrierten
// Tilde-Raum -- KORRIGIERT gegenueber der vorherigen Fassung (Fix 2). Die
// alte Variante (x_real = D*x_tilde; project_gauge(x_real); x_tilde =
// x_real/D) berechnete fuer die Kohomologie-Projektoren effektiv
//     x_tilde -= (c/D) * (c^T D x_tilde) / (c^T c)
// Das ist aber NICHT die orthogonale Projektion im Tilde-Raum entlang des
// tatsaechlichen Tilde-Nullvektors c_tilde = D^-1*c (folgt aus A*c=0 =>
// \tilde{A}*(D^-1*c) = D*A*D*D^-1*c = D*A*c = 0). Die korrekte orthogonale
// Projektion waere
//     x_tilde -= c_tilde * (c_tilde^T x_tilde) / (c_tilde^T c_tilde)
// mit c_tilde = c/D -- eine strukturell ANDERE Gewichtung (c_i/D_i statt
// c_i*D_i pro Eintrag), die bei stark variierendem D (siehe d_bem_inv_diag-
// Diagnose: mehrere Groessenordnungen) erheblich abweicht. Die alte Fassung
// war folglich eine schiefe (nicht orthogonale) Projektion im Tilde-Raum --
// project_gauge_tilde_orthogonal() unten berechnet stattdessen c_tilde
// direkt und projiziert damit orthogonal.
void CoupledMVOperator::project_gauge_tilde(const double* d_in_tilde, double* d_out_tilde) {
    if (!is_equilibrated || d_is_gauged == nullptr) {
        // Ohne aktive SDE-Skalierung ist Tilde == physikalisch (D = 1);
        // ohne aktive Eichung ist project_gauge() ohnehin Identitaet.
        project_gauge(d_in_tilde, d_out_tilde);
        return;
    }

    if (d_in_tilde != d_out_tilde) {
        const size_t sys_dofs = static_cast<size_t>(total_dofs) + static_cast<size_t>(num_bem_dofs);
        bf::memcpy_d(d_out_tilde, d_in_tilde, sys_dofs * sizeof(double),
                                    cudaMemcpyDeviceToDevice);
    }
    project_gauge_tilde_orthogonal(d_out_tilde);
}

// fuehrt die volle Eichungs-Projektion (Baum-Maske +
// Kohomologie) DIREKT im Tilde-Raum aus, orthogonal bzgl. des Standard-
// Skalarprodukts dort. Arbeitet in-place auf d_x_tilde. Die Baum-Maske
// (k_project_gauge) ist im Tilde-Raum identisch zur physikalischen Fassung,
// da D eine Diagonalmatrix ist (0 * D_i = 0, unabhaengig von D_i) -- nur
// die Kohomologie-Projektoren muessen mit c_tilde = c/D statt c arbeiten
// (siehe Kommentar bei project_gauge_tilde() oben). Nutzt d_apply_in_temp
// als Scratch fuer c_tilde -- sicher, wenn diese Funktion (wie an allen
// aktuellen Aufrufstellen) nicht waehrend eines aktiven apply()/
// project_gauge_tilde()-Aufrufs verschachtelt aufgerufen wird, der
// d_apply_in_temp noch selbst braucht.
void CoupledMVOperator::project_gauge_tilde_orthogonal(double* d_x_tilde) {
    if (d_is_gauged == nullptr) return;

    // 1. Baum-Koordinatenmaske (unveraendert gegenueber physikalisch).
    BF_LAUNCH_1D_PAR(k_project_gauge, total_dofs, d_x_tilde, d_is_gauged, d_x_tilde, total_dofs);

    // 2. Kohomologie-Projektoren, orthogonal mit c_tilde = D^-1 * c.
    for (const CohomologyProjector& proj : cohomology_projectors) {
        BF_LAUNCH_1D_PAR(k_vec_div, total_dofs,
                         total_dofs, proj.d_vector, d_D_scale.data(), d_apply_in_temp.data());

        double dot_num = 0.0, dot_den = 0.0;
        bf::ddot(cublas, total_dofs, d_x_tilde, d_apply_in_temp.data(), &dot_num);
        bf::ddot(cublas, total_dofs, d_apply_in_temp.data(), d_apply_in_temp.data(), &dot_den);

        if (dot_den > 1e-14) {
            const double scale = -(dot_num / dot_den);
            bf::daxpy(cublas, total_dofs, scale, d_apply_in_temp.data(), d_x_tilde);
        }
    }
}

// Fuellt d_bem_inv_diag mit dem sicheren Default 1.0 (= Identitaet).
// Wird in setup_bem_coupling() aufgerufen, bevor add_dense_block() ggf.
// einzelne Eintraege mit der echten (-mu_0*G)-Diagonale ueberschreibt. Ein
// BEM-DOF, der aus irgendeinem Grund (Luecke in der Block-Pair-Erzeugung)
// keinen Diagonalblock bekommt, faellt so auf das alte, funktionierende
// Identitaetsverhalten zurueck, statt mit 0 multipliziert (also komplett
// verworfen) zu werden.
BF_HD inline void k_fill_bem_inv_diag_default_body(int i, int n, double* inv_diag) {
    (void)n; inv_diag[i] = 1.0;
}
__global__ void k_fill_bem_inv_diag_default(int n, double* __restrict__ inv_diag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) k_fill_bem_inv_diag_default_body(i, n, inv_diag);
}

// Extrahiert die Diagonale eines dichten BEM-Blocks (Self-Patch,
// row_off == col_off) und invertiert sie fuer den Jacobi-Vorkonditionierer
// des K-Blocks: M_K = diag(-mu_0 * G), M_K^-1_ii = 1/(-mu_0 * G_ii).
BF_HD inline void k_extract_bem_diag_body(int i, int m, int row_off, const double* block,
                                          double* inv_diag, double mu_0, int num_bem_dofs) {
    // Bereichspruefung gegen die TATSAECHLICHE Laenge von inv_diag.
    //
    // Vorher wurde nur i < m geprueft, also gegen die Blockgroesse. Geschrieben
    // wird aber an inv_diag[row_off + i], und ob row_off + m innerhalb von
    // num_bem_dofs liegt, hat niemand geprueft. Im CPU-Modus schreibt ein
    // Ueberlauf still in fremden Heap-Speicher; im GPU-Modus liegt inv_diag im
    // Unified Memory, und ein Ueberlauf dort laesst den naechsten HOST-Zugriff
    // mit einem Speicherzugriffsfehler abstuerzen -- an einer voellig anderen
    // Stelle als der Ursache, naemlich in der Diagnoseausgabe.
    const int j = row_off + i;
    if (i >= m || j < 0 || j >= num_bem_dofs) return;
    const double val = block[i * m + i];   // m x m, column-major
    inv_diag[j] = 1.0 / (-mu_0 * val);
}
__global__ void k_extract_bem_diag(int m, int row_off, const double* __restrict__ block,
                                   double* __restrict__ inv_diag, double mu_0,
                                   int num_bem_dofs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    k_extract_bem_diag_body(i, m, row_off, block, inv_diag, mu_0, num_bem_dofs);
}

// Wendet die extrahierte (invertierte) Diagonale auf den K-Anteil des
// Vorkonditionierer-Eingabevektors an (Jacobi-Vorkonditionierung statt
// Identitaet, siehe apply_preconditioner()).
__global__ void k_apply_bem_jacobi(int num_bem_dofs, const double* __restrict__ inv_diag,
                                   const double* __restrict__ in, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_bem_dofs) return;
    out[i] = in[i] * inv_diag[i];
}

void CoupledMVOperator::setup_bem_coupling(int num_bem_dofs_in, const int* d_map, const int* d_sign) {
    if (num_bem_dofs_in <= 0 || d_map == nullptr || d_sign == nullptr) {
        throw std::runtime_error("setup_bem_coupling: ungueltige Argumente.");
    }
    num_bem_dofs  = num_bem_dofs_in;
    d_bem_to_fem  = d_map;
    d_bem_sign    = d_sign;
    d_x_bem.allocate(static_cast<size_t>(num_bem_dofs));
    d_y_bem.allocate(static_cast<size_t>(num_bem_dofs));

    // Diagonal-Puffer fuer den BEM-Jacobi-Vorkonditionierer -- jeden
    // Zeitschritt neu allozieren/fuellen (op wird pro Zeitschritt neu gebaut,
    // dense_blocks also auch neu befuellt, also muss auch diese Diagonale
    // neu extrahiert werden).
    d_bem_inv_diag.allocate(static_cast<size_t>(num_bem_dofs));
    BF_LAUNCH_1D_PAR(k_fill_bem_inv_diag_default, num_bem_dofs,
                     num_bem_dofs, d_bem_inv_diag.data());
}

// Geometrie fuer die T-Massenmatrix-Anwendung.
void CoupledMVOperator::setup_bem_mesh(int num_boundary_faces, const Mesh::QuadFace* d_faces, const double3* d_nodes) {
    if (num_boundary_faces <= 0 || d_faces == nullptr || d_nodes == nullptr) {
        throw std::runtime_error("setup_bem_mesh: ungueltige Argumente.");
    }
    m_num_boundary_faces = num_boundary_faces;
    d_mesh_faces = d_faces;
    d_mesh_nodes = d_nodes;
}

void CoupledMVOperator::add_dense_block_transposed(int m, int n, int row_off, int col_off,
                                                   double* d_mat) {
    DenseBlock b = {m, n, row_off, col_off, d_mat, /*transposed=*/true};
    dense_blocks.push_back(b);
    // Kein Diagonalabgriff: ein transponierter Block ist per Konstruktion ein
    // Nichtdiagonalblock (row_off != col_off), die Selbstbloecke werden
    // weiterhin ueber add_dense_block() eingetragen.
}

void CoupledMVOperator::add_dense_block(int m, int n, int row_off, int col_off, double* d_mat) {
    DenseBlock b = {m, n, row_off, col_off, d_mat, /*transposed=*/false};
    dense_blocks.push_back(b);

    // Diagonalblock (Self-Patch, jede BEM-Kante liegt in genau einem
    // solchen Block) -> Diagonale fuer den Jacobi-Preconditioner extrahieren.
    if (row_off == col_off && m == n && num_bem_dofs > 0) {
        // Ueberlauf hier ist kein theoretischer Fall: der Puffer ist
        // num_bem_dofs lang, row_off kommt aus der Blockstruktur. Passen sie
        // nicht zusammen, soll das gemeldet und nicht geschrieben werden.
        if (row_off + m > num_bem_dofs) {
            std::cerr << "[CoupledSolver] WARNUNG: Diagonalblock ragt ueber "
                         "d_bem_inv_diag hinaus (row_off " << row_off << " + m " << m
                      << " > num_bem_dofs " << num_bem_dofs
                      << "). Der ueberstehende Teil wird nicht extrahiert.\n";
        }
        BF_LAUNCH_1D_PAR(k_extract_bem_diag, m,
                         m, row_off, d_mat, d_bem_inv_diag.data(), m_mu_0, num_bem_dofs);
    }
}

void CoupledMVOperator::add_aca_block(int m, int n, int k, int row_off, int col_off, double* d_U, double* d_V) {
    AcaBlock b = {m, n, k, row_off, col_off, d_U, d_V};
    aca_blocks.push_back(b);
    
    // Puffer ggf. vergrößern, falls der Rang k gigantisch ist
    // Vergleich bewusst mit Cast: k ist int, size() ist size_t. Ohne Cast
    // warnt GCC (-Wsign-compare), und ein negatives k wuerde in eine sehr
    // grosse vorzeichenlose Zahl konvertiert, die Bedingung also falsch greifen.
    if (k < 0) {
        throw std::runtime_error("add_aca_block: negativer Rang k.");
    }
    if (static_cast<size_t>(k) > d_temp_z.size()) {
        d_temp_z.allocate(k * 2);
        // Guertel und Hosentraeger: allocate() initialisiert nicht. Der Puffer
        // wird zwar mit beta = 0 ueberschrieben, aber ein uninitialisierter
        // Puffer ist eine Falle, sobald irgendwo doch gelesen wird (siehe den
        // beta==0-Fehler in bf::dgemv).
        d_temp_z.fill_zero(); 
    }
}

// H_BEM-Bloecke (Double-Layer). Dieselbe d_temp_z-Pufferlogik
// wie bei add_aca_block, da beide (G und H) denselben Zwischenpuffer fuer
// z = V^T*x in apply() benutzen.
void CoupledMVOperator::add_dense_block_H(int m, int n, int row_off, int col_off, double* d_mat) {
    DenseBlock b = {m, n, row_off, col_off, d_mat};
    dense_blocks_H.push_back(b);
}

// Diagnose -- siehe Header fuer den Zweck.
void CoupledMVOperator::debug_print_bem_diag_stats() const {
    if (num_bem_dofs <= 0) return;

    std::vector<double> host_diag(static_cast<size_t>(num_bem_dofs));
    bf::memcpy_d(host_diag.data(), d_bem_inv_diag.data(),
                                static_cast<size_t>(num_bem_dofs) * sizeof(double),
                                cudaMemcpyDeviceToHost);

    double min_abs = std::numeric_limits<double>::infinity();
    double max_abs = 0.0;
    double sum_abs = 0.0;
    int num_default = 0; // noch exakt 1.0 (nie ueberschrieben)
    int min_idx = -1, max_idx = -1;

    for (int i = 0; i < num_bem_dofs; ++i) {
        const double a = std::abs(host_diag[i]);
        if (host_diag[i] == 1.0) ++num_default;
        if (a < min_abs) { min_abs = a; min_idx = i; }
        if (a > max_abs) { max_abs = a; max_idx = i; }
        sum_abs += a;
    }

    std::cout << "  [Diagnose] d_bem_inv_diag: |min| = " << min_abs << " (idx " << min_idx
              << "), |max| = " << max_abs << " (idx " << max_idx
              << "), Mittelwert(|.|) = " << (sum_abs / num_bem_dofs)
              << ", noch auf Default (1.0): " << num_default << " / " << num_bem_dofs << "\n";
}

// Zerlegt die Norm eines Residuumvektors (Tilde-Raum, volle
// gekoppelte Laenge get_total_dofs()) in drei Anteile: geeichte (Baum-)
// Kanten, freie (nicht geeichte) FEM-Kanten, und den BEM/K-Block. Soll
// lokalisieren, in welchem Teil des Systems ein Konvergenz-Plateau
// tatsaechlich sitzt, statt es zu vermuten -- z.B. ob es an den geeichten
// Kanten haengt (Eichungs-/Nullraum-Problem), an den physikalisch freien
// FEM-Kanten (ILU/Vernetzungsproblem) oder am BEM-Block (Kopplungs-/
// Preconditioner-Problem). Bewusst simpel gehalten (Host-Reduktion statt
// eigenem Reduktions-Kernel) -- wird nur alle 50 Iterationen aufgerufen,
// Performance ist hier nicht kritisch.
void CoupledMVOperator::debug_print_gauge_split_norm(const double* r, const char* label) const {
    if (d_is_gauged == nullptr) {
        double total_norm = 0.0;
        const int n = total_dofs + num_bem_dofs;
        bf::dnrm2(cublas, n, r, &total_norm);
        std::cout << "    [Diagnose-Split] " << label << ": keine Eichung aktiv, ||r|| = "
                  << total_norm << "\n";
        return;
    }

    std::vector<char> host_mask(static_cast<size_t>(total_dofs));
    bf::memcpy_d(host_mask.data(), d_is_gauged,
                                static_cast<size_t>(total_dofs) * sizeof(char),
                                cudaMemcpyDeviceToHost);
    std::vector<double> host_r(static_cast<size_t>(total_dofs));
    bf::memcpy_d(host_r.data(), r,
                                static_cast<size_t>(total_dofs) * sizeof(double),
                                cudaMemcpyDeviceToHost);

    double sumsq_gauged = 0.0, sumsq_free = 0.0;
    int n_gauged = 0, n_free = 0;
    for (int i = 0; i < total_dofs; ++i) {
        const double v = host_r[i];
        if (host_mask[i]) { sumsq_gauged += v * v; ++n_gauged; }
        else { sumsq_free += v * v; ++n_free; }
    }

    double norm_K = 0.0;
    if (num_bem_dofs > 0) {
        bf::dnrm2(cublas, num_bem_dofs, r + total_dofs, &norm_K);
    }

    std::cout << "    [Diagnose-Split] " << label
              << ": ||r_geeicht|| = " << std::sqrt(sumsq_gauged) << " (" << n_gauged << " Kanten), "
              << "||r_frei|| = " << std::sqrt(sumsq_free) << " (" << n_free << " Kanten), "
              << "||r_K|| = " << norm_K << " (" << num_bem_dofs << " BEM-DOFs)\n";
}

void CoupledMVOperator::add_aca_block_H(int m, int n, int k, int row_off, int col_off, double* d_U, double* d_V) {
    AcaBlock b = {m, n, k, row_off, col_off, d_U, d_V};
    aca_blocks_H.push_back(b);

    if (k < 0) {
        throw std::runtime_error("add_aca_block_H: negativer Rang k.");
    }
    if (static_cast<size_t>(k) > d_temp_z.size()) {
        d_temp_z.allocate(k * 2);
        // Guertel und Hosentraeger: allocate() initialisiert nicht. Der Puffer
        // wird zwar mit beta = 0 ueberschrieben, aber ein uninitialisierter
        // Puffer ist eine Falle, sobald irgendwo doch gelesen wird (siehe den
        // beta==0-Fehler in bf::dgemv).
        d_temp_z.fill_zero();
    }
}

// =================================================================================
// Kopplungs-Massenmatrix [T]: hier stehen nur noch die __global__ Huellen.
// =================================================================================
//
// Die Ruempfe (und die vollstaendige Herleitung der Vorzeichenkonvention)
// liegen in include/CouplingKernels.cuh -- ausgelagert, damit
// tests/test_coupling_sign_convention.cu sie direkt auf dem Host aufrufen und
// die Invariante zwischen Operator (-T) und Vorkonditionierer (+T) pruefen
// kann. Die Huellen muessen dagegen in DIESER Translation Unit bleiben: das
// Projekt kompiliert ohne Separable Compilation (-rdc=true), ein __global__
// Kernel aus einer anderen .cu-Datei liesse sich nicht linken.
// =================================================================================

__global__ void apply_T_matrix_kernel(
    int num_boundary_faces,
    const Mesh::QuadFace* faces, const double3* nodes,
    const int* bem_to_fem_map, const int* bem_edge_signs,
    const double* x_A, const double* x_K,
    double* y_A, double* y_K)
{
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_boundary_faces) return;
    apply_T_matrix_kernel_body(face_idx, num_boundary_faces, faces, nodes,
                               bem_to_fem_map, bem_edge_signs, x_A, x_K, y_A, y_K);
}

__global__ void apply_T_to_A_kernel(
    int num_boundary_faces,
    const Mesh::QuadFace* faces, const double3* nodes,
    const int* bem_to_fem_map, const int* bem_edge_signs,
    const double* x_K, double* y_A)
{
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (face_idx >= num_boundary_faces) return;
    apply_T_to_A_kernel_body(face_idx, num_boundary_faces, faces, nodes,
                             bem_to_fem_map, bem_edge_signs, x_K, y_A);
}

// =================================================================================
// Der 2x2-gekoppelte Operator. Loest
//     ( A_sys      -T     ) (A_FEM)   (RHS_FEM)
//     ( 0.5*T+D_BEM -mu_0*G) (K_BEM) = (   0   )
// d_x/d_y sind Vektoren der Laenge get_total_dofs() = total_dofs (FEM,
// erste total_dofs Eintraege = A) + num_bem_dofs (Rest = K). Ohne aktive
// BEM-Kopplung (num_bem_dofs == 0) reduziert sich alles auf das reine
// FEM-System von vorher.
// =================================================================================
// =================================================================================
// Aufbau der globalen Skalierungsmatrix D
// =================================================================================
void CoupledMVOperator::setup_equilibration() {
    const size_t coupled_dofs = static_cast<size_t>(total_dofs) + static_cast<size_t>(num_bem_dofs);

    d_D_scale.allocate(coupled_dofs);
    d_apply_in_temp.allocate(coupled_dofs);
    d_apply_out_temp.allocate(coupled_dofs);
    d_tilde_proj_temp.allocate(coupled_dofs);

    // 1. FEM-Anteil: D_ii = 1/sqrt(|A_ii|) aus der rohen FEM-Diagonale.
    BF_LAUNCH_1D_PAR(k_extract_fem_diag_and_build_D, total_dofs,
        total_dofs, m_row_offsets, m_col_indices, m_values, d_D_scale.data());

    // 2. BEM-Anteil (falls Kopplung aktiv): D_ii = sqrt(|(-mu_0*G_ii)^-1|),
    //    aus der bereits extrahierten d_bem_inv_diag.
    if (num_bem_dofs > 0) {
        BF_LAUNCH_1D_PAR(k_build_D_bem, num_bem_dofs,
            num_bem_dofs, total_dofs, d_bem_inv_diag.data(), d_D_scale.data());
    }

    bf::sync();
    is_equilibrated = true;
    std::cout << "[Solver] System wurde symmetrisch diagonal aequilibriert (SDE).\n";

    // Groessenordnung von D_scale ausgeben, getrennt FEM/BEM.
    // Soll klaeren, ob ein Kopplungsterm der Form D_K*T_rc*D_A (siehe BGS-
    // Vorkonditionierer-Versuche) durch die D-Faktoren selbst schon auf eine
    // unplausible Groessenordnung gehoben wird, statt das nur zu vermuten.
    {
        std::vector<double> host_D(coupled_dofs);
        bf::memcpy_d(host_D.data(), d_D_scale.data(),
                                     coupled_dofs * sizeof(double), cudaMemcpyDeviceToHost);

        double fem_min = std::numeric_limits<double>::infinity(), fem_max = 0.0;
        for (int i = 0; i < total_dofs; ++i) {
            const double v = std::abs(host_D[i]);
            fem_min = std::min(fem_min, v);
            fem_max = std::max(fem_max, v);
        }
        // Die min/max-Werte unten koennen NaN nicht sehen -- deshalb vorher
        // explizit zaehlen.
        audit_finite("D_scale (FEM-Anteil)", host_D.data(), total_dofs);
        if (num_bem_dofs > 0) {
            audit_finite("D_scale (BEM-Anteil)", host_D.data() + total_dofs, num_bem_dofs);
        }
        std::cout << "  [Diagnose] D_scale FEM-Anteil: min = " << fem_min
                  << ", max = " << fem_max << "\n";

        if (num_bem_dofs > 0) {
            double bem_min = std::numeric_limits<double>::infinity(), bem_max = 0.0;
            for (int i = total_dofs; i < total_dofs + num_bem_dofs; ++i) {
                const double v = std::abs(host_D[i]);
                bem_min = std::min(bem_min, v);
                bem_max = std::max(bem_max, v);
            }
            std::cout << "  [Diagnose] D_scale BEM-Anteil: min = " << bem_min
                      << ", max = " << bem_max << "\n";
            std::cout << "  [Diagnose] D_K*D_A (grobe obere Abschaetzung fuer den "
                      << "Kopplungsterm 0.5*D_K*T_rc*D_A ohne T_rc-Faktor): bis zu "
                      << (fem_max * bem_max) << "\n";
        }
    }
}

// oeffentliche Helfer, damit der Aufrufer (fem.cu) b/x0 ausserhalb
// des Solvers skalieren kann, ohne d_D_scale selbst (privat) anfassen zu
// muessen.
void CoupledMVOperator::scale_by_D(double* v) {
    if (!is_equilibrated) return;
    const int n = static_cast<int>(static_cast<size_t>(total_dofs) + static_cast<size_t>(num_bem_dofs));
    BF_LAUNCH_1D_PAR(k_vec_mult, n, n, v, d_D_scale.data(), v);
}

void CoupledMVOperator::scale_by_D_inverse(double* v) {
    if (!is_equilibrated) return;
    const int n = static_cast<int>(static_cast<size_t>(total_dofs) + static_cast<size_t>(num_bem_dofs));
    BF_LAUNCH_1D_PAR(k_vec_div, n, n, v, d_D_scale.data(), v);
}

// Wrapper fuer den BiCGStab -- transformiert x_tilde (skalierter
// Raum) nach physikalisch, ruft den unveraenderten physikalischen Operator
// auf, transformiert das Ergebnis zurueck. Ohne aktive Aequilibrierung
// (setup_equilibration() nicht aufgerufen) reines Pass-Through.
void CoupledMVOperator::apply(const double* d_x_tilde, double* d_y_tilde) {
    if (!is_equilibrated) {
        apply_physical(d_x_tilde, d_y_tilde);
        return;
    }

    const size_t coupled_dofs = static_cast<size_t>(total_dofs) + static_cast<size_t>(num_bem_dofs);

    // Audit der Eingabe im Tilde-Raum, VOR jeder Umrechnung -- trennt "der
    // Loeser reicht schon etwas Kaputtes herein" von "der Operator erzeugt es".
    const bool audit_wrapper = (m_apply_call_index == 0);
    if (audit_wrapper) {
        audit_finite("apply: EINGABE x_tilde", d_x_tilde, static_cast<int>(coupled_dofs));
    }

    // 1. Ruecktransformation: x_real = D * x_tilde
    BF_LAUNCH_1D_PAR(k_vec_mult, coupled_dofs,
        static_cast<int>(coupled_dofs), d_x_tilde, d_D_scale.data(), d_apply_in_temp.data());

    // 2. Physikalischen Operator anwenden -- OHNE dessen eigenen (I-P)*x_real
    //    Nullraum-Penalty-Term (siehe m_skip_nullspace_penalty-Deklaration).
    m_skip_nullspace_penalty = true;
    apply_physical(d_apply_in_temp.data(), d_apply_out_temp.data());
    m_skip_nullspace_penalty = false;

    if (audit_wrapper) {
        audit_finite("apply: y_real nach apply_physical",
                     d_apply_out_temp.data(), static_cast<int>(coupled_dofs));
    }

    // 3. Hintransformation: y_tilde = D * y_real
    BF_LAUNCH_1D_PAR(k_vec_mult, coupled_dofs,
        static_cast<int>(coupled_dofs), d_apply_out_temp.data(), d_D_scale.data(), d_y_tilde);

    // 4. Nullraum-Penalty korrekt im Tilde-Raum addieren:
    //    y_tilde += (x_tilde - P_tilde*x_tilde). Verschiebt den Nullraum-
    //    Eigenwert im Tilde-Raum auf exakt 1.0 (statt des verzerrten D_i^2
    //    aus der physikalischen Variante). project_gauge_tilde() liefert
    //    P_tilde bereits inklusive Kohomologie-Projektoren; der BEM-Anteil
    //    kommt dabei unveraendert durch (siehe dortiger Kommentar), der Term
    //    traegt dort also automatisch 0 bei.
    if (d_is_gauged != nullptr) {
        project_gauge_tilde(d_x_tilde, d_tilde_proj_temp.data());
        BF_LAUNCH_1D_PAR(k_add_null_space_residual_tilde, coupled_dofs,
            static_cast<int>(coupled_dofs), d_x_tilde, d_tilde_proj_temp.data(), d_y_tilde);
    }
}

// Endlichkeits-Audit der BEM-Blockdaten selbst.
//
// Anlass: die Audit-Kette in apply_physical zeigt, dass x_bem noch endlich in
// die D_BEM/H-Bloecke hineingeht und y_K danach zu 16544/16544 NaN ist. Damit
// bleiben genau zwei Moeglichkeiten -- entweder stehen in den Blockdaten
// (U, V, dichte Matrizen) bereits NaN, oder die Daten sind sauber und die
// Anwendung (Offsets, Dimensionen, Raenge) greift daneben. Dieses Audit
// entscheidet das.
//
// Die ACA-Bloecke sind dabei der interessante Teil: die Kreuzapproximation
// bricht ab, wenn ein Pivot null wird, und teilt durch dieses Pivot. Bleibt
// das unbemerkt, stehen in U/V NaN, ohne dass irgendwo eine Meldung erschien.
// Bisher geprueft wurde davon nur die Diagonale der dichten G-Selbstbloecke
// (d_bem_inv_diag) -- die Fernfeldbloecke gar nicht.
double CoupledMVOperator::relative_gradient_content(const double* d_v_tilde) const {
    if (m_edges == nullptr || m_num_nodes <= 0 || m_num_edges <= 0) return -1.0;
    bf::sync();

    // G^T ist die Inzidenz Kante -> Knoten: (G^T v)_n = Summe der v_e, die in n
    // enden, minus derer, die in n starten. Genau wie die G^T b-Pruefung in
    // fem.cu, nur auf den Kopplungsterm statt auf die Quelle angewandt.
    //
    // Gerechnet wird im PHYSIKALISCHEN Raum: der uebergebene Vektor ist eine
    // rechte Seite im Tilde-Raum, also v_phys = v_tilde / D_A.
    std::vector<double> gtv(static_cast<size_t>(m_num_nodes), 0.0);
    double v2 = 0.0;
    const double* D = is_equilibrated ? d_D_scale.data() : nullptr;

    for (int e = 0; e < m_num_edges && e < total_dofs; ++e) {
        double ve = d_v_tilde[e];
        if (D != nullptr) {
            const double d = D[e];
            ve = (d != 0.0) ? (ve / d) : 0.0;
        }
        v2 += ve * ve;
        const int2 ed = m_edges[e];
        gtv[ed.y] += ve;
        gtv[ed.x] -= ve;
    }
    if (!(v2 > 0.0)) return 0.0;

    double g2 = 0.0;
    for (double g : gtv) g2 += g * g;
    return std::sqrt(g2 / v2);
}

void CoupledMVOperator::audit_bem_blocks() const {
    bf::sync();

    // Die Blockdaten (d_matrix, d_U, d_V) liegen im GERAETESPEICHER. Diese
    // Lambda las sie frueher direkt -- im CPU-Modus richtig, auf der GPU ein
    // Segfault, und zwar mitten im ersten apply(). Der Puffer liegt ausserhalb
    // der Schleife, damit nicht je Block neu alloziert wird: es sind 464
    // dichte und 248 ACA-Bloecke.
    std::vector<double> scratch;
    auto scan = [&scratch](const double* v, size_t n, long& n_nan, long& n_inf) {
        if (v == nullptr || n == 0) return;
        const double* p = host_readable(v, n, scratch);
        if (p == nullptr) return;
        for (size_t i = 0; i < n; ++i) {
            if (std::isnan(p[i]))      ++n_nan;
            else if (std::isinf(p[i])) ++n_inf;
        }
    };

    auto report_dense = [&](const char* name, const std::vector<DenseBlock>& blocks) {
        long n_nan = 0, n_inf = 0, bad_blocks = 0;
        int first = -1;
        for (size_t b = 0; b < blocks.size(); ++b) {
            long a = 0, i2 = 0;
            scan(blocks[b].d_matrix,
                 static_cast<size_t>(blocks[b].m) * static_cast<size_t>(blocks[b].n), a, i2);
            if (a || i2) { ++bad_blocks; if (first < 0) first = static_cast<int>(b); }
            n_nan += a; n_inf += i2;
        }
        if (n_nan == 0 && n_inf == 0) return;
        std::cout << "  [AUDIT] " << name << ": " << bad_blocks << " von " << blocks.size()
                  << " Bloecken betroffen, " << n_nan << " NaN und " << n_inf
                  << " Inf; erster Block " << first;
        if (first >= 0) {
            std::cout << " (m=" << blocks[first].m << ", n=" << blocks[first].n
                      << ", row_off=" << blocks[first].row_offset
                      << ", col_off=" << blocks[first].col_offset << ")";
        }
        std::cout << "\n";
    };

    auto report_aca = [&](const char* name, const std::vector<AcaBlock>& blocks) {
        long nan_U = 0, inf_U = 0, nan_V = 0, inf_V = 0, bad_blocks = 0;
        int first = -1;
        for (size_t b = 0; b < blocks.size(); ++b) {
            long aU = 0, iU = 0, aV = 0, iV = 0;
            scan(blocks[b].d_U,
                 static_cast<size_t>(blocks[b].m) * static_cast<size_t>(blocks[b].k), aU, iU);
            scan(blocks[b].d_V,
                 static_cast<size_t>(blocks[b].n) * static_cast<size_t>(blocks[b].k), aV, iV);
            if (aU || iU || aV || iV) { ++bad_blocks; if (first < 0) first = static_cast<int>(b); }
            nan_U += aU; inf_U += iU; nan_V += aV; inf_V += iV;
        }
        if (nan_U == 0 && inf_U == 0 && nan_V == 0 && inf_V == 0) return;
        std::cout << "  [AUDIT] " << name << ": " << bad_blocks << " von " << blocks.size()
                  << " Bloecken betroffen; U: " << nan_U << " NaN / " << inf_U
                  << " Inf, V: " << nan_V << " NaN / " << inf_V << " Inf; erster Block " << first;
        if (first >= 0) {
            std::cout << " (m=" << blocks[first].m << ", n=" << blocks[first].n
                      << ", k=" << blocks[first].k
                      << ", row_off=" << blocks[first].row_offset
                      << ", col_off=" << blocks[first].col_offset << ")";
        }
        std::cout << "\n";
    };

    report_dense("D_BEM/H dichte Bloecke", dense_blocks_H);
    report_aca  ("D_BEM/H ACA-Bloecke",   aca_blocks_H);
    report_dense("G_BEM dichte Bloecke",  dense_blocks);
    report_aca  ("G_BEM ACA-Bloecke",     aca_blocks);
}

void CoupledMVOperator::apply_physical(const double* d_x, double* d_y) {
    const double alpha = 1.0;
    const double beta_zero = 0.0;
    const double beta_one = 1.0;

    const double* d_A_in = d_x;
    const double* d_K_in = d_x + total_dofs;
    double* d_y_A = d_y;
    double* d_y_K = d_y + total_dofs;

    const size_t coupled_dofs = static_cast<size_t>(total_dofs) + static_cast<size_t>(num_bem_dofs);
    bf::memset_d(d_y, 0, coupled_dofs * sizeof(double));

    // Stufenweises Endlichkeits-Audit beim ersten Aufruf. Die vier Audits an
    // den Entstehungsorten (Matrix, D_scale, ILU, rechte Seite) schweigen alle
    // -- die NaN entsteht also erst hier im Operator. Diese Kette benennt die
    // Stufe. Sie schweigt im Normalfall und kostet dann nur einen Durchlauf
    // ueber den Vektor, einmal pro Zeitschritt.
    const bool audit_this_call = (m_apply_call_index == 0);
    if (audit_this_call) {
        audit_finite("apply: EINGABE x (FEM-Anteil)", d_A_in, total_dofs);
        if (num_bem_dofs > 0) audit_finite("apply: EINGABE x (BEM-Anteil)", d_K_in, num_bem_dofs);
        audit_bem_blocks();
    }

    // -------------------------------------------------------------------------
    // Oben Links: y_A = FEM * A_in (mit Eichung wie bisher, nur auf dem
    // A-Block -- project_gauge() liest/schreibt ohnehin nur die ersten
    // total_dofs Eintraege, siehe dortiger Kommentar).
    // -------------------------------------------------------------------------
    const double* x_use = d_A_in;
    if (d_is_gauged != nullptr) {
        project_gauge(d_A_in, d_x_proj.data());
        x_use = d_x_proj.data();
        if (audit_this_call) audit_finite("apply: nach project_gauge(x)", x_use, total_dofs);
    }

    if (bf::on_cpu()) {
        // dieselbe Operation ohne cuSPARSE. beta ist hier
        // ohnehin 0 (y_A wird ueberschrieben, nicht akkumuliert) und alpha 1,
        // daher genuegt das schlichte y = A*x von bf::csr_spmv.
        bf::csr_spmv(total_dofs, m_row_offsets, m_col_indices, m_values, x_use, d_y_A);
        if (audit_this_call) audit_finite("apply: nach FEM-SpMV (y_A)", d_y_A, total_dofs);
    } else {
        cusparseDnVecDescr_t vec_x, vec_y;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_x, total_dofs, (void*)x_use, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_y, total_dofs, (void*)d_y_A, CUDA_R_64F));

        if (cusparse_buffer_size == 0) {
            CUSPARSE_CHECK(cusparseSpMV_bufferSize(
                cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, mat_FEM, vec_x, &beta_zero, vec_y, CUDA_R_64F,
                CUSPARSE_SPMV_ALG_DEFAULT, &cusparse_buffer_size));
            CUDA_CHECK_ERROR(cudaMalloc(&d_cusparse_buffer, cusparse_buffer_size));
        }

        CUSPARSE_CHECK(cusparseSpMV(
            cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_FEM, vec_x, &beta_zero, vec_y, CUDA_R_64F,
            CUSPARSE_SPMV_ALG_DEFAULT, d_cusparse_buffer));

        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_x));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_y));
    }

    // Diagnose -- Norm des reinen FEM-Anteils, unveraendert aus den
    // vorherigen Schritten uebernommen (siehe dortige Begruendung fuer
    // m_apply_call_index == 1).
    if (m_apply_call_index == 1) {
        double fem_only_norm = 0.0;
        bf::dnrm2(cublas, total_dofs, d_y_A, &fem_only_norm);
        std::cout << "  [Diagnose] ||FEM-Anteil von y|| (vor BEM-Scatter) = "
                  << fem_only_norm << "\n";
    }

    const bool have_bem = num_bem_dofs > 0;
    if (!have_bem) {
        // Kein BEM: reines FEM-System, Eichungsabschluss wie zuvor.
        // unterdrueckt, wenn apply() den Tilde-Ersatz-Term
        // selbst hinzufuegt (siehe dort).
        if (d_is_gauged != nullptr && !m_skip_nullspace_penalty) {
            BF_LAUNCH_1D_PAR(k_add_null_space_residual, total_dofs,
                             d_A_in, d_x_proj.data(), d_y_A, total_dofs);
        }
        ++m_apply_call_index;
        return;
    }

    if (m_num_boundary_faces <= 0 || d_mesh_faces == nullptr || d_mesh_nodes == nullptr) {
        throw std::runtime_error(
            "CoupledMVOperator::apply: BEM-Bloecke vorhanden, aber setup_bem_mesh() "
            "wurde nicht aufgerufen -- die T-Kopplungsmatrix braucht die Rand-Quad-Geometrie.");
    }

    // -------------------------------------------------------------------------
    // Kopplungs-Massenmatrix T: y_A -= T*K_in, y_K += 0.5*T*x_use (siehe
    // apply_T_matrix_kernel oben fuer die Herleitung der Vorzeichen).
    // WICHTIG: fuer die Randspur von A wird dieselbe geeichte Version x_use
    // verwendet wie fuer den FEM-Sparse-Anteil, nicht das rohe d_A_in.
    // -------------------------------------------------------------------------
    // Seriell im CPU-Modus (bf::atomic_add auf y_A, siehe Kernelkommentar).
    BF_LAUNCH_1D(apply_T_matrix_kernel, m_num_boundary_faces,
        m_num_boundary_faces, d_mesh_faces, d_mesh_nodes,
        d_bem_to_fem, d_bem_sign, x_use, d_K_in, d_y_A, d_y_K);
    if (audit_this_call) {
        audit_finite("apply: nach T-Kopplung (y_A)", d_y_A, total_dofs);
        audit_finite("apply: nach T-Kopplung (y_K)", d_y_K, num_bem_dofs);
    }

    // -------------------------------------------------------------------------
    // Randspur von A (x_use) in den BEM-Elementraum gathern -- gebraucht als
    // Eingabe fuer D_BEM (H_BEM = 0.5*T + D_BEM, T-Anteil kommt vom Kernel
    // oben, D_BEM hier ueber die dichten/ACA-Bloecke).
    // -------------------------------------------------------------------------
    BF_LAUNCH_1D_PAR(k_gather_bem, num_bem_dofs,
        x_use, d_bem_to_fem, d_bem_sign, d_x_bem.data(), num_bem_dofs);
    if (audit_this_call) audit_finite("apply: nach k_gather_bem (x_bem)", d_x_bem.data(), num_bem_dofs);

    // -------------------------------------------------------------------------
    // D_BEM (Double-Layer, dichte + ACA-Bloecke) * x_bem -> y_K, additiv.
    // Anders als G/T wird hier NICHT zurueck auf FEM-Kanten skattert -- der
    // Output lebt direkt im K-Block (y_K ist bereits BEM-lokal).
    // Vorzeichen auf -1.0, konsistent mit der
    // Negation der gesamten K-Gleichung (siehe apply_T_matrix_kernel oben und
    // G_BEM-Block unten fuer dieselbe Herleitung).
    // -------------------------------------------------------------------------
    const double alpha_minus_one = -1.0;
    for (const auto& block : dense_blocks_H) {
        const double* d_x_sub = d_x_bem.data() + block.col_offset;
        double* d_y_sub = d_y_K + block.row_offset;
        bf::dgemv(cublas, /*transpose=*/false, block.m, block.n,
              alpha_minus_one, block.d_matrix, block.m,
              d_x_sub, beta_one, d_y_sub);
    }
    for (const auto& aca : aca_blocks_H) {
        const double* d_x_sub = d_x_bem.data() + aca.col_offset;
        double* d_y_sub = d_y_K + aca.row_offset;
        double* d_z = d_temp_z.data();

        bf::dgemv(cublas, /*transpose=*/true, aca.n, aca.k,
              alpha, aca.d_V, aca.n,
              d_x_sub, beta_zero, d_z);
        bf::dgemv(cublas, /*transpose=*/false, aca.m, aca.k,
              alpha_minus_one, aca.d_U, aca.m,
              d_z, beta_one, d_y_sub);
    }

    if (audit_this_call) audit_finite("apply: nach D_BEM/H-Bloecken (y_K)", d_y_K, num_bem_dofs);

    // -------------------------------------------------------------------------
    // +mu_0 * G_BEM (Single-Layer, dichte + ACA-Bloecke) * K_in -> y_K,
    // additiv. G/K sind beide bereits BEM-lokal (K ist ja direkt die
    // BEM-Unbekannte) -- kein Gather/Scatter noetig, anders als frueher (als
    // dieselben Bloecke noch auf die von A abgeleitete Groesse wirkten).
    // Vorzeichen von -mu_0 auf +mu_0 gedreht,
    // konsistent mit der Negation der gesamten K-Gleichung (siehe
    // apply_T_matrix_kernel und H_BEM-Block oben). \tilde{M}_KK ist damit
    // im Tilde-Raum exakt +1.0 statt -1.0 -- apply_preconditioner() braucht
    // dafuer keine Negation mehr, siehe dort.
    // -------------------------------------------------------------------------
    const double plus_mu0 = m_mu_0;
    for (const auto& block : dense_blocks) {
        const double* d_x_sub = d_K_in + block.col_offset;
        double* d_y_sub = d_y_K + block.row_offset;
        // block.transposed: der Block teilt sich den Speicher mit seinem
        // Spiegelpartner (I,J) <-> (J,I) und wird als dessen Transponierte
        // angewandt. Damit ist G_BEM per Konstruktion exakt symmetrisch.
        // Gespeichert liegt dann eine (n x m)-Matrix, Leading Dimension n.
        bf::dgemv(cublas, /*transpose=*/block.transposed,
              block.transposed ? block.n : block.m,
              block.transposed ? block.m : block.n,
              plus_mu0, block.d_matrix,
              block.transposed ? block.n : block.m,
              d_x_sub, beta_one, d_y_sub);
    }
    for (const auto& aca : aca_blocks) {
        const double* d_x_sub = d_K_in + aca.col_offset;
        double* d_y_sub = d_y_K + aca.row_offset;
        double* d_z = d_temp_z.data();

        bf::dgemv(cublas, /*transpose=*/true, aca.n, aca.k,
              alpha, aca.d_V, aca.n,
              d_x_sub, beta_zero, d_z);
        bf::dgemv(cublas, /*transpose=*/false, aca.m, aca.k,
              plus_mu0, aca.d_U, aca.m,
              d_z, beta_one, d_y_sub);
    }

    if (audit_this_call) audit_finite("apply: nach G_BEM-Bloecken (y_K)", d_y_K, num_bem_dofs);

    // Diagnose -- Norm des reinen BEM(K)-Anteils von y, unmittelbar nach
    // G/H/T, vor der Eichung. War bisher an d_y_bem gemessen (Zwischenpuffer
    // vor dem Scatter); der ist im 2x2-System nicht mehr im Signalpfad, daher
    // jetzt direkt an y_K gemessen.
    if (m_apply_call_index == 1) {
        double bem_only_norm = 0.0;
        bf::dnrm2(cublas, num_bem_dofs, d_y_K, &bem_only_norm);
        std::cout << "  [Diagnose] ||BEM(K)-Anteil von y||                 = "
                  << bem_only_norm << "\n";
    }

    ++m_apply_call_index;

    // -------------------------------------------------------------------------
    // Eichungsabschluss nur auf dem A-Block: y_A = P*A_sys*P*A_in + (I-P)*A_in.
    // Der K-Block bleibt unangetastet -- die Eichung betrifft ausschliesslich
    // den Nullraum des FEM-Curl-Curl-Operators, nicht die BEM-Unbekannte.
    // -------------------------------------------------------------------------
    // unterdrueckt, wenn apply() den Tilde-Ersatz-Term selbst
    // hinzufuegt (siehe dort) -- sonst wuerde dieser physikalische (I-P)*x_real
    // Term ein zweites Mal mit D skaliert und den Nullraum-Eigenwert von 1.0
    // auf D_i^2 verzerren.
    if (d_is_gauged != nullptr && !m_skip_nullspace_penalty) {
        BF_LAUNCH_1D_PAR(k_add_null_space_residual, total_dofs,
                         d_A_in, d_x_proj.data(), d_y_A, total_dofs);
    }
}

// =================================================================================
// ILU(0)-Vorkonditionierung des FEM-Sparse-Anteils
// =================================================================================
//
// M = L*U approximiert nur die FEM-Matrix (gleiche Sparsity, L unit-lower,
// U non-unit-upper). BEM-Kopplung, Eichung und Kohomologie-Projektoren aus
// apply() fliessen bewusst NICHT ein -- ein Vorkonditionierer muss A^-1 nur
// billig annaehern, nicht exakt nachbilden.
// Gershgorin-Shift (Row-Norm-Shift) auf der ILU-Kopie der FEM-
// Werte.
// -------------------------------------------------------------------------
// A_sys = K + C/dt ist dort singulaer bzw. fast singulaer, wo kappa = 0 ist
// (nichtleitende Gebiete): uebrig bleibt der Curl-Curl-Operator, dessen
// Nullraum genau die geeichten Baum-Kanten sind (siehe TreeCotreeGauge.h).
// ILU(0) sieht davon nichts als "exakte" Nullpivots (cusparseXcsrilu02_zeroPivot
// schlaegt nicht an), sondern nur als winzige Pivots -- die Faktorisierung
// dividiert dadurch, und L\U explodieren.
//
// Ein fixer absoluter Shift waere von Einheiten, Zeitschrittweite und
// Materialwerten abhaengig (in Aluminium-Zeilen vernachlaessigbar, in
// Luft-Zeilen u.U. dominant -- oder umgekehrt, je nach Skala). Stattdessen
// addieren wir pro Zeile einen festen Anteil epsilon der eigenen Zeilennorm
// (Betragssumme) auf die Diagonale -- nach dem Gershgorin-Kreis-Theorem
// erzwingt das eine strikte Diagonaldominanz dieser Zeile, unabhaengig von
// ihrer absoluten Groessenordnung: eine "grosse" Aluminium-Zeile bekommt
// einen grossen, eine "kleine" Luft-Zeile einen kleinen Shift. Die relative
// physikalische Gewichtung zwischen den Regionen bleibt damit erhalten, nur
// die Kopie in d_ilu_values wird veraendert (m_values / mat_FEM bleiben
// unangetastet). Der eingefuehrte Fehler wird danach durch project_gauge()
// auf p_hat/s_hat wieder herausprojiziert (siehe BiCGStabSolver::solve(),
// Fix 2).
// Gauge-Aware ILU-Setup.
// -------------------------------------------------------------------------
// Der reine Gershgorin-Shift (siehe alte Version dieses Kernels) hielt die
// Pivots zwar von Null weg, liess die (fast singulaere) Baum-Teilmatrix aber
// weiterhin Teil der Faktorisierung sein -- ILU produzierte dadurch weiterhin
// eine quantitativ schlechte Naeherung (siehe ||p_hat||/||p||-Diagnose: die
// Amplifikation blieb klein, aber ||p|| selbst explodierte trotzdem ueber die
// Iterationen). Diese Version isoliert die geeichten (Baum-)Kanten VOR der
// Faktorisierung komplett: Zeile auf Identitaet (Diagonale 1, Rest 0). Da
// csrilu02 Zeile fuer Zeile eliminiert, kann eine solche Identitaetszeile
// weder von anderen Zeilen verschmutzt werden (ihre Eintraege ausserhalb der
// Diagonale sind ja schon vor der Faktorisierung 0) noch selbst andere Zeilen
// verschmutzen (ihre U-Zeile ist ueberall 0 ausser der Diagonale, jeder
// Eliminationsschritt gegen sie multipliziert sich also mit 0). Der
// physikalische Kobaum-Teilraum wird dadurch von der topologischen
// Nullraum-Richtung sauber entkoppelt; ILU(0) arbeitet nur noch auf dem gut
// konditionierten physikalischen Teilsystem. Passt zusammen mit
// project_gauge(): p/r/v/t sind an den geeichten Kanten ohnehin schon (nahezu)
// exakt 0, bevor sie hier ankommen -- die Identitaetszeile gibt dann exakt 0
// =============================================================================
// AMS-Vorkonditionierer aufbauen
// =============================================================================
void CoupledMVOperator::setup_ams(const Mesh& mesh) {
    if (m_values == nullptr || m_row_offsets == nullptr) {
        throw std::runtime_error("setup_ams: setup_fem_matrix() wurde nicht aufgerufen.");
    }

    // Die FEM-Matrix liegt als Managed Memory vor; der AMS arbeitet host-seitig.
    // Bewusst eine Kopie: der AMS baut daraus Hilfsmatrizen und ein AMG auf,
    // die ohnehin eigenen Speicher brauchen, und die Kopie kostet gegenueber
    // dem AMG-Aufbau nichts.
    HostCsr A;
    A.num_rows = m_ilu_num_rows;
    A.num_cols = m_ilu_num_rows;
    A.row_offsets.assign(m_row_offsets, m_row_offsets + m_ilu_num_rows + 1);
    A.col_indices.assign(m_col_indices, m_col_indices + m_ilu_nnz);
    A.values.assign(m_values, m_values + m_ilu_nnz);

    m_ams.setup(A, mesh);
    m_ams_ready = true;
    m_ams_in.assign(static_cast<size_t>(m_ilu_num_rows), 0.0);
    m_ams_out.assign(static_cast<size_t>(m_ilu_num_rows), 0.0);

    auto report = [](const char* name, int dofs, int empty, double cx,
                     const std::vector<int>& lv) {
        std::cout << "[AMS] " << name << ": " << dofs << " DOF, davon "
                  << empty << " mit LEERER Diagonale ("
                  << std::fixed << std::setprecision(1)
                  << (100.0 * empty / std::max(1, dofs)) << " %)\n"
                  << "[AMS]   Ebenen ";
        for (size_t i = 0; i < lv.size(); ++i)
            std::cout << lv[i] << (i + 1 < lv.size() ? " -> " : "");
        std::cout << ",  Op.Komplexitaet " << std::setprecision(2) << cx << "\n"
                  << std::defaultfloat;
    };
    report("skalarer Hilfsraum   ", m_ams.scalar_dofs(), m_ams.scalar_empty_rows(),
           m_ams.scalar_complexity(), m_ams.scalar_level_sizes());
    // Die aussagekraeftigere Zahl: nicht die EXAKT leeren Diagonalen, sondern
    // die rundungskleinen. Ausserhalb der Leiter ist A_G = G^T A_cc G bis auf
    // Rundung null, weil A_cc * G = 0 gilt -- exakt null wird dabei fast
    // nichts, rundungsklein dagegen alles.
    std::cout << "[AMS]   davon rundungsklein: " << m_ams.scalar_tiny_rows()
              << ", ausgeblendet: " << m_ams.scalar_masked_rows()
              << " (dort traegt der skalare Hilfsraum nichts bei)\n";
    report("vektorieller Hilfsraum", m_ams.vector_dofs(), m_ams.vector_empty_rows(),
           m_ams.vector_complexity(), m_ams.vector_level_sizes());

    // Die entscheidende Kennzahl: loest das AMG sein eigenes System?
    // Level-Groessen und Operatorkomplexitaet beantworten das NICHT.
    std::cout << "[AMS] Konvergenzfaktor je V-Zyklus (auf Zufalls-RHS gemessen): "
              << "skalar " << m_ams.scalar_convergence()
              << " (ohne Maske " << m_ams.scalar_convergence_unmasked() << ")"
              << ", vektoriell " << m_ams.vector_convergence() << "\n"
              << "[AMS]   Deutung: unter ~0.5 brauchbar, nahe 1.0 traegt der "
                 "Hilfsraum nichts bei.\n";
    // Die frueher hier stehende Warnung stuetzte sich auf die
    // Operatorkomplexitaet ("unter 1.3 -> Vergroeberung kollabiert"). Sie war
    // miskalibriert und hat die Suche in die Irre gefuehrt: die Schwelle stammt
    // aus einem Modellproblem mit viel sanfterer Vergroeberung ueber mehr
    // Ebenen. Bei 3D-Aggregation mit 3x3x3-Aggregaten und vier Ebenen ist ein
    // Wert um 1.1 zu erwarten und voellig in Ordnung -- gemessen am
    // 3D-Laplace (tests/test_amg_convergence) erreicht dieselbe
    // Implementierung damit einen Konvergenzfaktor von 0.30.
    //
    // Gewarnt wird jetzt an der Kennzahl, die es entscheidet.
    const double conv_G  = m_ams.scalar_convergence();
    const double conv_Pi = m_ams.vector_convergence();
    if ((conv_G > 0.6 && conv_G <= 1.0) || (conv_Pi > 0.6 && conv_Pi <= 1.0)) {
        std::cout << "[AMS] HINWEIS: mindestens ein Hilfsraum liegt ueber 0.6 "
                     "(Bezugswert auf dem 3D-Laplace: 0.30).\n"
                     "[AMS]   Die drei bekannten Ursachen sind inzwischen behoben:\n"
                     "[AMS]     * A_G ist ausserhalb der Leiter ausgeblendet (dort nur "
                     "Rundungsrauschen)\n"
                     "[AMS]     * A_Pi wird als Vektor-Poisson-Operator gebildet, nicht "
                     "als Pi^T A Pi\n"
                     "[AMS]     * die Aggregation benutzt einen zeilenrelativen "
                     "Staerkemaszstab (Anisotropie)\n"
                     "[AMS]   Was bleibt, ist der Rest -- und der bewegt den AEUSSEREN "
                     "Loeser erfahrungsgemaess kaum:\n"
                     "[AMS]   die drei Verbesserungen oben haben die Hilfsraeume von "
                     "0.93 auf 0.62 bzw. von 0.72 auf 0.52\n"
                     "[AMS]   gebracht, das Residuum nach 990 Iterationen aber nur von "
                     "0.025 auf 0.026. Der Engpass liegt\n"
                     "[AMS]   nicht im FEM-Block.\n";
    }
}


// zurueck, konsistent mit dem Rest des Operators.
BF_HD inline void k_ilu_gauge_aware_setup_body(int row, int num_rows, const int* row_offsets,
                                               const int* col_indices,
                                               double* ilu_values,
                                               const char* is_gauged,
                                               double epsilon) {
    (void)num_rows;

    const int start = row_offsets[row];
    const int end   = row_offsets[row + 1];

    if (is_gauged != nullptr && is_gauged[row]) {
        // Baum-Kante (Nullraum): Zeile auf Identitaet zwingen.
        for (int i = start; i < end; ++i) {
            if (col_indices[i] == row) {
                ilu_values[i] = 1.0;
            } else {
                ilu_values[i] = 0.0;
            }
        }
    } else {
        // Kobaum-Kante (Physik): Matrix belassen, nur leicht stabilisieren
        // (skaleninvarianter Gershgorin-Shift wie zuvor, aber jetzt auf einem
        // bereits gut konditionierten Teilsystem -- ein kleineres epsilon
        // reicht, siehe Aufrufstelle in compute_ilu()).
        double row_sum = 0.0;
        int diag_idx = -1;

        for (int i = start; i < end; ++i) {
            row_sum += fabs(ilu_values[i]);
            if (col_indices[i] == row) {
                diag_idx = i;
            }
        }

        if (diag_idx != -1) {
            ilu_values[diag_idx] += epsilon * row_sum;
        }
    }
}
__global__ void k_ilu_gauge_aware_setup(int num_rows, const int* __restrict__ row_offsets,
                                        const int* __restrict__ col_indices,
                                        double* __restrict__ ilu_values,
                                        const char* __restrict__ is_gauged,
                                        double epsilon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    k_ilu_gauge_aware_setup_body(row, num_rows, row_offsets, col_indices,
                                 ilu_values, is_gauged, epsilon);
}

void CoupledMVOperator::compute_ilu() {
    if (m_row_offsets == nullptr || m_col_indices == nullptr || m_values == nullptr) {
        throw std::runtime_error("compute_ilu: setup_fem_matrix() muss zuerst aufgerufen werden.");
    }

    // Zaehler zuruecksetzen, damit der naechste apply()-Aufruf (der
    // erste dieses Zeitschritts) wieder das FEM/BEM-Norm-Diagnose-Log ausgibt.
    m_apply_call_index = 0;
    // dito fuer die einmalige Ausgabe des blocktriangulaeren Zweigs (Anzahl
    // innerer CG-Iterationen, siehe apply_preconditioner()).
    m_pc_call_index = 0;

    // 1. Einmalige Initialisierung von Deskriptoren/Puffern, die ueber
    //    mehrere Zeitschritte hinweg gueltig bleiben (Sparsity aendert sich
    //    zwischen Zeitschritten nicht, nur die Werte).
    if (!d_ilu_values) {
        if (bf::on_cpu()) {
            // Arbeitskopie der Werte in einem std::vector, und
            // d_ilu_values zeigt darauf. Damit bleiben die beiden folgenden
            // Vorbereitungsschritte (SDE-Skalierung, Gauge-Aware-Setup) voellig
            // unveraendert -- sie laufen ueber BF_LAUNCH_1D_PAR und arbeiten
            // ohnehin nur auf d_ilu_values. Nur die Faktorisierung selbst
            // verzweigt weiter unten.
            //
            // Der Vektor wird ab hier nie mehr umdimensioniert (die Sparsity
            // aendert sich zwischen Zeitschritten nicht), der Zeiger bleibt
            // also gueltig. Freigegeben wird er vom Vektor selbst; der
            // Destruktor kehrt im CPU-Modus vorher zurueck und ruft daher kein
            // cudaFree auf diesem Host-Zeiger auf.
            m_cpu_ilu_values.assign(static_cast<size_t>(m_ilu_nnz), 0.0);
            m_cpu_ilu_temp.assign(static_cast<size_t>(m_ilu_num_rows), 0.0);
            d_ilu_values = m_cpu_ilu_values.data();
        } else {
            CUDA_CHECK_ERROR(cudaMalloc(&d_ilu_values, static_cast<size_t>(m_ilu_nnz) * sizeof(double)));

            CUSPARSE_CHECK(cusparseCreateMatDescr(&ilu_descr));
            CUSPARSE_CHECK(cusparseSetMatType(ilu_descr, CUSPARSE_MATRIX_TYPE_GENERAL));
            CUSPARSE_CHECK(cusparseSetMatIndexBase(ilu_descr, CUSPARSE_INDEX_BASE_ZERO));

            CUSPARSE_CHECK(cusparseCreateCsrilu02Info(&ilu_info));
        }
    }

    // 2. FEM-Werte in den ILU-Puffer kopieren -- die Faktorisierung laeuft in-place
    bf::memcpy_d(d_ilu_values, m_values,
                 static_cast<size_t>(m_ilu_nnz) * sizeof(double),
                 cudaMemcpyDeviceToDevice);

    // Arbeitskopie der Matrix mit D skalieren, BEVOR der Gauge-
    // Aware-Shift/die Faktorisierung laeuft. \tilde{A} = D*A*D. Erfordert,
    // dass setup_equilibration() bereits gelaufen ist (siehe fem.cu: Aufruf-
    // reihenfolge wurde entsprechend umgestellt). Eine geeichte (Baum-)Zeile
    // bleibt danach weiterhin eine Identitaetszeile, da D_ii*1*D_ii = D_ii^2
    // erst durch den nachfolgenden Gauge-Aware-Schritt ohnehin wieder auf 1
    // gesetzt wird -- die Reihenfolge der beiden Schritte ist also unkritisch.
    if (is_equilibrated) {
        BF_LAUNCH_1D_PAR(k_scale_csr_matrix, m_ilu_num_rows,
            m_ilu_num_rows, m_row_offsets, m_col_indices, d_ilu_values, d_D_scale.data());
        bf::sync();
    }

    // geeichte Zeilen auf Identitaet zwingen, den
    // physikalischen Rest nur leicht stabilisieren -- nur auf der Kopie
    // d_ilu_values, m_values/mat_FEM bleiben unveraendert. epsilon = 1e-4
    // (statt vorher 1e-3) reicht jetzt, da die vormals problematische
    // Nullraum-Richtung komplett isoliert ist, nicht mehr mitgeshiftet werden
    // muss. Siehe Kommentar bei k_ilu_gauge_aware_setup oben fuer die
    // Begruendung. Erfordert, dass setup_gauge() VOR compute_ilu() aufgerufen
    // wurde (siehe fem.cu) -- ohne aktive Eichung (d_is_gauged == nullptr)
    // verhaelt sich der Kernel identisch zum reinen Gershgorin-Shift von vorher.
    {
        const double epsilon = 1e-4;
        BF_LAUNCH_1D_PAR(k_ilu_gauge_aware_setup, m_ilu_num_rows,
            m_ilu_num_rows, m_row_offsets, m_col_indices, d_ilu_values, d_is_gauged, epsilon);
        bf::sync();
    }

    // -------------------------------------------------------------------------
    // eigene ILU(0)-Faktorisierung, danach fertig.
    // -------------------------------------------------------------------------
    // Numerisch dasselbe Verfahren wie cusparseDcsrilu02: ILU(0) ohne Fill-in,
    // L mit Einheitsdiagonale, in-place auf derselben Sparsity. Die Nullpivot-
    // Behandlung entspricht cusparseXcsrilu02_zeroPivot -- inklusive der
    // Unterscheidung, ob der Pivot strukturell fehlt (Diagonaleintrag gar nicht
    // im Muster) oder numerisch verschwindet.
    if (bf::on_cpu()) {
        const int zero_pivot = bf::csr_ilu0(m_ilu_num_rows, m_row_offsets, m_col_indices,
                                            d_ilu_values, m_cpu_ilu_diag);
        if (zero_pivot >= 0) {
            throw std::runtime_error("compute_ilu (CPU): Nullpivot in Zeile "
                + std::to_string(zero_pivot) + ". ILU(0) bricht hier zusammen -- "
                "Diagonale der FEM-Matrix pruefen.");
        }
        // Der Nullpivot-Test oben faengt nur exakte Nullen. Ein NaN auf der
        // Diagonalen laeuft durch ihn hindurch (jeder Vergleich mit NaN ist
        // falsch) und vergiftet danach jede Dreieckssubstitution.
        audit_finite("ILU-Faktorisierung (CPU)", d_ilu_values, m_ilu_nnz);
        ilu_ready = true;
        return;
    }

    // 3. Puffergroesse fuer die ILU(0)-Faktorisierung ermitteln
    int pBufferSize = 0;
    CUSPARSE_CHECK(cusparseDcsrilu02_bufferSize(
        cusparse, m_ilu_num_rows, m_ilu_nnz, ilu_descr,
        d_ilu_values, m_row_offsets, m_col_indices, ilu_info, &pBufferSize));

    if (d_ilu_buffer) {
        cudaFree(d_ilu_buffer);
        d_ilu_buffer = nullptr;
    }
    CUDA_CHECK_ERROR(cudaMalloc(&d_ilu_buffer, static_cast<size_t>(pBufferSize)));

    // 4. Analyse-Phase (Abhaengigkeitsstruktur von L/U)
    int structural_zero = -1;
    CUSPARSE_CHECK(cusparseDcsrilu02_analysis(
        cusparse, m_ilu_num_rows, m_ilu_nnz, ilu_descr,
        d_ilu_values, m_row_offsets, m_col_indices, ilu_info,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_ilu_buffer));

    cusparseStatus_t zp_status = cusparseXcsrilu02_zeroPivot(cusparse, ilu_info, &structural_zero);
    if (zp_status == CUSPARSE_STATUS_ZERO_PIVOT) {
        throw std::runtime_error("compute_ilu: struktureller Nullpivot in Zeile "
            + std::to_string(structural_zero) + " (Analyse-Phase).");
    }
    CUSPARSE_CHECK(zp_status);

    // 5. Faktorisierung: ueberschreibt d_ilu_values in-place mit L\U
    //    (Diagonale von L ist implizit 1, siehe CUSPARSE_DIAG_TYPE_UNIT unten)
    CUSPARSE_CHECK(cusparseDcsrilu02(
        cusparse, m_ilu_num_rows, m_ilu_nnz, ilu_descr,
        d_ilu_values, m_row_offsets, m_col_indices, ilu_info,
        CUSPARSE_SOLVE_POLICY_USE_LEVEL, d_ilu_buffer));

    int numerical_zero = -1;
    zp_status = cusparseXcsrilu02_zeroPivot(cusparse, ilu_info, &numerical_zero);
    if (zp_status == CUSPARSE_STATUS_ZERO_PIVOT) {
        throw std::runtime_error("compute_ilu: numerischer Nullpivot in Zeile "
            + std::to_string(numerical_zero) + " (Faktorisierungs-Phase). "
            "ILU(0) bricht hier zusammen -- Diagonale der FEM-Matrix pruefen.");
    }
    CUSPARSE_CHECK(zp_status);

    // 6. SpSV-Deskriptoren fuer L und U einmalig anlegen
    if (!mat_L) {
        CUSPARSE_CHECK(cusparseCreateCsr(&mat_L, m_ilu_num_rows, m_ilu_num_rows, m_ilu_nnz,
                                         m_row_offsets, m_col_indices, d_ilu_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        // Achtung: cusparseSpMatSetAttribute erwartet einen Zeiger auf ein
        // lvalue -- die Adresse eines Enum-Literals wie
        // "&CUSPARSE_FILL_MODE_LOWER" ist kein gueltiges C++ und liesse sich
        // gar nicht kompilieren. Daher ueber lokale Variablen.
        cusparseFillMode_t fill_lower = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diag_unit  = CUSPARSE_DIAG_TYPE_UNIT;
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(mat_L, CUSPARSE_SPMAT_FILL_MODE, &fill_lower, sizeof(fill_lower)));
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(mat_L, CUSPARSE_SPMAT_DIAG_TYPE, &diag_unit, sizeof(diag_unit)));

        CUSPARSE_CHECK(cusparseCreateCsr(&mat_U, m_ilu_num_rows, m_ilu_num_rows, m_ilu_nnz,
                                         m_row_offsets, m_col_indices, d_ilu_values,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
        cusparseFillMode_t fill_upper   = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag_nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(mat_U, CUSPARSE_SPMAT_FILL_MODE, &fill_upper, sizeof(fill_upper)));
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(mat_U, CUSPARSE_SPMAT_DIAG_TYPE, &diag_nonunit, sizeof(diag_nonunit)));

        CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsv_info_L));
        CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsv_info_U));
    }

    // Die SpSV-Analyse-Phasen haengen (anders als bei mat_L/mat_U-Anlage) auch
    // von den Zahlenwerten ab, nicht nur von der Sparsity -- deshalb nach
    // JEDER Neufaktorisierung wiederholen, nicht nur beim ersten Mal.
    const double alpha_one = 1.0;
    cusparseDnVecDescr_t vec_probe_in, vec_probe_out;
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_probe_in, m_ilu_num_rows, d_ilu_temp.data(), CUDA_R_64F));
    // GETRENNTER Puffer: cuSPARSE verlangt fuer SpSV, dass X und Y nicht
    // derselbe Speicher sind. Vorher zeigten beide auf d_ilu_temp.
    CUSPARSE_CHECK(cusparseCreateDnVec(&vec_probe_out, m_ilu_num_rows, d_ilu_probe.data(), CUDA_R_64F));

    size_t bufferSize_L = 0;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_one, mat_L, vec_probe_in, vec_probe_out, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_info_L, &bufferSize_L));
    if (d_spsv_buffer_L) { cudaFree(d_spsv_buffer_L); d_spsv_buffer_L = nullptr; }
    CUDA_CHECK_ERROR(cudaMalloc(&d_spsv_buffer_L, bufferSize_L));
    CUSPARSE_CHECK(cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_one, mat_L, vec_probe_in, vec_probe_out, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_info_L, d_spsv_buffer_L));

    size_t bufferSize_U = 0;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_one, mat_U, vec_probe_in, vec_probe_out, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_info_U, &bufferSize_U));
    if (d_spsv_buffer_U) { cudaFree(d_spsv_buffer_U); d_spsv_buffer_U = nullptr; }
    CUDA_CHECK_ERROR(cudaMalloc(&d_spsv_buffer_U, bufferSize_U));
    CUSPARSE_CHECK(cusparseSpSV_analysis(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha_one, mat_U, vec_probe_in, vec_probe_out, CUDA_R_64F,
        CUSPARSE_SPSV_ALG_DEFAULT, spsv_info_U, d_spsv_buffer_U));

    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_probe_in));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vec_probe_out));

    audit_finite("ILU-Faktorisierung", d_ilu_values, m_ilu_nnz);
    ilu_ready = true;
}

// =================================================================================
// Der FEM-Block-Vorkonditionierer A~^-1, herausgeloest
// =================================================================================
//
// Inhaltlich unveraendert (AMS-Zweig, CPU-ILU-Zweig, cuSPARSE-ILU-Zweig), nur
// nicht mehr direkt in apply_preconditioner(): der blocktriangulaere Zweig
// braucht dieselbe Operation auf einer ANDEREN rechten Seite (r_A + T*z_K
// statt r_A), und zwei Kopien derselben drei Zweige waeren die klassische
// Quelle dafuer, dass eine spaetere Aenderung nur an einer Stelle landet.
//
// Arbeitet ausschliesslich auf den ersten m_ilu_num_rows Eintraegen. d_in_A
// und d_out_A duerfen sich NICHT ueberlappen (gilt an allen Aufrufstellen).
void CoupledMVOperator::apply_fem_preconditioner(const double* d_in_A, double* d_out_A) {
    // -------------------------------------------------------------------------
    // AMS-Zweig (--ams)
    // -------------------------------------------------------------------------
    // Der AMS ist zur PHYSIKALISCHEN Matrix gebaut, gearbeitet wird aber im
    // aequilibrierten Tilde-Raum. Dort gilt B_tilde = D^-1 B D^-1, es sind also
    // genau zwei Divisionen durch D_scale einzurahmen. Die Hilfsmatrizen
    // bleiben davon unberuehrt, denn (D^-1 G)^T (D A D) (D^-1 G) = G^T A G.
    if (m_ams_ready) {
        bf::sync();                       // Host greift auf Managed Memory zu
        d_D_scale.prefetch_to_host();
        bf::sync();
        const double* D = d_D_scale.data();

        for (int i = 0; i < m_ilu_num_rows; ++i) {
            const double d = D[i];
            m_ams_in[i] = (d != 0.0) ? (d_in_A[i] / d) : 0.0;
        }
        m_ams.apply(m_ams_in.data(), m_ams_out.data());
        for (int i = 0; i < m_ilu_num_rows; ++i) {
            const double d = D[i];
            d_out_A[i] = (d != 0.0) ? (m_ams_out[i] / d) : 0.0;
        }
        return;
    }

    if (bf::on_cpu()) {
        // dieselben zwei Dreieckssubstitutionen wie unten,
        // nur ohne cusparseSpSV. Bewusst rein sequenziell -- eine
        // Dreieckssubstitution ist von Natur aus rekursiv (x[i] haengt von
        // x[j<i] ab); die Level-Scheduling-Parallelisierung, die cuSPARSE in
        // der Analysephase aufbaut, waere hier reiner Mehraufwand ohne
        // Gegenwert bei einem einzelnen CPU-Kern pro Substitution.
        //
        // d_in_A und d_out_A duerfen nicht ueberlappen (gilt an allen
        // Aufrufstellen in BiCGStab/GMRES: p -> p_hat, s -> s_hat, v_j -> z_j,
        // sowie d_pc_rA -> d_out im blocktriangulaeren Zweig).
        bf::csr_lu_solve(m_ilu_num_rows, m_row_offsets, m_col_indices,
                         d_ilu_values, m_cpu_ilu_diag,
                         d_in_A, d_out_A, m_cpu_ilu_temp.data());
    } else {
        const double alpha = 1.0;

        cusparseDnVecDescr_t vec_in, vec_temp, vec_out;
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_in, m_ilu_num_rows, const_cast<double*>(d_in_A), CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_temp, m_ilu_num_rows, d_ilu_temp.data(), CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vec_out, m_ilu_num_rows, d_out_A, CUDA_R_64F));

        // 1. Vorwaertseinsetzen: L * temp = in
        CUSPARSE_CHECK(cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_L, vec_in, vec_temp, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_info_L));

        // 2. Rueckwaertseinsetzen: U * out = temp
        CUSPARSE_CHECK(cusparseSpSV_solve(cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, mat_U, vec_temp, vec_out, CUDA_R_64F, CUSPARSE_SPSV_ALG_DEFAULT, spsv_info_U));

        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_in));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_temp));
        CUSPARSE_CHECK(cusparseDestroyDnVec(vec_out));
    }

}

// =================================================================================
// Blocktriangulaerer Vorkonditionierer -- Bausteine
// =================================================================================

void CoupledMVOperator::ensure_block_tri_buffers() {
    if (num_bem_dofs <= 0) return;

    const size_t n_fem = static_cast<size_t>(total_dofs);
    const size_t n_bem = static_cast<size_t>(num_bem_dofs);

    if (d_pc_rA.size() != n_fem) {
        d_pc_rA.allocate(n_fem);
        d_pc_TzK.allocate(n_fem);
    }
    if (d_pc_cg_r.size() != n_bem) {
        d_pc_S_tmp.allocate(n_bem);
        d_pc_K_phys.allocate(n_bem);
        d_pc_cg_r.allocate(n_bem);
        d_pc_cg_p.allocate(n_bem);
        d_pc_cg_Ap.allocate(n_bem);
        d_pc_cg_best.allocate(n_bem);
    }
}

void CoupledMVOperator::set_block_triangular_preconditioner(bool enable, int inner_iters,
                                                            double inner_tol) {
    m_block_tri_pc   = enable;
    m_pc_inner_iters = (inner_iters > 0) ? inner_iters : 1;
    m_pc_inner_tol   = (inner_tol > 0.0) ? inner_tol : 1e-2;
    m_pc_call_index  = 0;

    if (!enable) return;

    if (num_bem_dofs <= 0) {
        std::cout << "[Solver] Blocktriangulaerer Vorkonditionierer angefordert, aber keine "
                     "BEM-Kopplung aktiv -- Schalter bleibt wirkungslos.\n";
        return;
    }
    if (dense_blocks.empty() && aca_blocks.empty()) {
        throw std::runtime_error(
            "set_block_triangular_preconditioner: es liegen keine G_BEM-Bloecke vor. "
            "Die Schur-Naeherung S~ = mu_0*G_BEM laesst sich ohne sie nicht anwenden -- "
            "der Aufruf muss NACH den add_dense_block()/add_aca_block()-Aufrufen stehen.");
    }

    ensure_block_tri_buffers();

    std::cout << "[Solver] Blocktriangulaerer Vorkonditionierer aktiv "
              << "(S~ = mu_0*G_BEM, inneres CG: max " << m_pc_inner_iters
              << " Iterationen, Toleranz " << m_pc_inner_tol << ").\n";
}

// y_K = S~ * x_K, im selben Raum, in dem der aeussere Loeser rechnet.
//
// S~ = mu_0 * G_BEM ist der Single-Layer-Block, also genau der (2,2)-Block des
// Operators (siehe apply_physical(): +mu_0 auf dense_blocks/aca_blocks). Er ist
// symmetrisch positiv definit -- deshalb genuegt innen ein CG und kein zweites
// GMRES. Bei aktiver Aequilibrierung ist die Diagonale von D_K*(mu_0*G)*D_K
// per Konstruktion exakt 1.0, das innere System ist also bereits Jacobi-
// vorkonditioniert.
void CoupledMVOperator::apply_schur_tilde(const double* d_x_K, double* d_y_K) {
    const double alpha    = 1.0;
    const double beta_one = 1.0;
    const double beta_zero = 0.0;

    // 1. In den physikalischen Raum: x_phys = D_K .* x_tilde
    const double* x_use = d_x_K;
    if (is_equilibrated) {
        BF_LAUNCH_1D_PAR(k_vec_mult, num_bem_dofs,
            num_bem_dofs, d_x_K, d_D_scale.data() + total_dofs, d_pc_S_tmp.data());
        x_use = d_pc_S_tmp.data();
    }

    // 2. y_phys = mu_0 * G_BEM * x_phys (dichte Bloecke + ACA-Bloecke)
    bf::memset_d(d_y_K, 0, static_cast<size_t>(num_bem_dofs) * sizeof(double));

    const double plus_mu0 = m_mu_0;
    for (const auto& block : dense_blocks) {
        // Bei transposed liegt die (n x m)-Matrix des Partners im Speicher;
        // dgemv mit transpose=true wendet sie als (m x n) an. Leading
        // Dimension ist dann n, nicht m.
        bf::dgemv(cublas, /*transpose=*/block.transposed,
                  block.transposed ? block.n : block.m,
                  block.transposed ? block.m : block.n,
                  plus_mu0, block.d_matrix,
                  block.transposed ? block.n : block.m,
                  x_use + block.col_offset, beta_one, d_y_K + block.row_offset);
    }
    for (const auto& aca : aca_blocks) {
        double* d_z = d_temp_z.data();
        bf::dgemv(cublas, /*transpose=*/true, aca.n, aca.k,
                  alpha, aca.d_V, aca.n,
                  x_use + aca.col_offset, beta_zero, d_z);
        bf::dgemv(cublas, /*transpose=*/false, aca.m, aca.k,
                  plus_mu0, aca.d_U, aca.m,
                  d_z, beta_one, d_y_K + aca.row_offset);
    }

    // 3. Zurueck in den Tilde-Raum: y_tilde = D_K .* y_phys (elementweise,
    //    in-place unbedenklich).
    if (is_equilibrated) {
        BF_LAUNCH_1D_PAR(k_vec_mult, num_bem_dofs,
            num_bem_dofs, d_y_K, d_D_scale.data() + total_dofs, d_y_K);
    }
}

// Prueft, ob die Ausgabe des Vorkonditionierers durchgehend endlich ist.
// Bewusst auf dem Host: der Vektor liegt in Managed/Host-Speicher, und ein
// einzelner Durchlauf ist gegenueber den Dreieckssubstitutionen und dem
// inneren CG nicht messbar.
bool CoupledMVOperator::preconditioner_output_is_finite(const double* d_out) const {
    bf::sync();
    const int n = m_ilu_num_rows + num_bem_dofs;
    // Ueber host_readable(), nicht direkt: der Kommentar oben nimmt an, dass
    // d_out in Managed- oder Host-Speicher liegt. Das gilt fuer DeviceArray,
    // aber nicht fuer jeden Puffer, der hier ankommen kann -- und ein
    // cudaMalloc-Zeiger faellt hier mit einem Segfault auf, nicht mit einer
    // Fehlermeldung. Im CPU-Modus und bei Managed Memory ist der Aufruf ein
    // No-Op ohne Kopie.
    std::vector<double> scratch;
    const double* p = host_readable(d_out, static_cast<size_t>(n), scratch);
    if (p == nullptr) return true;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) return false;
    }
    return true;
}

// Zaehlt nicht endliche Eintraege getrennt nach FEM- und BEM-Block und meldet
// den ersten Fundort. Getrennt zu zaehlen ist der Punkt: es beantwortet die
// Frage, die man sonst nur raten kann -- kommt der Wert aus dem FEM-Block
// (ILU/AMS, Eichung, rechte Seite) oder aus dem BEM-Block (inneres CG,
// G_BEM/ACA)?
void CoupledMVOperator::report_non_finite(const char* label, const double* v) const {
    bf::sync();
    int bad_fem = 0, bad_bem = 0, first = -1;
    for (int i = 0; i < m_ilu_num_rows; ++i) {
        if (!std::isfinite(v[i])) { ++bad_fem; if (first < 0) first = i; }
    }
    for (int i = 0; i < num_bem_dofs; ++i) {
        const int g = m_ilu_num_rows + i;
        if (!std::isfinite(v[g])) { ++bad_bem; if (first < 0) first = g; }
    }
    std::cout << "  [Blocktri]   " << label << ": "
              << bad_fem << " / " << m_ilu_num_rows << " im FEM-Block, "
              << bad_bem << " / " << num_bem_dofs << " im BEM-Block";
    if (first >= 0) std::cout << ", erster bei Index " << first << " = " << v[first];
    if (bad_fem == 0 && bad_bem == 0) std::cout << "  (alles endlich)";
    std::cout << "\n";
}

// Inneres CG auf S~ z_K = r_K, Startvektor 0.// Inneres CG auf S~ z_K = r_K, Startvektor 0.
//
// Startvektor 0 ist hier nicht Bequemlichkeit, sondern Voraussetzung: ein
// Vorkonditionierer MUSS eine (naeherungsweise) LINEARE Abbildung von r auf z
// sein. Ein Warmstart aus dem vorherigen Aufruf wuerde eine Abhaengigkeit von
// der Aufrufhistorie einbauen; selbst das flexible GMRES vertraegt zwar einen
// wechselnden, aber keinen von z abhaengigen Operator.
int CoupledMVOperator::solve_schur_block(const double* d_r_K, double* d_z_K) {
    m_last_inner_rel_res = -1.0;
    const int n = num_bem_dofs;

    bf::memset_d(d_z_K, 0, static_cast<size_t>(n) * sizeof(double));
    bf::dcopy(cublas, n, d_r_K, d_pc_cg_r.data());

    double rho = 0.0;
    bf::ddot(cublas, n, d_pc_cg_r.data(), d_pc_cg_r.data(), &rho);
    if (!(rho > 0.0)) return 0;                    // r == 0 (oder NaN): nichts zu tun

    const double r0_norm = std::sqrt(rho);

    // BESTE ITERIERTE MITFUEHREN statt auf eine Schranke zu setzen.
    //
    // Vorgeschichte, zwei eigene Fehler: zuerst brach ich ab, sobald das
    // Residuum ueber den Startwert wuchs. Das ist fuer CG falsch -- die 2-Norm
    // faellt dort nicht monoton, ein Anstieg um Faktor 2 in den ersten
    // Schritten ist normal. Daraufhin lockerte ich auf einen Faktor 1e8, und
    // damit feuerte die Schranke nie: gemessen erreichte das innere CG mit 100
    // Iterationen relative Residuen von 9.17 und 6.19, also das 84-fache in
    // rho -- weit unter 1e8. Der aeussere Loeser brach dabei von 0.026 auf
    // 0.416 ein.
    //
    // Eine Schranke richtig zu treffen ist hier gar nicht noetig. Es genuegt,
    // die beste bisher erreichte Iterierte mitzufuehren und am Ende diese
    // zurueckzugeben. Das Ergebnis ist damit NIE schlechter als der Startwert
    // z_K = 0, unabhaengig davon, was CG unterwegs anstellt. Der Preis ist ein
    // zusaetzlicher Vektor der Laenge num_bem_dofs.
    double rho_best = rho;               // z_K = 0 ist die erste "beste" Iterierte
    bf::memset_d(d_pc_cg_best.data(), 0, static_cast<size_t>(n) * sizeof(double));
    bf::dcopy(cublas, n, d_pc_cg_r.data(), d_pc_cg_p.data());

    int it = 0;
    for (; it < m_pc_inner_iters; ++it) {
        apply_schur_tilde(d_pc_cg_p.data(), d_pc_cg_Ap.data());

        double pAp = 0.0;
        bf::ddot(cublas, n, d_pc_cg_p.data(), d_pc_cg_Ap.data(), &pAp);

        // Verlust der positiven Definitheit. G_BEM ist als Single-Layer-
        // Operator SPD, die ACA-Approximation der Fernfeldbloecke ist es aber
        // nicht exakt (Block (I,J) und (J,I) werden unabhaengig voneinander
        // approximiert). Statt CG hier weiterlaufen zu lassen -- das erzeugt
        // beliebigen Unsinn -- wird abgebrochen; das bis hierher aufgebaute
        // z_K ist eine gueltige, nur groebere Naeherung, und der aeussere
        // FGMRES vertraegt einen schwaecheren Vorkonditionierer problemlos.
        if (!(pAp > 0.0)) {
            if (m_pc_call_index == 1) {
                std::cout << "  [Blocktri] inneres CG bei Iteration " << it
                          << " abgebrochen (p^T S~ p <= 0) -- S~ ist durch die "
                             "ACA-Approximation nicht exakt definit.\n";
            }
            return it;
        }

        const double alpha_cg = rho / pAp;
        bf::daxpy(cublas, n,  alpha_cg, d_pc_cg_p.data(),  d_z_K);
        bf::daxpy(cublas, n, -alpha_cg, d_pc_cg_Ap.data(), d_pc_cg_r.data());

        double rho_new = 0.0;
        bf::ddot(cublas, n, d_pc_cg_r.data(), d_pc_cg_r.data(), &rho_new);

        // Divergenz statt Konvergenz.
        //
        // Die Schranke war mit "rho_new > 1e6 * rho" viel zu weit gefasst und
        // hat deshalb nie ausgeloest. Gemessen am echten Problem erreichte das
        // innere CG relative Residuen von 1.7 bis 2.1 -- bei einem Ziel von
        // 0.01. Es hat das Residuum also VERGROESSERT, um 70 bis 110 %, und der
        // aeussere Loeser bekam eine Korrektur, die schlechter ist als gar
        // keine. Ein Wachstum um Faktor 2 liegt weit unterhalb von 1e6 und fiel
        // durch.
        //
        // Der Grund dafuer ist inhaltlich: S~ = mu_0*G_BEM ist als
        // Single-Layer-Operator analytisch symmetrisch positiv definit, seine
        // ACA-Approximation aber nicht -- Block (I,J) und (J,I) werden
        // unabhaengig voneinander approximiert. CG setzt SPD voraus und darf
        // auf einem indefiniten Operator divergieren.
        //
        // Neue Schranke: sobald das Residuum groesser ist als das der
        // Startnaeherung z_K = 0, wird der letzte Schritt zurueckgenommen und
        // abgebrochen. Damit ist garantiert, dass der Vorkonditionierer nie
        // schlechter wird als das blockdiagonale Verhalten. Das Residuum von CG
        // ist auch im gutartigen Fall nicht monoton, deshalb wird gegen rho_0
        // verglichen und nicht gegen den Vorschritt -- ein zwischenzeitlicher
        // Anstieg bleibt zulaessig.
        // KORREKTUR eines Fehlers meinerseits: hier stand zuvor
        // "rho_new > rho_0", also Abbruch sobald die Residuumsnorm ueber den
        // Startwert waechst. Das ist fuer CG FALSCH. CG minimiert die
        // Energienorm des Fehlers, die 2-Norm des Residuums faellt dabei NICHT
        // monoton -- ein Anstieg um Faktor 2 bis 3 in den ersten Schritten ist
        // bei einem Konditionszahl-behafteten Operator voellig normal und kein
        // Zeichen von Divergenz.
        //
        // Mit dem zu scharfen Kriterium brach das innere CG schon in Iteration
        // 0 ab, lieferte z_K ~ 1e-17 und der aeussere Loeser blieb bei
        // Rel. Res. 0.996 stehen -- deutlich schlechter als ohne den
        // Vorkonditionierer.
        //
        // Abgebrochen wird jetzt nur bei echter Divergenz: nicht endlichen
        // Werten oder einem Anstieg um mehr als vier Groessenordnungen.
        if (!std::isfinite(rho_new)) {
            if (m_pc_call_index <= 3) {
                std::cout << "  [Blocktri]   inneres CG bei Iteration " << it
                          << " abgebrochen: nicht endliches Residuum.\n";
            }
            break;
        }

        if (rho_new < rho_best) {
            rho_best = rho_new;
            bf::dcopy(cublas, n, d_z_K, d_pc_cg_best.data());
        }

        if (std::sqrt(rho_new) <= m_pc_inner_tol * r0_norm) { ++it; break; }

        // p = r + beta*p
        const double beta_cg = rho_new / rho;
        bf::dscal(cublas, n, beta_cg, d_pc_cg_p.data());
        bf::daxpy(cublas, n, 1.0, d_pc_cg_r.data(), d_pc_cg_p.data());
        rho = rho_new;
    }

    // Die beste Iterierte zurueckgeben, nicht die letzte.
    bf::dcopy(cublas, n, d_pc_cg_best.data(), d_z_K);
    m_last_inner_rel_res = std::sqrt(rho_best) / r0_norm;

    // Hat CG den Startwert nicht unterboten, ist z_K = 0 herausgekommen. Der
    // Aufrufer faellt dann auf die Identitaet zurueck (siehe
    // apply_preconditioner) -- das ist genau das blockdiagonale Verhalten und
    // damit die Untergrenze, unter die der Vorkonditionierer nicht faellt.
    return (rho_best < r0_norm * r0_norm) ? it : 0;
}

// out_A = D_A .* ( T * (D_K .* z_K) ) -- der Kopplungsterm der
// Rueckwaertssubstitution. Siehe apply_T_to_A_kernel fuer das Vorzeichen (+T).
void CoupledMVOperator::apply_coupling_T_tilde(const double* d_z_K, double* d_out_A) {
    if (m_num_boundary_faces <= 0 || d_mesh_faces == nullptr || d_mesh_nodes == nullptr) {
        throw std::runtime_error(
            "apply_coupling_T_tilde: setup_bem_mesh() wurde nicht aufgerufen -- der "
            "Kopplungsblock T des Vorkonditionierers braucht die Rand-Quad-Geometrie.");
    }

    // 1. z_K in den physikalischen Raum.
    const double* z_use = d_z_K;
    if (is_equilibrated) {
        BF_LAUNCH_1D_PAR(k_vec_mult, num_bem_dofs,
            num_bem_dofs, d_z_K, d_D_scale.data() + total_dofs, d_pc_K_phys.data());
        z_use = d_pc_K_phys.data();
    }

    // 2. out_A = T * z_K (physikalisch). Der Kernel akkumuliert per
    //    atomic_add, der Puffer muss daher vorher genullt werden.
    bf::memset_d(d_out_A, 0, static_cast<size_t>(total_dofs) * sizeof(double));
    BF_LAUNCH_1D(apply_T_to_A_kernel, m_num_boundary_faces,
        m_num_boundary_faces, d_mesh_faces, d_mesh_nodes,
        d_bem_to_fem, d_bem_sign, z_use, d_out_A);

    // 3. Zurueck in den Tilde-Raum.
    if (is_equilibrated) {
        BF_LAUNCH_1D_PAR(k_vec_mult, total_dofs,
            total_dofs, d_out_A, d_D_scale.data(), d_out_A);
    }
}

// =================================================================================
// apply_preconditioner -- Weiche zwischen blockdiagonal und blocktriangulaer
// =================================================================================
void CoupledMVOperator::apply_preconditioner(const double* d_in, double* d_out) {
    // Diagnose-Bypass -- reine Identitaet auf dem GESAMTEN (gekoppelten)
    // Vektor, kein ILU/SpSV. Dient dazu, den ILU(0)-Preconditioner als
    // Ursache eines Konvergenzproblems auszuschliessen oder zu bestaetigen.
    if (m_ilu_bypass) {
        const size_t n = static_cast<size_t>(m_ilu_num_rows) + static_cast<size_t>(num_bem_dofs);
        bf::memcpy_d(d_out, d_in, n * sizeof(double), cudaMemcpyDeviceToDevice);
        return;
    }

    if (!ilu_ready) {
        throw std::runtime_error("apply_preconditioner: compute_ilu() wurde noch nicht (erfolgreich) aufgerufen.");
    }

    ++m_pc_call_index;

    // -------------------------------------------------------------------------
    // Blocktriangulaerer Zweig
    // -------------------------------------------------------------------------
    //
    //     P^-1 r = ( A~^-1 (r_A + T z_K) )   mit   z_K = S~^-1 r_K
    //              (        z_K           )         S~  = mu_0 * G_BEM
    //
    // Die drei Schritte werden SEQUENZIELL ausgefuehrt -- genau das loest das
    // Skalierungsproblem, an dem der Jacobi-Versuch auf dem K-Block gescheitert
    // ist: z_A und z_K werden nicht unabhaengig voneinander skaliert, sondern
    // z_A sieht den bereits korrigierten K-Anteil ueber den Term T*z_K. Die
    // relative Gewichtung der beiden Bloecke im gemeinsamen Krylov-Vektorraum
    // ergibt sich damit aus der Physik der Kopplung und nicht aus zwei
    // unabhaengig gewaehlten Diagonalskalierungen.
    if (block_triangular_preconditioner_active()) {
        double* d_z_K = d_out + m_ilu_num_rows;

        // 1. z_K = S~^-1 r_K
        const int inner_iters = solve_schur_block(d_in + m_ilu_num_rows, d_z_K);

        // KORREKTUR eines zweiten Fehlers meinerseits: hat das innere CG nichts
        // erreicht (Abbruch in Iteration 0, etwa weil p^T S~ p <= 0), blieb
        // z_K ~ 0 stehen. Das ist NICHT "so gut wie nichts tun", sondern
        // deutlich schlechter: der blockdiagonale Zweig setzt z_K = r_K, also
        // die Identitaet. Mit z_K = 0 bekommt der K-Block ueberhaupt keine
        // Vorkonditionierung, GMRES kann den K-Anteil des Residuums nicht mehr
        // reduzieren und stagniert bei Rel. Res. ~1 -- genau so beobachtet.
        //
        // Der richtige Rueckfall ist die Identitaet. Damit gilt die Invariante
        // aus tests/test_block_triangular_consistency auch im Fehlerfall: der
        // blocktriangulaere Zweig ist nie schlechter als der blockdiagonale.
        if (inner_iters == 0) {
            bf::memcpy_d(d_z_K, d_in + m_ilu_num_rows,
                         static_cast<size_t>(num_bem_dofs) * sizeof(double),
                         cudaMemcpyDeviceToDevice);
            if (m_pc_call_index <= 3) {
                std::cout << "  [Blocktri]   inneres CG ohne Fortschritt -> "
                             "Rueckfall auf z_K = r_K (Identitaet, wie im "
                             "blockdiagonalen Zweig).\n";
            }
        }

        // 2. r_A' = r_A + T z_K
        apply_coupling_T_tilde(d_z_K, d_pc_TzK.data());

        // Der Streu-Kernel schreibt auch auf geeichte Kanten -- dort lautet die
        // Gleichung aber x_i = 0, ein Beitrag von T*z_K waere dort reiner
        // Nullraum-Muell in der rechten Seite des ILU/AMS.
        //
        // WICHTIG: projiziert wird NUR der Korrekturterm T*z_K, nicht die Summe.
        // P ist linear und die eingehende Groesse r_A ist bereits geeicht (GMRES
        // projiziert r, v_j und z_j), also gilt P(r_A + T z_K) = r_A + P(T z_K)
        // -- mathematisch dasselbe, numerisch aber nicht: die Projektion auf die
        // Summe rechnet die Kohomologie-Anteile von r_A ein ZWEITES Mal heraus.
        // Das ist keine Theorie, es ist gemessen (tests/test_block_triangular_
        // consistency): mit z_K = 0 lieferte die Variante auf der Summe
        // ||z_A|| = 1.3956 statt der 1.6609 des blockdiagonalen Zweigs, obwohl
        // beide in diesem Fall identisch sein MUESSEN. Daraus folgt die
        // Invariante, die der Test festhaelt: r_K = 0  =>  blocktriangulaer
        // == blockdiagonal, exakt.
        if (d_is_gauged != nullptr) {
            project_gauge_tilde_orthogonal(d_pc_TzK.data());
        }

        bf::dcopy(cublas, m_ilu_num_rows, d_in, d_pc_rA.data());
        bf::daxpy(cublas, m_ilu_num_rows, 1.0, d_pc_TzK.data(), d_pc_rA.data());

        // 3. z_A = A~^-1 r_A'
        apply_fem_preconditioner(d_pc_rA.data(), d_out);

        // -------------------------------------------------------------------------
        // Notbremse: nicht endliche Ausgabe
        // -------------------------------------------------------------------------
        // Das innere CG laeuft auf dem ACA-approximierten G_BEM. Der
        // Single-Layer-Operator ist analytisch SPD, seine ACA-Approximation aber
        // nicht exakt -- Block (I,J) und (J,I) werden unabhaengig voneinander
        // approximiert. Neben dem bereits abgefangenen Fall p^T S~ p <= 0 kann
        // p^T S~ p auch positiv, aber winzig sein; alpha = rho/pAp wird dann
        // riesig und z_K kann ueberlaufen. Ein einziges Inf im Vorkonditionierer
        // vergiftet ueber w = A*z den kompletten Arnoldi-Prozess: die
        // Hessenberg-Matrix wird NaN, die Rueckwaertssubstitution liefert NaN,
        // und der Loeser meldet am Ende irgendeine Zahl, ohne dass irgendwo eine
        // Fehlermeldung stand.
        //
        // Statt das geschehen zu lassen: pruefen, und im Ernstfall auf den
        // blockdiagonalen Zweig zurueckfallen. Der ist schwaecher, aber
        // definiert -- ein schwacher Vorkonditionierer kostet Iterationen, ein
        // NaN kostet den ganzen Zeitschritt.
        if (!preconditioner_output_is_finite(d_out)) {
            if (!m_block_tri_warned) {
                std::cout << "  [Blocktri] WARNUNG: nicht endliche Ausgabe (NaN/Inf) im "
                             "blocktriangulaeren Vorkonditionierer.\n"
                             "  [Blocktri] Fallback auf den blockdiagonalen Zweig fuer "
                             "diesen Aufruf. Wahrscheinlichste Ursache: das innere CG auf\n"
                             "  [Blocktri] S~ = mu_0*G_BEM divergiert, weil die "
                             "ACA-Approximation von G_BEM nicht positiv definit ist.\n"
                             "  [Blocktri] Gegenmittel zum Ausprobieren: --pc-inner-iters=3 "
                             "(kuerzeres CG kann nicht weit divergieren).\n";

                // Herkunftsanalyse. Entscheidend ist die EINGABE: ist die schon
                // nicht endlich, hat der Vorkonditionierer den Wert nicht
                // erzeugt, sondern nur weitergereicht -- dann liegt die Ursache
                // vor GMRES (rechte Seite, Eichung, Aequilibrierung) und hat
                // mit --blocktri nichts zu tun.
                //
                // Zweiter Schluessel ist ||r_K||: ist es null, war z_K = 0 und
                // T*z_K = 0, der blocktriangulaere Zweig also rechnerisch
                // IDENTISCH zum blockdiagonalen (Invariante aus
                // tests/test_block_triangular_consistency). Ein Defekt kann
                // dann nicht aus der Blocktriangularitaet stammen.
                double r_K_norm = 0.0;
                bf::dnrm2(cublas, num_bem_dofs, d_in + m_ilu_num_rows, &r_K_norm);
                std::cout << "  [Blocktri]   inneres CG lief " << inner_iters
                          << " Iterationen, ||r_K|| = " << r_K_norm << "\n";
                report_non_finite("EINGABE  r", d_in);
                report_non_finite("AUSGABE  z", d_out);
                report_non_finite("T*z_K     ", d_pc_TzK.data());
                if (r_K_norm == 0.0) {
                    std::cout << "  [Blocktri]   ||r_K|| == 0 -> dieser Aufruf ist rechnerisch "
                                 "identisch zum blockdiagonalen Zweig.\n"
                                 "  [Blocktri]   Die Ursache liegt damit NICHT in der "
                                 "Blocktriangularitaet. Kontrolllauf ohne --blocktri.\n";
                }
                m_block_tri_warned = true;
            }
            apply_fem_preconditioner(d_in, d_out);
            if (num_bem_dofs > 0) {
                bf::memcpy_d(d_out + m_ilu_num_rows, d_in + m_ilu_num_rows,
                             static_cast<size_t>(num_bem_dofs) * sizeof(double),
                             cudaMemcpyDeviceToDevice);
            }
            return;
        }

        // Diagnose fuer die ersten Aufrufe je Zeitschritt: wie weit kommt das
        // innere CG, und in welcher Groessenordnung liegen die beiden Bloecke?
        // Bleibt die Iterationszahl konstant bei m_pc_inner_iters, ist die
        // Toleranz nie erreicht worden. Waechst ||z_K|| von Aufruf zu Aufruf um
        // Groessenordnungen, divergiert das innere CG.
        if (m_pc_call_index <= 3) {
            // Die Zwischengroesse ||T*z_K|| ist hier der eigentliche Punkt.
            // Beobachtet wurde ||z_A|| ~ 1e4 bei ||r|| = 1, also eine enorme
            // Verstaerkung durch den Vorkonditionierer. Dafuer gibt es genau
            // zwei Erklaerungen, und diese Zeile trennt sie:
            //
            //   (a) der Kopplungsterm T*z_K ist selbst schon riesig
            //       -> Skalierungsfehler in apply_coupling_T_tilde
            //   (b) T*z_K ist massvoll und erst A~^-1 blaest ihn auf
            //       -> der Term liegt (teilweise) im Gradientenanteil, den der
            //          Curl-Curl-Operator nicht sieht. A~^-1 ist auf so einer
            //          rechten Seite beliebig gross. Genau dagegen prueft
            //          fem.cu bei der physikalischen rechten Seite mit G^T b --
            //          fuer T*z_K findet diese Pruefung bisher nicht statt.
            double r_K_norm = 0.0, z_K_norm = 0.0, z_A_norm = 0.0;
            double r_A_norm = 0.0, TzK_norm = 0.0;
            bf::dnrm2(cublas, m_ilu_num_rows, d_in, &r_A_norm);
            bf::dnrm2(cublas, num_bem_dofs, d_in + m_ilu_num_rows, &r_K_norm);
            bf::dnrm2(cublas, num_bem_dofs, d_z_K, &z_K_norm);
            bf::dnrm2(cublas, m_ilu_num_rows, d_pc_TzK.data(), &TzK_norm);
            bf::dnrm2(cublas, m_ilu_num_rows, d_out, &z_A_norm);
            std::cout << "  [Blocktri] Aufruf " << m_pc_call_index
                      << " | inneres CG: " << inner_iters << " Iter."
                      << " | ||r_A|| = " << r_A_norm
                      << ", ||r_K|| = " << r_K_norm
                      << " -> ||z_K|| = " << z_K_norm
                      << ", ||T*z_K|| = " << TzK_norm
                      << " -> ||z_A|| = " << z_A_norm
                      << "  (Verstaerkung durch A~^-1: "
                      << (TzK_norm > 0.0 ? z_A_norm / TzK_norm : 0.0) << ")\n";

            // Was das innere CG tatsaechlich erreicht hat. Die Annahme hinter
            // S~ = mu_0*G_BEM ist, dass der Single-Layer-Operator im
            // Tilde-Raum gutartig ist (Diagonale exakt 1) und CG in wenigen
            // Schritten konvergiert. Am Modellproblem stimmte das. Hier laeuft
            // das CG in JEDEM Aufruf die volle Iterationszahl, ohne die
            // Toleranz zu erreichen -- diese Zahl sagt, wie weit es ueberhaupt
            // kommt. Bleibt sie nahe 1, ist S~ nicht das gutartige Objekt, fuer
            // das wir es halten, und die ganze Schur-Naeherung steht in Frage.
            std::cout << "  [Blocktri]   inneres CG: erreichtes rel. Residuum = "
                      << m_last_inner_rel_res
                      << " (Ziel " << m_pc_inner_tol << ")\n";

            // Der eigentliche Test: wie viel Gradientenanteil steckt im
            // Kopplungsterm? Vergleichsmassstab ist die physikalische rechte
            // Seite, fuer die fem.cu 2.1e-04 misst (praktisch divergenzfrei).
            // Liegt der Wert hier um Groessenordnungen hoeher, ist die
            // Verstaerkung erklaert: A~^-1 wird auf einen Anteil angewandt,
            // den der Curl-Curl-Operator auf null abbildet.
            const double gc_r = relative_gradient_content(d_in);
            const double gc_T = relative_gradient_content(d_pc_TzK.data());
            if (gc_r >= 0.0) {
                std::cout << "  [Blocktri]   Gradientenanteil ||G^T v||/||v||: "
                          << "r_A = " << gc_r << ", T*z_K = " << gc_T
                          << "   (physikalische rechte Seite zum Vergleich: 2.1e-04)\n";
            }
        }
        return;
    }

    // -------------------------------------------------------------------------
    // Blockdiagonaler Zweig (bisheriges Verhalten)
    // -------------------------------------------------------------------------
    apply_fem_preconditioner(d_in, d_out);

    // BEM/K-Anteil unveraendert durchreichen (Identitaets-
    // Vorkonditionierer). GEAENDERT ZURUECK von einem Jacobi-Versuch
    // (M_K = diag(-mu_0*G)): die Diagonalwerte selbst waren korrekt und
    // pathologiefrei (siehe debug_print_bem_diag_stats(): 2,5e8 bis 9,3e9,
    // keine Ausreisser, keine Luecken), aber das Ergebnis war empirisch
    // SCHLECHTER als die Identitaet (Plateau bei ~0,1 statt ~2e-4). Grund:
    // p/r/v/s/t sind EIN gemeinsamer Vektor ueber A- und K-Block, mit EINEM
    // gemeinsamen Skalarprodukt/einer gemeinsamen Norm fuer die gesamte
    // BiCGStab-Rekursion. Eine ~1e9-Skalierung des K-Anteils (waehrend der
    // A-Anteil bei ~1e-8 bleibt) laesst den K-Anteil diese Skalarprodukte/
    // Normen dominieren -- das Verfahren minimiert dann faktisch fast nur
    // noch den K-Fehler und vernachlaessigt den physikalisch relevanten
    // A-Fehler. "Pro Block gut skaliert" reicht bei einem gemeinsamen
    // Vektorraum nicht aus, wenn die Blocks dadurch relativ zueinander massiv
    // ungleich gewichtet werden. d_in/d_out haben bei aktiver Kopplung die
    // volle gekoppelte Laenge total_dofs+num_bem_dofs; nur die ersten
    // m_ilu_num_rows (== total_dofs, FEM) wurden oben per ILU vorkonditioniert.
    // BEM/K-Anteil des Preconditioners negieren statt zu
    // kopieren. Solange KEINE Aequilibrierung aktiv ist, war Identitaet
    // korrekt (M_K = diag(-mu_0*G) hat stark variierende, aber jeweils
    // negative Werte -- Identitaet ist dort nur eine grobe Naeherung).
    // MIT aktiver SDE-Skalierung (siehe setup_equilibration()) ist die
    // skalierte BEM-Diagonale \tilde{M}_KK_ii = D_i * M_KK_ii * D_i EXAKT
    // -1.0 fuer jeden Eintrag (D_i = sqrt(|1/M_KK_ii|), also D_i^2 * M_KK_ii
    // = sign(M_KK_ii) = -1, da G positiv definit und M_KK = -mu_0*G). Die
    // korrekte (hier sogar exakte, nicht nur approximative) Diagonal-Inverse
    // ist folglich -I, nicht +I. Eine Identitaet mit falschem Vorzeichen
    // behandelt einen negativ-definiten Block mit einem positiv-definiten
    // Vorkonditionierer -- das spaltet das Spektrum um 0 und fuehrt zu den
    // (r_hat,v)~0 / rho~0 Near-Breakdowns von BiCGStab. Ohne SDE (d_D_scale
    // nicht aufgebaut) waere diese Negation FALSCH -- daher der Bezug auf
    // is_equilibrated.
    // ein Block-Gauss-Seidel-Kopplungsterm
    // (M_KA_scaled*z_A - r_K, ueber die lokale Randmassenmatrix T) wurde
    // versucht, um die A<->K-Kopplung im Preconditioner abzubilden --
    // empirisch aber eine deutliche Verschlechterung (Amplifikation
    // ~30 -> ~130, Rel.Res divergierte statt zu fallen). Vermutlich war die
    // Annahme "Kopplung ~1.5, aehnlich der Diagonale" falsch: D_K liegt laut
    // Diagnose (d_bem_inv_diag: 2.5e8 bis 9.3e9) bei ~1.6e4 bis ~9.6e4, damit
    // ist D_K*T*D_A vermutlich um Groessenordnungen groesser als angenommen.
    // apply_physical() negiert jetzt die
    // gesamte K-Gleichung (H_BEM- und G_BEM-Vorzeichen gedreht, siehe dort --
    // loesungserhaltend, da RHS_K == 0). Die skalierte BEM-Diagonale
    // \tilde{M}_KK ist dadurch exakt +1.0 statt -1.0 (Herleitung wie oben,
    // nur mit umgekehrtem Vorzeichen von M_KK_ii). Die korrekte Diagonal-
    // Inverse ist folglich wieder +I -- zurueck zur reinen Identitaets-Kopie,
    // k_negate_vector wird hier nicht mehr gebraucht.
    if (num_bem_dofs > 0) {
        bf::memcpy_d(d_out + m_ilu_num_rows, d_in + m_ilu_num_rows,
                                         static_cast<size_t>(num_bem_dofs) * sizeof(double),
                                         cudaMemcpyDeviceToDevice);
    }
}

// =================================================================================
// Abbruch bei nicht endlichen Werten (BiCGStab, GMRES)
// =================================================================================
//
// Die Abbruchkriterien haben die Form "res/b_norm <= tol". Fuer NaN ist das
// falsch, der Loeser meldet also keine Konvergenz -- das ist die sichere
// Richtung, aber er lief bis max_iter weiter (bis zu 1000 Iterationen samt
// Vorkonditionierer) und lieferte am Ende ein NaN-x. Siehe
// docs/design/nan-and-thresholds.md. Jetzt bricht er sofort ab.
static void report_nonfinite_abort(const char* solver, const char* what, int iter) {
    std::cout << "[" << solver << "] ABBRUCH: " << what
              << " nicht endlich (NaN/Inf) bei Iteration " << iter
              << ". Matrix, rechte Seite oder Vorkonditionierer enthalten"
                 " nicht endliche Werte; weiterrechnen bringt nichts.\n";
}

// =================================================================================
// Implementierung: BiCGStabSolver (Aufgabe 6.2)
// =================================================================================

BiCGStabSolver::BiCGStabSolver(int N) {
    r.allocate(N);
    r_hat.allocate(N);
    p.allocate(N);
    v.allocate(N);
    s.allocate(N);
    t.allocate(N);
    p_hat.allocate(N); // M^-1 * p (ILU(0)-Vorkonditionierung)
    s_hat.allocate(N); // M^-1 * s
    x_best.allocate(N); // bestes bisher gefundenes x
}

void BiCGStabSolver::solve(CoupledMVOperator& op, const double* d_b, double* d_x, double tol, int max_iter) {
    int N = op.get_total_dofs();
    cublasHandle_t cublas = op.get_cublas_handle();

    const double c_one = 1.0;
    const double c_minus_one = -1.0;

    double b_norm = 0.0;
    bf::dnrm2(cublas, N, d_b, &b_norm);
    if (!std::isfinite(b_norm)) {
        report_nonfinite_abort("BiCGStab", "||b||", 0);
        return;
    }
    if (b_norm == 0.0) b_norm = 1.0;

    // 1. Initiales Residuum: r = b - A*x
    op.apply(d_x, r.data()); // r = A*x

    // A*x eichen. apply() liefert bei bereits gauge-konsistentem
    // x (siehe Fix 2: p_hat/s_hat werden vor jedem apply()-Aufruf projiziert,
    // x bleibt dadurch bei exakt 0 auf den geeichten Kanten) auf diesen Zeilen
    // keinen sinnvollen Wert mehr -- die (I-P)*x-Korrektur in apply() greift
    // dort nur, wenn x selbst noch ungeeichte Anteile enthaelt. Ohne diese
    // Projektion haeuft sich genau dieser "Muell" im Nullraum in r auf und
    // taeuscht ein hohes Residuum vor, obwohl der physikalisch relevante
    // (freie) Anteil laengst konvergiert ist.
    op.project_gauge_tilde(r.data(), r.data());

    // r = -r + b
    bf::dscal(cublas, N, c_minus_one, r.data());
    bf::daxpy(cublas, N, c_one, d_b, r.data());

    // r_hat = r
    bf::dcopy(cublas, N, r.data(), r_hat.data());

    double rho = 1.0, rho_prev = 1.0;
    double alpha = 1.0, omega = 1.0;
    double r_norm = 0.0;

    bf::dnrm2(cublas, N, r.data(), &r_norm);
    if (!std::isfinite(r_norm)) {
        report_nonfinite_abort("BiCGStab", "Startresiduum", 0);
        return;
    }
    if ((r_norm / b_norm) <= tol) {
        std::cout << "[BiCGStab] Converged at iteration 0\n";
        return;
    }

    // bestes x bisher = Startvektor (Sicherheitsnetz -- falls ein
    // Near-Breakdown spaeter im Verlauf zu einer voruebergehenden
    // Verschlechterung fuehrt, geben wir am Ende nicht das zuletzt berechnete,
    // sondern das mit dem kleinsten beobachteten Rel.Res. zurueck).
    double best_rel_res = r_norm / b_norm;
    bf::dcopy(cublas, N, d_x, x_best.data());
    int restart_count = 0;
    // Relative Schwelle fuer "Near-Breakdown": exakte Nullchecks (== 0.0)
    // greifen in double precision praktisch nie, obwohl Werte in der
    // Groessenordnung von Rundungsfehlern bereits zu riesigen alpha/omega und
    // damit zu den beobachteten Oszillationen (Rel.Res. springt zwischen ~13
    // und ~591) fuehren. Van der Vorst u.a. empfehlen bei sowas einen Restart
    // (r_hat = r, p = v = 0) statt eines harten Abbruchs.
    const double breakdown_eps = 1e-12;

    // p = 0, v = 0
    bf::memset_d(p.data(), 0, N * sizeof(double));
    bf::memset_d(v.data(), 0, N * sizeof(double));

    for (int iter = 1; iter <= max_iter; ++iter) {
        // rho = (r_hat, r)
        bf::ddot(cublas, N, r_hat.data(), r.data(), &rho);

        double r_hat_norm = 0.0;
        bf::dnrm2(cublas, N, r_hat.data(), &r_hat_norm);

        if (std::abs(rho) < breakdown_eps * r_hat_norm * r_norm) {
            // Near-Breakdown: r ist (fast) orthogonal zu r_hat geworden. Ein
            // exaktes Null waere nur bei Unterlauf (||r|| < ca. 1.5e-154)
            // moeglich -- das haette die Konvergenzpruefung oben laengst
            // gefangen. Heilmittel: Restart mit frischem r_hat = r.
            ++restart_count;
            std::cout << "  [BiCGStab] Near-Breakdown (rho ~ 0) bei Iteration " << iter
                      << ", Restart Nr. " << restart_count << " (r_hat = r).\n";
            bf::dcopy(cublas, N, r.data(), r_hat.data());
            bf::memset_d(p.data(), 0, N * sizeof(double));
            bf::memset_d(v.data(), 0, N * sizeof(double));
            rho_prev = 1.0; alpha = 1.0; omega = 1.0;
            continue;
        }

        if (iter == 1) {
            // p = r
            bf::dcopy(cublas, N, r.data(), p.data());
        } else {
            // beta = (rho / rho_prev) * (alpha / omega)
            double beta = (rho / rho_prev) * (alpha / omega);

            // p = r + beta * (p - omega * v)
            // 1. p = p - omega * v
            double minus_omega = -omega;
            bf::daxpy(cublas, N, minus_omega, v.data(), p.data());
            // 2. p = beta * p
            bf::dscal(cublas, N, beta, p.data());
            // 3. p = p + r
            bf::daxpy(cublas, N, c_one, r.data(), p.data());
        }

        // rechts-Vorkonditionierung, M * p_hat = p
        op.apply_preconditioner(p.data(), p_hat.data());

        // Diagnose -- Amplifikationsfaktor ||p_hat||/||p|| des
        // Preconditioners in den ersten Iterationen. Ein vernuenftiger
        // Preconditioner sollte die Groessenordnung von p ungefaehr erhalten
        // (Faktor nahe 1, evtl. 0.1-10). Ein Faktor von vielen
        // Groessenordnungen deutet auf kaputte L/U-Werte hin (echter Bug),
        // nicht nur auf einen schwachen/unpassenden Preconditioner.
        if (iter <= 3) {
            double p_norm = 0.0, p_hat_norm = 0.0;
            bf::dnrm2(cublas, N, p.data(), &p_norm);
            bf::dnrm2(cublas, N, p_hat.data(), &p_hat_norm);
            std::cout << "  [Diagnose] iter " << iter << ": ||p|| = " << p_norm
                      << ", ||p_hat|| = " << p_hat_norm
                      << ", Amplifikation = " << (p_hat_norm / p_norm) << "\n";
        }

        // p_hat re-eichen. Der Diagonal-Shift in compute_ilu()
        // macht M nichtsingulaer, aber M ist nicht P*A*P -- auf den
        // geeichten/Kohomologie-Richtungen "erfindet" M^-1 dort Anteile, die
        // apply() sofort wieder herausprojizieren wuerde. Ohne diese
        // Re-Eichung haeufen sich die erfundenen Anteile ueber die
        // Iterationen auf (Ursache des NaN-Blowups). project_gauge ist
        // idempotent und identisch zu der Projektion, die apply() intern
        // anwendet -- p_hat bleibt also konsistent mit dem restlichen Operator.
        op.project_gauge_tilde(p_hat.data(), p_hat.data());

        // v = A * p_hat  (statt A*p -- rechts-vorkonditioniertes BiCGStab)
        op.apply(p_hat.data(), v.data());

        // v eichen, aus demselben Grund wie beim initialen
        // Residuum oben -- p_hat ist bereits gauge-konsistent (Fix 2), der
        // (I-P)*x-Korrekturterm in apply() liefert daher hier nur Rauschen.
        op.project_gauge_tilde(v.data(), v.data());

        // alpha = rho / (r_hat, v)
        double rhat_dot_v = 0.0;
        bf::ddot(cublas, N, r_hat.data(), v.data(), &rhat_dot_v);

        double v_norm = 0.0;
        bf::dnrm2(cublas, N, v.data(), &v_norm);

        if (std::abs(rhat_dot_v) < breakdown_eps * r_hat_norm * v_norm) {
            // Near-Breakdown: v ist (fast) orthogonal zu r_hat. Gleiches
            // Heilmittel wie oben -- Restart statt harter Exception.
            ++restart_count;
            std::cout << "  [BiCGStab] Near-Breakdown ((r_hat,v) ~ 0) bei Iteration " << iter
                      << ", Restart Nr. " << restart_count << " (r_hat = r).\n";
            bf::dcopy(cublas, N, r.data(), r_hat.data());
            bf::memset_d(p.data(), 0, N * sizeof(double));
            bf::memset_d(v.data(), 0, N * sizeof(double));
            rho_prev = 1.0; alpha = 1.0; omega = 1.0;
            continue;
        }
        alpha = rho / rhat_dot_v;

        // s = r - alpha * v
        bf::dcopy(cublas, N, r.data(), s.data());
        double minus_alpha = -alpha;
        bf::daxpy(cublas, N, minus_alpha, v.data(), s.data());

        // Early convergence check
        double s_norm = 0.0;
        bf::dnrm2(cublas, N, s.data(), &s_norm);
        if (!std::isfinite(s_norm)) {
            // x selbst kann seit dem letzten besten Stand schon verdorben
            // sein; zurueck auf den letzten endlichen besten Stand.
            report_nonfinite_abort("BiCGStab", "||s||", iter);
            bf::dcopy(cublas, N, x_best.data(), d_x);
            return;
        }
        if ((s_norm / b_norm) <= tol) {
            // x = x + alpha * p_hat  (mit dem vorkonditionierten Vektor updaten,
            // da rechts-vorkonditioniert mit x = M^-1 * u gerechnet wird)
            bf::daxpy(cublas, N, alpha, p_hat.data(), d_x);
            std::cout << "[BiCGStab] Converged at iteration " << iter << " (early exit)\n";
            return;
        }

        // rechts-Vorkonditionierung, M * s_hat = s
        op.apply_preconditioner(s.data(), s_hat.data());

        // s_hat re-eichen, aus demselben Grund wie bei p_hat oben.
        op.project_gauge_tilde(s_hat.data(), s_hat.data());

        // t = A * s_hat  (statt A*s)
        op.apply(s_hat.data(), t.data());

        // t eichen, aus demselben Grund wie bei v oben.
        op.project_gauge_tilde(t.data(), t.data());

        // omega = (t, s) / (t, t)
        double t_dot_s = 0.0, t_dot_t = 0.0;
        bf::ddot(cublas, N, t.data(), s.data(), &t_dot_s);
        bf::ddot(cublas, N, t.data(), t.data(), &t_dot_t);
        
        if (t_dot_t < breakdown_eps * breakdown_eps * b_norm * b_norm) {
            // Near-Breakdown: t = A*s_hat ist (fast) null, obwohl s selbst
            // (s_norm/b_norm > tol, siehe Check oben) noch nicht klein ist --
            // omega waere hier undefinierbar/riesig. Der alpha-Anteil ist
            // davon unabhaengig gueltig, wird also noch angewendet; danach
            // Restart mit r = s (kein omega-Term) und frischem r_hat = r.
            ++restart_count;
            std::cout << "  [BiCGStab] Near-Breakdown ((t,t) ~ 0) bei Iteration " << iter
                      << ", Restart Nr. " << restart_count
                      << " (x += alpha*p_hat, r = s, r_hat = r).\n";
            bf::daxpy(cublas, N, alpha, p_hat.data(), d_x);
            bf::dcopy(cublas, N, s.data(), r.data());
            bf::dcopy(cublas, N, r.data(), r_hat.data());
            bf::memset_d(p.data(), 0, N * sizeof(double));
            bf::memset_d(v.data(), 0, N * sizeof(double));
            rho_prev = 1.0; alpha = 1.0; omega = 1.0;

            bf::dnrm2(cublas, N, r.data(), &r_norm);
            if ((r_norm / b_norm) < best_rel_res) {
                best_rel_res = r_norm / b_norm;
                bf::dcopy(cublas, N, d_x, x_best.data());
            }
            continue;
        }
        omega = t_dot_s / t_dot_t;

        // x = x + alpha * p_hat + omega * s_hat
        // (mit den vorkonditionierten Vektoren p_hat/s_hat statt p/s --
        // rechts-vorkonditioniertes BiCGStab loest A*M^-1*u = b und rekonstruiert
        // x = M^-1*u inkrementell ueber genau diese beiden Terme)
        // 1. x = x + alpha * p_hat
        bf::daxpy(cublas, N, alpha, p_hat.data(), d_x);
        // 2. x = x + omega * s_hat
        bf::daxpy(cublas, N, omega, s_hat.data(), d_x);

        // r = s - omega * t
        bf::dcopy(cublas, N, s.data(), r.data());
        double minus_omega2 = -omega;
        bf::daxpy(cublas, N, minus_omega2, t.data(), r.data());

        // Normal convergence check
        bf::dnrm2(cublas, N, r.data(), &r_norm);
        if (!std::isfinite(r_norm)) {
            report_nonfinite_abort("BiCGStab", "Residuum", iter);
            bf::dcopy(cublas, N, x_best.data(), d_x);
            return;
        }

        // bestes x bisher merken (Sicherheitsnetz gegen voruebergehende
        // Verschlechterungen durch Near-Breakdowns, siehe oben).
        if ((r_norm / b_norm) < best_rel_res) {
            best_rel_res = r_norm / b_norm;
            bf::dcopy(cublas, N, d_x, x_best.data());
        }

        // Diagnose-Logging alle 50 Iterationen (plus die erste), um zu
        // unterscheiden, ob die Rel.Res. monoton (nur langsam, z.B. weil das
        // ILU(0) den BEM-Anteil nicht sieht) faellt oder tatsaechlich
        // stagniert/oszilliert (strukturelles Problem). Bewusst kein Logging
        // bei jeder Iteration, um das Log bei max_iter=1000 nicht zu fluten.
        if (iter == 1 || iter % 50 == 0) {
            std::cout << "  [BiCGStab] iter " << iter
                      << " | Rel. Res: " << (r_norm / b_norm) << "\n";
            // Residuum nach geeichten Kanten / freien Kanten /
            // BEM-Block aufschluesseln -- soll zeigen, WO das 0.13-Plateau
            // tatsaechlich sitzt, statt es zu vermuten.
            op.debug_print_gauge_split_norm(r.data(), "r");
        }

        if ((r_norm / b_norm) <= tol) {
            std::cout << "[BiCGStab] Converged at iteration " << iter 
                      << " | Rel. Res: " << (r_norm / b_norm) << "\n";
            return;
        }

        rho_prev = rho;
    }

    // Falls max_iter erreicht wurde, ohne dass der letzte Schritt der
    // beste war (z.B. weil ein Near-Breakdown-Restart kurz vor Schluss noch
    // einmal voruebergehend verschlechtert hat), das beste bisher gefundene
    // x zurueckgeben statt des zuletzt berechneten.
    const double final_rel_res = r_norm / b_norm;
    if (best_rel_res < final_rel_res) {
        std::cout << "[BiCGStab] Gebe bestes gefundenes x zurueck (Rel.Res " << best_rel_res
                  << ") statt des zuletzt berechneten (Rel.Res " << final_rel_res << ").\n";
        bf::dcopy(cublas, N, x_best.data(), d_x);
    }

    std::cout << "[BiCGStab] WARNING: Max iterations (" << max_iter 
              << ") reached without convergence. Rel. Res: "
              << std::min(final_rel_res, best_rel_res) << "\n";
}

// =================================================================================
// Implementierung GMRESSolver (Rechts-vorkonditioniertes GMRES(m))
// =================================================================================

GMRESSolver::GMRESSolver(int N_dofs, int m_restart)
    : N(N_dofs), m(m_restart)
{
    r.allocate(static_cast<size_t>(N));
    w.allocate(static_cast<size_t>(N));

    V_basis.allocate(static_cast<size_t>(N) * static_cast<size_t>(m + 1));
    Z_basis.allocate(static_cast<size_t>(N) * static_cast<size_t>(m));

    H.resize(static_cast<size_t>(m + 1) * static_cast<size_t>(m), 0.0);
    g.resize(static_cast<size_t>(m + 1), 0.0);
    cosines.resize(static_cast<size_t>(m), 0.0);
    sines.resize(static_cast<size_t>(m), 0.0);
}

void GMRESSolver::generate_givens_rotation(double dx, double dy, double& cs, double& sn) {
    if (dy == 0.0) {
        cs = 1.0;
        sn = 0.0;
    } else if (std::abs(dy) > std::abs(dx)) {
        double temp = dx / dy;
        sn = 1.0 / std::sqrt(1.0 + temp * temp);
        cs = temp * sn;
    } else {
        double temp = dy / dx;
        cs = 1.0 / std::sqrt(1.0 + temp * temp);
        sn = temp * cs;
    }
}

void GMRESSolver::apply_givens_rotation(double& h1, double& h2, double& cs, double& sn) {
    double temp = cs * h1 + sn * h2;
    h2 = -sn * h1 + cs * h2;
    h1 = temp;
}

void GMRESSolver::solve(CoupledMVOperator& op, const double* d_b, double* d_x, double tol, int max_iter) {
    cublasHandle_t cublas = op.get_cublas_handle();
    const double c_one = 1.0;
    const double c_minus_one = -1.0;

    double b_norm = 0.0;
    bf::dnrm2(cublas, N, d_b, &b_norm);
    if (!std::isfinite(b_norm)) {
        report_nonfinite_abort("GMRES", "||b||", 0);
        m_last_rel_res = std::numeric_limits<double>::quiet_NaN();
        m_last_iterations = 0;
        return;
    }
    if (b_norm == 0.0) b_norm = 1.0;

    int total_iters = 0;

    while (total_iters < max_iter) {
        // 1. Initiales Residuum: r = b - A*x. Dieselbe Reihenfolge wie in
        // BiCGStabSolver::solve() (siehe dortiger Kommentar): apply() liefert
        // bei bereits gauge-konsistentem x auf den geeichten Kanten keinen
        // sinnvollen Wert mehr -- ohne project_gauge_tilde() haeuft sich dort
        // Nullraum-Muell an und taeuscht ein zu hohes Residuum vor.
        op.apply(d_x, r.data());
        op.project_gauge_tilde(r.data(), r.data());

        bf::dscal(cublas, N, c_minus_one, r.data());
        bf::daxpy(cublas, N, c_one, d_b, r.data());

        double r_norm = 0.0;
        bf::dnrm2(cublas, N, r.data(), &r_norm);

        // Auch hier setzen: wer schon am Zyklusbeginn konvergiert (oder
        // abbricht), las bisher den Wert eines frueheren solve()-Aufrufs.
        m_last_rel_res = r_norm / b_norm;
        m_last_iterations = total_iters;
        if (!std::isfinite(r_norm)) {
            report_nonfinite_abort("GMRES", "Residuum am Zyklusbeginn", total_iters);
            return;
        }
        if ((r_norm / b_norm) <= tol) {
            std::cout << "[GMRES] Converged at iteration " << total_iters << "\n";
            return;
        }

        double inv_r_norm = 1.0 / r_norm;
        double* d_v0 = V_basis.data();
        bf::dcopy(cublas, N, r.data(), d_v0);
        bf::dscal(cublas, N, inv_r_norm, d_v0);

        std::fill(g.begin(), g.end(), 0.0);
        std::fill(H.begin(), H.end(), 0.0);
        g[0] = r_norm;

        int j = 0;
        for (; j < m && total_iters < max_iter; ++j, ++total_iters) {
            double* d_v_j = V_basis.data() + static_cast<size_t>(j) * N;
            double* d_z_j = Z_basis.data() + static_cast<size_t>(j) * N;
            double* d_v_next = V_basis.data() + static_cast<size_t>(j + 1) * N;

            // Rechts-Vorkonditionierung: z_j = M^-1 * v_j
            op.apply_preconditioner(d_v_j, d_z_j);
            op.project_gauge_tilde(d_z_j, d_z_j);

            // w = A * z_j
            op.apply(d_z_j, w.data());
            op.project_gauge_tilde(w.data(), w.data());

            // Arnoldi-Orthogonalisierung (Modified Gram-Schmidt)
            for (int i = 0; i <= j; ++i) {
                double* d_v_i = V_basis.data() + static_cast<size_t>(i) * N;
                double h_ij = 0.0;
                bf::ddot(cublas, N, w.data(), d_v_i, &h_ij);

                H[static_cast<size_t>(i) * m + j] = h_ij;

                double minus_h_ij = -h_ij;
                bf::daxpy(cublas, N, minus_h_ij, d_v_i, w.data());
            }

            double h_next = 0.0;
            bf::dnrm2(cublas, N, w.data(), &h_next);
            H[static_cast<size_t>(j + 1) * m + j] = h_next;

            // Spalte j sichern, BEVOR die Givens-Rotationen sie ueberschreiben.
            // Die Rotationen bringen H auf obere Dreiecksform; die
            // Hessenberg-Struktur und damit das Ritz-Spektrum waeren danach
            // verloren. Die Kopie kostet m^2 doubles einmalig.
            if (m_want_ritz) {
                if (m_H_raw.size() != H.size()) m_H_raw.assign(H.size(), 0.0);
                for (int i = 0; i <= j + 1; ++i)
                    m_H_raw[static_cast<size_t>(i) * m + j] = H[static_cast<size_t>(i) * m + j];
                m_ritz_dim = j + 1;
            }

            bf::dcopy(cublas, N, w.data(), d_v_next);
            if (h_next > 1e-14) {
                double inv_h = 1.0 / h_next;
                bf::dscal(cublas, N, inv_h, d_v_next);
            }

            // Givens-Rotationen: H auf obere Dreiecksform bringen
            for (int i = 0; i < j; ++i) {
                apply_givens_rotation(H[static_cast<size_t>(i) * m + j],
                                     H[static_cast<size_t>(i + 1) * m + j],
                                     cosines[i], sines[i]);
            }

            generate_givens_rotation(H[static_cast<size_t>(j) * m + j],
                                     H[static_cast<size_t>(j + 1) * m + j],
                                     cosines[j], sines[j]);
            apply_givens_rotation(H[static_cast<size_t>(j) * m + j],
                                  H[static_cast<size_t>(j + 1) * m + j],
                                  cosines[j], sines[j]);
            apply_givens_rotation(g[j], g[j + 1], cosines[j], sines[j]);

            double current_res = std::abs(g[j + 1]) / b_norm;

            // NaN oder Inf aus apply() oder dem Vorkonditionierer laufen
            // ueber H und die Givens-Drehungen in g. Ohne diese Pruefung ist
            // "current_res <= tol" nur falsch, der Zyklus laeuft bis m
            // weiter, und die Rueckwaertssubstitution macht x zu NaN.
            // Abbrechen, BEVOR x angefasst wird: x bleibt der letzte
            // endliche Stand, das NaN-Residuum meldet den Fehlschlag.
            if (!std::isfinite(current_res)) {
                report_nonfinite_abort("GMRES", "Hessenberg-Residuum", total_iters + 1);
                m_last_rel_res = std::numeric_limits<double>::quiet_NaN();
                m_last_iterations = total_iters + 1;
                return;
            }

            if (total_iters == 1 || total_iters % 10 == 0) {
                std::cout << "  [GMRES] iter " << total_iters << " | Rel. Res: " << current_res << "\n";
            }

            if (current_res <= tol) {
                ++j;
                break;
            }
        }

        // Kleines Hessenberg-System H(0:j-1,0:j-1) * y = g(0:j-1) rueckwaerts
        // einsetzen (H ist nach den Givens-Rotationen obere Dreiecksform).
        std::vector<double> y(static_cast<size_t>(j), 0.0);
        // Massstab fuer die Pivotschranke: der groesste Diagonaleintrag des
        // aufgebauten Hessenberg-Systems.
        double h_diag_max = 0.0;
        for (int i = 0; i < j; ++i)
            h_diag_max = std::fmax(h_diag_max, std::fabs(H[static_cast<size_t>(i) * m + i]));
        const double h_diag_floor = 1e-13 * (h_diag_max > 0.0 ? h_diag_max : 1.0);
        int num_dropped = 0;

        for (int i = j - 1; i >= 0; --i) {
            y[i] = g[i];
            for (int k = i + 1; k < j; ++k) {
                y[i] -= H[static_cast<size_t>(i) * m + k] * y[k];
            }
            // Ungeschuetzt war das eine Falle: bei Stagnation oder "happy
            // breakdown" wird H[i][i] winzig oder null, y[i] damit inf, und
            // x += Z*y macht die gesamte Loesung zu NaN -- ohne Fehlermeldung,
            // weil danach niemand mehr auf Endlichkeit prueft.
            //
            // Die Schranke ist relativ zum groessten Diagonaleintrag des
            // Hessenberg-Systems, nicht "ungleich null": ein Pivot von 1e-300
            // ist genauso unbrauchbar wie eine Null, kommt aber durch jede
            // Gleichheitspruefung.
            const double h_ii = H[static_cast<size_t>(i) * m + i];
            if (std::fabs(h_ii) > h_diag_floor) {
                y[i] /= h_ii;
            } else {
                // Dieser Anteil der Suchrichtung ist nicht bestimmbar. Ihn
                // wegzulassen ist die konservative Wahl -- x bleibt gueltig,
                // der Zyklus bringt nur weniger Fortschritt.
                y[i] = 0.0;
                ++num_dropped;
            }
        }
        if (num_dropped > 0) {
            std::cout << "  [GMRES] " << num_dropped << " von " << j
                      << " Hessenberg-Pivots unbrauchbar (|H_ii| <= "
                      << h_diag_floor << ") -- Anteile weggelassen. "
                      << "Das Verfahren stagniert.\n";
        }

        // x += Z * y
        for (int i = 0; i < j; ++i) {
            double* d_z_i = Z_basis.data() + static_cast<size_t>(i) * N;
            bf::daxpy(cublas, N, y[i], d_z_i, d_x);
        }

        // Echtes Residuum nach dem Restart-Zyklus pruefen (nicht nur die
        // Hessenberg-Schaetzung |g[j]|, die durch Rundungsfehler ueber viele
        // Restarts hinweg driften kann).
        op.apply(d_x, r.data());
        op.project_gauge_tilde(r.data(), r.data());
        bf::dscal(cublas, N, c_minus_one, r.data());
        bf::daxpy(cublas, N, c_one, d_b, r.data());
        bf::dnrm2(cublas, N, r.data(), &r_norm);

        m_last_rel_res = r_norm / b_norm;
        m_last_iterations = total_iters;
        if ((r_norm / b_norm) <= tol) {
            std::cout << "[GMRES] Converged at iteration " << total_iters
                      << " | Rel. Res: " << (r_norm / b_norm) << "\n";
            return;
        } else if (j == m) {
            std::cout << "  [GMRES] Restarting (Rel. Res: " << (r_norm / b_norm) << ")...\n";

            // Blockweise Aufschluesselung des echten Residuums.
            //
            // Der Grund dafuer ist ein Befund aus der Vorkonditionierer-
            // Diagnose: die Krylov-Vektoren tragen ||r_A|| ~ 0.02 gegen
            // ||r_K|| ~ 1, das Residuum sitzt also fast vollstaendig im
            // BEM-Block. Und dessen rechte Seite ist KONSTRUKTIONSBEDINGT NULL
            // (siehe fem.cu: RHS_K wird nie gefuellt).
            //
            // Damit ist die Gesamtnorm r_norm/b_norm irrefuehrend: b besteht
            // praktisch nur aus dem FEM-Anteil, ein Residuum von 0.007
            // relativ zu ||b|| kann also im K-Block einem RELATIVEN Fehler von
            // 100 % entsprechen -- die BEM-Gleichungen waeren dann gar nicht
            // geloest. Genau das laesst sich hier ablesen und nicht an der
            // Gesamtnorm.
            //
            // Der FEM-Anteil wird gegen ||b_A|| gemessen, der BEM-Anteil
            // absolut sowie im Verhaeltnis zum FEM-Anteil des Residuums --
            // ein relatives Mass gibt es dort nicht, weil b_K = 0 ist.
            if (m_report_block_residual && m_num_bem_for_report > 0) {
                const int n_fem = N - m_num_bem_for_report;
                double rA = 0.0, rK = 0.0, bA = 0.0, bK = 0.0;
                bf::dnrm2(cublas, n_fem, r.data(), &rA);
                bf::dnrm2(cublas, m_num_bem_for_report, r.data() + n_fem, &rK);
                bf::dnrm2(cublas, n_fem, d_b, &bA);
                bf::dnrm2(cublas, m_num_bem_for_report, d_b + n_fem, &bK);
                std::cout << "  [GMRES]   davon FEM-Block: " << rA
                          << " von ||b_A|| = " << bA
                          << "  (relativ " << (bA > 0.0 ? rA / bA : 0.0) << ")\n"
                          << "  [GMRES]   davon BEM-Block: " << rK
                          << " von ||b_K|| = " << bK;
                if (!(bK > 0.0)) {
                    std::cout << "  (b_K ist null -- ein relatives Mass gibt es"
                                 " hier nicht; entscheidend ist rK gegen rA)";
                }
                std::cout << "\n";
            }
            // Ritz-Spektrum genau EINMAL auswerten, nach dem ersten vollen
            // Zyklus. Dann ist der Krylov-Raum maximal, ohne dass die Ausgabe
            // sich bei jedem Restart wiederholt.
            if (m_want_ritz && m_ritz_dim > 1) {
                std::vector<std::complex<double>> ritz;
                std::vector<double> Hc = m_H_raw;
                hessenberg_eigenvalues(Hc, m_ritz_dim, m, ritz);
                report_ritz_spectrum("vorkonditionierter Operator", ritz);
                m_want_ritz = false;
            }
        }
    }

    std::cout << "[GMRES] WARNING: Max iterations reached without convergence.\n";
    m_last_iterations = max_iter;
}