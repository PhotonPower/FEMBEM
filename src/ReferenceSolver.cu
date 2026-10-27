// =============================================================================
// ReferenceSolver.cu -- Baseline: Q1-Knoten-FEM + vollbesetzte BEM
// =============================================================================
// Herleitung, Vorzeichen und die bekannten Grenzen stehen in
// include/ReferenceSolver.cuh bzw. include/NodalBem.cuh. Hier nur die
// Umsetzung.
// =============================================================================

#include "ReferenceSolver.cuh"

#include "Backend.cuh"
#include "DenseLinalg.cuh"
#include "HostLinalg.cuh"
#include "Kinematics.cuh"
#include "NodalBem.cuh"
#include "NodalHexahedron.cuh"
#include "BiotSavart.cuh"
#include "FieldVerification.cuh"
#include "NodalMaxwell.cuh"
#include "PhysicalObject.h"
#include "TransientSolver.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

constexpr double MU_0 = 4.0e-7 * 3.14159265358979323846;

// =============================================================================
// 1. Randtopologie
// =============================================================================

/// Mitte eines Hexaeders -- fuer die Orientierungsbestimmung der Panels.
double3 hex_center(const Mesh& mesh, int hex_id) {
    const Hexahedron h = mesh.hex_nodes[hex_id];
    double3 c = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < 8; ++i) {
        const double3 p = mesh.nodes_current[h.v[i]];
        c.x += p.x; c.y += p.y; c.z += p.z;
    }
    c.x *= 0.125; c.y *= 0.125; c.z *= 0.125;
    return c;
}

} // namespace

ReferenceBoundary build_reference_boundary(const Mesh& mesh) {
    ReferenceBoundary rb;

    mesh.boundary_faces.prefetch_to_host();
    mesh.nodes_current.prefetch_to_host();
    mesh.hex_nodes.prefetch_to_host();

    std::set<int> unique;
    for (size_t f = 0; f < mesh.num_boundary_faces; ++f) {
        const Mesh::QuadFace& q = mesh.boundary_faces[f];
        for (int k = 0; k < 4; ++k) unique.insert(q.v[k]);
    }
    rb.bnode_ids.assign(unique.begin(), unique.end());

    rb.node_to_bnode.assign(mesh.num_nodes, -1);
    for (size_t i = 0; i < rb.bnode_ids.size(); ++i) {
        rb.node_to_bnode[rb.bnode_ids[i]] = static_cast<int>(i);
    }

    // Orientierung geometrisch bestimmen: die aus der Knotenumlaufrichtung
    // gebildete Normale muss von der Mitte des Elternhexaeders WEG zeigen.
    // Damit ist das Vorzeichen unabhaengig davon, wie QuadFace::v sortiert
    // wurde -- eine Tabelle waere hier eine stille Fehlerquelle.
    rb.face_orientation.assign(mesh.num_boundary_faces, 1.0);
    int flipped = 0;
    for (size_t f = 0; f < mesh.num_boundary_faces; ++f) {
        const Mesh::QuadFace& q = mesh.boundary_faces[f];
        double3 nd[4];
        for (int k = 0; k < 4; ++k) nd[k] = mesh.nodes_current[q.v[k]];

        double3 y, nrm; double jac;
        bem_quad_geometry(nd, 0.0, 0.0, y, nrm, jac);

        const double3 hc = hex_center(mesh, q.parent_hex_id);
        const double3 outward = sub3(y, hc);
        if (dot3(nrm, outward) < 0.0) { rb.face_orientation[f] = -1.0; ++flipped; }
    }

    std::cout << "[Referenz] Rand: " << rb.bnode_ids.size() << " Knoten, "
              << mesh.num_boundary_faces << " Panels ("
              << flipped << " Panels umorientiert)\n";
    const double gib = 2.0 * static_cast<double>(rb.bnode_ids.size())
                     * static_cast<double>(rb.bnode_ids.size()) * 8.0 / (1024.0*1024.0*1024.0);
    std::cout << "[Referenz] Vollbesetzte BEM: V und K je "
              << rb.bnode_ids.size() << "^2 -> " << std::fixed << std::setprecision(3)
              << gib << " GiB zusammen\n" << std::defaultfloat;

    // Riegel gegen einen aussichtslosen Lauf. Gebraucht werden DREI dichte
    // Matrizen (V, dann C das zu D wird, und T) plus Arbeitsspeicher fuer
    // cuSOLVER. Eine Verfeinerungsstufe bringt N_bem von 4138 auf 16542, also
    // 2.04 GiB je Matrix und 6.1 GiB zusammen -- das passt in 16 GiB. Zwei
    // Stufen bringen rund 66000 Knoten und 33 GiB JE MATRIX; das scheitert
    // sonst mitten im Lauf an einer Allokation, ohne die Ursache zu nennen.
    {
        const double need = 3.2 * static_cast<double>(rb.bnode_ids.size())
                                * static_cast<double>(rb.bnode_ids.size()) * 8.0
                                / (1024.0*1024.0*1024.0);
        size_t free_b = 0, total_b = 0;
        double budget = 15.0;
        if (bf::on_gpu() && cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
            budget = 0.92 * static_cast<double>(free_b) / (1024.0*1024.0*1024.0);
        }
        std::cout << "[Referenz] Bedarf dichte Algebra ca. " << std::fixed
                  << std::setprecision(2) << need << " GiB, verfuegbar ca. "
                  << budget << " GiB\n" << std::defaultfloat;
        if (need > budget) {
            throw std::runtime_error(
                "Referenzloeser: die vollbesetzte BEM braucht bei " +
                std::to_string(rb.bnode_ids.size()) + " Randknoten etwa " +
                std::to_string(need) + " GiB, verfuegbar sind etwa " +
                std::to_string(budget) + " GiB. Der Speicherbedarf waechst mit "
                "N_bem^2, eine Verfeinerungsstufe vervierfacht N_bem also die "
                "16-fache Matrix. Moeglichkeiten: eine Stufe weniger "
                "(--refine), oder der produktive ACA-Pfad ohne --reference.");
        }
    }

    return rb;
}

// =============================================================================
// 2. CSR-Struktur fuer das Knotensystem (3 DOF je Knoten)
// =============================================================================
namespace {

struct NodalCsr {
    int num_rows = 0;                  // 3 * num_nodes
    int nnz      = 0;
    DeviceArray<int>    row_offsets;
    DeviceArray<int>    col_indices;
    DeviceArray<double> values;
};

void build_nodal_csr(const Mesh& mesh, NodalCsr& csr) {
    const int nn = static_cast<int>(mesh.num_nodes);

    // Knotennachbarschaft ueber die Hexaeder
    std::vector<std::set<int>> adj(nn);
    mesh.hex_nodes.prefetch_to_host();
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const Hexahedron hx = mesh.hex_nodes[h];
        for (int a = 0; a < 8; ++a)
            for (int b = 0; b < 8; ++b)
                adj[hx.v[a]].insert(hx.v[b]);
    }

    csr.num_rows = 3 * nn;
    // Je Knotenpaar ein 3x3-Block. Bei beta = 1 ist der Block diagonal, die
    // Nulleintraege werden aber mitgefuehrt: die Struktur bleibt so
    // unabhaengig von beta, und ILU(0) profitiert nicht davon, sie zu sparen.
    long long nnz = 0;
    for (int i = 0; i < nn; ++i) nnz += 9LL * static_cast<long long>(adj[i].size());
    csr.nnz = static_cast<int>(nnz);

    csr.row_offsets.allocate(static_cast<size_t>(csr.num_rows) + 1);
    csr.col_indices.allocate(static_cast<size_t>(csr.nnz));
    csr.values.allocate(static_cast<size_t>(csr.nnz));
    csr.row_offsets.prefetch_to_host();
    csr.col_indices.prefetch_to_host();

    int pos = 0;
    for (int i = 0; i < nn; ++i) {
        for (int ci = 0; ci < 3; ++ci) {
            csr.row_offsets[3 * i + ci] = pos;
            // Spaltenindizes MUESSEN je Zeile aufsteigend sein -- bf::csr_ilu0
            // setzt das voraus. adj ist ein std::set, also sortiert; die
            // Komponente laeuft innen, damit 3*j+cj monoton bleibt.
            for (int j : adj[i]) {
                for (int cj = 0; cj < 3; ++cj) csr.col_indices[pos++] = 3 * j + cj;
            }
        }
    }
    csr.row_offsets[csr.num_rows] = pos;
    csr.col_indices.prefetch_to_device();
    csr.row_offsets.prefetch_to_device();

    std::cout << "[Referenz] FEM-System: " << csr.num_rows << " Zeilen, "
              << csr.nnz << " Nichtnullen\n";
}

// =============================================================================
// 3. FEM-Assemblierung
// =============================================================================
//
// Elementbeitrag (24x24, DOF-Reihenfolge lokal: knoten*3 + komponente):
//
//   Laplace:  delta_kl * Int (1/mu_0) gradN_i . gradN_j
//   Strafe:   (beta-1)/mu_0 * Int (dN_i/dx_k)(dN_j/dx_l)      [div-div]
//   Leitwert: delta_kl * Int (sigma/dt) N_i N_j
//   Quelle:   Int J_s,k N_i        (rechte Seite)
//   Traegheit:delta_kl * Int (sigma/dt) N_i N_j * A_prev       (rechte Seite)
//
// Ein Kernel pro Hexaeder, Ergebnis wird atomar in CSR und RHS gestreut.
// Das ist derselbe Aufbau wie im produktiven Pfad; im CPU-Modus laeuft er
// deshalb seriell (bf::atomic_add), siehe Backend.cuh.

BF_HD inline void reference_assemble_body(
    int hex_id,
    int num_hexes,
    const Hexahedron* hex_nodes,
    const double3*    nodes,
    const MaterialProperties* materials,
    const double*     A_prev,
    double            dt,
    double            beta,
    double            time_now,
    const int*        csr_row_offsets,
    const int*        csr_col_indices,
    double*           csr_values,
    double*           rhs)
{
    (void)num_hexes;

    const Hexahedron hx = hex_nodes[hex_id];
    double3 nd[8];
    for (int i = 0; i < 8; ++i) nd[i] = nodes[hx.v[i]];

    const MaterialProperties mat = materials[hex_id];
    const double mu     = MU_0 * ((mat.mu_relative > 0.0) ? mat.mu_relative : 1.0);
    const double inv_mu = 1.0 / mu;
    const double sig_dt = mat.kappa / dt;

    // Strafterm elementweise: beta wird als NEGATIV uebergeben, wenn er in
    // leitenden Elementen entfallen soll (--no-conductor-penalty). Der
    // Vorzeichentrick vermeidet ein zusaetzliches Kernelargument.
    double beta_e = (beta < 0.0) ? -beta : beta;
    if (beta < 0.0 && mat.kappa > 0.0) beta_e = 0.0;

    double Ke[24][24];
    double Fe[24];
    for (int a = 0; a < 24; ++a) { Fe[a] = 0.0; for (int b = 0; b < 24; ++b) Ke[a][b] = 0.0; }

    const int ng = nodal_gauss_count();
    for (int g = 0; g < ng; ++g) {
        double xi, eta, zeta, wg;
        nodal_gauss_point(g, xi, eta, zeta, wg);

        double3 gradN[8]; double detJ;
        evaluate_nodal_gradients(nd, xi, eta, zeta, gradN, detJ);
        if (detJ <= 0.0) continue;   // entartetes Element: Beitrag verwerfen

        double N[8];
        evaluate_nodal_shape(xi, eta, zeta, N);

        const double w = wg * detJ;

        // ------ Steifigkeit + Strafterm + Leitwert
        for (int i = 0; i < 8; ++i) {
            const double gi[3] = { gradN[i].x, gradN[i].y, gradN[i].z };
            for (int j = 0; j < 8; ++j) {
                const double gj[3] = { gradN[j].x, gradN[j].y, gradN[j].z };
                const double lap  = inv_mu * (gi[0]*gj[0] + gi[1]*gj[1] + gi[2]*gj[2]);
                const double mass = sig_dt * N[i] * N[j];

                for (int k = 0; k < 3; ++k) {
                    Ke[i*3+k][j*3+k] += w * (lap + mass);
                    if (beta_e != 1.0) {
                        for (int l = 0; l < 3; ++l) {
                            Ke[i*3+k][j*3+l] += w * (beta_e - 1.0) * inv_mu * gi[k] * gj[l];
                        }
                    }
                }
            }
        }

        // ------ Quellstromdichte
        // Dieselbe Auswertung wie im produktiven Pfad: eine Ringspule traegt
        // einen AZIMUTALEN Strom, nur der ist im Spulengebiet divergenzfrei
        // (siehe MaterialTypes.cuh). Ein konstanter kartesischer Vektor waere
        // physikalisch falsch.
        double3 Js = make_double3(0.0, 0.0, 0.0);
        if (mat.type == RegionType::COIL_CURRENT) {
            const double3 p = nodal_map_to_physical(nd, xi, eta, zeta);
            if (mat.current_is_azimuthal) {
                const double rx = p.x - mat.current_axis_origin.x;
                const double ry = p.y - mat.current_axis_origin.y;
                const double r  = sqrt(rx*rx + ry*ry);
                if (r > 1e-12) {
                    const double amp = mat.current_density_x;   // Betrag im x-Feld
                    Js.x = -amp * ry / r;
                    Js.y =  amp * rx / r;
                    Js.z =  0.0;
                }
            } else {
                Js = make_double3(mat.current_density_x,
                                  mat.current_density_y,
                                  mat.current_density_z);
            }
        }
        (void)time_now;   // Zeitabhaengigkeit steckt bereits in current_density_*

        // ------ rechte Seite: Quelle + Traegheit sigma/dt * A_prev
        for (int i = 0; i < 8; ++i) {
            const double wn = w * N[i];
            Fe[i*3+0] += wn * Js.x;
            Fe[i*3+1] += wn * Js.y;
            Fe[i*3+2] += wn * Js.z;

            if (sig_dt != 0.0) {
                // sigma/dt * Int N_i N_j * A_prev_j
                for (int j = 0; j < 8; ++j) {
                    const double m = w * sig_dt * N[i] * N[j];
                    for (int k = 0; k < 3; ++k) {
                        Fe[i*3+k] += m * A_prev[hx.v[j] * 3 + k];
                    }
                }
            }
        }
    }

    // ------ Streuen
    for (int a = 0; a < 24; ++a) {
        const int gi = hx.v[a / 3] * 3 + (a % 3);
        bf::atomic_add(&rhs[gi], Fe[a]);

        const int start = csr_row_offsets[gi];
        const int end   = csr_row_offsets[gi + 1];
        for (int b = 0; b < 24; ++b) {
            const double v = Ke[a][b];
            if (v == 0.0) continue;
            const int gj = hx.v[b / 3] * 3 + (b % 3);
            // Binaere Suche: die Spalten sind je Zeile aufsteigend.
            int lo = start, hi = end - 1, found = -1;
            while (lo <= hi) {
                const int mid = (lo + hi) / 2;
                const int c   = csr_col_indices[mid];
                if (c == gj) { found = mid; break; }
                if (c <  gj) lo = mid + 1; else hi = mid - 1;
            }
            if (found >= 0) bf::atomic_add(&csr_values[found], v);
        }
    }
}

__global__ void reference_assemble(
    int num_hexes, const Hexahedron* hex_nodes, const double3* nodes,
    const MaterialProperties* materials, const double* A_prev,
    double dt, double beta, double time_now,
    const int* csr_row_offsets, const int* csr_col_indices,
    double* csr_values, double* rhs)
{
    int hex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hex_id >= num_hexes) return;
    reference_assemble_body(hex_id, num_hexes, hex_nodes, nodes, materials, A_prev,
                            dt, beta, time_now, csr_row_offsets, csr_col_indices,
                            csr_values, rhs);
}

// =============================================================================
// 4. BEM-Assemblierung (vollbesetzt)
// =============================================================================
//
// Eine Zeile je Randknoten (Kollokation). Die Zeile wird von EINEM Thread
// gerechnet: sie schreibt ausschliesslich in ihre eigene Zeile von V und K,
// also rennfrei und ohne Atomics. Zeilenweise Parallelitaet reicht -- bei 4138
// Zeilen ist die GPU gut ausgelastet, und der Aufwand steckt in der inneren
// Panel-Schleife.
BF_HD inline void reference_bem_row_body(
    int row,
    int num_bnodes,
    int num_faces,
    const int*    bnode_ids,
    const int*    node_to_bnode,
    const Mesh::QuadFace* faces,
    const double* face_orientation,
    const double3* nodes,
    double* V,          // num_bnodes x num_bnodes, ZEILEN-major
    double* K)
{
    const int    nb = num_bnodes;
    const double3 x = nodes[bnode_ids[row]];
    const int    xg = bnode_ids[row];

    double* Vrow = V + static_cast<size_t>(row) * nb;
    double* Krow = K + static_cast<size_t>(row) * nb;
    for (int j = 0; j < nb; ++j) { Vrow[j] = 0.0; Krow[j] = 0.0; }

    for (int f = 0; f < num_faces; ++f) {
        const Mesh::QuadFace q = faces[f];
        double3 nd[4];
        for (int k = 0; k < 4; ++k) nd[k] = nodes[q.v[k]];

        int corner = -1;
        for (int k = 0; k < 4; ++k) if (q.v[k] == xg) corner = k;

        double lV[4], lK[4];
        if (corner >= 0) {
            nodal_bem_panel_singular(nd, corner, face_orientation[f], lV, lK);
        } else {
            const double3 c = nodal_bem_panel_center(nd);
            const double  d = norm3(sub3(x, c));
            const double  h = nodal_bem_panel_size(nd);
            nodal_bem_panel_regular(x, nd, face_orientation[f],
                                    nodal_bem_subdivisions(d, h), lV, lK);
        }

        for (int k = 0; k < 4; ++k) {
            const int col = node_to_bnode[q.v[k]];
            if (col < 0) continue;      // kann nicht vorkommen, aber billig
            Vrow[col] += lV[k];
            Krow[col] += lK[k];
        }
    }

    // Diagonale von K aus der Zeilensummenbedingung (1/2 I + K) * 1 = 0.
    // Siehe NodalBem.cuh: das vermeidet das stark singulaere Integral und
    // erfasst den Raumwinkel an Kanten und Ecken automatisch.
    double off = 0.0;
    for (int j = 0; j < nb; ++j) if (j != row) off += Krow[j];
    Krow[row] = -0.5 - off;
}

__global__ void reference_bem_row(
    int num_bnodes, int num_faces, const int* bnode_ids, const int* node_to_bnode,
    const Mesh::QuadFace* faces, const double* face_orientation,
    const double3* nodes, double* V, double* K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_bnodes) return;
    reference_bem_row_body(row, num_bnodes, num_faces, bnode_ids, node_to_bnode,
                           faces, face_orientation, nodes, V, K);
}

// -----------------------------------------------------------------------------
// Randmassenmatrix T_ij = Int_Gamma N_i N_j dGamma  (dicht, num_bnodes^2)
// -----------------------------------------------------------------------------
BF_HD inline void reference_bem_mass_body(
    int f, int num_faces, int num_bnodes,
    const Mesh::QuadFace* faces, const int* node_to_bnode,
    const double3* nodes, double* T)
{
    (void)num_faces;
    const Mesh::QuadFace q = faces[f];
    double3 nd[4];
    for (int k = 0; k < 4; ++k) nd[k] = nodes[q.v[k]];

    double Me[4][4];
    for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) Me[a][b] = 0.0;

    const int n = bem_gl_n();
    for (int gu = 0; gu < n; ++gu) {
        for (int gv = 0; gv < n; ++gv) {
            const double u = bem_gl_x(gu), v = bem_gl_x(gv);
            const double w = bem_gl_w(gu) * bem_gl_w(gv);

            double3 y, nrm; double jac;
            bem_quad_geometry(nd, u, v, y, nrm, jac);
            double N[4];
            bem_quad_shape(u, v, N);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b)
                    Me[a][b] += w * jac * N[a] * N[b];
        }
    }

    for (int a = 0; a < 4; ++a) {
        const int ia = node_to_bnode[q.v[a]];
        if (ia < 0) continue;
        for (int b = 0; b < 4; ++b) {
            const int ib = node_to_bnode[q.v[b]];
            if (ib < 0) continue;
            bf::atomic_add(&T[static_cast<size_t>(ia) * num_bnodes + ib], Me[a][b]);
        }
    }
}

__global__ void reference_bem_mass(
    int num_faces, int num_bnodes, const Mesh::QuadFace* faces,
    const int* node_to_bnode, const double3* nodes, double* T)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;
    reference_bem_mass_body(f, num_faces, num_bnodes, faces, node_to_bnode, nodes, T);
}


} // namespace

// =============================================================================
// 5. Gekoppelter Operator und GMRES
// =============================================================================
namespace {


// -----------------------------------------------------------------------------
// Gather/Scatter zwischen FEM-Vektor und Randknotenvektor
// -----------------------------------------------------------------------------
// Als KERNEL, nicht als Host-Schleife. Der Grund ist der Fehler, der den ersten
// Referenzlauf zum Divergieren gebracht hat: Kernelstarts und cuBLAS-Aufrufe
// sind ASYNCHRON. Eine Host-Schleife, die unmittelbar danach denselben
// Managed-Memory-Bereich liest, laeuft in ein Datenrennen -- der Operator ist
// dann nicht mehr deterministisch, die Arnoldi-Beziehung bricht, und GMRES
// liefert beliebigen Unsinn statt zu konvergieren. So bleibt apply()
// vollstaendig auf dem Geraet und alles ist ueber den Stream geordnet.
BF_HD inline void ref_gather_body(int i, int nb, const int* bnode_ids,
                                 const double* x, double* out)
{
    const int g = bnode_ids[i] * 3;
    out[static_cast<size_t>(0) * nb + i] = x[g + 0];
    out[static_cast<size_t>(1) * nb + i] = x[g + 1];
    out[static_cast<size_t>(2) * nb + i] = x[g + 2];
}
__global__ void ref_gather(int nb, const int* bnode_ids, const double* x, double* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    ref_gather_body(i, nb, bnode_ids, x, out);
}

BF_HD inline void ref_scatter_body(int i, int nb, const int* bnode_ids,
                                   const double* in, double scale, double* y)
{
    const int g = bnode_ids[i] * 3;
    y[g + 0] += scale * in[static_cast<size_t>(0) * nb + i];
    y[g + 1] += scale * in[static_cast<size_t>(1) * nb + i];
    y[g + 2] += scale * in[static_cast<size_t>(2) * nb + i];
}
__global__ void ref_scatter(int nb, const int* bnode_ids, const double* in,
                            double scale, double* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    ref_scatter_body(i, nb, bnode_ids, in, scale, y);
}

/// Diagonale des dichten Randblocks (T*D)_ii, fuer den Vorkonditionierer.
/// T ist zeilen-major und symmetrisch, D column-major:
///     (T*D)_ii = sum_k T(i,k) D(k,i) = sum_k T[i*nb+k] * D[i*nb+k]
/// Beide Indizes fallen zusammen -- ein einziger Durchlauf je Zeile.
BF_HD inline void ref_dtn_diag_body(int i, int nb, const double* T, const double* D,
                                    double* diag)
{
    double acc = 0.0;
    const size_t off = static_cast<size_t>(i) * nb;
    for (int k = 0; k < nb; ++k) acc += T[off + k] * D[off + k];
    diag[i] = acc;
}
__global__ void ref_dtn_diag(int nb, const double* T, const double* D, double* diag) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    ref_dtn_diag_body(i, nb, T, D, diag);
}

/// y = S_csr * x  -  (1/mu_0) * Streuung( T * D * Sammlung(x) ) je Komponente.
/// G = T * D ist der vollbesetzte Randblock (num_bnodes x num_bnodes,
/// ZEILEN-major), derselbe fuer alle drei Komponenten.
struct ReferenceOperator {
    int num_rows = 0;                    // 3 * num_nodes
    int nb       = 0;                    // num_bnodes
    const int*    row_offsets = nullptr;
    const int*    col_indices = nullptr;
    const double* values      = nullptr;
    const int*    bnode_ids   = nullptr;
    // D = V^-1 (K - 1/2 I), COLUMN-major. T = Randmassenmatrix, symmetrisch.
    // Bewusst NICHT als Produkt G = T*D vorab ausmultipliziert: das waere
    // O(nb^3) = 7e10 FLOP je Zeitschritt fuer nichts. Zwei gemv im Mat-Vec
    // kosten stattdessen 2*nb^2 und sparen zusaetzlich eine ganze Matrix.
    const double* D           = nullptr;
    const double* T           = nullptr;

    // Echter cuBLAS-Handle. Die bf::-Wrapper leiten im GPU-Modus an cuBLAS
    // weiter und ignorieren den Handle nur im CPU-Modus -- ein nullptr waere
    // dort also ein Fehler, nicht eine Abkuerzung.
    cublasHandle_t cublas = nullptr;

    // Managed Memory, NICHT std::vector: cuBLAS und die Kernel brauchen
    // geraetezugreifbare Zeiger. Host-Speicher aus std::vector ist das nicht.
    mutable DeviceArray<double> gather;   // 3 * nb
    mutable DeviceArray<double> mid;      // 3 * nb
    mutable DeviceArray<double> scatter;  // 3 * nb

    void allocate_buffers() {
        if (nb <= 0) return;
        gather.allocate(static_cast<size_t>(3) * nb);
        mid.allocate(static_cast<size_t>(3) * nb);
        scatter.allocate(static_cast<size_t>(3) * nb);
    }

    void apply(const double* x, double* y) const {
        // Sparse-Anteil
        bf::csr_spmv(num_rows, row_offsets, col_indices, values, x, y);

        if (nb == 0 || D == nullptr || T == nullptr) return;

        BF_LAUNCH_1D_PAR(ref_gather, nb, nb, bnode_ids, x, gather.data());

        // q = D * A_Gamma, dann Randterm = T * q.
        // D liegt column-major -> OP_N. T ist symmetrisch, Layout gleichgueltig.
        for (int c = 0; c < 3; ++c) {
            const size_t off = static_cast<size_t>(c) * nb;
            bf::dgemv(cublas, /*transpose=*/false, nb, nb,
                      1.0, D, nb, gather.data() + off, 0.0, mid.data() + off);
            bf::dgemv(cublas, /*transpose=*/false, nb, nb,
                      1.0, T, nb, mid.data() + off, 0.0, scatter.data() + off);
        }

        BF_LAUNCH_1D_PAR(ref_scatter, nb, nb, bnode_ids, scatter.data(),
                         -1.0 / MU_0, y);
    }
};

/// GMRES(m) mit ILU(0)-Vorkonditionierer auf dem Sparse-Anteil.
/// Bewusst eine eigene, kurze Implementierung statt CoupledMVOperator: dessen
/// Krylov-Teil ist mit Aequilibrierung, Eichprojektoren und Nullraumstrafe
/// verwoben, und genau die soll dieser Loeser NICHT haben.
int reference_gmres(const ReferenceOperator& op,
                    const double* b, double* x,
                    const std::vector<double>& ilu_values,
                    const std::vector<int>&    ilu_diag,
                    const int* row_offsets, const int* col_indices,
                    bool use_ilu, int step)
{
    const int n = op.num_rows;
    const int m = REFERENCE_GMRES_RESTART;
    cublasHandle_t cublas = op.cublas;

    // Alle Vektoren, die in bf::-BLAS-Aufrufe oder Kernel gehen, MUESSEN im
    // Managed Memory liegen. Die Hessenberg-Matrix und die Givens-Groessen
    // bleiben Host-Skalare -- sie sind winzig und werden nur auf dem Host
    // gebraucht.
    DeviceArray<double> r(static_cast<size_t>(n));
    DeviceArray<double> tmp(static_cast<size_t>(n));
    DeviceArray<double> w(static_cast<size_t>(n));
    DeviceArray<double> ilu_tmp(static_cast<size_t>(n));
    std::vector<DeviceArray<double>> Vk(static_cast<size_t>(m) + 1);
    for (int i = 0; i <= m; ++i) Vk[i].allocate(static_cast<size_t>(n));

    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
    std::vector<double> cs(m, 0.0), sn(m, 0.0), g(m + 1, 0.0), yv(m, 0.0);

    // ILU(0)-Anwendung: zwei Dreieckssubstitutionen, von Natur aus rekursiv
    // und damit nicht parallelisierbar. Sie laeuft deshalb auf dem HOST, auch
    // im GPU-Modus. Das ist vertretbar, weil Managed Memory den Zugriff
    // erlaubt und die bewegten Datenmengen klein sind: die ILU-Werte (rund
    // 10 MB) wandern einmal und bleiben, die beiden Vektoren sind je 168 kB.
    auto precond = [&](const double* in, double* out) {
        if (!use_ilu) { bf::dcopy(cublas, n, in, out); return; }
        // Die Substitution laeuft auf dem HOST. 'in' wurde gerade von einem
        // Kernel bzw. cuBLAS geschrieben, beides asynchron -- ohne dieses
        // Abwarten liest der Host halbfertige Daten.
        bf::sync();
        bf::csr_lu_solve(n, row_offsets, col_indices, ilu_values.data(), ilu_diag,
                         in, out, ilu_tmp.data());
    };

    double bnorm = 0.0;
    bf::dnrm2(cublas, n, b, &bnorm);
    if (bnorm == 0.0) { std::fill(x, x + n, 0.0); return 0; }

    // Erstes Residuum als Bezug fuer die Divergenzerkennung.
    double rel_first = -1.0;

    int total = 0;
    for (int outer = 0; outer * m < REFERENCE_GMRES_MAX_ITER; ++outer) {
        // r = b - A x, per BLAS: eine Host-Schleife ueber Managed Memory
        // wuerde die Seiten in jeder Iteration hin und her wandern lassen.
        op.apply(x, tmp.data());
        bf::dcopy(cublas, n, b, r.data());
        bf::daxpy(cublas, n, -1.0, tmp.data(), r.data());

        precond(r.data(), w.data());
        double beta_norm = 0.0;
        bf::dnrm2(cublas, n, w.data(), &beta_norm);
        if (beta_norm == 0.0) break;

        bf::dcopy(cublas, n, w.data(), Vk[0].data());
        bf::dscal(cublas, n, 1.0 / beta_norm, Vk[0].data());
        std::fill(g.begin(), g.end(), 0.0);
        g[0] = beta_norm;

        int k_used = 0;
        for (int k = 0; k < m; ++k) {
            ++total;
            op.apply(Vk[k].data(), tmp.data());
            precond(tmp.data(), w.data());

            for (int i = 0; i <= k; ++i) {
                bf::ddot(cublas, n, w.data(), Vk[i].data(), &H[i][k]);
                bf::daxpy(cublas, n, -H[i][k], Vk[i].data(), w.data());
            }
            bf::dnrm2(cublas, n, w.data(), &H[k + 1][k]);

            if (H[k + 1][k] > 1e-300) {
                bf::dcopy(cublas, n, w.data(), Vk[k + 1].data());
                bf::dscal(cublas, n, 1.0 / H[k + 1][k], Vk[k + 1].data());
            } else {
                Vk[k + 1].fill_zero();
            }

            // Givens-Rotationen
            for (int i = 0; i < k; ++i) {
                const double t = cs[i] * H[i][k] + sn[i] * H[i + 1][k];
                H[i + 1][k]    = -sn[i] * H[i][k] + cs[i] * H[i + 1][k];
                H[i][k]        = t;
            }
            const double d = std::sqrt(H[k][k]*H[k][k] + H[k+1][k]*H[k+1][k]);
            if (d > 0.0) { cs[k] = H[k][k] / d; sn[k] = H[k+1][k] / d; }
            else         { cs[k] = 1.0; sn[k] = 0.0; }
            H[k][k]     = d;
            H[k + 1][k] = 0.0;
            g[k + 1]    = -sn[k] * g[k];
            g[k]        =  cs[k] * g[k];

            k_used = k + 1;
            if (std::fabs(g[k + 1]) / beta_norm < REFERENCE_GMRES_TOL) break;
        }

        // Dreieckssystem loesen und aktualisieren
        for (int i = k_used - 1; i >= 0; --i) {
            double s = g[i];
            for (int j = i + 1; j < k_used; ++j) s -= H[i][j] * yv[j];
            yv[i] = (H[i][i] != 0.0) ? (s / H[i][i]) : 0.0;
        }
        for (int i = 0; i < k_used; ++i) {
            bf::daxpy(cublas, n, yv[i], Vk[i].data(), x);
        }

        // Echtes Residuum (nicht das aus den Givens-Rotationen geschaetzte)
        op.apply(x, tmp.data());
        bf::dcopy(cublas, n, b, r.data());
        bf::daxpy(cublas, n, -1.0, tmp.data(), r.data());
        double rn = 0.0;
        bf::dnrm2(cublas, n, r.data(), &rn);
        const double rel = rn / bnorm;

        if (step <= 3 || outer == 0) {
            std::cout << "  [Ref-GMRES] " << total << " Iterationen, rel. Residuum "
                      << std::scientific << std::setprecision(3) << rel
                      << std::defaultfloat << "\n";
        }
        if (rel < REFERENCE_GMRES_TOL * 1e3) break;

        // Divergenzschutz. GMRES minimiert das Residuum ueber den Krylov-Raum;
        // bei korrektem, linearem Operator und festem Vorkonditionierer KANN es
        // nicht wachsen. Tut es das doch, ist etwas kaputt -- typischerweise ein
        // nicht deterministischer Operator (fehlende Synchronisation zwischen
        // asynchronen Kerneln und Host-Zugriffen) oder ein singulaerer
        // Vorkonditionierer. Weiterrechnen erzeugt dann Kraefte von 1e85 N und
        // verschleiert die Ursache; deshalb hier Abbruch mit Diagnose.
        if (rel_first < 0.0) rel_first = rel;
        if (!std::isfinite(rel) || rel > 10.0 * rel_first) {
            throw std::runtime_error(
                "Referenz-GMRES divergiert (rel. Residuum " + std::to_string(rel) +
                " gegen " + std::to_string(rel_first) + " nach dem ersten Zyklus). "
                "GMRES kann bei korrektem linearem Operator nicht divergieren. Zu "
                "pruefen: (a) fehlt eine Synchronisation zwischen asynchronen "
                "Kernel-/cuBLAS-Aufrufen und Host-Zugriffen auf Managed Memory, "
                "(b) ist der ILU-Vorkonditionierer singulaer.");
        }
    }
    return total;
}

} // namespace

// =============================================================================
// 6. Zeitschleife
// =============================================================================
int run_reference_solver(Mesh& mesh,
                         DeviceArray<MaterialProperties>& d_materials,
                         PhysicalObject& plate,
                         const std::vector<PhysicalObject*>& sources,
                         int max_steps,
                         double dt)
{
    std::cout << "\n========================================================\n"
              << " REFERENZLOESER: Q1-Knoten-FEM + vollbesetzte BEM        \n"
              << "========================================================\n";

    // --- Voraussetzung pruefen: mu_r == 1 ueberall
    d_materials.prefetch_to_host();
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        if (std::fabs(d_materials[h].mu_relative - 1.0) > 1e-12) {
            throw std::runtime_error(
                "Referenzloeser: mu_relative != 1 in Hexaeder " + std::to_string(h) +
                ". Die komponentenweise Laplace-Formulierung mit Strafterm-Eichung "
                "setzt mu = const voraus (siehe include/ReferenceSolver.cuh). "
                "Mit magnetischem Material ist dieser Loeser NICHT gueltig.");
        }
    }
    std::cout << "[Referenz] mu_r == 1 in allen " << mesh.num_hexes
              << " Elementen -- Formulierung zulaessig.\n";
    if (bf::no_conductor_penalty()) {
        std::cout << "[Referenz] --no-conductor-penalty: div-div-Strafterm in "
                     "leitenden Elementen ABGESCHALTET (Diagnose, Randterme der "
                     "BEM-Kopplung dann nicht exakt konsistent).\n";
    }
    std::cout << "[Referenz] div-div-Strafterm beta = " << REFERENCE_PENALTY_BETA
              << (REFERENCE_PENALTY_BETA == 1.0
                  ? "  (konsistenter Fall)\n"
                  : "  (ACHTUNG: != 1, Randterme nicht mehr exakt konsistent)\n");

    const int nn  = static_cast<int>(mesh.num_nodes);
    const int ndof = 3 * nn;

    // Startlage der Plattenunterkante. Die Aufgabenstellung definiert die
    // Schwebehoehe als Abstand zwischen Plattenunterkante und Spulenoberkante,
    // und die Spulenoberkante liegt im Netz bei z = 0. Die absolute z-Koordinate
    // der Unterkante IST damit die Schwebehoehe -- waehrend
    // props.translation.z nur die Verschiebung gegenueber der Startlage ist.
    // Ohne diesen Bezug ist die ausgegebene Zahl nicht mit Tabelle I
    // vergleichbar.
    double plate_z0 = 0.0;
    {
        mesh.nodes_initial.prefetch_to_host();
        mesh.hex_nodes.prefetch_to_host();
        bf::sync();
        bool first = true;
        for (int h : plate.get_hex_indices()) {
            const Hexahedron H = mesh.hex_nodes[h];
            for (int n = 0; n < 8; ++n) {
                const double z = mesh.nodes_initial[H.v[n]].z;
                if (first || z < plate_z0) { plate_z0 = z; first = false; }
            }
        }
        std::cout << "[Referenz] Plattenunterkante bei t=0: z = " << std::fixed
                  << std::setprecision(3) << plate_z0 * 1000.0
                  << " mm ueber der Spulenoberkante (= Schwebehoehe nach "
                     "Aufgabenstellung)\n" << std::defaultfloat;
    }

    // --plate-z: Platte auf eine vorgegebene Schwebehoehe setzen, VOR dem
    // Aufbau der Randtopologie -- dann wird die Panelorientierung gleich fuer
    // die endgueltige Lage bestimmt.
    if (bf::plate_z_mm() > 0.0) {
        const double target = bf::plate_z_mm() * 1e-3;
        plate.set_translation_z(target - plate_z0);
        apply_kinematics(mesh, plate);
        bf::sync();
        std::cout << "[Referenz] --plate-z: Platte auf Schwebehoehe "
                  << std::fixed << std::setprecision(3) << bf::plate_z_mm()
                  << " mm gesetzt (Verschiebung "
                  << (target - plate_z0) * 1000.0 << " mm gegen die Netzlage).\n"
                  << std::defaultfloat;
        if (std::fabs(bf::plate_z_mm() - 11.3) < 1e-9 && bf::fixed_plate()) {
            std::cout << "[Referenz] Das ist die gemessene Ruhelage. Das Zyklusmittel "
                         "der Vertikalkraft MUSS dort m*g = 1.0497 N sein.\n";
        }
    }

    ReferenceBoundary rb = build_reference_boundary(mesh);
    const int nb = static_cast<int>(rb.num_bnodes());

    NodalCsr csr;
    build_nodal_csr(mesh, csr);

    // Randtopologie auf das Gerat
    DeviceArray<int> d_bnode_ids(static_cast<size_t>(nb));
    DeviceArray<int> d_node_to_bnode(static_cast<size_t>(nn));
    DeviceArray<double> d_face_orient(mesh.num_boundary_faces);
    d_bnode_ids.prefetch_to_host();
    d_node_to_bnode.prefetch_to_host();
    d_face_orient.prefetch_to_host();
    for (int i = 0; i < nb; ++i) d_bnode_ids[i] = rb.bnode_ids[i];
    for (int i = 0; i < nn; ++i) d_node_to_bnode[i] = rb.node_to_bnode[i];
    for (size_t f = 0; f < mesh.num_boundary_faces; ++f) d_face_orient[f] = rb.face_orientation[f];
    d_bnode_ids.prefetch_to_device();
    d_node_to_bnode.prefetch_to_device();
    d_face_orient.prefetch_to_device();

    // Vollbesetzte Matrizen
    const size_t nb2 = static_cast<size_t>(nb) * static_cast<size_t>(nb);
    DeviceArray<double> d_V(nb2), d_K(nb2), d_T(nb2);
    DeviceArray<double> d_dtn_diag(static_cast<size_t>(nb));

    DeviceArray<double> d_A(static_cast<size_t>(ndof));
    DeviceArray<double> d_A_prev(static_cast<size_t>(ndof));
    DeviceArray<double> d_rhs(static_cast<size_t>(ndof));
    d_A.fill_zero(); d_A_prev.fill_zero();

    std::vector<double> ilu_values;
    std::vector<int>    ilu_diag;

    // cuBLAS-Handle fuer den Krylov-Teil. Im CPU-Modus bleibt er nullptr und
    // wird von den bf::-Wrappern ignoriert; im GPU-Modus MUSS er echt sein --
    // genau das war der Fehler, der beim ersten Lauf zu
    // "cuBLAS Error at HostLinalg.cu:53" gefuehrt hat: bf::dnrm2 entscheidet
    // nach dem BACKEND, nicht nach dem Handle, und gab nullptr an cuBLAS weiter.
    cublasHandle_t cublas = nullptr;
    if (bf::on_gpu()) {
        if (cublasCreate(&cublas) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Referenzloeser: cublasCreate fehlgeschlagen.");
        }
    }

    ReferenceOperator op;
    op.num_rows    = csr.num_rows;
    op.nb          = nb;
    op.row_offsets = csr.row_offsets.data();
    op.col_indices = csr.col_indices.data();
    op.values      = csr.values.data();
    op.bnode_ids   = d_bnode_ids.data();
    op.T           = d_T.data();
    op.cublas      = cublas;
    op.allocate_buffers();

    // -------------------------------------------------------------------------
    // Zyklusmittelwert der Vertikalkraft
    // -------------------------------------------------------------------------
    // DIE Groesse fuer den Vergleich mit der Messung. Ein Momentanwert taugt
    // dafuer nicht: die Kraft ist quadratisch im Strom und pulsiert mit 2*f0,
    // sie schwingt also innerhalb jeder Periode zwischen negativen und
    // positiven Werten. Nur der Mittelwert traegt die Platte, und er muss im
    // Schwebezustand exakt m*g sein.
    //
    // Wichtig ist ausserdem, spaete Perioden anzusehen. Die Wirbelstrom-
    // Zeitkonstante der Platte ist tau = mu_0*sigma*d*a/2 ~ 4.2 ms; in den
    // ersten Millisekunden ist der Loeser im Einschwingen, dort gilt die
    // Beziehung "Momentanwert bei T/4 = 2 * Mittelwert" nicht.
    // Die BEM-Matrizen haengen NUR von der Randgeometrie ab. Bewegt sich die
    // Platte nicht, sind sie unveraendert und werden einmal gebaut. Das ist bei
    // --fixed-plate immer der Fall und macht den Lauf um ein Vielfaches
    // schneller -- die Zeitmessung zeigt BEM 85 s + LU 113 s von 300 s
    // Gesamtzeit fuer 400 Schritte, das ist also der Hauptposten.
    double q_mz_at_zero = 0.0;   // |m_z| beim Stromnulldurchgang
    double q_mz_at_max  = 0.0;   // |m_z| beim Strommaximum

    bool   bem_valid    = false;
    double bem_built_at = 0.0;   // translation.z beim letzten Aufbau

    const int steps_per_period = static_cast<int>(std::lround(1.0 / (50.0 * dt)));
    std::vector<double> ring_max(static_cast<size_t>(steps_per_period), 0.0);
    std::vector<double> ring_lor(static_cast<size_t>(steps_per_period), 0.0);
    int ring_fill = 0;

    // -------------------------------------------------------------------------
    // --verify-field: magnetostatische Probe
    // -------------------------------------------------------------------------
    // sigma = 0 in ALLEN Elementen -> keine Wirbelstroeme, der Massenterm
    // sigma/dt verschwindet, und uebrig bleibt das reine Quellfeld der Spulen:
    //     L A - (1/mu_0) T D A = Int J_s . A'
    // Nichtsingulaer bleibt das System durch den Aussenraum-DtN-Term (L allein
    // ist der Neumann-Laplace mit konstantem Nullraum je Koerper).
    //
    // Gerechnet wird bei t = 5 ms, also einer Viertelperiode: sin(2*pi*50*0.005)
    // = 1, der Spulenstrom steht damit auf seinem Maximum von 20 A. Das ist
    // genau der Wert, gegen den die Biot-Savart-Referenz gestellt wird.
    if (bf::fixed_plate()) {
        std::cout << "[Referenz] --fixed-plate: Platte wird nicht bewegt. "
                     "BEM-Matrizen bleiben damit unveraendert und werden nur "
                     "einmal gebaut.\n";
    }

    if (bf::verify_field()) {
        // Gleichstromprobe, nicht sigma = 0 -- Begruendung in mainloop().
        // Der Spulenstrom wird konstant gehalten (t in jedem Schritt 5 ms), und
        // nach dem Abklingen der L/R-Transiente ist J = 0 und das Feld rein
        // statisch. Die Wirbelstromdiagnose je Schritt (m_z, Verlustleistung)
        // zeigt das Abklingen und ist damit die Gueltigkeitspruefung des Tests.
        if (max_steps < 300) max_steps = 300;
        std::cout << "[Verify] Gleichstromprobe: 20 A konstant, " << max_steps
                  << " Schritte (tau = 1.92 ms, also 15.7 Zeitkonstanten).\n";
    }

    std::cout << "[Referenz] Starte " << max_steps << " Zeitschritte, dt = "
              << dt << " s\n";
    std::cout << "[Referenz] Eine Stromperiode = " << steps_per_period
              << " Zeitschritte; Zyklusmittel ab Schritt " << steps_per_period << "\n\n";

    // Zeitmessung je Phase. Vor jeder Optimierung will man wissen, wohin die
    // Zeit geht -- bei 17000 Schritten bis zur Ruhelage entscheidet das, ob
    // sich Aufwand lohnt und wo.
    using Clock = std::chrono::steady_clock;
    double t_fem = 0.0, t_bem = 0.0, t_lu = 0.0, t_solve = 0.0, t_force = 0.0;

    for (int step = 1; step <= max_steps; ++step) {
        // Im Pruefmodus die Viertelperiode ansteuern, damit der Strom auf 20 A
        // steht; sonst der normale Zeitschritt.
        const double t_now = bf::verify_field() ? 0.005 : step * dt;
        const auto step_t0 = Clock::now();
        auto phase_t0 = Clock::now();
        auto phase = [&](double& acc) {
            const auto now = Clock::now();
            acc += std::chrono::duration<double>(now - phase_t0).count();
            phase_t0 = now;
        };
        std::cout << "--- Referenz-Zeitschritt " << step
                  << " (t = " << t_now * 1000.0 << " ms) ---\n";

        // 1. Zeitabhaengige Quellen fortschreiben.
        //    WICHTIG: d_materials haelt KOPIEN der Eigenschaften. update_state()
        //    aendert nur das Objekt; ohne das Zurueckschreiben in die
        //    Hexaeder-Eintraege bliebe der Spulenstrom auf dem Anfangswert
        //    stehen und die rechte Seite waere zeitlich konstant. Genauso macht
        //    es der produktive Pfad.
        d_materials.prefetch_to_host();
        for (PhysicalObject* src : sources) {
            src->update_state(t_now, dt);
            for (int hex_id : src->get_hex_indices()) {
                d_materials[hex_id] = src->get_properties();
            }
        }
        d_materials.prefetch_to_device();

        // 2. FEM assemblieren
        csr.values.fill_zero();
        d_rhs.fill_zero();
        BF_LAUNCH_1D(reference_assemble, mesh.num_hexes,
            static_cast<int>(mesh.num_hexes), mesh.hex_nodes.data(),
            mesh.nodes_current.data(), d_materials.data(), d_A_prev.data(),
            dt, bf::no_conductor_penalty() ? -REFERENCE_PENALTY_BETA
                                           :  REFERENCE_PENALTY_BETA, t_now,
            csr.row_offsets.data(), csr.col_indices.data(),
            csr.values.data(), d_rhs.data());
        bf::sync();
        phase(t_fem);

        // 3. BEM assemblieren (V, K und Randmassenmatrix), nur wenn noetig
        const double tr_now = plate.get_properties().translation.z;
        const bool need_bem = !bem_valid || (tr_now != bem_built_at);
        if (!need_bem) {
            if (step <= 3 || step % 50 == 0) {
                std::cout << "  [Ref-BEM] Geometrie unveraendert, V/K/T/D "
                             "wiederverwendet.\n";
            }
            phase(t_bem);
            phase(t_lu);
        }
        // Der folgende Block bis "Ende if (need_bem)" ist absichtlich nicht
        // weiter eingerueckt, damit der Diff gegen die vorige Fassung lesbar
        // bleibt -- inhaltlich ist er unveraendert und nur bedingt geworden.
        if (need_bem) {
        BF_LAUNCH_1D_PAR(reference_bem_row, nb,
            nb, static_cast<int>(mesh.num_boundary_faces),
            d_bnode_ids.data(), d_node_to_bnode.data(),
            mesh.boundary_faces.data(), d_face_orient.data(),
            mesh.nodes_current.data(), d_V.data(), d_K.data());
        bf::sync();

        d_T.fill_zero();
        BF_LAUNCH_1D(reference_bem_mass, mesh.num_boundary_faces,
            static_cast<int>(mesh.num_boundary_faces), nb,
            mesh.boundary_faces.data(), d_node_to_bnode.data(),
            mesh.nodes_current.data(), d_T.data());
        bf::sync();
        phase(t_bem);

        // 4. C = K - 1/2 I, dann D = V^-1 C, dann G = T * D
        //    d_K wird zu C, danach zu D, danach wird G gebildet.
        d_K.prefetch_to_host();
        bf::sync();   // cudaMemPrefetchAsync ist asynchron
        for (int i = 0; i < nb; ++i) d_K[static_cast<size_t>(i) * nb + i] -= 0.5;
        d_K.prefetch_to_device();

        {
            // DenseLU erwartet column-major. V ist symmetrisch aufgebaut? NEIN --
            // Kollokation liefert ein unsymmetrisches V. Zeilen-major V ist als
            // column-major gelesen V^T, also loesen wir V^T X = C^T, was
            // (C V^-1)^T ... das ist NICHT das Gesuchte. Deshalb wird V hier
            // explizit transponiert, damit die column-major-Sicht wirklich V ist.
            d_V.prefetch_to_host();
            bf::sync();
            for (int i = 0; i < nb; ++i) {
                for (int j = i + 1; j < nb; ++j) {
                    std::swap(d_V[static_cast<size_t>(i) * nb + j],
                              d_V[static_cast<size_t>(j) * nb + i]);
                }
            }
            d_V.prefetch_to_device();

            // C ebenso: zeilen-major -> column-major
            d_K.prefetch_to_host();
            bf::sync();
            for (int i = 0; i < nb; ++i) {
                for (int j = i + 1; j < nb; ++j) {
                    std::swap(d_K[static_cast<size_t>(i) * nb + j],
                              d_K[static_cast<size_t>(j) * nb + i]);
                }
            }
            d_K.prefetch_to_device();

            bf::DenseLU lu;
            lu.factorize(d_V.data(), nb);
            lu.solve(d_K.data(), nb);      // d_K := D = V^-1 C  (column-major)
        }
        bem_valid    = true;
        bem_built_at = tr_now;
        phase(t_lu);
        }   // Ende if (need_bem)

        // 5. ILU(0) fuer den Vorkonditionierer
        //
        // ACHTUNG, das war der zweite Fehler des ersten Laufs: der Sparse-Anteil
        // ALLEIN ist SINGULAER. Er ist der reine Neumann-Laplace, sein Nullraum
        // sind die konstanten Felder -- und zwar je Koerper und je Komponente.
        // Das Netz enthaelt DREI getrennte Koerper (Platte und zwei Spulen), in
        // den Spulen ist sigma = 0, dort fehlt also auch der Massenterm
        // sigma/dt, der den Nullraum brechen wuerde. Nullraumdimension damit
        // 2 Koerper x 3 Komponenten = 6.
        //
        // Nichtsingulaer wird das System erst durch den dichten
        // Dirichlet-zu-Neumann-Block. ILU(0) auf dem Sparse-Anteil allein
        // faktorisiert also eine singulaere Matrix: die Zerlegung laeuft
        // moeglicherweise ohne exakten Nullpivot durch, liefert aber winzige
        // Pivots und damit einen Vorkonditionierer mit riesiger Norm -- GMRES
        // auf M^-1 A ist dann wertlos.
        //
        // Abhilfe: die DIAGONALE des dichten Blocks in die ILU-Matrix
        // uebernehmen, mit genau demselben Vorfaktor, den apply() benutzt. Fuer
        // einen konstanten Vektor v auf einem Koerper gilt dann
        // v^T (L + D_korr) v = sum_{i in Rand} d_i > 0 statt 0 -- der Nullraum
        // ist weg, und der Vorkonditionierer liegt naeher am wahren Operator.
        {
            BF_LAUNCH_1D_PAR(ref_dtn_diag, nb, nb, d_T.data(), d_K.data(),
                             d_dtn_diag.data());
            bf::sync();
        }

        csr.values.prefetch_to_host();
        csr.row_offsets.prefetch_to_host();
        csr.col_indices.prefetch_to_host();
        d_dtn_diag.prefetch_to_host();
        bf::sync();   // prefetch ist asynchron -- vor dem Host-Zugriff abwarten

        ilu_values.assign(csr.values.data(), csr.values.data() + csr.nnz);
        for (int i = 0; i < nb; ++i) {
            const double d = -d_dtn_diag[i] / MU_0;   // wie in apply()
            for (int c = 0; c < 3; ++c) {
                const int row = rb.bnode_ids[i] * 3 + c;
                for (int k = csr.row_offsets[row]; k < csr.row_offsets[row + 1]; ++k) {
                    if (csr.col_indices[k] == row) { ilu_values[k] += d; break; }
                }
            }
        }

        const int zp = bf::csr_ilu0(csr.num_rows, csr.row_offsets.data(),
                                    csr.col_indices.data(), ilu_values.data(), ilu_diag);
        const bool use_ilu = (zp < 0);
        if (!use_ilu) {
            std::cout << "  [Referenz] ILU(0) Nullpivot in Zeile " << zp
                      << " -- GMRES laeuft ohne Vorkonditionierer weiter.\n";
        }

        // 6. Loesen. d_K enthaelt nach dem Solve D = V^-1 C.
        op.D = d_K.data();

        // Selbsttest im ersten Schritt: der gekoppelte Operator MUSS positiv
        // definit sein (L ist positiv semidefinit, der Aussenraum-DtN-Term
        // liefert +(1/mu_0) * Int_Aussen |grad u|^2 >= 0 und beseitigt den
        // Nullraum). Zwei Applikationen kosten nichts und haetten den
        // divergierenden ersten Lauf sofort erklaert.
        if (step == 1) {
            DeviceArray<double> tv(static_cast<size_t>(csr.num_rows));
            DeviceArray<double> tw(static_cast<size_t>(csr.num_rows));
            tv.prefetch_to_host(); bf::sync();
            unsigned seed = 12345u;
            for (int i = 0; i < csr.num_rows; ++i) {
                seed = seed * 1103515245u + 12345u;
                tv[i] = ((seed >> 16) & 0x7fff) / 16383.5 - 1.0;
            }
            tv.prefetch_to_device();
            op.apply(tv.data(), tw.data());
            double quad = 0.0, nv = 0.0;
            bf::ddot(cublas, csr.num_rows, tv.data(), tw.data(), &quad);
            bf::dnrm2(cublas, csr.num_rows, tv.data(), &nv);
            const double rayleigh = quad / (nv * nv);
            std::cout << "  [Ref-Selbsttest] Rayleigh-Quotient x^T A x / ||x||^2 = "
                      << std::scientific << std::setprecision(4) << rayleigh
                      << std::defaultfloat << (rayleigh > 0.0 ? "  (positiv, gut)\n"
                                                             : "  NEGATIV -- Operator ist nicht definit!\n");
            if (!(rayleigh > 0.0)) {
                throw std::runtime_error(
                    "Referenzloeser: der gekoppelte Operator ist nicht positiv definit. "
                    "Erwartet wird L - (1/mu_0) T D > 0. Moegliche Ursachen: Vorzeichen des "
                    "Kopplungsterms, Panelorientierung, oder V^-1 C falsch gebildet. "
                    "GMRES weiterlaufen zu lassen waere sinnlos.");
            }
        }
        // KEIN prefetch_to_host hier: der Krylov-Teil laeuft im GPU-Modus auf
        // dem Geraet, nur die ILU-Substitution greift vom Host zu.
        reference_gmres(op, d_rhs.data(), d_A.data(), ilu_values, ilu_diag,
                        csr.row_offsets.data(), csr.col_indices.data(),
                        use_ilu, step);
        bf::sync();
        phase(t_solve);

        // -----------------------------------------------------------------
        // Pruefmodus: im LETZTEN Schritt B_z auf der Achse gegen Biot-Savart
        // -----------------------------------------------------------------
        if (bf::verify_field() && step == max_steps) {
            mesh.nodes_current.prefetch_to_host();
            mesh.hex_nodes.prefetch_to_host();
            d_A.prefetch_to_host();
            bf::sync();

            // Die achsennaechsten Plattenelemente heraussuchen. Die Platte ist
            // eine VOLLSCHEIBE ohne Bohrung (TEAM 28 hat kein Loch). Im
            // Standardnetz ueberdeckt je Lage eine Mittelzelle die Achse; es
            // liegt nur kein Knoten genau auf ihr (der naechste bei r = 1.84 mm).
            // Die Schwerpunkte dieser Mittelzellen liegen bei r ~ 1.6 mm; dort
            // ist B_z bis auf O((r/L)^2) ~ 0.2 % der Achsenwert, was gegenueber
            // den gesuchten Abweichungen von zehn Prozent nicht ins Gewicht
            // faellt. Auswertepunkte: die achsennaechsten Plattenelemente.
            // Begruendung zum O((r/L)^2)-Fehler in include/FieldVerification.cuh.
            std::vector<FieldSample> samples;
            double r_min = 1e30;
            for (int h : plate.get_hex_indices()) {
                const Hexahedron H = mesh.hex_nodes[h];
                double cx = 0.0, cy = 0.0;
                for (int n = 0; n < 8; ++n) {
                    cx += mesh.nodes_current[H.v[n]].x;
                    cy += mesh.nodes_current[H.v[n]].y;
                }
                cx *= 0.125; cy *= 0.125;
                r_min = std::fmin(r_min, std::sqrt(cx*cx + cy*cy));
            }
            for (int h : plate.get_hex_indices()) {
                const Hexahedron H = mesh.hex_nodes[h];
                double3 nd[8], Ah[8];
                double cx = 0.0, cy = 0.0, cz = 0.0;
                for (int n = 0; n < 8; ++n) {
                    nd[n] = mesh.nodes_current[H.v[n]];
                    Ah[n] = make_double3(d_A[H.v[n]*3+0], d_A[H.v[n]*3+1], d_A[H.v[n]*3+2]);
                    cx += nd[n].x; cy += nd[n].y; cz += nd[n].z;
                }
                cx *= 0.125; cy *= 0.125; cz *= 0.125;
                const double rr = std::sqrt(cx*cx + cy*cy);
                if (rr > r_min * 1.05) continue;

                double3 gradN[8]; double detJ;
                evaluate_nodal_gradients(nd, 0.0, 0.0, 0.0, gradN, detJ);
                double3 B = make_double3(0.0, 0.0, 0.0);
                for (int n = 0; n < 8; ++n) {
                    const double3 c = cross3(gradN[n], Ah[n]);
                    B.x += c.x; B.y += c.y; B.z += c.z;
                }
                samples.push_back({rr, cz, B.z});
            }

            // Restliche Verlustleistung als Gueltigkeitspruefung mitgeben.
            double res_pw = -1.0;
            {
                double *mz, *pw, *aj;
                bf::malloc_d(&mz, sizeof(double)); bf::malloc_d(&pw, sizeof(double));
                bf::malloc_d(&aj, sizeof(double));
                bf::memset_d(mz,0,sizeof(double)); bf::memset_d(pw,0,sizeof(double));
                bf::memset_d(aj,0,sizeof(double));
                BF_LAUNCH_1D(nodal_eddy_diag, mesh.num_hexes,
                    static_cast<int>(mesh.num_hexes), mesh.hex_nodes.data(),
                    mesh.nodes_current.data(), d_materials.data(),
                    d_A.data(), d_A_prev.data(), dt, mz, pw, aj);
                bf::sync();
                bf::memcpy_d(&res_pw, pw, sizeof(double), cudaMemcpyDeviceToHost);
                bf::free_d(mz); bf::free_d(pw); bf::free_d(aj);
            }

            report_field_verification("Q1-Knoten-FEM + vollbesetzte BEM",
                                      samples, 20.0, r_min, res_pw);

            if (cublas != nullptr) cublasDestroy(cublas);
            return 0;
        }

        // 7. Kraft aus dem Maxwell-Spannungstensor (nodale Fassung)
        {
            double *fx, *fy, *fz, *mx, *my, *mz;
            bf::malloc_d(&fx, sizeof(double)); bf::malloc_d(&fy, sizeof(double));
            bf::malloc_d(&fz, sizeof(double)); bf::malloc_d(&mx, sizeof(double));
            bf::malloc_d(&my, sizeof(double)); bf::malloc_d(&mz, sizeof(double));
            bf::memset_d(fx, 0, sizeof(double)); bf::memset_d(fy, 0, sizeof(double));
            bf::memset_d(fz, 0, sizeof(double)); bf::memset_d(mx, 0, sizeof(double));
            bf::memset_d(my, 0, sizeof(double)); bf::memset_d(mz, 0, sizeof(double));

            const double3 cm = plate.get_properties().center_of_mass;
            BF_LAUNCH_1D(reference_maxwell, mesh.num_component_surface_faces,
                static_cast<int>(mesh.num_component_surface_faces),
                mesh.component_surface_faces.data(), mesh.hex_nodes.data(),
                mesh.nodes_current.data(), d_A.data(), cm,
                fx, fy, fz, mx, my, mz);
            bf::sync();

            MechanicalLoads loads;
            bf::memcpy_d(&loads.force.x,  fx, sizeof(double), cudaMemcpyDeviceToHost);
            bf::memcpy_d(&loads.force.y,  fy, sizeof(double), cudaMemcpyDeviceToHost);
            bf::memcpy_d(&loads.force.z,  fz, sizeof(double), cudaMemcpyDeviceToHost);
            bf::memcpy_d(&loads.torque.x, mx, sizeof(double), cudaMemcpyDeviceToHost);
            bf::memcpy_d(&loads.torque.y, my, sizeof(double), cudaMemcpyDeviceToHost);
            bf::memcpy_d(&loads.torque.z, mz, sizeof(double), cudaMemcpyDeviceToHost);
            bf::free_d(fx); bf::free_d(fy); bf::free_d(fz);
            bf::free_d(mx); bf::free_d(my); bf::free_d(mz);

            // ---- Zweite, unabhaengige Rechnung: Lorentzkraft als Volumenintegral
            // Auf grobem Netz weichen die beiden Verfahren typischerweise ab.
            // Der Spannungstensor braucht das Feld genau auf der
            // Materialgrenzflaeche und bildet dort die Differenz zweier grosser,
            // fast gleicher Terme; das Volumenintegral mittelt ueber den ganzen
            // Leiter. Fuer unmagnetisches Material ist Letzteres das empfohlene
            // Verfahren -- es wird deshalb auch fuer die Bewegung benutzt.
            // Zwei Quadraturordnungen: stimmen sie ueberein, ist die Quadratur
            // nicht die Fehlerquelle. Kostet fast nichts (27 statt 8 Punkte je
            // Element) und spart im Zweifel eine lange Suche an der falschen
            // Stelle.
            MechanicalLoads lorentz, lorentz_hi;
            for (int pass = 0; pass < 2; ++pass) {
                const int order = (pass == 0) ? 2 : 3;
                MechanicalLoads& dst = (pass == 0) ? lorentz : lorentz_hi;
                double *gx,*gy,*gz,*hx_,*hy_,*hz_;
                bf::malloc_d(&gx, sizeof(double)); bf::malloc_d(&gy, sizeof(double));
                bf::malloc_d(&gz, sizeof(double)); bf::malloc_d(&hx_, sizeof(double));
                bf::malloc_d(&hy_, sizeof(double)); bf::malloc_d(&hz_, sizeof(double));
                bf::memset_d(gx,0,sizeof(double)); bf::memset_d(gy,0,sizeof(double));
                bf::memset_d(gz,0,sizeof(double)); bf::memset_d(hx_,0,sizeof(double));
                bf::memset_d(hy_,0,sizeof(double)); bf::memset_d(hz_,0,sizeof(double));

                BF_LAUNCH_1D(nodal_lorentz, mesh.num_hexes,
                    static_cast<int>(mesh.num_hexes), mesh.hex_nodes.data(),
                    mesh.nodes_current.data(), d_materials.data(),
                    d_A.data(), d_A_prev.data(), dt, order, cm,
                    gx, gy, gz, hx_, hy_, hz_);
                bf::sync();

                bf::memcpy_d(&dst.force.x,  gx, sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(&dst.force.y,  gy, sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(&dst.force.z,  gz, sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(&dst.torque.x, hx_, sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(&dst.torque.y, hy_, sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(&dst.torque.z, hz_, sizeof(double), cudaMemcpyDeviceToHost);
                bf::free_d(gx); bf::free_d(gy); bf::free_d(gz);
                bf::free_d(hx_); bf::free_d(hy_); bf::free_d(hz_);
            }

            const double weight = 0.107 * 9.81;
            std::cout << std::scientific << std::setprecision(5)
                      << "  [Ref-Kraft] Maxwell-Spannungstensor F = ("
                      << loads.force.x << ", " << loads.force.y << ", " << loads.force.z << ") N\n"
                      << "  [Ref-Kraft] Lorentz-Volumenintegral  F = ("
                      << lorentz.force.x << ", " << lorentz.force.y << ", " << lorentz.force.z << ") N\n"
                      << std::defaultfloat << std::fixed << std::setprecision(4)
                      << "  [Ref-Kraft] F_z/(m*g):  Maxwell " << loads.force.z / weight
                      << "   Lorentz " << lorentz.force.z / weight << "\n"
                      << std::defaultfloat;
            if (std::fabs(lorentz.force.z) > 1e-30) {
                const double ratio = loads.force.z / lorentz.force.z;
                std::cout << "  [Ref-Kraft] Verhaeltnis Maxwell/Lorentz = "
                          << std::fixed << std::setprecision(3) << ratio << std::defaultfloat;
                if (std::fabs(ratio - 1.0) > 0.25) {
                    // Frueher stand hier "das Netz loest die Materialgrenzflaeche
                    // schlecht auf". Das war die falsche Erklaerung. Die Pruefung
                    // an der gemessenen Ruhelage z = 11.3 mm, wo das Zyklusmittel
                    // exakt m*g sein MUSS, hat es entschieden:
                    //     Lorentz 0.930 * m*g   (7 % Abweichung)
                    //     Maxwell 0.172 * m*g   (83 % Abweichung)
                    // und bei 18.2 mm ist der Maxwell-Mittelwert sogar negativ.
                    // Eine Netzverfeinerung aenderte daran fast nichts (das
                    // Verhaeltnis ging nur von 0.894 auf 0.946). Der
                    // Oberflaechen-Spannungstensor ist als Schaetzer des
                    // GLEICHANTEILS untauglich, nicht schlecht aufgeloest: er
                    // braucht das Feld genau auf der Unstetigkeitsflaeche und
                    // bildet dort die Differenz zweier grosser, fast gleicher
                    // Terme, waehrend der gesuchte Gleichanteil nur etwa ein
                    // Drittel der Schwankungsamplitude ist.
                    std::cout << "   -- erwartet: der Spannungstensor taugt fuer den "
                                 "Gleichanteil nicht, siehe Kommentar im Quelltext";
                }
                std::cout << "\n";
            }
            // Quadraturprobe
            {
                const double a = lorentz.force.z, b = lorentz_hi.force.z;
                const double rel = (std::fabs(b) > 1e-30) ? std::fabs(a - b) / std::fabs(b) : 0.0;
                std::cout << std::scientific << std::setprecision(5)
                          << "  [Ref-Quadratur] F_z Lorentz mit 2x2x2 = " << a
                          << " N,  mit 3x3x3 = " << b << " N"
                          << std::defaultfloat << std::fixed << std::setprecision(2)
                          << "   Abweichung " << rel * 100.0 << " %"
                          << (rel > 0.02 ? "  -- Quadratur ist mitschuldig!\n"
                                         : "  -- Quadratur ist NICHT die Fehlerquelle\n")
                          << std::defaultfloat;
            }

            std::cout << "  [Ref-Kraft] Moment (Lorentz) = (" << std::scientific
                      << std::setprecision(3) << lorentz.torque.x << ", "
                      << lorentz.torque.y << ", " << lorentz.torque.z << ") Nm\n"
                      << std::defaultfloat;

            // ---- Wirbelstromdiagnose
            {
                double *mz, *pw, *aj;
                bf::malloc_d(&mz, sizeof(double)); bf::malloc_d(&pw, sizeof(double));
                bf::malloc_d(&aj, sizeof(double));
                bf::memset_d(mz,0,sizeof(double)); bf::memset_d(pw,0,sizeof(double));
                bf::memset_d(aj,0,sizeof(double));

                BF_LAUNCH_1D(nodal_eddy_diag, mesh.num_hexes,
                    static_cast<int>(mesh.num_hexes), mesh.hex_nodes.data(),
                    mesh.nodes_current.data(), d_materials.data(),
                    d_A.data(), d_A_prev.data(), dt, mz, pw, aj);
                bf::sync();

                double h_mz = 0.0, h_pw = 0.0, h_aj = 0.0;
                bf::memcpy_d(&h_mz, mz, sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(&h_pw, pw, sizeof(double), cudaMemcpyDeviceToHost);
                bf::memcpy_d(&h_aj, aj, sizeof(double), cudaMemcpyDeviceToHost);
                bf::free_d(mz); bf::free_d(pw); bf::free_d(aj);

                const double phase_rad = 2.0 * M_PI * 50.0 * t_now;
                const double s_i = std::sin(phase_rad);   // Spulenstrom / 20 A
                std::cout << std::scientific << std::setprecision(4)
                          << "  [Ref-Wirbel] m_z = " << h_mz << " A m^2"
                          << "   Verlustleistung = " << h_pw << " W\n"
                          << std::defaultfloat << std::fixed << std::setprecision(4)
                          << "  [Ref-Wirbel] Spulenstrom = " << s_i * 20.0 << " A"
                          << "   (Phase " << phase_rad / M_PI << " pi)\n"
                          << std::defaultfloat;

                // ---- Q = w*tau direkt aus zwei Phasenpunkten
                // Mit i1 = sin(wt) ist der induzierte Strom im stationaeren
                // Zustand i2 ~ -(cos wt + Q sin wt). Also
                //     |i2| beim Stromnulldurchgang (wt = k*pi)      ~ 1
                //     |i2| beim Strommaximum     (wt = pi/2 + k*pi) ~ Q
                // Damit ist Q = |m_z(Maximum)| / |m_z(Nulldurchgang)| -- zwei
                // Messpunkte je Halbperiode, ohne Modellannahme ueber die
                // Bewegung. Genau deshalb braucht es --fixed-plate: bei freier
                // Bewegung aendert sich die Geometrie zwischen den beiden
                // Punkten und die Ablesung waere wertlos.
                const int half = steps_per_period / 2;      // 100 Schritte
                if (half > 0) {
                    if (step % half == 0)               q_mz_at_zero = std::fabs(h_mz);
                    if (step % half == half / 2)        q_mz_at_max  = std::fabs(h_mz);

                    if (q_mz_at_zero > 0.0 && q_mz_at_max > 0.0) {
                        const double q_meas = q_mz_at_max / q_mz_at_zero;
                        std::cout << std::fixed << std::setprecision(4)
                                  << "  [Ref-Q] |m_z| Maximum " << q_mz_at_max
                                  << " / Nulldurchgang " << q_mz_at_zero
                                  << "  ->  Q_eff = " << q_meas << "\n"
                                  << "  [Ref-Q] Hinweis: Q_eff ist ein EFFEKTIVER Wert. "
                                     "Die Ein-Moden-Ableitung setzt eine einzige "
                                     "Zeitkonstante voraus; eine Scheibe hat ein "
                                     "Modenkontinuum, und welche Moden angeregt werden "
                                     "haengt vom Feldprofil und damit von der Hoehe ab. "
                                     "Gemessen wurden 0.60 / 0.80 / 0.37 bei 3.8 / 11.3 / "
                                     "18.2 mm -- das ist zu erwarten, kein Fehler.\n"
                                  << std::defaultfloat;
                        if (!bf::fixed_plate()) {
                            std::cout << "  [Ref-Q] ACHTUNG: nur mit --fixed-plate "
                                         "aussagekraeftig, die Platte bewegt sich.\n";
                        }
                    }
                }
            }

            // ---- Zyklusmittel fortschreiben und ausgeben
            {
                const int slot = (step - 1) % steps_per_period;
                ring_max[slot] = loads.force.z;
                ring_lor[slot] = lorentz.force.z;
                if (ring_fill < steps_per_period) ++ring_fill;

                if (ring_fill == steps_per_period) {
                    double sm = 0.0, sl = 0.0;
                    double lo = ring_lor[0], hi = ring_lor[0];
                    for (int i = 0; i < steps_per_period; ++i) {
                        sm += ring_max[i]; sl += ring_lor[i];
                        lo = std::fmin(lo, ring_lor[i]);
                        hi = std::fmax(hi, ring_lor[i]);
                    }
                    sm /= steps_per_period; sl /= steps_per_period;
                    std::cout << std::fixed << std::setprecision(4)
                              << "  [Ref-Zyklusmittel] ueber die letzten " << steps_per_period
                              << " Schritte:  F_z(Maxwell) = " << sm << " N"
                              << "   F_z(Lorentz) = " << sl << " N\n"
                              << "  [Ref-Zyklusmittel] F_z(Lorentz)/(m*g) = " << sl / weight
                              << (bf::fixed_plate() && std::fabs(bf::plate_z_mm() - 11.3) < 1e-9
                                  ? "   <-- MUSS 1.0000 sein (gemessene Ruhelage)\n"
                                  : "   (1.0 = Schweben; bei 3.8 mm fordert die Messung ~1.62)\n")
                              << "  [Ref-Zyklusmittel] Schwankung Lorentz: min " << lo
                              << " N, max " << hi << " N\n"
                              << std::defaultfloat;
                }
            }

            // 8. Bewegung -- Lorentzkraft mit der hoeheren Quadraturordnung.
            if (!bf::fixed_plate()) plate.apply_newtons_law(lorentz_hi.force, lorentz_hi.torque, dt);
            if (!bf::fixed_plate() && plate.is_moving()) {
                apply_kinematics(mesh, plate);
                bf::sync();
                // Die Randgeometrie hat sich geaendert -> Panelorientierung neu
                // bestimmen. V, K und T werden im naechsten Schritt ohnehin
                // komplett neu aufgebaut.
                rb = build_reference_boundary(mesh);
                d_face_orient.prefetch_to_host();
                for (size_t f = 0; f < mesh.num_boundary_faces; ++f) {
                    d_face_orient[f] = rb.face_orientation[f];
                }
                d_face_orient.prefetch_to_device();

                const double3 tr = plate.get_properties().translation;
                const double height_mm = (plate_z0 + tr.z) * 1000.0;
                std::cout << std::fixed << std::setprecision(4)
                          << "  [Ref-Kinematik] Verschiebung gegen Startlage = "
                          << tr.z * 1000.0 << " mm\n"
                          << "  [Ref-Kinematik] SCHWEBEHOEHE z = " << height_mm
                          << " mm  (Aufgabenstellung: Unterkante ueber Spulenoberkante,"
                          << " Ruhelage 3.8, Endwert 11.3, Maximum 18.2)\n"
                          << std::defaultfloat;
                if (plate.at_lower_stop()) {
                    std::cout << "  [Ref-Kinematik] Platte liegt auf der Wickelform auf.\n";
                }
            }
        }

        phase(t_force);

        if (step % 50 == 0 || step <= 3) {
            const double dtot = std::chrono::duration<double>(Clock::now() - step_t0).count();
            std::cout << std::fixed << std::setprecision(3)
                      << "  [Ref-Zeit] Schritt " << dtot << " s   kumuliert: FEM "
                      << t_fem << "  BEM " << t_bem << "  LU " << t_lu
                      << "  GMRES " << t_solve << "  Kraft " << t_force << " s"
                      << "   -> " << max_steps * dtot / 3600.0
                      << " h fuer " << max_steps << " Schritte\n"
                      << std::defaultfloat;
        }

        bf::memcpy_d(d_A_prev.data(), d_A.data(),
                     static_cast<size_t>(ndof) * sizeof(double),
                     cudaMemcpyDeviceToDevice);
    }

    if (cublas != nullptr) cublasDestroy(cublas);

    std::cout << "\n[Referenz] Lauf beendet.\n";
    return 0;
}
