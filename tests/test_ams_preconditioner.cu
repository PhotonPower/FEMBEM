// =============================================================================
// test_ams_preconditioner.cu -- wirkt der AMS ueberhaupt?
// =============================================================================
//
// Gemessen wird, worauf es ankommt: die Zahl der CG-Iterationen auf einem
// echten curl-curl + Masse System, einmal mit Jacobi, einmal mit ILU(0) und
// einmal mit AMS. Ein Vorkonditionierer, der die Iterationszahl nicht senkt,
// ist keiner -- alle Strukturpruefungen der Bausteine (test_discrete_gradient)
// sagen darueber nichts.
//
// Entscheidend ist die Abhaengigkeit von sigma. Der Nahnullraum aus
// Gradientenfeldern wird nur durch den Massenterm sigma/dt gebrochen; je
// kleiner sigma, desto schlechter arbeiten Punktglaetter und desto mehr muss
// der Hilfsraum leisten. Deshalb wird ueber mehrere Groessenordnungen von
// sigma gemessen.
//
// Braucht keine GPU.
// =============================================================================

#include "AmsPreconditioner.cuh"
#include "HostLinalg.cuh"
#include "NedelecDispatch.cuh"   // piola_covariant/_contravariant; bindet NedelecHexahedron.cuh ein

#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
void ck(bool ok, const char* w) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", w);
    if (!ok) ++g_fail;
}

const int LOC[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},
                         {0,4},{1,5},{2,6},{3,7} };

struct SmallMesh {
    Mesh mesh;
    std::vector<int> hex_edge, hex_sign;
};

void build(int n, SmallMesh& sm) {
    std::map<std::array<int,3>, int> lut;
    std::vector<double3> pts;
    auto nid = [&](int i, int j, int k) {
        std::array<int,3> a{i,j,k};
        auto it = lut.find(a);
        if (it != lut.end()) return it->second;
        const int id = static_cast<int>(pts.size());
        pts.push_back(make_double3(i * 0.01 + 0.0009 * std::sin(1.3 * j + k),
                                   j * 0.01 + 0.0006 * k,
                                   k * 0.01 + 0.0004 * i * j));
        lut[a] = id; return id;
    };
    std::vector<Hexahedron> hx;
    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        Hexahedron h{};
        h.v[0]=nid(i,j,k);   h.v[1]=nid(i+1,j,k);   h.v[2]=nid(i+1,j+1,k);   h.v[3]=nid(i,j+1,k);
        h.v[4]=nid(i,j,k+1); h.v[5]=nid(i+1,j,k+1); h.v[6]=nid(i+1,j+1,k+1); h.v[7]=nid(i,j+1,k+1);
        hx.push_back(h);
    }
    std::map<std::pair<int,int>, int> emap;
    for (const auto& h : hx) for (int le = 0; le < 12; ++le) {
        const int a = h.v[LOC[le][0]], b = h.v[LOC[le][1]];
        emap.emplace(std::make_pair(std::min(a,b), std::max(a,b)), 0);
    }
    int idx = 0; for (auto& kv : emap) kv.second = idx++;

    sm.mesh.num_nodes = pts.size(); sm.mesh.num_hexes = hx.size(); sm.mesh.num_edges = emap.size();
    sm.mesh.nodes_current.allocate(pts.size());
    sm.mesh.hex_nodes.allocate(hx.size());
    sm.mesh.edges.allocate(emap.size());
    for (size_t q = 0; q < pts.size(); ++q) sm.mesh.nodes_current[q] = pts[q];
    for (size_t q = 0; q < hx.size(); ++q)  sm.mesh.hex_nodes[q] = hx[q];
    for (const auto& kv : emap) sm.mesh.edges[kv.second] = make_int2(kv.first.first, kv.first.second);

    sm.hex_edge.resize(hx.size()*12); sm.hex_sign.resize(hx.size()*12);
    for (size_t h = 0; h < hx.size(); ++h) for (int le = 0; le < 12; ++le) {
        const int a = hx[h].v[LOC[le][0]], b = hx[h].v[LOC[le][1]];
        sm.hex_edge[h*12+le] = emap.at({std::min(a,b), std::max(a,b)});
        sm.hex_sign[h*12+le] = (a < b) ? 1 : -1;
    }
}

/// curl-curl / mu_0 + sigma/dt * Masse
HostCsr assemble(const SmallMesh& sm, double sigma_over_dt) {
    const double inv_mu = 1.0 / (4e-7 * 3.14159265358979323846);
    const int ne = static_cast<int>(sm.mesh.num_edges);
    std::vector<std::map<int,double>> rows(ne);
    const double gp = 1.0 / std::sqrt(3.0);

    for (size_t h = 0; h < sm.mesh.num_hexes; ++h) {
        double3 nd[8];
        for (int i = 0; i < 8; ++i) nd[i] = sm.mesh.nodes_current[sm.mesh.hex_nodes[h].v[i]];
        double Ke[12][12] = {};
        for (int a = 0; a < 2; ++a) for (int b = 0; b < 2; ++b) for (int c = 0; c < 2; ++c) {
            const double xi=(a?gp:-gp), et=(b?gp:-gp), ze=(c?gp:-gp);
            double J[3][3], invJ[3][3], detJ;
            compute_jacobian(nd, xi, et, ze, J, invJ, detJ);
            double3 cw[12], w[12];
            evaluate_nedelec_curl(xi, et, ze, cw);
            evaluate_nedelec_basis(xi, et, ze, w);
            // Dieselben Piola-Transformationen wie die produktive
            // Assemblierung in fem.cu. Hier stand bis 31.10.2026 die
            // transponierte Form (W spaltenweise ueber invJ, rot W zeilenweise
            // ueber J) -- auf dem gestoerten Netz ist das ein etwas anderer
            // Operator als der, den der Loeser vorkonditionieren muss.
            double3 rc[12], wp[12];
            for (int i = 0; i < 12; ++i) {
                rc[i] = piola_contravariant(J, detJ, cw[i]);
                wp[i] = piola_covariant(invJ, w[i]);
            }
            for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
                Ke[i][j] += detJ * (inv_mu*(rc[i].x*rc[j].x+rc[i].y*rc[j].y+rc[i].z*rc[j].z)
                                  + sigma_over_dt*(wp[i].x*wp[j].x+wp[i].y*wp[j].y+wp[i].z*wp[j].z));
            }
        }
        for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
            const int gi = sm.hex_edge[h*12+i], gj = sm.hex_edge[h*12+j];
            rows[gi][gj] += sm.hex_sign[h*12+i]*sm.hex_sign[h*12+j]*Ke[i][j];
        }
    }
    HostCsr A; A.num_rows = ne; A.num_cols = ne; A.row_offsets.resize(ne+1);
    for (int i = 0; i < ne; ++i) {
        A.row_offsets[i] = static_cast<int>(A.col_indices.size());
        for (const auto& kv : rows[i]) { A.col_indices.push_back(kv.first); A.values.push_back(kv.second); }
    }
    A.row_offsets[ne] = static_cast<int>(A.col_indices.size());
    return A;
}

/// Vorkonditioniertes CG. Rueckgabe: Iterationen bis rel. Residuum < tol,
/// oder max_it falls nicht erreicht.
template <class Prec>
int pcg(const HostCsr& A, const std::vector<double>& b, Prec prec,
        double tol, int max_it, double* out_rel)
{
    const int n = A.num_rows;
    std::vector<double> x(n,0.0), r=b, z(n), p(n), Ap(n);
    double bn = 0.0; for (double v : b) bn += v*v; bn = std::sqrt(bn);
    if (bn == 0.0) { *out_rel = 0.0; return 0; }
    prec(r.data(), z.data());
    p = z;
    double rz = 0.0; for (int i=0;i<n;++i) rz += r[i]*z[i];
    for (int it = 1; it <= max_it; ++it) {
        apply_host_csr(A, p.data(), Ap.data());
        double pAp = 0.0; for (int i=0;i<n;++i) pAp += p[i]*Ap[i];
        if (std::fabs(pAp) < 1e-300) { *out_rel = -1.0; return it; }
        const double al = rz/pAp;
        for (int i=0;i<n;++i) { x[i] += al*p[i]; r[i] -= al*Ap[i]; }
        double rn = 0.0; for (double v : r) rn += v*v; rn = std::sqrt(rn);
        *out_rel = rn/bn;
        if (rn/bn < tol) return it;
        prec(r.data(), z.data());
        double rz2 = 0.0; for (int i=0;i<n;++i) rz2 += r[i]*z[i];
        const double be = rz2/rz; rz = rz2;
        for (int i=0;i<n;++i) p[i] = z[i] + be*p[i];
    }
    return max_it;
}

} // namespace

int main() {
    bf::g_backend = bf::Backend::CPU;
    std::puts("========================================================");
    std::puts(" AMS-Vorkonditionierer: Iterationszahlen im Vergleich    ");
    std::puts("========================================================");

    SmallMesh sm; build(4, sm);
    const int ne = static_cast<int>(sm.mesh.num_edges);
    std::printf("\nTestnetz: %zu Knoten, %zu Hexaeder, %d Kanten\n",
                sm.mesh.num_nodes, sm.mesh.num_hexes, ne);
    std::puts("\nsigma/dt klein = starker Gradienten-Nahnullraum = harter Fall.");
    std::puts("Zum Vergleich: 1/mu_0 = 7.96e+05.\n");
    std::puts("  Abbruch bei 3000 Iterationen = nicht konvergiert (mit ! markiert).");
    std::printf("  %-12s %11s %11s %11s   %s\n",
                "sigma/dt", "Jacobi", "ILU(0)", "AMS", "AMS gegen Jacobi");

    bool ams_best_everywhere = true;
    for (double sdt : {1e6, 1e4, 1e2, 1.0}) {
        HostCsr A = assemble(sm, sdt);

        std::mt19937 rng(11);
        std::uniform_real_distribution<double> uni(-1,1);
        std::vector<double> b(ne);
        for (auto& v : b) v = uni(rng);

        // --- Jacobi
        std::vector<double> dinv(ne, 0.0);
        for (int i = 0; i < ne; ++i)
            for (int k = A.row_offsets[i]; k < A.row_offsets[i+1]; ++k)
                if (A.col_indices[k]==i) { dinv[i] = (A.values[k]!=0.0)?1.0/A.values[k]:0.0; break; }
        double rj = 0.0;
        const int it_j = pcg(A, b, [&](const double* r, double* z){
            for (int i = 0; i < ne; ++i) z[i] = dinv[i]*r[i]; }, 1e-8, 3000, &rj);

        // --- ILU(0)
        std::vector<double> iv = A.values; std::vector<int> idg;
        const int zp = bf::csr_ilu0(ne, A.row_offsets.data(), A.col_indices.data(), iv.data(), idg);
        std::vector<double> tmp(ne);
        double ri = 0.0;
        const int it_i = (zp < 0) ? pcg(A, b, [&](const double* r, double* z){
            bf::csr_lu_solve(ne, A.row_offsets.data(), A.col_indices.data(),
                             iv.data(), idg, r, z, tmp.data()); }, 1e-8, 3000, &ri) : -1;

        // --- AMS
        // Dieselben Koeffizienten wie in assemble() an den Alpha-Poisson-
        // Operator geben (nu = 1/mu_0, beta = sigma/dt), wie es src/fem.cu fuer
        // den produktiven Pfad tut. Ohne sie faellt der Hilfsraum auf nu = 1,
        // beta = 0 zurueck und wird schaedlich: gemessen 1174 Iterationen
        // gegen 298 mit Jacobi bei sigma/dt = 1e6. Der Test stammte aus der
        // Zeit vor Alpha-Poisson als Vorgabe und war darauf nicht angepasst.
        const std::vector<double> nu(sm.mesh.num_hexes, 1.0 / (4e-7 * 3.14159265358979323846));
        const std::vector<double> beta(sm.mesh.num_hexes, sdt);
        AmsPreconditioner ams;
        ams.set_alpha_coefficients(nu.data(), beta.data());
        ams.setup(A, sm.mesh);
        double ra = 0.0;
        const int it_a = pcg(A, b, [&](const double* r, double* z){ ams.apply(r,z); },
                             1e-8, 3000, &ra);

        auto tag = [](int it, char* buf) {
            std::snprintf(buf, 16, "%d%s", it, (it >= 3000) ? "!" : "");
            return buf;
        };
        char b1[16], b2[16], b3[16];
        std::printf("  %-12.0e %11s %11s %11s   %8.2fx\n", sdt,
                    tag(it_j,b1), tag(it_i,b2), tag(it_a,b3),
                    (it_a > 0) ? double(it_j)/double(it_a) : 0.0);
        // Der Vergleich gegen Jacobi ist der aussagekraeftige: ILU(0)
        // konvergiert auf dem UNgeeichten curl-curl-System ueberhaupt nicht,
        // ein Verhaeltnis dagegen waere bedeutungslos.
        if (!(it_a < it_j)) ams_best_everywhere = false;

        if (sdt == 1.0) {
            std::printf("\n  Aufbau bei sigma/dt = 1:\n"
                        "    skalarer Hilfsraum     %5d DOF, %d AMG-Ebenen, Op.Kompl %.2f\n"
                        "    vektorieller Hilfsraum %5d DOF, %d AMG-Ebenen, Op.Kompl %.2f\n"
                        "    max|A_Pi|/max|A_G| = %.2e\n",
                        ams.scalar_dofs(), ams.scalar_levels(), ams.scalar_complexity(),
                        ams.vector_dofs(), ams.vector_levels(), ams.vector_complexity(),
                        ams.auxiliary_scale_ratio());
        }
    }

    std::puts("");
    ck(ams_best_everywhere, "AMS braucht bei JEDEM sigma weniger Iterationen als Jacobi");
    std::puts("\n  Einordnung: ILU(0) konvergiert auf dem UNGEEICHTEN curl-curl-System in");
    std::puts("  keinem Fall. Das ist zu erwarten -- die Matrix ist dort extrem schlecht");
    std::puts("  konditioniert (der Gradienten-Nahnullraum wird nur durch den kleinen");
    std::puts("  Massenterm gebrochen), und ILU(0) erzeugt darauf winzige oder negative");
    std::puts("  Pivots. Genau deshalb braucht der produktive Pfad bisher die Eichung.");
    std::puts("  AMS kommt ohne aus.");
    std::puts("");
    std::puts("  Die Hilfsprobleme werden mit Aggregations-AMG geloest. Eine erste");
    std::puts("  Fassung benutzte dort ILU(0) mit Shift und kam nur auf rund Faktor 2");
    std::puts("  gegenueber Jacobi -- die Hilfsloesung war der Engpass, nicht die");
    std::puts("  AMS-Struktur.");

    std::puts("\n========================================================");
    if (g_fail == 0) { std::puts(" ALLE PRUEFUNGEN BESTANDEN");
                       std::puts("========================================================"); return 0; }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
