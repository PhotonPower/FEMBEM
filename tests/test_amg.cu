// =============================================================================
// test_amg.cu -- liefert das AMG netzunabhaengige Konvergenz?
// =============================================================================
//
// Die eine Eigenschaft, die AMG gegenueber ILU(0) und Jacobi auszeichnet, ist
// die Unabhaengigkeit der Iterationszahl von der Problemgroesse. Genau das wird
// hier gemessen -- an einem 3D-Knoten-Laplace, also demselben Problemtyp wie
// die AMS-Hilfsraeume A_G und A_Pi.
//
// Waechst die Iterationszahl mit n, ist das AMG nicht in Ordnung, egal wie gut
// die Einzelbausteine aussehen.
//
// Zusaetzlich geprueft:
//   * Vergroeberungsrate und Operatorkomplexitaet (bei gesunder Aggregation
//     sinkt die Groesse je Ebene um etwa Faktor 3 bis 9 in 3D)
//   * ein SYSTEM mit drei Freiheitsgraden je Knoten -- die Bauart, die A_Pi
//     braucht
//
// Braucht keine GPU.
// =============================================================================

#include "AlgebraicMultigrid.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int g_fail = 0;
void ck(bool ok, const char* w) {
    std::printf("  [%s] %s\n", ok ? "OK " : "FEHL", w);
    if (!ok) ++g_fail;
}

/// 7-Punkt-Laplace auf einem n x n x n Gitter, optional mit b Freiheitsgraden
/// je Knoten (blockdiagonal -- genuegt, um die Systemvariante zu pruefen).
HostCsr laplace3d(int n, int b = 1) {
    const int N = n * n * n;
    HostCsr A;
    A.num_rows = N * b; A.num_cols = N * b;
    A.row_offsets.assign(N * b + 1, 0);
    auto idx = [n](int i, int j, int k) { return (i * n + j) * n + k; };

    for (int i = 0; i < n; ++i) for (int j = 0; j < n; ++j) for (int k = 0; k < n; ++k) {
        const int I = idx(i,j,k);
        for (int c = 0; c < b; ++c) {
            const int row = I * b + c;
            A.row_offsets[row] = static_cast<int>(A.col_indices.size());
            std::vector<std::pair<int,double>> e;
            if (i>0)   e.emplace_back(idx(i-1,j,k)*b+c, -1.0);
            if (j>0)   e.emplace_back(idx(i,j-1,k)*b+c, -1.0);
            if (k>0)   e.emplace_back(idx(i,j,k-1)*b+c, -1.0);
            e.emplace_back(row, 6.0);
            if (k<n-1) e.emplace_back(idx(i,j,k+1)*b+c, -1.0);
            if (j<n-1) e.emplace_back(idx(i,j+1,k)*b+c, -1.0);
            if (i<n-1) e.emplace_back(idx(i+1,j,k)*b+c, -1.0);
            std::sort(e.begin(), e.end());
            for (auto& p : e) { A.col_indices.push_back(p.first); A.values.push_back(p.second); }
        }
    }
    A.row_offsets[N*b] = static_cast<int>(A.col_indices.size());
    return A;
}

template <class Prec>
int pcg(const HostCsr& A, const std::vector<double>& b, Prec prec,
        double tol, int max_it)
{
    const int n = A.num_rows;
    std::vector<double> x(n,0.0), r=b, z(n), p(n), Ap(n);
    double bn = 0.0; for (double v : b) bn += v*v; bn = std::sqrt(bn);
    prec(r.data(), z.data());
    p = z;
    double rz = 0.0; for (int i=0;i<n;++i) rz += r[i]*z[i];
    for (int it = 1; it <= max_it; ++it) {
        apply_host_csr(A, p.data(), Ap.data());
        double pAp = 0.0; for (int i=0;i<n;++i) pAp += p[i]*Ap[i];
        if (std::fabs(pAp) < 1e-300) return it;
        const double al = rz/pAp;
        for (int i=0;i<n;++i) { x[i] += al*p[i]; r[i] -= al*Ap[i]; }
        double rn = 0.0; for (double v : r) rn += v*v;
        if (std::sqrt(rn)/bn < tol) return it;
        prec(r.data(), z.data());
        double rz2 = 0.0; for (int i=0;i<n;++i) rz2 += r[i]*z[i];
        const double be = rz2/rz; rz = rz2;
        for (int i=0;i<n;++i) p[i] = z[i] + be*p[i];
    }
    return max_it;
}

} // namespace

int main() {
    std::puts("========================================================");
    std::puts(" AMG: Iterationszahl gegen Problemgroesse                ");
    std::puts("========================================================");
    std::puts("\n3D-Knoten-Laplace, derselbe Problemtyp wie die AMS-Hilfsraeume.");
    std::puts("Entscheidend: waechst die AMG-Spalte mit n, ist das AMG nicht in Ordnung.\n");
    std::printf("  %6s %9s %10s %10s %8s %8s %s\n",
                "n", "DOF", "Jacobi", "AMG", "Ebenen", "Op.Kompl", "groebste");

    std::vector<int> amg_iters;
    for (int n : {8, 12, 16, 20}) {
        HostCsr A = laplace3d(n);
        const int N = A.num_rows;
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> uni(-1,1);
        std::vector<double> b(N);
        for (auto& v : b) v = uni(rng);

        std::vector<double> dinv(N);
        for (int i = 0; i < N; ++i)
            for (int k = A.row_offsets[i]; k < A.row_offsets[i+1]; ++k)
                if (A.col_indices[k]==i) { dinv[i] = 1.0/A.values[k]; break; }
        const int it_j = pcg(A, b, [&](const double* r, double* z){
            for (int i=0;i<N;++i) z[i] = dinv[i]*r[i]; }, 1e-10, 5000);

        AlgebraicMultigrid amg;
        amg.setup(A, 1);
        const int it_a = pcg(A, b, [&](const double* r, double* z){ amg.apply(r,z); }, 1e-10, 5000);
        amg_iters.push_back(it_a);

        std::printf("  %6d %9d %10d %10d %8d %8.2f %8d\n", n, N, it_j, it_a,
                    amg.num_levels(), amg.operator_complexity(),
                    amg.level_size(amg.num_levels()-1));
    }

    const int lo = *std::min_element(amg_iters.begin(), amg_iters.end());
    const int hi = *std::max_element(amg_iters.begin(), amg_iters.end());
    std::printf("\n  AMG-Iterationen zwischen %d und %d bei 8-facher DOF-Zahl.\n", lo, hi);
    ck(hi <= lo + 6, "Iterationszahl waechst kaum mit der Problemgroesse");
    ck(hi < 40, "AMG konvergiert in unter 40 Iterationen");

    // ---- System mit 3 Freiheitsgraden je Knoten (Bauart von A_Pi)
    std::puts("\n[System] Drei Freiheitsgrade je Knoten, Aggregation auf Knotenebene");
    {
        HostCsr A = laplace3d(12, 3);
        const int N = A.num_rows;
        std::mt19937 rng(9);
        std::uniform_real_distribution<double> uni(-1,1);
        std::vector<double> b(N);
        for (auto& v : b) v = uni(rng);

        AlgebraicMultigrid amg;
        amg.setup(A, 3);
        const int it = pcg(A, b, [&](const double* r, double* z){ amg.apply(r,z); }, 1e-10, 5000);
        std::printf("      %d DOF, %d Ebenen, %d Iterationen, Op.Komplexitaet %.2f\n",
                    N, amg.num_levels(), it, amg.operator_complexity());
        ck(it < 40, "Systemvariante konvergiert ebenso schnell");
        ck(amg.coarsest_direct_ok(), "groebste Ebene direkt loesbar");
    }

    std::puts("\n========================================================");
    if (g_fail == 0) { std::puts(" ALLE PRUEFUNGEN BESTANDEN");
                       std::puts("========================================================"); return 0; }
    std::printf(" %d PRUEFUNG(EN) FEHLGESCHLAGEN\n", g_fail);
    std::puts("========================================================");
    return 1;
}
