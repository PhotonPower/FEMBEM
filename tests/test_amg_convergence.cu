// =============================================================================
// Abnahmeprüfung: löst das AMG ein Problem, bei dem es funktionieren MUSS?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_amg_convergence
//   ./build/test_amg_convergence
//
// Reiner Host-Code, keine GPU, keine Datei.
//
// HINTERGRUND
// -----------
// Der AMS-Zweig wurde lange nach Level-Größen und Operatorkomplexität
// beurteilt. Beide sagen über die Qualität nichts:
//
//   * "vektorieller Hilfsraum: 21039 -> 948" klingt nach absurder
//     Vergröberung. Bei bs = 3 sind das aber 316 Aggregate aus 7013 Knoten,
//     also 22-fach auf Knotenebene -- und ein 3x3x3-Aggregat im
//     Hexaedernetz ist 27-fach, mithin Lehrbuch für 3D-Aggregation.
//   * Die Warnschwelle 1.3 für die Operatorkomplexität stammt aus einem
//     Modellproblem mit viel sanfterer Vergröberung über mehr Ebenen. Bei
//     dieser Vergröberung und vier Ebenen ist ein Wert um 1.1 zu erwarten.
//
// Beide Kennzahlen haben die Suche in die Irre geführt. Was zählt, ist der
// KONVERGENZFAKTOR: um welchen Faktor fällt das Residuum je V-Zyklus?
//
// Dieser Test misst ihn auf dem 3D-Laplace, wo jedes brauchbare AMG
// funktionieren muss, und hält damit zweierlei fest: dass die Implementierung
// arbeitet, und den Bezugswert, gegen den die Messung am echten Problem
// (AmsPreconditioner::scalar_convergence / vector_convergence) zu lesen ist.
//
// Gemessen wurden 0.30 je Zyklus, praktisch netzunabhängig (1000 / 4096 / 8000
// Freiheitsgrade). Netzunabhängigkeit ist dabei die eigentliche Aussage: sie
// unterscheidet ein echtes Mehrgitterverfahren von einem geglätteten
// Einzelgitterverfahren, das bei feinerem Netz einbricht.
//
// Die Schranke ist mit 0.6 bewusst locker: der Test soll einen KAPUTTEN
// Hierarchieaufbau fangen, nicht eine Verschlechterung um zehn Prozent
// melden.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "AlgebraicMultigrid.cuh"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

// 3D-Laplace auf einem n x n x n Gitter, 7-Punkt-Stern.
static HostCsr laplace3d(int n) {
    const int N = n*n*n;
    HostCsr A; A.num_rows = N; A.num_cols = N;
    A.row_offsets.assign(N+1, 0);
    auto id = [&](int i,int j,int k){ return (k*n + j)*n + i; };
    for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
        const int row = id(i,j,k);
        A.row_offsets[row] = (int)A.col_indices.size();
        std::vector<std::pair<int,double>> e;
        if (k>0)   e.push_back({id(i,j,k-1),-1.0});
        if (j>0)   e.push_back({id(i,j-1,k),-1.0});
        if (i>0)   e.push_back({id(i-1,j,k),-1.0});
        e.push_back({row, 6.0});
        if (i<n-1) e.push_back({id(i+1,j,k),-1.0});
        if (j<n-1) e.push_back({id(i,j+1,k),-1.0});
        if (k<n-1) e.push_back({id(i,j,k+1),-1.0});
        for (auto& p : e) { A.col_indices.push_back(p.first); A.values.push_back(p.second); }
    }
    A.row_offsets[N] = (int)A.col_indices.size();
    return A;
}

static double measure(const AlgebraicMultigrid& amg, const HostCsr& A, int cycles) {
    const int N = A.num_rows;
    std::mt19937 rng(1); std::uniform_real_distribution<double> uni(-1,1);
    std::vector<double> x(N,0.0), b(N), r(N), e(N), Ax(N);
    for (int i=0;i<N;++i) b[i]=uni(rng);
    auto rn = [&](){ apply_host_csr(A,x.data(),Ax.data()); double s=0;
        for(int i=0;i<N;++i){r[i]=b[i]-Ax[i]; s+=r[i]*r[i];} return std::sqrt(s); };
    const double r0 = rn(); double rk = r0;
    for (int c=0;c<cycles;++c){ amg.apply(r.data(),e.data());
        for(int i=0;i<N;++i) x[i]+=e[i]; rk = rn(); }
    return std::pow(rk/r0, 1.0/cycles);
}

int main() {
    std::printf("\n=== AMG auf dem 3D-Laplace: Konvergenzfaktor je V-Zyklus ===\n\n");
    std::printf("  %8s %10s %10s %18s\n", "n", "DOF", "Ebenen", "Faktor/V-Zyklus");

    int failures = 0;
    double worst = 0.0, best = 1.0;
    for (int n : {10, 16, 20}) {
        HostCsr A = laplace3d(n);
        AlgebraicMultigrid amg;
        AlgebraicMultigrid::Options o;
        amg.setup(A, 1, o);
        const double f = measure(amg, A, 5);
        std::printf("  %8d %10d %10d %18.4f\n", n, A.num_rows, amg.num_levels(), f);
        worst = std::fmax(worst, f);
        best  = std::fmin(best, f);
        if (!(f < 0.6)) ++failures;
    }
    std::printf("\n");
    std::printf("  [%s] AMG konvergiert auf allen Groessen (Faktor < 0.6)\n",
                failures == 0 ? " OK " : "FEHL");

    // Netzunabhaengigkeit: das ist die eigentliche Mehrgitter-Eigenschaft.
    // Ein Verfahren, dessen Faktor mit dem Netz waechst, ist keins.
    const bool mesh_indep = (worst - best) < 0.15;
    std::printf("  [%s] netzunabhaengig (Spanne %.4f ueber Faktor 8 in der DOF-Zahl)\n",
                mesh_indep ? " OK " : "FEHL", worst - best);
    if (!mesh_indep) ++failures;

    std::printf("\n");
    if (failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", failures);
    return 1;
}
