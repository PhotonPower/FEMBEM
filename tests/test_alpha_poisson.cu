// =============================================================================
// Abnahmeprüfung: taugt der Vektor-Poisson-Operator als AMG-Grundlage?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_alpha_poisson
//   ./build/test_alpha_poisson
//
// Reiner Host-Code, keine GPU, kein Netz von der Platte -- das Hexaedernetz
// wird hier erzeugt.
//
// HINTERGRUND
// -----------
// Der vektorielle AMS-Hilfsraum benutzte bisher A_Pi = Pi^T A Pi. Gemessener
// AMG-Konvergenzfaktor am TEAM-28-Problem: 0.93 je V-Zyklus, also rund 30
// Zyklen je Dekade -- der Hilfsraum trug praktisch nichts bei.
//
// Die AMG-Implementierung ist dabei nachweislich gesund: auf dem 3D-Laplace
// erreicht sie 0.30, netzunabhängig (tests/test_amg_convergence.cu). Es lag
// also am Operator. A_Pi = Pi^T A Pi erbt den Nullraum des Curl-Curl-Anteils
// und ist dort, wo sigma = 0 ist, in einem ganzen UNTERRAUM entartet (den
// Gradientenanteilen im knotenweisen Vektorfeld). Glatte Aggregation kann so
// einen Nahnullraum nicht darstellen.
//
// Ersatz ist der knotenweise Vektor-Poisson-Operator mit nu (in hypre:
// HYPRE_AMSSetAlphaPoissonMatrix), der überall positiv definit ist.
//
// GEMESSEN auf strukturierten Hexaedernetzen, mit gelumptem Massenterm in der
// unteren Hälfte (also mit derselben Leiter/Luft-Struktur wie das echte
// Problem):
//
//   Zellen   Knoten   DOF     Faktor je V-Zyklus
//    8^3        729   2187          0.206
//   12^3       2197   6591          0.246
//   16^3       4913  14739          0.256
//
// Zum Vergleich: der 7-Punkt-Laplace liegt bei 0.30
// (tests/test_amg_convergence.cu), Pi^T A Pi am echten Problem bei 0.93.
//
// Der Massenterm ist dabei nicht nebensächlich, sondern der Kern der Sache.
// OHNE ihn (reine Steifigkeit, nu = 1) lag der Faktor bei 0.48 bis 0.58 --
// immer noch brauchbar als AMG-Grundlage, aber als VORKONDITIONIERER
// unbrauchbar, weil der Operator dann nicht spektral äquivalent zu Pi^T A Pi
// ist. Der äußere GMRES stagnierte damit. Genau deshalb misst
// tests/test_preconditioner_end_to_end.cu die Iterationszahl und nicht diesen
// Faktor: ein guter Faktor auf dem falschen Operator hilft nichts.
//
// WAS DIESER TEST NICHT ZEIGT: ob AMS damit am echten Problem den
// blockdiagonalen ILU schlägt. Das Netz hier ist strukturiert und
// gleichmäßig; das TEAM-28-Netz ist es nicht.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "DiscreteGradient.cuh"
#include "AlgebraicMultigrid.cuh"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static double measure(const AlgebraicMultigrid& amg, const HostCsr& A, int cycles) {
    const int N = A.num_rows;
    std::mt19937 rng(7); std::uniform_real_distribution<double> uni(-1,1);
    std::vector<double> x(N,0.0), b(N), r(N), e(N), Ax(N);
    for (int i=0;i<N;++i) b[i]=uni(rng);
    auto rn=[&](){ apply_host_csr(A,x.data(),Ax.data()); double s=0;
        for(int i=0;i<N;++i){r[i]=b[i]-Ax[i]; s+=r[i]*r[i];} return std::sqrt(s); };
    const double r0=rn(); double rk=r0;
    for(int c=0;c<cycles;++c){ amg.apply(r.data(),e.data());
        for(int i=0;i<N;++i) x[i]+=e[i]; rk=rn(); }
    return std::pow(rk/r0, 1.0/cycles);
}

int main() {
    // Host-Test: ohne GPU lauffaehig. Ohne diese Zeile legt DeviceArray den
    // Speicher mit cudaMallocManaged an und bricht ohne Grafikkarte ab.
    bf::g_backend = bf::Backend::CPU;
    std::printf("\n=== Vektor-Poisson-Operator: AMG-Konvergenzfaktor ===\n\n");
    int failures = 0;
    double f_coarse = 0.0, f_fine = 0.0;

    // Strukturiertes Hexaedernetz n x n x n Zellen
    for (int n : {8, 12, 16}) {
        const int np = n + 1;
        Mesh mesh;
        mesh.num_nodes = (size_t)np*np*np;
        mesh.num_hexes = (size_t)n*n*n;
        mesh.nodes_current.allocate(mesh.num_nodes);
        mesh.hex_nodes.allocate(mesh.num_hexes);
        auto nid=[&](int i,int j,int k){ return (k*np + j)*np + i; };
        for (int k=0;k<np;++k) for(int j=0;j<np;++j) for(int i=0;i<np;++i)
            mesh.nodes_current.data()[nid(i,j,k)] =
                make_double3(i/(double)n, j/(double)n, k/(double)n);
        int h=0;
        for (int k=0;k<n;++k) for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
            Hexahedron hx;
            hx.v[0]=nid(i,j,k);     hx.v[1]=nid(i+1,j,k);
            hx.v[2]=nid(i+1,j+1,k); hx.v[3]=nid(i,j+1,k);
            hx.v[4]=nid(i,j,k+1);   hx.v[5]=nid(i+1,j,k+1);
            hx.v[6]=nid(i+1,j+1,k+1); hx.v[7]=nid(i,j+1,k+1);
            mesh.hex_nodes.data()[h++] = hx;
        }

        // Zwei Faelle: reine Steifigkeit (nu = 1, kein Leiter) und mit
        // gelumptem Massenterm in der unteren Haelfte. Der Massenterm ist der
        // Teil, ohne den der Operator am echten Problem unbrauchbar war --
        // er gehoert deshalb in die Pruefung.
        std::vector<double> nu_h((size_t)n*n*n, 1.0);
        std::vector<double> beta_h((size_t)n*n*n, 0.0);
        for (int kk = 0; kk < n; ++kk)
            for (int jj = 0; jj < n; ++jj)
                for (int ii = 0; ii < n; ++ii)
                    beta_h[(size_t)((kk*n + jj)*n + ii)] = (kk < n/2) ? 1.0e3 : 0.0;

        HostCsr A = build_alpha_poisson_matrix(mesh, nu_h.data(), beta_h.data());
        // Dirichlet-artige Regularisierung: der reine Neumann-Poisson ist
        // singulaer (Konstanten im Nullraum). Ein kleiner Massenterm auf der
        // Diagonalen macht ihn definit, ohne die Struktur zu aendern.
        double dmax=0.0;
        for (int i=0;i<A.num_rows;++i)
            for (int k2=A.row_offsets[i];k2<A.row_offsets[i+1];++k2)
                if (A.col_indices[k2]==i) dmax=std::fmax(dmax,std::fabs(A.values[k2]));
        for (int i=0;i<A.num_rows;++i)
            for (int k2=A.row_offsets[i];k2<A.row_offsets[i+1];++k2)
                if (A.col_indices[k2]==i) A.values[k2] += 1e-3*dmax;

        AlgebraicMultigrid amg; AlgebraicMultigrid::Options o;
        amg.setup(A, 3, o);
        const double f = measure(amg, A, 5);
        std::printf("  n=%2d  Knoten %6zu  DOF %7d  Ebenen %d  Faktor %.4f\n",
                    n, mesh.num_nodes, A.num_rows, amg.num_levels(), f);
        if (n == 8)  f_coarse = f;
        if (n == 16) f_fine   = f;
        if (!(f < 0.6)) ++failures;
    }

    std::printf("\n");
    std::printf("  [%s] alle Netzgroessen unter der Brauchbarkeitsschwelle 0.6\n",
                failures == 0 ? " OK " : "FEHL");

    // Der Faktor darf mit feinerem Netz nicht davonlaufen -- sonst waere es
    // kein Mehrgitterverfahren, sondern ein geglaettetes Einzelgitterverfahren.
    const bool ok_scaling = (f_fine <= f_coarse + 0.05);
    std::printf("  [%s] Faktor laeuft mit feinerem Netz nicht davon "
                "(%.4f bei 8^3 -> %.4f bei 16^3)\n",
                ok_scaling ? " OK " : "FEHL", f_coarse, f_fine);
    if (!ok_scaling) ++failures;

    std::printf("\n");
    if (failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", failures);
    return 1;
}
