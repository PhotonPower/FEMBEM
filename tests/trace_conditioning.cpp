// =============================================================================
// Phase 1b, Schritt 5: Konditionierung der Bloecke und des Schurkomplements (nur messen, keine Gegenmassnahmen).
//   Aufruf:  trace_conditioning ball <n>
//            trace_conditioning team28 <vtk> [--mur-coil=<wert>]
//   ball:    Kugel n^3 Hexaeder (Cube-to-Ball), sigma = 0, mu_r in {1, 10, 100, 1000}, dimensionslos (mu0 = 1)
//   team28:  TEAM-28-Hexaedernetz (Millimeter), Aussenspule (Komp. 1), Innenspule (2), Platte (3, sigma = 3.4e7 S/m),
//            mu0 = 4 pi 1e-7, Frequenzen 0.01 ... 50 Hz (Dichte Matrizen in Double, komplexe Arithmetik im Block (1,1)).
// Gemessen: Eigenwerte (symmetrisch, exakt) von K (nach Baum-Kobaum), W (ohne Konstanten), V0 und Singulaerwerte des
// Schurkomplements S auf (a_Gamma, psi, c) -- exakt per SVD (n <= ~3000), Konditionszahlen auch mit Diagonalskalierung.
// =============================================================================
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "TraceHexMeshes.h"
#include "TraceBemOps.h"
#include "TraceFem.h"
#include "TraceSystem.h"

using namespace trace;
using Cd = std::complex<double>;
using CMat = Eigen::MatrixXcd;
using CVec = Eigen::VectorXcd;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Spec { double lmax, lmin_plus; int nzero; };
// symmetrische Eigenwerte: groesster, kleinster von null verschiedener (> 1e-10 lmax), Zahl der (fast) Nullwerte
static Spec sym_spec(const Mat& A)
{
    Eigen::SelfAdjointEigenSolver<Mat> es(A, Eigen::EigenvaluesOnly);
    const Vec ev = es.eigenvalues();
    Spec s{0, 0, 0};
    double amax = 0;
    for (int i = 0; i < ev.size(); ++i) amax = std::max(amax, std::fabs(ev[i]));
    s.lmax = amax;
    s.lmin_plus = 1e300;
    for (int i = 0; i < ev.size(); ++i) {
        if (std::fabs(ev[i]) <= 1e-10 * amax) ++s.nzero;
        else s.lmin_plus = std::min(s.lmin_plus, std::fabs(ev[i]));
    }
    return s;
}

static double cond_svd(const CMat& A)
{
    Eigen::BDCSVD<CMat> svd(A);
    const auto sv = svd.singularValues();
    return sv[0] / sv[sv.size() - 1];
}

static double cond_scaled(const CMat& A)
{
    CMat B = A;
    Vec d(A.rows());
    for (int i = 0; i < A.rows(); ++i) d[i] = 1.0 / std::sqrt(std::max(std::abs(A(i, i)), 1e-300));
    for (int i = 0; i < A.rows(); ++i) for (int j = 0; j < A.cols(); ++j) B(i, j) *= d[i] * d[j];
    return cond_svd(B);
}

// Schurkomplement des Gesamtsystems auf Gamma = (a auf Flaechenkanten, psi, c); innen = uebrige erhaltene FEM-Kanten.
struct Parts { int nG = 0, nI = 0; };
static CMat schur(const Coupled& C, const FemProblem& F, const Mat& Msig_scaled, double omega, Parts* parts, double* cond_Aii)
{
    const TraceSurface& S = F.surface;
    std::vector<char> on_surface(F.num_edges, 0);
    for (int e : F.surf_edge_to_fem) on_surface[e] = 1;
    std::vector<int> G, I;
    for (int idx : C.keep) {
        if (idx >= C.nE) G.push_back(idx);
        else if (on_surface[idx]) G.push_back(idx);
        else I.push_back(idx);
    }
    (void)S;
    const int nG = static_cast<int>(G.size()), nI = static_cast<int>(I.size());
    parts->nG = nG; parts->nI = nI;
    CMat AGG(nG, nG), AGI(nG, nI), AIG(nI, nG), AII(nI, nI);
    auto entry = [&](int i, int j) {
        Cd v = C.A(i, j);
        if (i < C.nE && j < C.nE) v += Cd(0.0, omega) * Msig_scaled(i, j);      // j omega sigma M, nur Block (1,1)
        return v;
    };
    for (int a = 0; a < nG; ++a) for (int b = 0; b < nG; ++b) AGG(a, b) = entry(G[a], G[b]);
    for (int a = 0; a < nG; ++a) for (int b = 0; b < nI; ++b) { AGI(a, b) = entry(G[a], I[b]); AIG(b, a) = entry(I[b], G[a]); }
    for (int a = 0; a < nI; ++a) for (int b = 0; b < nI; ++b) AII(a, b) = entry(I[a], I[b]);
    Eigen::PartialPivLU<CMat> lu(AII);
    *cond_Aii = 0.0;
    if (nI > 0 && nI <= 4000) { // Konditionsschaetzung von A_ii: sigma_max (Potenzmethode), sigma_min (inverse Iteration), je 25 Schritte
        CVec x = CVec::Random(nI); x.normalize();
        double smax = 0;
        for (int it = 0; it < 25; ++it) { CVec y = AII.adjoint() * (AII * x); smax = std::sqrt(y.norm()); x = y / y.norm(); }
        CVec z = CVec::Random(nI); z.normalize();
        double inv = 0;
        for (int it = 0; it < 25; ++it) { CVec y = lu.solve(z); CVec w = lu.adjoint().solve(y); inv = std::sqrt(w.norm()); z = w / w.norm(); }
        // inv ~ 1 / sigma_min^... : w = (A^H A)^{-1} z  => |w| -> 1/sigma_min^2
        *cond_Aii = smax * std::sqrt(inv);
    }
    return AGG - AGI * lu.solve(AIG);
}

static void report(const char* label, const FemProblem& F, const BoundaryOps& O, const DerivedOps& D, double mu0, double omega, bool want_blocks)
{
    const TraceSurface& S = F.surface;
    CoupledOptions opt; opt.sigma_factor = 0.0;
    const Vec f = Vec::Zero(F.num_edges);
    const Coupled C = build_coupled(F, O, D, f, opt);
    const Mat Msig = as_matrix(F.Msig, F.num_edges, F.num_edges);
    Parts parts;
    double cAii = 0;
    const double t0 = now();
    const CMat Sc = schur(C, F, Msig, omega, &parts, &cAii);
    const double cS = cond_svd(Sc), cSs = cond_scaled(Sc);
    std::printf("  %-34s omega=%9.3e  n_innen=%5d n_Gamma=%5d | cond(A_ii) ~ %.3e | cond(S) = %.3e, mit Diagonalskalierung %.3e  [%.0f s]\n",
                label, omega, parts.nI, parts.nG, cAii, cS, cSs, now() - t0);
    if (want_blocks) {
        // symmetrische Bloecke
        std::vector<char> gauged = F.gauged;
        Mat Kr; {
            std::vector<int> keep;
            for (int e = 0; e < F.num_edges; ++e) if (!gauged[e]) keep.push_back(e);
            const Mat K = dense_K(F, 0.0);
            Kr.resize(keep.size(), keep.size());
            for (size_t i = 0; i < keep.size(); ++i) for (size_t j = 0; j < keep.size(); ++j) Kr(i, j) = K(keep[i], keep[j]);
        }
        const Spec sK = sym_spec(Kr), sW = sym_spec(D.W), sV = sym_spec(D.V0);
        std::printf("      K (Baum-Kobaum): lmax %.3e, lmin+ %.3e, Kond %.3e, Nullwerte %d | W: lmax %.3e, lmin+ %.3e, Kond %.3e, Nullwerte %d (Konstanten: %d) | V0: lmax %.3e, lmin %.3e, Kond %.3e\n",
                    sK.lmax, sK.lmin_plus, sK.lmax / sK.lmin_plus, sK.nzero, sW.lmax, sW.lmin_plus, sW.lmax / sW.lmin_plus, sW.nzero, S.num_components, sV.lmax, sV.lmin_plus, sV.lmax / sV.lmin_plus);
        std::printf("      Verhaeltnis lmax(1/mu0 V0-Term)/lmax(K): siehe Spektren; mu0 = %.3e\n", mu0);
    }
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "Aufruf: trace_conditioning ball <n> | team28 <vtk> [--mur-coil=<wert>]\n"); return 2; }
    const std::string mode = argv[1];
    if (mode == "ball") {
        const int n = std::atoi(argv[2]);
        write_vtk("trace_cond_ball.vtk", ball_hex(n, 1.0), "ball");
        std::printf("Kugel n=%d (%d Hexaeder), sigma = 0, mu0 = 1, mu_r variiert (Ball, Material im ganzen Koerper)\n", n, n * n * n);
        for (double mur : {1.0, 10.0, 100.0, 1000.0}) {
            FemMaterials mats; mats.mu_r = {mur, mur}; mats.kappa = {0.0, 0.0};
            FemProblem F = build_fem_problem("trace_cond_ball.vtk", mats, 1.0, [](const Vec3&, int) { return v3(0, 0, 0); });
            TraceSurface& S = F.surface;
            compute_harmonic_fields(S);
            const BoundaryOps O = assemble_boundary_ops(S);
            const DerivedOps D = derive_ops(S, O);
            char label[64]; std::snprintf(label, sizeof label, "mu_r = %g", mur);
            report(label, F, O, D, 1.0, 0.0, true);
        }
    } else if (mode == "team28") {
        const std::string vtk = argv[2];
        double mur_coil = 1.0;
        for (int i = 3; i < argc; ++i) if (!std::strncmp(argv[i], "--mur-coil=", 11)) mur_coil = std::atof(argv[i] + 11);
        const double MU0r = 4e-7 * 3.14159265358979323846;
        FemMaterials mats; mats.mu_r = {1.0, mur_coil, mur_coil, 1.0}; mats.kappa = {0.0, 0.0, 0.0, 3.4e7};
        double t0 = now();
        FemProblem F = build_fem_problem(vtk, mats, MU0r, [](const Vec3&, int) { return v3(0, 0, 0); }, 1e-3);
        TraceSurface& S = F.surface;
        compute_harmonic_fields(S);
        const BoundaryOps O = assemble_boundary_ops(S);
        const DerivedOps D = derive_ops(S, O);
        std::printf("TEAM 28: %s: %d Hexaeder, %d FEM-Kanten, %d Randquads, %d Komponenten, nH = %d, mu_r(Spulen) = %g  [Aufbau %.0f s]\n",
                    vtk.c_str(), F.num_hexes, F.num_edges, S.num_quads(), S.num_components, S.nH, mur_coil, now() - t0);
        bool first = true;
        for (double fHz : {0.01, 0.1, 1.0, 10.0, 50.0}) {
            char label[64]; std::snprintf(label, sizeof label, "f = %g Hz", fHz);
            report(label, F, O, D, MU0r, 2.0 * 3.14159265358979323846 * fHz, first);
            first = false;
        }
    } else { std::fprintf(stderr, "unbekannter Modus\n"); return 2; }
    return 0;
}
