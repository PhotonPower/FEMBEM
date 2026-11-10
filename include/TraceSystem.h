#ifndef TRACE_SYSTEM_H
#define TRACE_SYSTEM_H

// Abgeleitete Bloecke und gekoppeltes System der Maxwell-Spur-Kopplung (Phase 1b). Nur in .cpp-Dateien
// einbinden (Eigen). Theorie: docs/theory/maxwell-trace-coupling.md, Abschnitte 3.3, 5.2, 5.3.
#include <Eigen/Dense>

#include "TraceBemOps.h"
#include "TraceFem.h"
#include "TraceSurface.h"

namespace trace {

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;

Mat as_matrix(const std::vector<double>& d, int rows, int cols);   // zeilenweise -> Eigen

// Blocksystem der Randunbekannten (S):  W = G^T Vrot G (V x V), Z = -G^T Vrot H, Y = H^T Vrot H, Q = (1/2 Mrot + Crot) H,
// N = 1/2 Mt + Dt (F x V). Mit flat = true wird Vrot durch die Variante mit koplanarem Skalarprodukt ersetzt (Gegenprobe).
struct DerivedOps {
    Mat G, R, H;      // E x V, F x E, E x nH
    Mat W, Z, Y, Q, N, P;
    Mat V0;
};
DerivedOps derive_ops(const TraceSurface& S, const BoundaryOps& O, bool flat = false);

// Gekoppeltes Blocksystem in den Unbekannten x = (a [alle FEM-Kanten], psi [Flaechenknoten], c [harmonische Koeffizienten]).
//   symmetrisch (S):  [K + (1/mu0) B_R^T V0 B_R,  B_R^T N,  Q ;  N^T B_R, -mu0 W, -mu0 Z ;  Q^T, -mu0 Z^T, -mu0 Y]
//   unsymmetrisch (J): [K, B_R^T Mt, P ; N^T B_R, -mu0 W, -mu0 Z ; Q^T, -mu0 Z^T, -mu0 Y]
// mit B_R = R * (Einbettung Flaechenkante -> FEM-Kante), K = K_curl + sigma_factor * M_sigma (sigma_factor = 1/dt bzw. omega-aehnlicher
// reeller Faktor; Tests rechnen reell). 'keep' listet die Freiheitsgrade nach Eichung (Baum-Kobaum) und Festlegung je einer psi-Konstante je Komponente.
struct Coupled {
    Mat A;                 // volle Matrix (nE + nV + nH)^2
    Vec b;                 // rechte Seite (mu0 f, 0, 0) bei scaled, sonst (f, 0, 0)
    bool scaled = true;    // siehe CoupledOptions::scaled
    double mu0 = 1.0;
    int nE = 0, nV = 0, nH = 0;
    std::vector<int> keep; // erhaltene Indizes
    Mat reduced() const;   // A(keep, keep)
    Vec solve() const;     // Loesung der reduzierten Gleichung in den URSPRUENGLICHEN Variablen (a, psi, c): bei scaled wird psi~ und c~ durch mu0 geteilt; entfernte Indizes = 0
};
struct CoupledOptions {
    bool unsymmetric = false;   // (J) statt (S)
    bool gauge_tree = true;     // Baum-Kobaum im sigma = 0-Gebiet
    bool pin_psi = true;        // eine Konstante je Flaechenkomponente festlegen
    bool include_bem = true;    // false: nur K (alte Eichung ohne Kopplung, fuer Gegenproben)
    double sigma_factor = 1.0;
    // scaled (Phase 2, Schritt 0a): Unbekannte (a, psi~, c~) = (a, mu0 psi, mu0 c), Gesamtmatrix mit mu0 multipliziert:
    //   [mu0 K + R^T V0 R, R^T N, Q; N^T R, -W, -Z; Q^T, -Z^T, -Y], rechte Seite (mu0 f, 0, 0); symmetrisch, alle Bloecke geometrisch (kein mu0, kein Einheitenfaktor ausser
    //   der Laenge). false: die unskalierte Fassung aus Phase 1b (zum Vergleich).
    bool scaled = true;
    bool flip_lambda = false;   // Gegenprobe: Vorzeichen der Kopplung (1,2), (1,3) umkehren
};
Coupled build_coupled(const FemProblem& F, const BoundaryOps& O, const DerivedOps& D, const Vec& f, const CoupledOptions& opt = CoupledOptions());

Mat dense_K(const FemProblem& F, double sigma_factor);

} // namespace trace

#endif
