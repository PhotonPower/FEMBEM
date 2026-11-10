#ifndef TRACE_SYSTEM_H
#define TRACE_SYSTEM_H

// Abgeleitete Bloecke und gekoppeltes System der Maxwell-Spur-Kopplung (Phase 1b). Nur in .cpp-Dateien
// einbinden (Eigen). Theorie: docs/theory/maxwell-trace-coupling.md, Abschnitte 3.3, 5.2, 5.3.
#include <Eigen/Dense>

#include "TraceBemOps.h"
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

} // namespace trace

#endif
