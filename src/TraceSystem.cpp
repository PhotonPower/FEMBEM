#include "TraceSystem.h"

namespace trace {

Mat as_matrix(const std::vector<double>& d, int rows, int cols)
{
    Mat M(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j) M(i, j) = d[static_cast<size_t>(i) * cols + j];
    return M;
}

DerivedOps derive_ops(const TraceSurface& S, const BoundaryOps& O, bool flat)
{
    DerivedOps D;
    D.G = as_matrix(S.G_dense(), O.E, O.V);
    D.R = as_matrix(S.R_dense(), O.F, O.E);
    D.H = as_matrix(S.H, O.E, S.nH);
    const Mat Vr = as_matrix(flat ? O.VrotFlat : O.Vrot, O.E, O.E);
    const Mat Cr = as_matrix(O.Crot, O.E, O.E), Mr = as_matrix(O.Mrot, O.E, O.E);
    D.W = D.G.transpose() * Vr * D.G;
    // Z = (-G)^T Vrot H: lambda = n x sum c_e W_e mit c = -G psi + H c_h, also Kreuzblock der Gram-Matrix [G^T V G, -G^T V H; ., H^T V H]
    D.Z = -D.G.transpose() * Vr * D.H;
    D.Y = D.H.transpose() * Vr * D.H;
    D.Q = (0.5 * Mr + Cr) * D.H;      // <W_e, (1/2 + n x C) h_l>, h_l = n x (H_l-Feld)
    D.P = Mr * D.H;                    // P_{e,k} = <h_k, W_e> = int (n x H_k-Feld).W_e = (Mrot H)_{ek}   (Variante (J))
    D.N = 0.5 * as_matrix(O.Mt, O.F, O.V) + as_matrix(O.Dt, O.F, O.V);
    D.V0 = as_matrix(O.V0, O.F, O.F);
    return D;
}

} // namespace trace
