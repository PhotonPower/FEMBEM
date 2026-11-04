#ifndef BEM_ROW_CHECK_RUN_H
#define BEM_ROW_CHECK_RUN_H

#include <vector>

#include "BemCoupling.h"
#include "CoupledSolver.cuh"
#include "Mesh.cuh"
#include "fem.cuh"

// Gegenprobe 2 im produktiven harmonischen BEM-FEM-Pfad (--bem-row-check).
// Nur lesend: wendet die bestehenden Operatoren auf das exakte Biot-Savart-Paar
// an und druckt Residuentabellen (siehe docs/design/bem-row-check.md).
//
//   op          fertig aufgesetzter Operator (BEM-Kopplung, setup_bem_mesh)
//   body_of_hex Koerperkennung je Hexaeder: 1 Platte, 2 Innen-, 3 Aussenspule
//   csr_W       W = K + omega*C (physikalisch), csr_C die Massenmatrix C
//   b_phys      rechte Seite im physikalischen Raum (Laenge total_dofs)
//   spacing_m   Fadenabstand der Spulenquelle
//
// Rueckgabe 0 bei Erfolg. Aendert weder op noch die Matrizen.
int run_bem_row_check(CoupledMVOperator& op, Mesh& mesh, const BemCouplingMap& map,
                      const std::vector<int>& body_of_hex, CSRMatrix& csr_W, CSRMatrix& csr_C,
                      double omega, const std::vector<double>& b_phys, double spacing_m);

#endif  // BEM_ROW_CHECK_RUN_H
