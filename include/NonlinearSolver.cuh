#ifndef NONLINEAR_SOLVER_CUH
#define NONLINEAR_SOLVER_CUH

#include "CoupledSolver.cuh"
#include "Mesh.cuh"
#include "MaterialTypes.cuh"
#include <vector>

// -----------------------------------------------------------------------------
// Struktur für die Spline-Interpolation der B-H-Kurve auf der GPU
// -----------------------------------------------------------------------------
struct BHSplinePoint {
    double b_sq;      // |B|^2 (Vermeidet teure Wurzeln auf der GPU)
    double nu;        // Reluktivität nu(|B|)
    double dnu_db_sq; // Ableitung d(nu)/d(|B|^2)
};

class NewtonRaphsonSolver {
private:
    int total_dofs;
    
    // Arrays für den Solver (Device Memory)
    DeviceArray<double> d_residual;
    DeviceArray<double> d_delta_A;
    
    // B-H-Kurven für verschiedene nichtlineare Regionen
    std::vector<DeviceArray<BHSplinePoint>> bh_curves;

public:
    NewtonRaphsonSolver(int N);
    
    // Fügt eine materialabhängige B-H-Kurve hinzu
    void add_bh_curve(const std::vector<double>& b_values, const std::vector<double>& h_values);

    // Der BEM-Teil bleibt während der Newton-Iteration konstant!
    // Erzeugt nur einen linearen Beitrag zum Residuum.
    void solve_nonlinear_step(
        Mesh& mesh, 
        MaterialProperties* d_materials, 
        CoupledMVOperator& jacobian_op, 
        BiCGStabSolver& linear_solver,
        double* d_A_current,     // In/Out: Aktuelle Lösung A^(k)
        double* d_b_rhs,         // Rechte Seite (Quellströme, Zeitschritt-Historie)
        double tol = 1e-4, 
        int max_newton_iter = 20);
};

#endif // NONLINEAR_SOLVER_CUH