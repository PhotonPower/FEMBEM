#include "NonlinearSolver.cuh"
#include <iostream>
#include <cmath>

NewtonRaphsonSolver::NewtonRaphsonSolver(int N) : total_dofs(N) {
    d_residual.allocate(N);
    d_delta_A.allocate(N);
}

void NewtonRaphsonSolver::solve_nonlinear_step(
    Mesh& mesh, 
    MaterialProperties* d_materials, 
    CoupledMVOperator& jacobian_op, 
    BiCGStabSolver& linear_solver,
    double* d_A_current,
    double* d_b_rhs, 
    double tol, 
    int max_newton_iter) 
{
    // Der nichtlineare Loeser ist unfertig: die Assemblierungsschleife weiter
    // unten ist auskommentiert und mainloop() ruft ihn nicht auf. Bis dahin
    // bleiben diese Parameter ungenutzt; explizit markiert, damit die Warnungen
    // die echten nicht ueberdecken.
    (void)mesh; (void)d_materials; (void)d_b_rhs;
    cublasHandle_t cublas = jacobian_op.get_cublas_handle();
    const double c_minus_one = -1.0;
    const double c_one = 1.0;

    for (int iter = 0; iter < max_newton_iter; ++iter) {
        
        // 1. Matrizen leeren und auf der GPU neu aufbauen (Pseudocode)
        // jacobian_op.reset_fem_matrix();
        // launch_linear_assembly_kernel(...);
        // launch_nonlinear_jacobian_kernel(..., d_A_current, local_Jacobian, local_Residual);
        // scatter_to_csr(local_Jacobian, ...);
        // scatter_to_vector(local_Residual, d_residual.data());

        // 2. Das totale Residuum R(A) = F_FEM(A) + F_BEM(A) - b_rhs berechnen
        // BEM ist linear: F_BEM(A) = Matrix_BEM * A
        // Wir nutzen den Operator temporär, um nur den BEM-Teil auf d_A_current anzuwenden
        // (Die FEM-Matrix im Operator muss dafür temporär auf Null gesetzt werden)
        
        // ... r_total = F_FEM + F_BEM - b_rhs ...
        
        double res_norm = 0.0;
        bf::dnrm2(cublas, total_dofs, d_residual.data(), &res_norm);
        
        std::cout << "  [Newton] Iter " << iter << " | Residual Norm: " << res_norm << "\n";
        
        if (res_norm < tol) {
            std::cout << "[Newton] Converged.\n";
            break;
        }

        // 3. Löse das lineare System: J(A_k) * delta_A = -R(A_k)
        // Negiere das Residuum für die rechte Seite
        bf::dscal(cublas, total_dofs, c_minus_one, d_residual.data());
        
        CUDA_CHECK_ERROR(cudaMemset(d_delta_A.data(), 0, total_dofs * sizeof(double)));
        
        // Der BiCGStab nutzt den aktuellen jacobian_op
        linear_solver.solve(jacobian_op, d_residual.data(), d_delta_A.data(), 1e-4, 500);

        // 4. Update der Lösung: A_{k+1} = A_k + delta_A
        bf::daxpy(cublas, total_dofs, c_one, d_delta_A.data(), d_A_current);
    }
}