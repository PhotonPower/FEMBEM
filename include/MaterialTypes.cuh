#ifndef MATERIAL_TYPES_CUH
#define MATERIAL_TYPES_CUH

#include <cuda_runtime.h>

// Typen von physikalischen Regionen
enum class RegionType {
    AIR = 0,
    LINEAR_METAL = 1,
    NONLINEAR_METAL = 2,
    COIL_CURRENT = 3,
    COIL_VOLTAGE = 4
};

// Struktur, die exakt EINER Materialregion zugeordnet wird.
// Diese Struktur kann bedenkenlos an CUDA-Kernels übergeben werden.
// WICHTIG: alle Felder haben Vorbelegungen. Die abgeleiteten Klassen in
// PhysicalObject.h setzen jeweils nur die fuer sie relevanten (LinearMetal etwa
// nur type, kappa, mu_relative). Solange die Stromdichtefelder von niemandem
// gelesen wurden, war das harmlos. Seit assemble_local_matrices_kernel den
// Quellterm Integral(W_i . J_s) auswertet, liest der Kernel sie fuer JEDEN
// Hexaeder -- ein uninitialisiertes current_is_azimuthal (bool, kann jedes
// Bitmuster haben) und Muell als Stromdichte fuehrten zu einem unsinnigen
// rechten Seitenvektor und damit zu "BiCGStab breakdown: rho == 0".
struct MaterialProperties {
    RegionType type = RegionType::AIR;
    
    double kappa = 0.0;        // Elektrische Leitfähigkeit [S/m]
    double mu_relative = 1.0;  // Relative Permeabilität (für lineare Materialien)
    
    // Für Spulen: eingeprägte Quellstromdichte J_s
    //
    // current_is_azimuthal == false: J_s ist der konstante Vektor
    //   (current_density_x, current_density_y, current_density_z).
    //
    // current_is_azimuthal == true: die drei Komponenten tragen nur den BETRAG
    //   im x-Feld, und J_s = |J| * phi_hat um die z-Achse durch
    //   current_axis_origin. Das ist die physikalisch richtige Wahl für
    //   Ringspulen wie bei TEAM 28: nur ein azimutaler Strom ist im Spulengebiet
    //   divergenzfrei, denn div(f(r) * phi_hat) = (1/r) * d/dphi f = 0. Eine
    //   konstante kartesische Richtung ist es nicht -- der Strom müsste durch
    //   die Spulenoberfläche ein- und austreten, und die A-Formulierung setzt
    //   div J_s = 0 voraus.
    double current_density_x = 0.0;
    double current_density_y = 0.0;
    double current_density_z = 0.0;
    bool   current_is_azimuthal = false;
    double3 current_axis_origin = {0.0, 0.0, 0.0};
    
    // Kinematik (wird vom CPU-Objekt aktualisiert und an die GPU geschickt)
    double3 center_of_mass = {0.0, 0.0, 0.0};
    double3 translation = {0.0, 0.0, 0.0};
    double rot_matrix[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    bool is_moving = false;
};

#endif // MATERIAL_TYPES_CUH