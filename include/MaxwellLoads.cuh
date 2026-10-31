#ifndef MAXWELL_LOADS_CUH
#define MAXWELL_LOADS_CUH

// =============================================================================
// Maxwell-Spannungstensor auf einer Vierecksflaeche mit Hexaeder-Nachbar
// =============================================================================
//
// Rumpf von compute_maxwell_loads_kernel (TransientSolver.cu), unveraendert
// hierher verschoben. Grund: der Rumpf war nur innerhalb von
// TransientSolver.cu erreichbar, und TransientSolver.cu zieht ueber
// CoupledSolver.cuh den halben Loeser mit. Ein Test, der den PRODUKTIVEN
// Kraftkoerper auf einem gedrehten und gescherten Hexaeder pruefen will
// (tests/test_piola_production), braucht ihn ohne diesen Ballast. Der Kernel
// und der Aufrufer bleiben in TransientSolver.cu.
//
// Warum der Test das braucht: an genau dieser Stelle stand frueher die
// zeilenweise (transponierte) Piola-Transformation, und sie fiel erst ueber
// eine Querkraft F_x = 17 % von F_z auf. Auf einem achsenparallelen Quader
// ist J symmetrisch und die Transposition unsichtbar.
//
// mu_0 als eigene Konstante statt des Makros MU_0, das TransientSolver.cu
// definiert: ein Makro in einem Header wuerde mit den constexpr-Konstanten
// gleichen Namens in BiotSavart.cpp und ReferenceSolver.cu kollidieren.
// Der Ausdruck ist derselbe wie im Makro, der Wert also bitgleich.
// =============================================================================

#include "Backend.cuh"
#include "Mesh.cuh"
#include "NedelecDispatch.cuh"

#include <math.h>   // sqrt

constexpr double TRANSIENT_MU_0 = 4.0 * 3.14159265358979323846 * 1e-7;

// Rumpf gemeinsam fuer beide Pfade. Reduziert per bf::atomic_add auf sechs
// Skalare -- auf der CPU deshalb seriell starten (BF_LAUNCH_1D).
BF_HD inline void compute_maxwell_loads_kernel_body(
    int face_idx,
    int num_surface_faces,  // Anzahl Flaechen der Materialgrenzflaeche (nicht mehr die aeussere BEM-Randflaeche)
    const Mesh::QuadFace* faces,
    const Hexahedron* hex_nodes,
    const double3* current_nodes,
    const int* hex_edges,
    const int* hex_edge_signs,
    const double* A_current,
    double3 center_of_mass,
    double* out_force_x, double* out_force_y, double* out_force_z,
    double* out_torque_x, double* out_torque_y, double* out_torque_z
) {
    (void)num_surface_faces;

    Mesh::QuadFace quad = faces[face_idx];
    int parent_hex = quad.parent_hex_id;
    int local_face = quad.local_face_id;

    Hexahedron hex = hex_nodes[parent_hex];
    double3 h_nodes[8];
    for(int i = 0; i < 8; ++i) h_nodes[i] = current_nodes[hex.v[i]];

    // Hole lokale Vektorpotentiale des angrenzenden Hexaeders
    double A_loc[12];
    for(int i = 0; i < 12; ++i) {
        int g_edge = hex_edges[parent_hex * 12 + i];
        int sign = hex_edge_signs[parent_hex * 12 + i];
        if (g_edge < 0) { A_loc[i] = 0.0; continue; }
        A_loc[i] = A_current[g_edge] * sign;
    }

    // VTK Mapping von 2D-Quad Koordinaten (u,v) zu 3D-Hex Koordinaten (xi, eta, zeta)
    // Basierend auf local_face (0 bis 5)
    double3 ref_nodes[8] = {
        {-1,-1,-1}, { 1,-1,-1}, { 1, 1,-1}, {-1, 1,-1},
        {-1,-1, 1}, { 1,-1, 1}, { 1, 1, 1}, {-1, 1, 1}
    };
    
    // Die 4 lokalen Knotenindizes des Hexaeders, die diese Fläche bilden (vgl. Aufgabe 2.3)
    const int local_face_to_nodes[6][4] = {
        {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, 
        {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}
    };

    int n0 = local_face_to_nodes[local_face][0];
    int n1 = local_face_to_nodes[local_face][1];
    int n2 = local_face_to_nodes[local_face][2];
    int n3 = local_face_to_nodes[local_face][3];

    // 4-Punkt Gauss Quadratur über das 2D Quad
    const double gp = 0.5773502691896257; 
    const double gauss_pts[4][2] = { {-gp, -gp}, { gp, -gp}, { gp, gp}, {-gp, gp} };

    double force_x = 0, force_y = 0, force_z = 0;
    double torque_x = 0, torque_y = 0, torque_z = 0;

    for (int g = 0; g < 4; ++g) {
        double u = gauss_pts[g][0];
        double v = gauss_pts[g][1];

        // 1. Bilineare Interpolation im Referenzraum (finde xi, eta, zeta)
        double N0 = 0.25 * (1.0 - u) * (1.0 - v);
        double N1 = 0.25 * (1.0 + u) * (1.0 - v);
        double N2 = 0.25 * (1.0 + u) * (1.0 + v);
        double N3 = 0.25 * (1.0 - u) * (1.0 + v);

        double xi = N0*ref_nodes[n0].x + N1*ref_nodes[n1].x + N2*ref_nodes[n2].x + N3*ref_nodes[n3].x;
        double eta= N0*ref_nodes[n0].y + N1*ref_nodes[n1].y + N2*ref_nodes[n2].y + N3*ref_nodes[n3].y;
        double zeta=N0*ref_nodes[n0].z + N1*ref_nodes[n1].z + N2*ref_nodes[n2].z + N3*ref_nodes[n3].z;

        // 2. Evaluierung des B-Felds im Hexaeder an der Stelle (xi, eta, zeta)
        double J[3][3], invJ[3][3], detJ_hex;
        compute_jacobian(h_nodes, xi, eta, zeta, J, invJ, detJ_hex);

        double3 curlW[12];
        evaluate_nedelec_curl(xi, eta, zeta, curlW);

        double3 B_field = make_double3(0,0,0);
        for(int i = 0; i < 12; ++i) {
            // KONTRAVARIANTE Piola-Transformation, SPALTENweise.
            //
            // compute_jacobian() baut J TRANSPONIERT auf -- der Kommentar dort
            // sagt es woertlich: J[i][j] = d x_j / d xi_i, also J = J_std^T.
            // Die kontravariante Transformation lautet
            //     rot W_phys = (1/detJ) * J_std * rot W_ref
            // und mit J_std = J^T ist das die SPALTENweise Form ueber das hier
            // vorliegende J.
            //
            // Hier stand die zeilenweise Form. Derselbe Fehler wurde in der
            // K-Assemblierung (fem.cu) bereits gefunden und dort mit einer
            // Nachrechnung am Hexaeder 1200 der Innenspule belegt -- der
            // Kraftpfad ist bei der Korrektur uebersehen worden.
            //
            // Wirkung: J^T statt J mischt die Komponenten von B ueber Kreuz.
            // Bei einer achsensymmetrischen Anordnung wandert damit ein Teil
            // der axialen Komponente in die Querrichtungen, und genau das war
            // zu sehen: F_x = 17 Prozent von F_z beim Maxwell-Pfad, beim
            // Volumenintegral sogar 64-mal groesser als F_z, wo beide
            // praktisch null sein muessen.
            //
            // Die Formel steht nur noch in piola_contravariant
            // (NedelecDispatch.cuh); tests/test_piola_production prueft
            // diesen Koerper auf einem gedrehten und gescherten Hexaeder.
            const double3 r_curl = piola_contravariant(J, detJ_hex, curlW[i]);

            B_field.x += A_loc[i] * r_curl.x;
            B_field.y += A_loc[i] * r_curl.y;
            B_field.z += A_loc[i] * r_curl.z;
        }

        // 3. Flächennormale n und physikalischer Punkt r_phys berechnen
        double3 p0 = current_nodes[quad.v[0]];
        double3 p1 = current_nodes[quad.v[1]];
        double3 p2 = current_nodes[quad.v[2]];
        double3 p3 = current_nodes[quad.v[3]];

        double3 r_phys = make_double3(
            N0*p0.x + N1*p1.x + N2*p2.x + N3*p3.x,
            N0*p0.y + N1*p1.y + N2*p2.y + N3*p3.y,
            N0*p0.z + N1*p1.z + N2*p2.z + N3*p3.z
        );

        double3 dr_du = make_double3(
            0.25 * (-(1-v)*p0.x + (1-v)*p1.x + (1+v)*p2.x - (1+v)*p3.x),
            0.25 * (-(1-v)*p0.y + (1-v)*p1.y + (1+v)*p2.y - (1+v)*p3.y),
            0.25 * (-(1-v)*p0.z + (1-v)*p1.z + (1+v)*p2.z - (1+v)*p3.z)
        );

        double3 dr_dv = make_double3(
            0.25 * (-(1-u)*p0.x - (1+u)*p1.x + (1+u)*p2.x + (1-u)*p3.x),
            0.25 * (-(1-u)*p0.y - (1+u)*p1.y + (1+u)*p2.y + (1-u)*p3.y),
            0.25 * (-(1-u)*p0.z - (1+u)*p1.z + (1+u)*p2.z + (1-u)*p3.z)
        );

        double3 normal = make_double3(
            dr_du.y * dr_dv.z - dr_du.z * dr_dv.y,
            dr_du.z * dr_dv.x - dr_du.x * dr_dv.z,
            dr_du.x * dr_dv.y - dr_du.y * dr_dv.x
        );

        double dS = sqrt(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        double3 n_unit = make_double3(normal.x / dS, normal.y / dS, normal.z / dS);

        // 4. Maxwell Spannungstensor
        double B_dot_n = B_field.x*n_unit.x + B_field.y*n_unit.y + B_field.z*n_unit.z;
        double B_sq = B_field.x*B_field.x + B_field.y*B_field.y + B_field.z*B_field.z;

        // Kraftdichte: f = (1 / mu0) * [ B * (B \cdot n) - 0.5 * |B|^2 * n ]
        double3 f_density = make_double3(
            (B_field.x * B_dot_n - 0.5 * B_sq * n_unit.x) / TRANSIENT_MU_0,
            (B_field.y * B_dot_n - 0.5 * B_sq * n_unit.y) / TRANSIENT_MU_0,
            (B_field.z * B_dot_n - 0.5 * B_sq * n_unit.z) / TRANSIENT_MU_0
        );

        // Integration (Gauss Gewicht = 1.0 * dS)
        force_x += f_density.x * dS;
        force_y += f_density.y * dS;
        force_z += f_density.z * dS;

        // 5. Drehmoment M = (r - r_cm) x f
        double3 r_lever = make_double3(
            r_phys.x - center_of_mass.x,
            r_phys.y - center_of_mass.y,
            r_phys.z - center_of_mass.z
        );

        torque_x += (r_lever.y * f_density.z - r_lever.z * f_density.y) * dS;
        torque_y += (r_lever.z * f_density.x - r_lever.x * f_density.z) * dS;
        torque_z += (r_lever.x * f_density.y - r_lever.y * f_density.x) * dS;
    }

    // Sicheres Schreiben in globale Variablen per atomarer Addition
    bf::atomic_add(out_force_x, force_x);
    bf::atomic_add(out_force_y, force_y);
    bf::atomic_add(out_force_z, force_z);

    bf::atomic_add(out_torque_x, torque_x);
    bf::atomic_add(out_torque_y, torque_y);
    bf::atomic_add(out_torque_z, torque_z);
}

#endif // MAXWELL_LOADS_CUH
