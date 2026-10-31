#include "NedelecDispatch.cuh"   // piola_contravariant; bindet NedelecHexahedron.cuh ein
#include "Backend.cuh"
#include "MaterialTypes.cuh"
#include "NonlinearSolver.cuh"

// Hilfsfunktion zur Auswertung der B-H Kurve (Binäre Suche)
// BF_HD, nicht __device__: aufgerufen wird sie aus
// assemble_nonlinear_jacobian_kernel_body(), und das ist seit der Einfuehrung
// des CPU-Modus eine __host__ __device__ Funktion. nvcc verbietet den Aufruf
// einer reinen __device__-Funktion von dort ("calling a __device__ function
// from a __host__ __device__ function is not allowed") -- und zwar
// unabhaengig davon, ob der Host-Zweig je genommen wird, denn die Pruefung
// ist rein statisch.
BF_HD inline void evaluate_bh_curve(
    const BHSplinePoint* curve, int curve_size, double b_sq, 
    double& nu, double& dnu_db_sq) 
{
    // FIX (Absicherung): curve[0] wurde ohne Pruefung auf curve_size == 0
    // gelesen. add_bh_curve() wird von mainloop() nirgends aufgerufen (der
    // nichtlineare Loeser ist unfertig, siehe src/NonlinearSolver.cpp), sodass
    // bh_curves im Live-Zustand leer ist. Wird dieser Kernel je aktiviert, ohne
    // vorher eine Kurve zu laden, waere dies ein Zugriff auf nicht
    // allozierten Speicher. Reluktivitaet von Luft (nu = 1/mu_0) als
    // Rueckfallwert, der Aufrufer bekommt aber ueber dnu_db_sq = 0 ein
    // lineares statt nichtlineares Verhalten -- korrekt fuer "keine
    // BH-Kurve vorhanden" ist eigentlich ein Abbruch, aber in Device-Code
    // kann nicht geworfen werden. Der Rueckfall bleibt auch im CPU-Modus
    // bestehen, obwohl dort eine Ausnahme moeglich waere: beide Pfade sollen
    // sich identisch verhalten, sonst waere ein Fehler nur in einem von
    // beiden sichtbar. Der Aufrufer sollte curve_size pruefen, bevor der
    // Kernel gestartet wird.
    if (curve_size <= 0) {
        nu = 1.0 / (4.0 * M_PI * 1e-7);
        dnu_db_sq = 0.0;
        return;
    }

    // Einfache lineare Interpolation zwischen den Stützstellen
    // (In einem Produktionscode würde man hier Kubische Splines implementieren)
    if (b_sq <= curve[0].b_sq) {
        nu = curve[0].nu;
        dnu_db_sq = 0.0;
        return;
    }
    
    for (int i = 0; i < curve_size - 1; ++i) {
        if (b_sq >= curve[i].b_sq && b_sq < curve[i+1].b_sq) {
            double t = (b_sq - curve[i].b_sq) / (curve[i+1].b_sq - curve[i].b_sq);
            nu = (1.0 - t) * curve[i].nu + t * curve[i+1].nu;
            dnu_db_sq = curve[i].dnu_db_sq; // Konstante Steigung in diesem Intervall
            return;
        }
    }
    
    // Starke Sättigung (Luft-Verhalten am Ende der Kurve)
    nu = curve[curve_size - 1].nu;
    dnu_db_sq = 0.0;
}

// -----------------------------------------------------------------------------
// Baut J(A) und F(A) für nichtlineare Elemente auf
// -----------------------------------------------------------------------------
// Wie alle Kernel dieses Projekts in Rumpf + Indexbildung getrennt (siehe
// include/Backend.cuh), damit der CPU-Modus (--cpu) denselben Code ausfuehrt.
// HINWEIS: Dieser Pfad ist derzeit nicht in mainloop() verdrahtet -- TEAM 28
// rechnet mit linearem Aluminium. Die Trennung ist hier vorgenommen, damit ein
// spaeteres Verdrahten nicht versehentlich einen GPU-only-Kernel in den
// CPU-Modus einschleppt.
BF_HD inline void assemble_nonlinear_jacobian_kernel_body(
    int hex_id,
    int num_hexes,
    const Hexahedron* hex_nodes,
    const double3* current_nodes,
    const MaterialProperties* materials,
    const int* hex_edges,
    const int* hex_edge_signs,
    const double* A_current, // Die globalen Freiheitsgrade der aktuellen Newton-Iteration
    const BHSplinePoint* bh_curve,
    int curve_size,
    double* local_Jacobian,  // Output: num_hexes * 144
    double* local_Residual   // Output: num_hexes * 12
) {
    (void)num_hexes;

    if (materials[hex_id].type != RegionType::NONLINEAR_METAL) {
        // Lineare Elemente werden im linearen Assembly-Kernel verarbeitet
        return; 
    }

    Hexahedron hex = hex_nodes[hex_id];
    double3 nodes[8];
    for(int i=0; i<8; ++i) nodes[i] = current_nodes[hex.v[i]];

    // Hole lokale A-Koeffizienten für dieses Element
    double A_loc[12];
    int signs[12];
    for(int i=0; i<12; ++i) {
        int global_edge = hex_edges[hex_id * 12 + i];
        // Ungenutzter Platz eines Tetraeders oder Prismas. Ohne diese Abfrage
        // waere hex_edges[..] == -1 ein Zugriff ausserhalb von A.
        if (global_edge < 0) { A_loc[i] = 0.0; signs[i] = 0; continue; }
        signs[i] = hex_edge_signs[hex_id * 12 + i];
        A_loc[i] = A_current[global_edge] * signs[i]; // Vorzeichenkorrektur anwenden!
    }

    double J_e[12][12] = {0.0};
    double F_e[12] = {0.0};

    // Gauss-Quadratur (8 Punkte)
    const double gp = 0.5773502691896257; 
    // FIX: Die Initialisierungsliste war leer (nur ein Platzhalterkommentar),
    // wodurch alle 24 Einträge 0.0 waren. Die Schleife wertete damit achtmal
    // den Elementmittelpunkt aus -- achtfache Mittelpunktsregel statt
    // 2x2x2-Gauss, Genauigkeitsordnung 1 statt 3. Deshalb war auch gp
    // ungenutzt (nvcc-Warnung #177-D).
    const double gauss_pts[8][3] = {
        {-gp, -gp, -gp}, { gp, -gp, -gp}, {-gp,  gp, -gp}, { gp,  gp, -gp},
        {-gp, -gp,  gp}, { gp, -gp,  gp}, {-gp,  gp,  gp}, { gp,  gp,  gp}
    };

    for(int g=0; g<8; ++g) {
        double xi = gauss_pts[g][0], eta = gauss_pts[g][1], zeta = gauss_pts[g][2];
        
        double J[3][3], invJ[3][3], detJ;
        compute_jacobian(nodes, xi, eta, zeta, J, invJ, detJ);
        
        double3 W[12], curlW[12];
        evaluate_nedelec_basis(xi, eta, zeta, W);
        evaluate_nedelec_curl(xi, eta, zeta, curlW);
        
        // Piola-Transformation
        double3 real_curlW[12];
        double3 B_field = make_double3(0,0,0);

        for(int i=0; i<12; ++i) {
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
            real_curlW[i] = piola_contravariant(J, detJ, curlW[i]);
            
            real_curlW[i].x *= signs[i]; real_curlW[i].y *= signs[i]; real_curlW[i].z *= signs[i];
            
            // Berechnung des B-Feldes an diesem Gauss-Punkt: B = sum(A_i * rot_W_i)
            B_field.x += A_loc[i] * real_curlW[i].x;
            B_field.y += A_loc[i] * real_curlW[i].y;
            B_field.z += A_loc[i] * real_curlW[i].z;
        }

        double b_sq = B_field.x*B_field.x + B_field.y*B_field.y + B_field.z*B_field.z;
        
        // Materialgesetz auswerten
        double nu, dnu_db_sq;
        evaluate_bh_curve(bh_curve, curve_size, b_sq, nu, dnu_db_sq);

        double weight = detJ;

        // Jacobi-Matrix und Residuum aufbauen
        for(int i=0; i<12; ++i) {
            double dot_B_curlWi = B_field.x*real_curlW[i].x + B_field.y*real_curlW[i].y + B_field.z*real_curlW[i].z;
            
            // Residuum: F_i = int( nu * B \cdot rot_W_i )
            F_e[i] += nu * dot_B_curlWi * weight;

            for(int j=0; j<12; ++j) {
                double dot_curl = real_curlW[i].x*real_curlW[j].x + real_curlW[i].y*real_curlW[j].y + real_curlW[i].z*real_curlW[j].z;
                double dot_B_curlWj = B_field.x*real_curlW[j].x + B_field.y*real_curlW[j].y + B_field.z*real_curlW[j].z;
                
                // Jacobi: J_ij = int( nu * rot_Wi \cdot rot_Wj + 2 * dnu * (B \cdot rot_Wi) * (B \cdot rot_Wj) )
                J_e[i][j] += (nu * dot_curl + 2.0 * dnu_db_sq * dot_B_curlWi * dot_B_curlWj) * weight;
            }
        }
    }

    // In globale Arrays schreiben
    int offset_J = hex_id * 144;
    int offset_F = hex_id * 12;
    for(int i=0; i<12; ++i) {
        local_Residual[offset_F + i] = F_e[i];
        for(int j=0; j<12; ++j) {
            local_Jacobian[offset_J + i*12 + j] = J_e[i][j];
        }
    }
}

__global__ void assemble_nonlinear_jacobian_kernel(
    int num_hexes,
    const Hexahedron* hex_nodes,
    const double3* current_nodes,
    const MaterialProperties* materials,
    const int* hex_edges,
    const int* hex_edge_signs,
    const double* A_current,
    const BHSplinePoint* bh_curve,
    int curve_size,
    double* local_Jacobian,
    double* local_Residual
) {
    int hex_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (hex_id >= num_hexes) return;
    assemble_nonlinear_jacobian_kernel_body(
        hex_id, num_hexes, hex_nodes, current_nodes, materials, hex_edges,
        hex_edge_signs, A_current, bh_curve, curve_size,
        local_Jacobian, local_Residual);
}
