#ifndef HARMONIC_LORENTZ_CUH
#define HARMONIC_LORENTZ_CUH

// =============================================================================
// Zeitgemittelte Lorentzkraft aus dem Phasor -- Volumenintegral, Nedelec
// =============================================================================
//
// WARUM NICHT DER MAXWELL-SPANNUNGSTENSOR
//
// Der vorhandene Oberflaechenpfad berechnet die Nettokraft als Differenz
// grosser, sich fast aufhebender Flaechenbeitraege. Wie gross der dabei
// verbleibende Diskretisierungsfehler ist, zeigt der Realteil allein:
//
//   F(A_r) = -0,122 N
//
// A_r ist im Wesentlichen das Spulenfeld, also das Feld OHNE Wirbelstroeme.
// Bei mu_r = 1 und stromfreiem Inneren muss das Oberflaechenintegral ueber
// eine geschlossene Flaeche dafuer EXAKT NULL ergeben. Es ergibt -0,122 N --
// auf 200 Flaechen eines 348-Hexaeder-Netzes ist der Auslöschungsfehler also
// groesser als das gesuchte Ergebnis, Vorzeichen eingeschlossen.
//
// Derselbe Effekt erklaert den Referenzlauf: Maxwell 0,1870 N gegen Lorentz
// 0,5014 N bei 11,3 mm, wo 1,0497 N stehen muessten.
//
// Das Volumenintegral hat diese Ausloeschung nicht: es wertet J x B dort aus,
// wo beide Groessen von null verschieden sind, naemlich im Leiter.
//
// DIE FORMEL
//
// Mit A(t) = Re(A^ e^{jwt}) = A_r cos(wt) - A_i sin(wt) folgt
//
//   J = -sigma dA/dt = sigma*w ( A_r sin(wt) + A_i cos(wt) )
//   B = rot A        = rot A_r cos(wt) - rot A_i sin(wt)
//
// Mit <cos^2> = <sin^2> = 1/2 und <sin cos> = 0 bleibt vom Zeitmittel:
//
//   <J x B> = (sigma*w/2) ( A_i x rot A_r  -  A_r x rot A_i )
//
// Kein Differenzenquotient, kein kuenstliches A_prev, keine Phasenschleife --
// ein Durchlauf ueber die Plattenhexaeder wertet A_r und A_i gemeinsam aus.
//
// Und die Formel ist genau dann von null verschieden, wenn A_i nicht
// parallel zu A_r liegt, also wenn es eine Phasenverzoegerung gibt. Ohne
// Wirbelstroeme keine Kraft -- die Physik steht der Formel an.
//
// GEGENPROBE
//
// Aus derselben Zerlegung folgt fuer den Oberflaechenpfad
//
//   <F_Maxwell> = 1/2 ( F(A_r) + F(A_i) )
//
// Das ist an den gemessenen Zahlen bestaetigt: (-0,122 - 0,0181)/2 = -0,0700
// gegen gemeldete -0,0702. Beide Verfahren muessen dasselbe liefern; tun sie
// es nicht, ist es der Oberflaechenpfad, denn seine Fehlerquelle ist
// benannt und quantifiziert.

#include "fem.cuh"
#include "NedelecHexahedron.cuh"
#include "NedelecDispatch.cuh"
#include "MaterialTypes.cuh"

/// Zeitgemittelte Lorentzkraftdichte eines Hexaeders, aufsummiert.
///
/// `excluded` markiert Elemente, die NICHT beitragen sollen (1 = ueberspringen,
/// null = keine Ausnahme).
///
/// WARUM DAS NOETIG IST -- die kappa > 0 Abfrage allein reicht nicht mehr.
/// Auf dem --fem-direct-Pfad setzt regularize_nonconducting_regions ALLEN
/// nichtleitenden Gebieten sigma = eps * sigma_Leiter, damit K auf ihren
/// Gradienten nicht singulaer ist. Danach hat jedes Element kappa > 0, und
/// der Test unten trennt nichts mehr: Luft und Spulen liefen ins
/// Volumenintegral mit.
///
/// Das ist kein kleiner Fehler -- aber auch kein grosser, und das war nicht
/// vorherzusehen. Eine Abschaetzung ueber F = |J|*|B|*V ergab 1,9 N und damit
/// mehr als die gesuchte Kraft. GEMESSEN sind es -3,8e-05 N.
///
/// Der Grund fuer den Unterschied ist strukturell: die Abschaetzung
/// unterstellt kohaerente Aufaddierung ueber das Volumen, ein Gradientenfeld
/// im Kreuzprodukt addiert sich aber nicht kohaerent. Mit
///
///   Int grad(phi) x B dV = Rand(n x phi B) - Int phi (rot B) dV
///
/// und rot B = 0 in der stromfreien Luft verschwindet der Volumenanteil
/// identisch, es bleibt ein Randterm.
///
/// Die Maske bleibt trotzdem. Sie macht den Ausschluss ausdruecklich statt
/// zufaellig, sie kostet einen zweiten Kernelaufruf, und die Scheinkraft ist
/// EPS-UNABHAENGIG (sigma_Luft faellt mit eps, der Scheingradient in A_i
/// waechst mit 1/eps, das Produkt bleibt) -- die eps-Abnahme koennte sie also
/// nicht aufdecken, wenn sie doch einmal gross wuerde.
BF_HD inline void harmonic_lorentz_body(
    int h, int num_hexes,
    const Hexahedron* hex_nodes, const double3* nodes,
    const int* hex_edges, const int* hex_edge_signs,
    const int* elem_types,                     // null = alles Hexaeder
    const MaterialProperties* materials,
    const unsigned char* excluded,             // null = keine Ausnahme
    const double* A_r, const double* A_i, double omega,
    double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    if (h >= num_hexes) return;
    if (excluded && excluded[h]) return;       // kuenstlich leitend gemacht
    const double sigma = materials[h].kappa;   // Feld heisst kappa, nicht conductivity
    if (!(sigma > 0.0)) return;   // Nichtleiter tragen nicht bei

    // Ohne die Typabfrage wuerde ein Tetraeder mit den Hexaeder-Basis-
    // funktionen ausgewertet -- also eine falsche Kraft, lautlos.
    const int etype   = element_type_or_hex(elem_types, h);
    const int n_nodes = element_num_nodes(etype);
    const int n_edges = element_num_edges(etype);

    const Hexahedron hex = hex_nodes[h];
    double3 h_nodes[8];
    for (int i = 0; i < n_nodes; ++i) h_nodes[i] = nodes[hex.v[i]];
    for (int i = n_nodes; i < 8; ++i) h_nodes[i] = make_double3(0.0, 0.0, 0.0);

    // Lokale Koeffizienten mit Kantenvorzeichen -- wie im Maxwell-Kernel.
    double ar[12], ai[12];
    for (int i = 0; i < 12; ++i) {
        const int g = hex_edges[h * 12 + i];
        const int s = hex_edge_signs[h * 12 + i];
        if (g < 0) { ar[i] = 0.0; ai[i] = 0.0; continue; }
        ar[i] = A_r[g] * s;
        ai[i] = A_i[g] * s;
    }

    // Quadratur je Elementtyp: Hexaeder 2x2x2 Gauss, Prisma 3x2, Tetraeder 4.
    double fx = 0.0, fy = 0.0, fz = 0.0;
    double mx = 0.0, my = 0.0, mz = 0.0;

    const int n_quad = element_num_quad_points(etype);

    for (int g = 0; g < n_quad; ++g) {
        double xi, eta, zeta, quad_w;
        element_quadrature_point(etype, g, xi, eta, zeta, quad_w);

        double J[3][3], invJ[3][3], detJ;
        compute_element_jacobian(etype, h_nodes, xi, eta, zeta, J, invJ, detJ);

        double3 W[12], curlW[12];
        evaluate_element_basis(etype, xi, eta, zeta, W);
        evaluate_element_curl(etype, xi, eta, zeta, curlW);

        double3 Ar = make_double3(0, 0, 0), Ai = make_double3(0, 0, 0);
        double3 Br = make_double3(0, 0, 0), Bi = make_double3(0, 0, 0);
        for (int i = 0; i < n_edges; ++i) {
            // A ist eine kovariante 1-Form: W_phys = invJ * W_ref, ZEILENweise.
            // B = rot A ist eine kontravariante 2-Form: (1/detJ) * J_std *
            // rot W_ref, und J_std = J^T, also SPALTENweise ueber das hier
            // vorliegende J.
            //
            // Die Index-Reihenfolge folgt VtkExporter.cu, das sie mit Bezug
            // auf die Aufbaukonvention von compute_jacobian() ausdruecklich
            // begruendet (J[i][j] = d x_j / d xi_i). Der Maxwell-Kernel in
            // TransientSolver.cu transformiert B dagegen ZEILENweise -- beide
            // koennen nicht recht haben. Wer diese Datei anfasst, sollte das
            // im Auge behalten; hier ist bewusst der begruendeten Variante
            // gefolgt.
            const double wx = invJ[0][0]*W[i].x + invJ[0][1]*W[i].y + invJ[0][2]*W[i].z;
            const double wy = invJ[1][0]*W[i].x + invJ[1][1]*W[i].y + invJ[1][2]*W[i].z;
            const double wz = invJ[2][0]*W[i].x + invJ[2][1]*W[i].y + invJ[2][2]*W[i].z;

            const double cx = (J[0][0]*curlW[i].x + J[1][0]*curlW[i].y + J[2][0]*curlW[i].z) / detJ;
            const double cy = (J[0][1]*curlW[i].x + J[1][1]*curlW[i].y + J[2][1]*curlW[i].z) / detJ;
            const double cz = (J[0][2]*curlW[i].x + J[1][2]*curlW[i].y + J[2][2]*curlW[i].z) / detJ;

            Ar.x += ar[i]*wx; Ar.y += ar[i]*wy; Ar.z += ar[i]*wz;
            Ai.x += ai[i]*wx; Ai.y += ai[i]*wy; Ai.z += ai[i]*wz;
            Br.x += ar[i]*cx; Br.y += ar[i]*cy; Br.z += ar[i]*cz;
            Bi.x += ai[i]*cx; Bi.y += ai[i]*cy; Bi.z += ai[i]*cz;
        }

        // <J x B> = (sigma*w/2) ( A_i x B_r - A_r x B_i )
        const double k = 0.5 * sigma * omega;
        const double dx = k * ((Ai.y*Br.z - Ai.z*Br.y) - (Ar.y*Bi.z - Ar.z*Bi.y));
        const double dy = k * ((Ai.z*Br.x - Ai.x*Br.z) - (Ar.z*Bi.x - Ar.x*Bi.z));
        const double dz = k * ((Ai.x*Br.y - Ai.y*Br.x) - (Ar.x*Bi.y - Ar.y*Bi.x));

        // Quadraturgewicht mal Volumenelement |detJ|. Der BETRAG ist
        // wesentlich: bei invertierter Knotenreihenfolge waere detJ negativ,
        // und die Kraft bekaeme fuer einen Teil der Elemente das falsche
        // Vorzeichen -- lautlos. Beim Hexaeder ist quad_w = 1 wie bisher,
        // beim Tetraeder 1/24, beim Prisma 1/6.
        const double w = quad_w * fabs(detJ);
        fx += dx * w; fy += dy * w; fz += dz * w;

        double3 r = make_double3(0, 0, 0);
        double Nsh[8];
        element_shape_functions(etype, xi, eta, zeta, Nsh);
        for (int n = 0; n < n_nodes; ++n) {
            r.x += Nsh[n]*h_nodes[n].x; r.y += Nsh[n]*h_nodes[n].y; r.z += Nsh[n]*h_nodes[n].z;
        }
        const double lx = r.x - center_of_mass.x;
        const double ly = r.y - center_of_mass.y;
        const double lz = r.z - center_of_mass.z;
        mx += (ly*dz - lz*dy) * w;
        my += (lz*dx - lx*dz) * w;
        mz += (lx*dy - ly*dx) * w;
    }

    bf::atomic_add(out_fx, fx); bf::atomic_add(out_fy, fy); bf::atomic_add(out_fz, fz);
    bf::atomic_add(out_mx, mx); bf::atomic_add(out_my, my); bf::atomic_add(out_mz, mz);
}

// static: der Header koennte aus mehreren Uebersetzungseinheiten eingebunden
// werden, und __global__-Funktionen mit externer Bindung im Header geben dann
// "Multiple definition" beim Linken -- genau der Fehler, den NodalMaxwell.cuh
// hatte.
// Der Kernel heisst harmonic_lorentz, die Rumpffunktion harmonic_lorentz_body.
// Das ist keine Kosmetik: BF_LAUNCH_1D haengt fuer den CPU-Pfad "_body" an den
// uebergebenen Namen an (KERNEL##_body). Ein Kernel harmonic_lorentz_kernel
// wuerde dort harmonic_lorentz_kernel_body suchen -- und der Fehler zeigt sich
// NUR beim Uebersetzen des Aufrufers, nicht dieses Headers.
static __global__ void harmonic_lorentz(
    int num_hexes,
    const Hexahedron* hex_nodes, const double3* nodes,
    const int* hex_edges, const int* hex_edge_signs,
    const int* elem_types,                     // null = alles Hexaeder
    const MaterialProperties* materials,
    const unsigned char* excluded,             // null = keine Ausnahme
    const double* A_r, const double* A_i, double omega,
    double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    harmonic_lorentz_body(h, num_hexes, hex_nodes, nodes, hex_edges, hex_edge_signs,
                          elem_types, materials, excluded, A_r, A_i, omega,
                          center_of_mass,
                          out_fx, out_fy, out_fz, out_mx, out_my, out_mz);
}

#endif // HARMONIC_LORENTZ_CUH
