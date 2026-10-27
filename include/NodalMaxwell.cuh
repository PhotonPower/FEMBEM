#ifndef NODAL_MAXWELL_CUH
#define NODAL_MAXWELL_CUH

// =============================================================================
// NodalMaxwell.cuh -- Maxwell-Spannungstensor fuer KNOTENfelder
// =============================================================================
// Gehoert zum Referenzloeser (--reference). Die Nedelec-Fassung in
// TransientSolver.cu liest Kantengrade und ist hier nicht verwendbar; beide
// bleiben parallel bestehen.
//
// In einem eigenen Header und NICHT im anonymen Namensraum von
// ReferenceSolver.cu, damit tests/test_reference_solver den Kernel direkt gegen
// analytisch bekannte Faelle pruefen kann. Ein Vorzeichenfehler in der Normale
// kehrt die Kraft um, ohne sonst aufzufallen -- das muss testbar sein.
// =============================================================================

#include <cuda_runtime.h>

#include "Backend.cuh"
#include "BemIntegration.cuh"
#include "MaterialTypes.cuh"
#include "Mesh.cuh"
#include "NodalBem.cuh"
#include "NodalHexahedron.cuh"

constexpr double NODAL_MU_0 = 4.0e-7 * 3.14159265358979323846;

// =============================================================================
// 4b. Maxwell-Spannungstensor fuer KNOTENfelder
// =============================================================================
//
// Integriert  F = Int_S T.n dS,  M = Int_S (x - x_cm) x (T.n) dS  ueber die
// Materialgrenzflaeche der Platte, mit
//
//     T.n = (1/mu_0) [ (B.n) B - 1/2 |B|^2 n ]
//
// B aus dem Knotenfeld:  A = sum_i N_i A_i  =>  B = curl A = sum_i grad N_i x A_i.
// (Nachrechnen: (curl A)_k = eps_kmn d_m (sum_i N_i A_i,n)
//                          = sum_i eps_kmn (d_m N_i) A_i,n = sum_i (grad N_i x A_i)_k.)
//
// Die Nedelec-Fassung in TransientSolver.cu ist hier nicht verwendbar -- sie
// liest Kantengrade. Beides bleibt parallel bestehen.
//
// Zwei Punkte, die hier stimmen muessen:
//
//   * Auswertungsseite. Der Spannungstensor gehoert eigentlich in die LUFT
//     direkt aussen. Weil mu_r == 1 UEBERALL ist (Voraussetzung dieses
//     Loesers), ist B ueber die Plattenoberflaeche vollstaendig stetig --
//     Normal- und Tangentialanteil. Die Auswertung vom Plattenelement aus
//     liefert deshalb denselben Wert. Bei mu_r != 1 waere das falsch.
//
//   * Normalenrichtung. Sie wird GEOMETRISCH bestimmt (Panelmitte gegen Mitte
//     des Elternhexaeders), nicht aus einer Tabelle. parent_hex_id gehoert per
//     Konstruktion zur Komponente, "weg von der Hexaedermitte" ist also
//     "aus der Platte heraus". Ein Vorzeichenfehler hier kehrt die Kraft um,
//     ohne sonst aufzufallen -- deshalb kein Tabellenvertrauen.
//
// Wie im produktiven Pfad wird auf sechs Skalare reduziert; im CPU-Modus
// laeuft der Kernel deshalb seriell (bf::atomic_add).
// Die Kernel sind STATIC, die _body-Funktionen inline.
//
// Ohne static hat jeder Kernel externe Bindung, und sobald der Header in mehr
// als einer .cu eingebunden wird, meldet nvlink "Multiple definition".
//
// static und nicht inline: __global__-Funktionen duerfen in CUDA nicht inline
// sein. static gibt jeder Uebersetzungseinheit ihre eigene Kopie; das kostet
// etwas Codegroesse und ist bei drei Kerneln ohne Belang.
//
// ACHTUNG BEIM WIEDERVERWENDEN: diese Kernel gehoeren zum KNOTENBASIERTEN
// Referenzpfad. Sie indizieren A[knoten*3 + komponente] und erwarten
// 3*num_nodes Werte. Der Hauptpfad rechnet mit KANTENBASIERTEN
// Nedelec-Elementen (num_edges Werte, ein Freiheitsgrad je Kante). Die beiden
// Feldrepraesentationen sind NICHT austauschbar -- ein Nedelec-Vektor an
// nodal_lorentz uebergeben liest ueber das Array hinaus und liefert Zahlen,
// die klein und plausibel aussehen statt NaN zu sein.

BF_HD inline void reference_maxwell_body(
    int face_idx,
    int num_faces,
    const Mesh::QuadFace* faces,
    const Hexahedron* hex_nodes,
    const double3* nodes,
    const double*  A,               // 3 * num_nodes
    double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    (void)num_faces;

    const Mesh::QuadFace q = faces[face_idx];

    // --- Geometrie des Elternhexaeders und seine Knotenwerte
    const Hexahedron hx = hex_nodes[q.parent_hex_id];
    double3 hn[8];
    double3 Ah[8];
    double3 hc = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < 8; ++i) {
        hn[i] = nodes[hx.v[i]];
        Ah[i] = make_double3(A[hx.v[i] * 3 + 0], A[hx.v[i] * 3 + 1], A[hx.v[i] * 3 + 2]);
        hc.x += hn[i].x; hc.y += hn[i].y; hc.z += hn[i].z;
    }
    hc.x *= 0.125; hc.y *= 0.125; hc.z *= 0.125;

    // --- Referenzkoordinaten der vier Flaechenknoten im Hexaeder.
    // Die Flaeche des Referenzwuerfels ist eben, die bilineare Interpolation
    // der Referenzkoordinaten ueber (u,v) ist daher EXAKT.
    const int lf2n[6][4] = {
        {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
        {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}
    };
    const int lf = q.local_face_id;
    double3 fref[4];
    double3 fpos[4];
    for (int k = 0; k < 4; ++k) {
        const int ln = lf2n[lf][k];
        fref[k] = make_double3(nodal_sign_xi(ln), nodal_sign_eta(ln), nodal_sign_zeta(ln));
        fpos[k] = hn[ln];
    }

    // --- Orientierung: Normale muss von der Hexaedermitte weg zeigen
    double orient = 1.0;
    {
        double3 y, nrm; double jac;
        bem_quad_geometry(fpos, 0.0, 0.0, y, nrm, jac);
        if (dot3(nrm, sub3(y, hc)) < 0.0) orient = -1.0;
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;
    double mx = 0.0, my = 0.0, mz = 0.0;

    const int ng = bem_gl_n();
    for (int gu = 0; gu < ng; ++gu) {
        for (int gv = 0; gv < ng; ++gv) {
            const double u = bem_gl_x(gu), v = bem_gl_x(gv);
            const double w = bem_gl_w(gu) * bem_gl_w(gv);

            double Nf[4];
            bem_quad_shape(u, v, Nf);

            // Referenzpunkt im Hexaeder
            double xi = 0.0, eta = 0.0, zeta = 0.0;
            for (int k = 0; k < 4; ++k) {
                xi   += Nf[k] * fref[k].x;
                eta  += Nf[k] * fref[k].y;
                zeta += Nf[k] * fref[k].z;
            }

            double3 gradN[8]; double detJ;
            evaluate_nodal_gradients(hn, xi, eta, zeta, gradN, detJ);
            if (detJ <= 0.0) continue;

            // B = sum_i grad N_i x A_i
            double3 B = make_double3(0.0, 0.0, 0.0);
            for (int i = 0; i < 8; ++i) {
                const double3 c = cross3(gradN[i], Ah[i]);
                B.x += c.x; B.y += c.y; B.z += c.z;
            }

            // Flaechenelement und Einheitsnormale
            double3 y, nrm; double jac;
            bem_quad_geometry(fpos, u, v, y, nrm, jac);
            if (jac <= 0.0) continue;
            const double3 nh = scale3(orient / jac, nrm);

            const double Bn  = dot3(B, nh);
            const double B2  = dot3(B, B);
            const double inv = 1.0 / NODAL_MU_0;
            const double3 t = make_double3(
                inv * (Bn * B.x - 0.5 * B2 * nh.x),
                inv * (Bn * B.y - 0.5 * B2 * nh.y),
                inv * (Bn * B.z - 0.5 * B2 * nh.z));

            const double dS = w * jac;
            fx += t.x * dS; fy += t.y * dS; fz += t.z * dS;

            const double3 r = sub3(y, center_of_mass);
            const double3 m = cross3(r, t);
            mx += m.x * dS; my += m.y * dS; mz += m.z * dS;
        }
    }

    bf::atomic_add(out_fx, fx); bf::atomic_add(out_fy, fy); bf::atomic_add(out_fz, fz);
    bf::atomic_add(out_mx, mx); bf::atomic_add(out_my, my); bf::atomic_add(out_mz, mz);
}

static __global__ void reference_maxwell(
    int num_faces, const Mesh::QuadFace* faces, const Hexahedron* hex_nodes,
    const double3* nodes, const double* A, double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_faces) return;
    reference_maxwell_body(f, num_faces, faces, hex_nodes, nodes, A, center_of_mass,
                           out_fx, out_fy, out_fz, out_mx, out_my, out_mz);
}


// =============================================================================
// Lorentzkraft als VOLUMENintegral -- zweite, unabhaengige Rechnung
// =============================================================================
//
//     F = Int_Leiter J x B dV,   J = -sigma dA/dt,   B = curl A
//     M = Int_Leiter (x - x_cm) x (J x B) dV
//
// Warum zusaetzlich zum Maxwell-Spannungstensor? Weil die beiden Verfahren auf
// grobem Netz sehr unterschiedlich genau sind:
//
//   * Der Spannungstensor braucht das Feld AUF der Materialgrenzflaeche, also
//     genau dort, wo die Loesung am ungenauesten ist -- und er bildet die
//     Differenz zweier grosser, fast gleicher Terme ((B.n)B gegen |B|^2 n/2).
//     Er ist deshalb bekannt empfindlich gegen Netzfehler.
//   * Das Volumenintegral mittelt ueber das ganze Leitervolumen und braucht
//     keinen Wert an der Unstetigkeitsflaeche. Fuer Wirbelstromprobleme in
//     UNMAGNETISCHEM Material (mu_r == 1, hier erfuellt) ist es das empfohlene
//     Verfahren.
//
// Stimmen beide Zahlen ueberein, ist das ein starkes Argument fuer die Loesung.
// Weichen sie ab, sagt die Abweichung, wie gut das Netz die Grenzflaeche
// aufloest -- eine Diagnose, die eine einzelne Zahl nicht liefern kann.
//
// J wird konsistent mit dem impliziten Euler des Zeitschritts gebildet:
// J^n = -sigma (A^n - A^(n-1)) / dt. Eine andere Differenz waere gegenueber
// dem geloesten System inkonsistent.
BF_HD inline void nodal_lorentz_body(
    int hex_id,
    int num_hexes,
    const Hexahedron* hex_nodes,
    const double3* nodes,
    const MaterialProperties* materials,
    const double* A,
    const double* A_prev,
    double dt,
    int quad_order,
    double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    (void)num_hexes;

    const double kappa = materials[hex_id].kappa;
    if (kappa <= 0.0) return;      // nur Leiter tragen bei

    const Hexahedron hx = hex_nodes[hex_id];
    double3 nd[8], Ah[8], Ap[8];
    for (int i = 0; i < 8; ++i) {
        nd[i] = nodes[hx.v[i]];
        Ah[i] = make_double3(A[hx.v[i]*3+0], A[hx.v[i]*3+1], A[hx.v[i]*3+2]);
        Ap[i] = make_double3(A_prev[hx.v[i]*3+0], A_prev[hx.v[i]*3+1], A_prev[hx.v[i]*3+2]);
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;
    double mx = 0.0, my = 0.0, mz = 0.0;

    const int ng = nodal_gauss_count_n(quad_order);
    for (int g = 0; g < ng; ++g) {
        double xi, eta, zeta, wg;
        nodal_gauss_point_n(quad_order, g, xi, eta, zeta, wg);

        double3 gradN[8]; double detJ;
        evaluate_nodal_gradients(nd, xi, eta, zeta, gradN, detJ);
        if (detJ <= 0.0) continue;

        double N[8];
        evaluate_nodal_shape(xi, eta, zeta, N);

        // J = -sigma * (A - A_prev)/dt, im Gausspunkt interpoliert
        double3 dAdt = make_double3(0.0, 0.0, 0.0);
        for (int i = 0; i < 8; ++i) {
            dAdt.x += N[i] * (Ah[i].x - Ap[i].x);
            dAdt.y += N[i] * (Ah[i].y - Ap[i].y);
            dAdt.z += N[i] * (Ah[i].z - Ap[i].z);
        }
        const double sc = -kappa / dt;
        const double3 J = make_double3(sc * dAdt.x, sc * dAdt.y, sc * dAdt.z);

        // B = sum_i grad N_i x A_i
        double3 B = make_double3(0.0, 0.0, 0.0);
        for (int i = 0; i < 8; ++i) {
            const double3 c = cross3(gradN[i], Ah[i]);
            B.x += c.x; B.y += c.y; B.z += c.z;
        }

        const double3 f  = cross3(J, B);
        const double  dV = wg * detJ;
        fx += f.x * dV; fy += f.y * dV; fz += f.z * dV;

        const double3 x = nodal_map_to_physical(nd, xi, eta, zeta);
        const double3 mm = cross3(sub3(x, center_of_mass), f);
        mx += mm.x * dV; my += mm.y * dV; mz += mm.z * dV;
    }

    bf::atomic_add(out_fx, fx); bf::atomic_add(out_fy, fy); bf::atomic_add(out_fz, fz);
    bf::atomic_add(out_mx, mx); bf::atomic_add(out_my, my); bf::atomic_add(out_mz, mz);
}

static __global__ void nodal_lorentz(
    int num_hexes, const Hexahedron* hex_nodes, const double3* nodes,
    const MaterialProperties* materials, const double* A, const double* A_prev,
    double dt, int quad_order, double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_hexes) return;
    nodal_lorentz_body(h, num_hexes, hex_nodes, nodes, materials, A, A_prev, dt,
                       quad_order, center_of_mass,
                       out_fx, out_fy, out_fz, out_mx, out_my, out_mz);
}


// =============================================================================
// Wirbelstromdiagnose: magnetisches Moment und Verlustleistung
// =============================================================================
//
//     m_z = 1/2 Int (x J_y - y J_x) dV      magnetisches Moment der Wirbelstroeme
//     P   = Int |J|^2 / sigma dV            ohmsche Verlustleistung
//
// Warum diese zwei Groessen? Weil die Staerke der Wirbelstroeme sonst nur
// indirekt ueber die Kraftschwankung ablesbar ist, und die ist schlecht
// konditioniert: der Gleichanteil der Kraft ist ein kleiner Rest zweier grosser
// Terme, ein Fehler von 5 % im Momentanwert wird dort zu 26 % im Zyklusmittel.
//
// m_z ist dagegen direkt und einordenbar: die Spulen haben Momente von
// +101 A m^2 (w1) und -277 A m^2 (w2), netto -176 A m^2. Das Verhaeltnis
// m_Wirbel / m_Spule ist der Abschirmfaktor, eine Groesse der Ordnung eins.
// Aus Amplitude UND Phase von m_z gegen den Spulenstrom laesst sich Q = w*tau
// direkt ablesen, ohne Umweg ueber die Kraft.
//
// J wird konsistent mit dem impliziten Euler gebildet: J = -sigma (A^n - A^(n-1))/dt.
BF_HD inline void nodal_eddy_diag_body(
    int hex_id,
    int num_hexes,
    const Hexahedron* hex_nodes,
    const double3* nodes,
    const MaterialProperties* materials,
    const double* A,
    const double* A_prev,
    double dt,
    double* out_mz,        // magnetisches Moment, z-Komponente
    double* out_power,     // Verlustleistung
    double* out_absJ)      // Int |J| dV, als Bezugsgroesse
{
    (void)num_hexes;

    const double kappa = materials[hex_id].kappa;
    if (kappa <= 0.0) return;

    const Hexahedron hx = hex_nodes[hex_id];
    double3 nd[8], dA[8];
    for (int i = 0; i < 8; ++i) {
        nd[i] = nodes[hx.v[i]];
        dA[i] = make_double3(A[hx.v[i]*3+0] - A_prev[hx.v[i]*3+0],
                             A[hx.v[i]*3+1] - A_prev[hx.v[i]*3+1],
                             A[hx.v[i]*3+2] - A_prev[hx.v[i]*3+2]);
    }

    double mz = 0.0, pw = 0.0, aj = 0.0;

    const int ng = nodal_gauss_count_n(3);
    for (int g = 0; g < ng; ++g) {
        double xi, eta, zeta, wg;
        nodal_gauss_point_n(3, g, xi, eta, zeta, wg);

        double3 gradN[8]; double detJ;
        evaluate_nodal_gradients(nd, xi, eta, zeta, gradN, detJ);
        if (detJ <= 0.0) continue;

        double N[8];
        evaluate_nodal_shape(xi, eta, zeta, N);

        double3 d = make_double3(0.0, 0.0, 0.0);
        for (int i = 0; i < 8; ++i) {
            d.x += N[i] * dA[i].x; d.y += N[i] * dA[i].y; d.z += N[i] * dA[i].z;
        }
        const double sc = -kappa / dt;
        const double3 J = make_double3(sc * d.x, sc * d.y, sc * d.z);

        const double3 x  = nodal_map_to_physical(nd, xi, eta, zeta);
        const double  dV = wg * detJ;

        mz += 0.5 * (x.x * J.y - x.y * J.x) * dV;
        pw += (J.x*J.x + J.y*J.y + J.z*J.z) / kappa * dV;
        aj += sqrt(J.x*J.x + J.y*J.y + J.z*J.z) * dV;
    }

    bf::atomic_add(out_mz, mz);
    bf::atomic_add(out_power, pw);
    bf::atomic_add(out_absJ, aj);
}

static __global__ void nodal_eddy_diag(
    int num_hexes, const Hexahedron* hex_nodes, const double3* nodes,
    const MaterialProperties* materials, const double* A, const double* A_prev,
    double dt, double* out_mz, double* out_power, double* out_absJ)
{
    int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_hexes) return;
    nodal_eddy_diag_body(h, num_hexes, hex_nodes, nodes, materials, A, A_prev, dt,
                         out_mz, out_power, out_absJ);
}


#endif // NODAL_MAXWELL_CUH
