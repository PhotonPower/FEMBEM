#ifndef HARMONIC_MAXWELL_CUH
#define HARMONIC_MAXWELL_CUH

// =============================================================================
// Zeitgemittelter Maxwell-Spannungstensor aus dem Phasor -- Oberflaechenpfad
// =============================================================================
//
// WOZU, WENN HarmonicLorentz.cuh SCHON DIE KRAFT LIEFERT
//
// Als UNABHAENGIGE Gegenprobe, nicht als Ersatz. Beide berechnen dieselbe
// Groesse ueber voellig verschiedene Wege: das Volumenintegral wertet J x B
// im Leiter aus, dieses hier integriert Spannungen ueber die Huelle. Stimmen
// sie ueberein, haengt die Kraftzahl an keinem Formelfehler in einem der
// beiden Pfade. Weichen sie ab, weiss man, wo zu suchen ist.
//
// Der Kopf von HarmonicLorentz.cuh nennt gute Gruende, dem Oberflaechenpfad
// WENIGER zu trauen: er bildet die Nettokraft als Differenz grosser, sich
// fast aufhebender Flaechenbeitraege. Das bleibt richtig. Eine Gegenprobe
// muss nicht genauer sein als das Geprueft; sie muss nur unabhaengig sein.
//
// WARUM NICHT compute_maxwell_loads AUS TransientSolver.cu
//
// Der dortige Kernel setzt Vierecks-Panels mit Hexaeder-Nachbarn voraus --
// 2x2-Gauss, local_face_to_nodes fuer Hexaeder, bilineare Flaechenabbildung.
// Auf einem Tetraedernetz gibt es Dreiecksflaechen, und er rechnet dann still
// falsch. Er prueft das ab und wirft. Diese Fassung arbeitet ueber
// element_local_face_nodes und element_reference_node fuer alle Elementtypen,
// mit einer Dreiecksregel auf Dreiecken und 2x2-Gauss auf Vierecken.
//
// DIE FORMEL
//
// Mit B(t) = B_r cos(wt) - B_i sin(wt), <cos^2> = <sin^2> = 1/2 und
// <sin cos> = 0 folgt aus  T.n = (1/mu_0) [ (B.n) B - 1/2 |B|^2 n ]
//
//   <T.n> = (1/(2 mu_0)) [ (B_r.n) B_r + (B_i.n) B_i
//                          - 1/2 (|B_r|^2 + |B_i|^2) n ]
//
// EIGENSCHAFT, AN DER SICH DIESER KERNEL PRUEFEN LAESST
//
// Fuer ein HOMOGENES B verschwindet das Integral ueber jede GESCHLOSSENE
// Flaeche exakt. Damit ist der Kernel gegen einen analytisch bekannten Fall
// pruefbar, und zwar genau in der Groesse, die hier gefaehrlich ist: der
// Ausloeschung. Ein Vorzeichenfehler in der Normale kehrt die Kraft um, ohne
// sonst aufzufallen -- siehe tests/test_patch_assembly.
//
// NORMALENRICHTUNG
//
// Geometrisch bestimmt, aus dem Kreuzprodukt der Flaechenkanten und dann
// gegen die Elementmitte orientiert: "weg von der Mitte des Elternelements"
// ist "aus der Platte heraus", weil parent_hex_id per Konstruktion zur
// Komponente gehoert. Nicht aus einer Tabelle, weil die
// Umlaufsinnkonvention von element_local_face_nodes je Elementtyp anders ist
// und ein Fehler dort nur das Vorzeichen dreht.
//
// AUSWERTUNGSSEITE
//
// Der Spannungstensor gehoert in die Luft direkt aussen. Weil mu_r == 1
// UEBERALL gilt (Voraussetzung dieses Loesers), ist B ueber die
// Plattenoberflaeche vollstaendig stetig, Normal- und Tangentialanteil. Die
// Auswertung vom Plattenelement aus liefert deshalb denselben Wert. Bei
// mu_r != 1 waere das falsch.
// =============================================================================

#include <math.h>   // sqrt -- im Geraetecode da, beim Host-Uebersetzen nicht

#include "Backend.cuh"
#include "MaterialTypes.cuh"
#include "Mesh.cuh"
#include "NedelecDispatch.cuh"

constexpr double HARMONIC_MU_0 = 4.0e-7 * 3.14159265358979323846;

/// Referenzkoordinaten des lokalen Knotens i. Die Konventionen sind je
/// Elementtyp verschieden und muessen zu den Formfunktionen passen:
///   TET   N = {1-r-s-t, r, s, t}          -> (0,0,0) (1,0,0) (0,1,0) (0,0,1)
///   PRISM L = {1-r-s, r, s} mal (1-+z)/2  -> Dreieck in (r,s), z in [-1,1]
///   HEX   Tensorprodukt                   -> (+-1, +-1, +-1)
BF_HD inline void element_reference_node(int type, int i,
                                         double& r, double& s, double& t)
{
    if (type == ELEM_TET) {
        const double tbl[4][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
        r = tbl[i][0]; s = tbl[i][1]; t = tbl[i][2];
        return;
    }
    if (type == ELEM_PRISM) {
        const double tri[3][2] = {{0,0},{1,0},{0,1}};
        r = tri[i % 3][0]; s = tri[i % 3][1];
        t = (i < 3) ? -1.0 : 1.0;
        return;
    }
    const double hex[8][3] = {
        {-1,-1,-1},{ 1,-1,-1},{ 1, 1,-1},{-1, 1,-1},
        {-1,-1, 1},{ 1,-1, 1},{ 1, 1, 1},{-1, 1, 1}};
    r = hex[i][0]; s = hex[i][1]; t = hex[i][2];
}

/// Zeitgemittelter Maxwell-Spannungstensor ueber eine Flaeche, aufsummiert.
///
/// `A_r` und `A_i` sind die globalen Kantengrade von Real- und Imaginaerteil.
BF_HD inline void harmonic_maxwell_body(
    int face_idx, int num_faces,
    const Mesh::QuadFace* faces,
    const Hexahedron* hex_nodes, const double3* nodes,
    const int* hex_edges, const int* hex_edge_signs,
    const int* elem_types,                     // null = alles Hexaeder
    const double* A_r, const double* A_i,
    double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    if (face_idx >= num_faces) return;

    const Mesh::QuadFace face = faces[face_idx];
    const int parent = face.parent_hex_id;
    const int etype  = element_type_or_hex(elem_types, parent);
    const int n_nodes = element_num_nodes(etype);
    const int n_edges = element_num_edges(etype);

    const Hexahedron hex = hex_nodes[parent];
    double3 e_nodes[8];
    for (int i = 0; i < n_nodes; ++i) e_nodes[i] = nodes[hex.v[i]];
    for (int i = n_nodes; i < 8; ++i) e_nodes[i] = make_double3(0.0, 0.0, 0.0);

    // Mitte des Elternelements -- nur zur Orientierung der Normale
    double3 e_center = make_double3(0.0, 0.0, 0.0);
    for (int i = 0; i < n_nodes; ++i) {
        e_center.x += e_nodes[i].x / n_nodes;
        e_center.y += e_nodes[i].y / n_nodes;
        e_center.z += e_nodes[i].z / n_nodes;
    }

    double ar[12] = {0.0}, ai[12] = {0.0};
    for (int i = 0; i < n_edges; ++i) {
        const int g = hex_edges[parent * 12 + i];
        if (g < 0) continue;
        const double sg = hex_edge_signs[parent * 12 + i];
        ar[i] = A_r[g] * sg;
        ai[i] = A_i[g] * sg;
    }

    int lf[4];
    const int nfn = element_local_face_nodes(etype, face.local_face_id, lf);

    // Referenzkoordinaten und Weltkoordinaten der Flaechenknoten
    double3 fr[4], fx[4];
    for (int i = 0; i < nfn; ++i) {
        double r, s, t;
        element_reference_node(etype, lf[i], r, s, t);
        fr[i] = make_double3(r, s, t);
        fx[i] = e_nodes[lf[i]];
    }

    // Quadratur. Auf dem Dreieck die 3-Punkt-Regel (genau bis Grad 2), auf
    // dem Viereck 2x2-Gauss. Die Gewichte sind so normiert, dass sie sich zu
    // 1 summieren; die Flaeche kommt ueber das Kreuzprodukt dazu.
    double qa[4], qb[4], qw[4];
    int nq;
    if (nfn == 3) {
        nq = 3;
        const double a = 2.0/3.0, b = 1.0/6.0;
        const double L[3][2] = {{a,b},{b,a},{b,b}};
        for (int g = 0; g < 3; ++g) { qa[g] = L[g][0]; qb[g] = L[g][1]; qw[g] = 1.0/3.0; }
    } else {
        nq = 4;
        const double gp = 0.5773502691896257;
        const double P[4][2] = {{-gp,-gp},{gp,-gp},{gp,gp},{-gp,gp}};
        for (int g = 0; g < 4; ++g) { qa[g] = P[g][0]; qb[g] = P[g][1]; qw[g] = 0.25; }
    }

    double f_x = 0, f_y = 0, f_z = 0, m_x = 0, m_y = 0, m_z = 0;

    for (int g = 0; g < nq; ++g) {
        // Formfunktionen der FLAECHE und ihre Ableitungen, um daraus sowohl
        // den Quadraturpunkt als auch die Flaechennormale zu bekommen.
        double N[4], dNa[4], dNb[4];
        if (nfn == 3) {
            const double l1 = qa[g], l2 = qb[g];
            N[0] = l1; N[1] = l2; N[2] = 1.0 - l1 - l2;
            dNa[0] = 1; dNa[1] = 0; dNa[2] = -1;
            dNb[0] = 0; dNb[1] = 1; dNb[2] = -1;
        } else {
            const double u = qa[g], v = qb[g];
            N[0] = 0.25*(1-u)*(1-v); N[1] = 0.25*(1+u)*(1-v);
            N[2] = 0.25*(1+u)*(1+v); N[3] = 0.25*(1-u)*(1+v);
            dNa[0] = -0.25*(1-v); dNa[1] =  0.25*(1-v);
            dNa[2] =  0.25*(1+v); dNa[3] = -0.25*(1+v);
            dNb[0] = -0.25*(1-u); dNb[1] = -0.25*(1+u);
            dNb[2] =  0.25*(1+u); dNb[3] =  0.25*(1-u);
        }

        // Punkt im Referenzraum des ELEMENTS und in Weltkoordinaten
        double rr = 0, ss = 0, tt = 0;
        double3 pos = make_double3(0, 0, 0);
        double3 ta = make_double3(0, 0, 0), tb = make_double3(0, 0, 0);
        for (int i = 0; i < nfn; ++i) {
            rr += N[i] * fr[i].x; ss += N[i] * fr[i].y; tt += N[i] * fr[i].z;
            pos.x += N[i] * fx[i].x; pos.y += N[i] * fx[i].y; pos.z += N[i] * fx[i].z;
            ta.x += dNa[i] * fx[i].x; ta.y += dNa[i] * fx[i].y; ta.z += dNa[i] * fx[i].z;
            tb.x += dNb[i] * fx[i].x; tb.y += dNb[i] * fx[i].y; tb.z += dNb[i] * fx[i].z;
        }

        // Flaechenelement aus dem Kreuzprodukt der Tangenten
        double3 nv = make_double3(ta.y*tb.z - ta.z*tb.y,
                                  ta.z*tb.x - ta.x*tb.z,
                                  ta.x*tb.y - ta.y*tb.x);
        double dS = sqrt(nv.x*nv.x + nv.y*nv.y + nv.z*nv.z);
        if (!(dS > 0.0)) continue;
        nv.x /= dS; nv.y /= dS; nv.z /= dS;
        // Auf dem Dreieck ist |ta x tb| bereits die doppelte Flaeche, und die
        // Gewichte summieren zu 1 -- der Faktor 1/2 gehoert also dazu.
        const double area = (nfn == 3) ? 0.5 * dS : dS;

        // Nach AUSSEN orientieren, weg von der Mitte des Elternelements.
        const double outward = nv.x * (pos.x - e_center.x)
                             + nv.y * (pos.y - e_center.y)
                             + nv.z * (pos.z - e_center.z);
        if (outward < 0.0) { nv.x = -nv.x; nv.y = -nv.y; nv.z = -nv.z; }

        // B = rot A, kontravariante Piola-Transformation SPALTENweise ueber J
        // und geteilt durch detJ -- dieselbe Form wie in
        // assemble_local_matrices_kernel und harmonic_lorentz_body.
        double Jm[3][3], invJ[3][3], detJ;
        compute_element_jacobian(etype, e_nodes, rr, ss, tt, Jm, invJ, detJ);
        double3 curlW[12];
        evaluate_element_curl(etype, rr, ss, tt, curlW);

        double3 Br = make_double3(0, 0, 0), Bi = make_double3(0, 0, 0);
        for (int i = 0; i < n_edges; ++i) {
            const double cx = (Jm[0][0]*curlW[i].x + Jm[1][0]*curlW[i].y + Jm[2][0]*curlW[i].z) / detJ;
            const double cy = (Jm[0][1]*curlW[i].x + Jm[1][1]*curlW[i].y + Jm[2][1]*curlW[i].z) / detJ;
            const double cz = (Jm[0][2]*curlW[i].x + Jm[1][2]*curlW[i].y + Jm[2][2]*curlW[i].z) / detJ;
            Br.x += ar[i]*cx; Br.y += ar[i]*cy; Br.z += ar[i]*cz;
            Bi.x += ai[i]*cx; Bi.y += ai[i]*cy; Bi.z += ai[i]*cz;
        }

        const double bn_r = Br.x*nv.x + Br.y*nv.y + Br.z*nv.z;
        const double bn_i = Bi.x*nv.x + Bi.y*nv.y + Bi.z*nv.z;
        const double b2   = Br.x*Br.x + Br.y*Br.y + Br.z*Br.z
                          + Bi.x*Bi.x + Bi.y*Bi.y + Bi.z*Bi.z;

        const double k = 1.0 / (2.0 * HARMONIC_MU_0);
        const double tx = k * (bn_r*Br.x + bn_i*Bi.x - 0.5*b2*nv.x);
        const double ty = k * (bn_r*Br.y + bn_i*Bi.y - 0.5*b2*nv.y);
        const double tz = k * (bn_r*Br.z + bn_i*Bi.z - 0.5*b2*nv.z);

        const double w = qw[g] * area;
        f_x += tx * w; f_y += ty * w; f_z += tz * w;

        const double lx = pos.x - center_of_mass.x;
        const double ly = pos.y - center_of_mass.y;
        const double lz = pos.z - center_of_mass.z;
        m_x += (ly*tz - lz*ty) * w;
        m_y += (lz*tx - lx*tz) * w;
        m_z += (lx*ty - ly*tx) * w;
    }

    bf::atomic_add(out_fx, f_x); bf::atomic_add(out_fy, f_y); bf::atomic_add(out_fz, f_z);
    bf::atomic_add(out_mx, m_x); bf::atomic_add(out_my, m_y); bf::atomic_add(out_mz, m_z);
}

#ifdef __CUDACC__
static __global__ void harmonic_maxwell(
    int num_faces,
    const Mesh::QuadFace* faces,
    const Hexahedron* hex_nodes, const double3* nodes,
    const int* hex_edges, const int* hex_edge_signs,
    const int* elem_types,
    const double* A_r, const double* A_i,
    double3 center_of_mass,
    double* out_fx, double* out_fy, double* out_fz,
    double* out_mx, double* out_my, double* out_mz)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    harmonic_maxwell_body(f, num_faces, faces, hex_nodes, nodes,
                          hex_edges, hex_edge_signs, elem_types,
                          A_r, A_i, center_of_mass,
                          out_fx, out_fy, out_fz, out_mx, out_my, out_mz);
}
#endif

#endif  // HARMONIC_MAXWELL_CUH
