// =============================================================================
// Abnahmepruefung: Piola-Transformation in den PRODUKTIVEN Feldkoerpern
//
//   cmake --build build --config Release --target test_piola_production
//   ./build/test_piola_production
//
// Reiner Host-Code, keine GPU, keine Dateien.
//
// ANLASS
// ------
// compute_jacobian() (und die Varianten fuer Tetraeder und Prisma) legen die
// Jacobi-Matrix TRANSPONIERT ab: J[i][j] = d x_j / d xi_i, also J = J_std^T
// und invJ = J_std^-T. Richtig ist damit
//     W_phys     = invJ * W_ref              (zeilenweise ueber invJ)
//     rot W_phys = J^T * rot W_ref / detJ    (spaltenweise ueber J)
// Die transponierte Form stand frueher in der Assemblierung (fem.cu) und im
// Maxwell-Kraftkoerper (TransientSolver.cu) und fiel dort erst als Querkraft
// F_x = 17 % von F_z auf. Bestehende Tests sahen das nicht:
//   - test_element_dispatch prueft piola_contravariant selbst, nicht die
//     Koerper, die die Formel frueher inline abschrieben;
//   - test_patch_assembly Pruefung 6 integriert ein homogenes B ueber eine
//     GESCHLOSSENE Huelle -- ein konstantes, falsches B gibt dort ebenfalls
//     die Kraft 0;
//   - auf einem achsenparallelen Quader ist J symmetrisch, die Transposition
//     also unsichtbar.
//
// WAS GEPRUEFT WIRD
// -----------------
// Ein Element je Typ (Hexaeder, Tetraeder, Prisma), AFFIN, aber gedreht und
// geschert, damit J deutlich unsymmetrisch ist. Kantengrade aus
//     A(x) = 1/2 b x x + c,   a_e = Int_e A . dx = A(Kantenmitte).(x_B - x_A)
// (exakt, weil A linear ist). Auf einem affinen Element liegt dieses A im
// Nedelec-Raum erster Art, also muss JEDER Koerper B = b und A = A(x) exakt,
// bis auf Rundung, zurueckgeben. Geprueft werden die PRODUKTIVEN Koerper:
//   1. compute_cell_centered_fields_kernel_body (VtkExporter): A, B, J im
//      Mittelpunkt, alle drei Typen;
//   2. compute_maxwell_loads_kernel_body (TransientSolver): Kraft und Moment
//      je Hexaederflaeche gegen (1/mu0)[b(b.n) - 1/2 |b|^2 n] * Flaeche;
//   3. harmonic_lorentz_body: zeitgemittelte Lorentzkraft und -moment gegen
//      die exakten Felder an denselben Quadraturpunkten, alle drei Typen;
//   4. harmonic_maxwell_body: zeitgemittelter Spannungstensor je Flaeche,
//      alle drei Typen.
// Toleranz 1e-12 relativ zur jeweiligen Bezugsgroesse; gemessen wird
// Rundungsniveau (2e-16 bis 1,4e-15, Stand 31.10.2026).
//
// Pruefung 4 hat beim ersten Lauf einen zweiten, von der Piola-Frage
// unabhaengigen Fehler gefunden: harmonic_maxwell_body gewichtete jede
// VIERECKflaeche mit einem Viertel (|ta x tb| bei u, v in [-1, 1] ist ein
// Viertel der Flaeche, die Gewichte summierten zu 1). Fehler je Flaeche
// 0,25 (Hexaeder) und 0,27 (Prisma) relativ zur Bezugsgroesse, auf den
// Dreiecksflaechen des Tetraeders 9e-16. Behoben in HarmonicMaxwell.cuh.
//
// KONTROLLFALL
// ------------
// Dieselbe Rekonstruktion mit absichtlich transponierter Formel (rot W
// zeilenweise ueber J, W spaltenweise ueber invJ) muss durchfallen, ebenso
// die Maxwell-Flaechenkraft daraus. Erwartete Ausgabe dort: DURCHGEFALLEN.
// Faellt der Kontrollfall NICHT durch, ist das Element zu symmetrisch, um
// die Transposition zu sehen, und der Test endet mit 1.
//
// NICHT ABGEDECKT
// ---------------
// verify_bz_nedelec_body und die Assemblierung (fem.cu) sowie der Koerper in
// NonlinearSolver.cu sind nur innerhalb ihrer Uebersetzungseinheit
// erreichbar. Sie rufen dieselben Helfer piola_covariant/piola_contravariant
// auf, die hier ueber die vier Koerper und in test_element_dispatch direkt
// geprueft sind.
// =============================================================================

#include "Backend.cuh"
#include "CellCenteredFields.cuh"
#include "HarmonicLorentz.cuh"
#include "HarmonicMaxwell.cuh"
#include "MaxwellLoads.cuh"
#include "NedelecDispatch.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

static int g_failures = 0;

static void check(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  %-60s %10.3e  (Grenze %8.1e)  %s\n",
                name, err, tol, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

// Kontrollfall: muss die Grenze REISSEN. Ein NaN zaehlt nicht als Riss.
static void check_control(const char* name, double err, double tol) {
    const bool failed_as_expected = std::isfinite(err) && (err > tol);
    std::printf("  %-60s %10.3e  (Grenze %8.1e)  %s\n", name, err, tol,
                failed_as_expected ? "DURCHGEFALLEN (erwartet)"
                                   : "BESTANDEN -- Kontrollfall wirkungslos");
    if (!failed_as_expected) ++g_failures;
}

static const char* type_name(int t) {
    return (t == ELEM_TET) ? "Tetraeder" : (t == ELEM_PRISM) ? "Prisma" : "Hexaeder";
}

// --- kleine Vektorhelfer ------------------------------------------------------
static double3 add(double3 a, double3 b) { return make_double3(a.x+b.x, a.y+b.y, a.z+b.z); }
static double3 sub(double3 a, double3 b) { return make_double3(a.x-b.x, a.y-b.y, a.z-b.z); }
static double3 scl(double s, double3 a)  { return make_double3(s*a.x, s*a.y, s*a.z); }
static double  dot(double3 a, double3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static double3 crs(double3 a, double3 b) {
    return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static double  nrm(double3 a) { return std::sqrt(dot(a, a)); }
static double  dist(double3 a, double3 b) { return nrm(sub(a, b)); }

// -----------------------------------------------------------------------------
// Geometrie: x = x0 + M * xi, M = Drehung * Scherung
// -----------------------------------------------------------------------------
// Drehung um eine schiefe Achse (nicht um eine Koordinatenachse, damit alle
// Komponenten mischen) mal einer oberen Dreiecksmatrix mit Scherung und
// ungleicher Streckung. det(M) = 1 * 1.3 * 0.8 > 0, Orientierung erhalten.
struct Affine {
    double M[3][3];
    double3 x0;
    double3 map(double3 xi) const {
        return make_double3(x0.x + M[0][0]*xi.x + M[0][1]*xi.y + M[0][2]*xi.z,
                            x0.y + M[1][0]*xi.x + M[1][1]*xi.y + M[1][2]*xi.z,
                            x0.z + M[2][0]*xi.x + M[2][1]*xi.y + M[2][2]*xi.z);
    }
    double det() const {
        return M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
             - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
             + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
    }
};

static Affine make_affine() {
    // Rodrigues: Achse (1,2,3)/|.|, Winkel 0.7 rad
    const double th = 0.7;
    double ax = 1.0, ay = 2.0, az = 3.0;
    const double an = std::sqrt(ax*ax + ay*ay + az*az);
    ax /= an; ay /= an; az /= an;
    const double c = std::cos(th), s = std::sin(th), C = 1.0 - c;
    const double R[3][3] = {
        {c + ax*ax*C,    ax*ay*C - az*s, ax*az*C + ay*s},
        {ay*ax*C + az*s, c + ay*ay*C,    ay*az*C - ax*s},
        {az*ax*C - ay*s, az*ay*C + ax*s, c + az*az*C}};
    const double S[3][3] = {{1.0, 0.6, 0.2}, {0.0, 1.3, 0.4}, {0.0, 0.0, 0.8}};
    Affine a;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            a.M[i][j] = 0.0;
            for (int k = 0; k < 3; ++k) a.M[i][j] += R[i][k] * S[k][j];
        }
    // Massstab 1e-2 m wie die Netze des Loesers, versetzt vom Ursprung, damit
    // der konstante Anteil c und das Moment nicht trivial werden.
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) a.M[i][j] *= 0.01;
    a.x0 = make_double3(0.031, -0.012, 0.047);
    return a;
}

// Ein Element samt "Netz" aus genau diesem Element: Knoten 0..n-1, lokale
// Kante i = globale Kante i mit Vorzeichen +1.
struct OneElement {
    int type;
    Affine G;
    int n_nodes, n_edges;
    std::vector<double3> nodes;
    Hexahedron hex;
    std::vector<int> edges, signs;   // je 12 Eintraege, wie im Loeser
    double3 ref_node(int i) const {
        double r, s, t;
        element_reference_node(type, i, r, s, t);
        return make_double3(r, s, t);
    }
};

static OneElement make_element(int type, const Affine& G) {
    OneElement e;
    e.type = type; e.G = G;
    e.n_nodes = element_num_nodes(type);
    e.n_edges = element_num_edges(type);
    e.nodes.resize(e.n_nodes);
    for (int i = 0; i < 8; ++i) e.hex.v[i] = (i < e.n_nodes) ? i : 0;
    for (int i = 0; i < e.n_nodes; ++i) e.nodes[i] = G.map(e.ref_node(i));
    e.edges.assign(12, -1);
    e.signs.assign(12, 1);
    for (int i = 0; i < e.n_edges; ++i) e.edges[i] = i;
    return e;
}

// Lineares Feld A(x) = 1/2 b x x + c, rot A = b
struct LinearField {
    double3 b, c;
    double3 A(double3 x) const { return add(scl(0.5, crs(b, x)), c); }
};

// Kantengrade a_e = Int_e A . dx, exakt fuer lineares A
static std::vector<double> edge_dofs(const OneElement& e, const LinearField& f) {
    std::vector<double> a(e.n_edges);
    for (int i = 0; i < e.n_edges; ++i) {
        int vA, vB;
        element_local_edge_nodes(e.type, i, vA, vB);
        const double3 xa = e.nodes[vA], xb = e.nodes[vB];
        a[i] = dot(f.A(scl(0.5, add(xa, xb))), sub(xb, xa));
    }
    return a;
}

// -----------------------------------------------------------------------------
// Absichtlich falsche (transponierte) Formeln fuer den Kontrollfall
// -----------------------------------------------------------------------------
static double3 piola_contravariant_transposed(const double J[3][3], double detJ, double3 c) {
    return make_double3((J[0][0]*c.x + J[0][1]*c.y + J[0][2]*c.z) / detJ,
                        (J[1][0]*c.x + J[1][1]*c.y + J[1][2]*c.z) / detJ,
                        (J[2][0]*c.x + J[2][1]*c.y + J[2][2]*c.z) / detJ);
}
static double3 piola_covariant_transposed(const double invJ[3][3], double3 w) {
    return make_double3(invJ[0][0]*w.x + invJ[1][0]*w.y + invJ[2][0]*w.z,
                        invJ[0][1]*w.x + invJ[1][1]*w.y + invJ[2][1]*w.z,
                        invJ[0][2]*w.x + invJ[1][2]*w.y + invJ[2][2]*w.z);
}

// A und B im Referenzpunkt mit der transponierten Formel
static void reconstruct_transposed(const OneElement& e, const std::vector<double>& a,
                                   double3 xi, double3& A, double3& B) {
    double J[3][3], invJ[3][3], detJ;
    compute_element_jacobian(e.type, e.nodes.data(), xi.x, xi.y, xi.z, J, invJ, detJ);
    double3 W[12], C[12];
    evaluate_element_basis(e.type, xi.x, xi.y, xi.z, W);
    evaluate_element_curl(e.type, xi.x, xi.y, xi.z, C);
    A = make_double3(0, 0, 0); B = make_double3(0, 0, 0);
    for (int i = 0; i < e.n_edges; ++i) {
        A = add(A, scl(a[i], piola_covariant_transposed(invJ, W[i])));
        B = add(B, scl(a[i], piola_contravariant_transposed(J, detJ, C[i])));
    }
}

// Ausgangsfelder
static const LinearField F_REAL = {make_double3( 0.30, -1.10,  0.70), make_double3( 2.0e-3,  5.0e-3, -4.0e-3)};
static const LinearField F_IMAG = {make_double3(-0.45,  0.25,  0.90), make_double3(-3.0e-3,  1.0e-3,  6.0e-3)};
static const LinearField F_PREV = {make_double3( 0.10,  0.40, -0.20), make_double3( 1.0e-3, -2.0e-3,  3.0e-3)};

// Aussenliegende Flaeche f: Mittelpunkt, Flaeche, aeussere Einheitsnormale.
// Alle Flaechen sind eben (affines Element); Viereck = Parallelogramm.
static void face_geometry(const OneElement& e, int f, double3& centroid,
                          double& area, double3& n_out) {
    int lf[4];
    const int nfn = element_local_face_nodes(e.type, f, lf);
    centroid = make_double3(0, 0, 0);
    for (int i = 0; i < nfn; ++i) centroid = add(centroid, scl(1.0 / nfn, e.nodes[lf[i]]));
    const double3 p0 = e.nodes[lf[0]], p1 = e.nodes[lf[1]];
    const double3 pl = e.nodes[lf[nfn - 1]];
    const double3 cr = crs(sub(p1, p0), sub(pl, p0));
    area = (nfn == 3) ? 0.5 * nrm(cr) : nrm(cr);
    n_out = scl(1.0 / nrm(cr), cr);
    // Unabhaengig von der Tabellenkonvention nach aussen drehen
    double3 ec = make_double3(0, 0, 0);
    for (int i = 0; i < e.n_nodes; ++i) ec = add(ec, scl(1.0 / e.n_nodes, e.nodes[i]));
    if (dot(n_out, sub(centroid, ec)) < 0.0) n_out = scl(-1.0, n_out);
}

// Maxwell-Zug (1/mu0)[b(b.n) - 1/2|b|^2 n]
static double3 traction(double3 b, double3 n, double mu0) {
    return scl(1.0 / mu0, sub(scl(dot(b, n), b), scl(0.5 * dot(b, b), n)));
}

// -----------------------------------------------------------------------------
// 1. VtkExporter: Zellmittelwerte A, B, J
// -----------------------------------------------------------------------------
static void test_cell_centered(int type, const Affine& G) {
    const OneElement e = make_element(type, G);
    const std::vector<double> ac = edge_dofs(e, F_REAL);
    const std::vector<double> ap = edge_dofs(e, F_PREV);

    MaterialProperties mat;
    mat.kappa = 3.5e7;
    mat.current_density_x = 1.0e5; mat.current_density_y = -2.0e5; mat.current_density_z = 0.5e5;
    const double dt = 1.0e-3;
    const int etype = type;

    double3 Ao, Bo, Jo;
    compute_cell_centered_fields_kernel_body(
        0, 1, dt, &e.hex, e.nodes.data(), e.edges.data(), e.signs.data(),
        &etype, &mat, ac.data(), ap.data(), &Ao, &Bo, &Jo);

    double r, s, t;
    element_reference_center(type, r, s, t);
    const double3 xc = G.map(make_double3(r, s, t));
    const double3 A_ex = F_REAL.A(xc);
    const double3 dA = scl(1.0 / dt, sub(F_REAL.A(xc), F_PREV.A(xc)));
    const double3 J_ex = add(scl(-mat.kappa, dA),
        make_double3(mat.current_density_x, mat.current_density_y, mat.current_density_z));

    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: VtkExporter B = b", type_name(type));
    check(buf, dist(Bo, F_REAL.b) / nrm(F_REAL.b), 1e-12);
    std::snprintf(buf, sizeof buf, "%s: VtkExporter A = A(x_Mitte)", type_name(type));
    check(buf, dist(Ao, A_ex) / nrm(A_ex), 1e-12);
    std::snprintf(buf, sizeof buf, "%s: VtkExporter J = -kappa dA/dt + J_s", type_name(type));
    check(buf, dist(Jo, J_ex) / nrm(J_ex), 1e-12);

    // Kontrollfall: dieselbe Auswertung mit transponierten Formeln
    double3 At, Bt;
    reconstruct_transposed(e, ac, make_double3(r, s, t), At, Bt);
    std::snprintf(buf, sizeof buf, "%s: Kontrolle transponiert, B", type_name(type));
    check_control(buf, dist(Bt, F_REAL.b) / nrm(F_REAL.b), 1e-12);
    std::snprintf(buf, sizeof buf, "%s: Kontrolle transponiert, A", type_name(type));
    check_control(buf, dist(At, A_ex) / nrm(A_ex), 1e-12);
}

// -----------------------------------------------------------------------------
// 2. TransientSolver: Maxwell-Spannungstensor je Hexaederflaeche
// -----------------------------------------------------------------------------
static void test_maxwell_transient(const Affine& G) {
    const OneElement e = make_element(ELEM_HEX, G);
    const std::vector<double> a = edge_dofs(e, F_REAL);
    const double3 com = make_double3(0.02, 0.01, 0.03);

    // Bezugsgroessen fuer die relativen Fehler
    double area_max = 0.0;
    for (int f = 0; f < 6; ++f) {
        double3 c, n; double ar;
        face_geometry(e, f, c, ar, n);
        area_max = std::fmax(area_max, ar);
    }
    const double F_ref = dot(F_REAL.b, F_REAL.b) / TRANSIENT_MU_0 * area_max;
    double3 ec = make_double3(0, 0, 0);
    for (int i = 0; i < 8; ++i) ec = add(ec, scl(0.125, e.nodes[i]));
    const double M_ref = F_ref * (dist(ec, com) + std::sqrt(area_max));

    double errF = 0.0, errM = 0.0, errCtl = 0.0;
    double3 sumF = make_double3(0, 0, 0);
    for (int f = 0; f < 6; ++f) {
        // QuadFace wie TopologyExtractor: v[] in der Reihenfolge von
        // local_face_to_nodes, damit Normale und Referenzabbildung zusammen-
        // passen.
        int lf[4];
        element_local_face_nodes(ELEM_HEX, f, lf);
        Mesh::QuadFace q;
        for (int i = 0; i < 4; ++i) q.v[i] = e.hex.v[lf[i]];
        q.parent_hex_id = 0;
        q.local_face_id = f;

        double fx = 0, fy = 0, fz = 0, mx = 0, my = 0, mz = 0;
        compute_maxwell_loads_kernel_body(
            0, 1, &q, &e.hex, e.nodes.data(), e.edges.data(), e.signs.data(),
            a.data(), com, &fx, &fy, &fz, &mx, &my, &mz);

        double3 c, n; double ar;
        face_geometry(e, f, c, ar, n);
        const double3 F_ex = scl(ar, traction(F_REAL.b, n, TRANSIENT_MU_0));
        const double3 M_ex = crs(sub(c, com), F_ex);   // Zug homogen, Flaeche eben
        errF = std::fmax(errF, dist(make_double3(fx, fy, fz), F_ex) / F_ref);
        errM = std::fmax(errM, dist(make_double3(mx, my, mz), M_ex) / M_ref);
        sumF = add(sumF, make_double3(fx, fy, fz));

        // Kontrollfall: Zug aus dem transponiert rekonstruierten B im
        // Flaechenmittelpunkt (B ist dort ohnehin konstant)
        double r = 0, s = 0, t = 0;
        for (int i = 0; i < 4; ++i) {
            double rr, ss, tt;
            element_reference_node(ELEM_HEX, lf[i], rr, ss, tt);
            r += 0.25 * rr; s += 0.25 * ss; t += 0.25 * tt;
        }
        double3 At, Bt;
        reconstruct_transposed(e, a, make_double3(r, s, t), At, Bt);
        const double3 F_t = scl(ar, traction(Bt, n, TRANSIENT_MU_0));
        errCtl = std::fmax(errCtl, dist(F_t, F_ex) / F_ref);
    }
    check("Hexaeder: Maxwell (TransientSolver) Kraft je Flaeche", errF, 1e-12);
    check("Hexaeder: Maxwell (TransientSolver) Moment je Flaeche", errM, 1e-12);
    check("Hexaeder: Maxwell (TransientSolver) Summe ueber Huelle = 0", nrm(sumF) / F_ref, 1e-12);
    check_control("Hexaeder: Kontrolle transponiert, Flaechenkraft", errCtl, 1e-12);
}

// -----------------------------------------------------------------------------
// 3. harmonic_lorentz_body: <J x B> = (sigma w/2)(A_i x B_r - A_r x B_i)
// -----------------------------------------------------------------------------
static void test_harmonic_lorentz(int type, const Affine& G) {
    const OneElement e = make_element(type, G);
    const std::vector<double> ar = edge_dofs(e, F_REAL);
    const std::vector<double> ai = edge_dofs(e, F_IMAG);
    MaterialProperties mat;
    mat.kappa = 3.5e7;
    const double omega = 2.0 * 3.14159265358979323846 * 50.0;
    const double3 com = make_double3(0.02, 0.01, 0.03);
    const int etype = type;

    double fx = 0, fy = 0, fz = 0, mx = 0, my = 0, mz = 0;
    harmonic_lorentz_body(0, 1, &e.hex, e.nodes.data(), e.edges.data(), e.signs.data(),
                          &etype, &mat, nullptr, ar.data(), ai.data(), omega, com,
                          &fx, &fy, &fz, &mx, &my, &mz);

    // Erwartung aus den EXAKTEN Feldern an denselben Quadraturpunkten. Die
    // Dichte ist linear in x, das Moment quadratisch -- beides integrieren
    // alle drei Regeln auf affinen Elementen exakt. |detJ| = |det M| bei
    // allen drei Typen (Referenzvolumen steckt in den Gewichten).
    const double k = 0.5 * mat.kappa * omega;
    const double detM = std::fabs(G.det());
    double3 F_ex = make_double3(0, 0, 0), M_ex = make_double3(0, 0, 0);
    double vol = 0.0;
    for (int q = 0; q < element_num_quad_points(type); ++q) {
        double r, s, t, w;
        element_quadrature_point(type, q, r, s, t, w);
        const double3 x = G.map(make_double3(r, s, t));
        const double3 d = scl(k, sub(crs(F_IMAG.A(x), F_REAL.b), crs(F_REAL.A(x), F_IMAG.b)));
        F_ex = add(F_ex, scl(w * detM, d));
        M_ex = add(M_ex, scl(w * detM, crs(sub(x, com), d)));
        vol += w * detM;
    }
    const double F_ref = k * nrm(F_REAL.b) * nrm(F_REAL.A(G.x0)) * vol;
    const double M_ref = F_ref * (dist(G.x0, com) + std::cbrt(vol));

    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: harmonic_lorentz Kraft", type_name(type));
    check(buf, dist(make_double3(fx, fy, fz), F_ex) / F_ref, 1e-12);
    std::snprintf(buf, sizeof buf, "%s: harmonic_lorentz Moment", type_name(type));
    check(buf, dist(make_double3(mx, my, mz), M_ex) / M_ref, 1e-12);
}

// -----------------------------------------------------------------------------
// 4. harmonic_maxwell_body: <T.n> = (1/(2 mu0))[(B_r.n)B_r + (B_i.n)B_i
//                                    - 1/2(|B_r|^2 + |B_i|^2) n]
// -----------------------------------------------------------------------------
static void test_harmonic_maxwell(int type, const Affine& G) {
    const OneElement e = make_element(type, G);
    const std::vector<double> ar = edge_dofs(e, F_REAL);
    const std::vector<double> ai = edge_dofs(e, F_IMAG);
    const double3 com = make_double3(0.02, 0.01, 0.03);
    const int etype = type;
    const int nf = element_num_faces(type);

    double area_max = 0.0;
    for (int f = 0; f < nf; ++f) {
        double3 c, n; double a;
        face_geometry(e, f, c, a, n);
        area_max = std::fmax(area_max, a);
    }
    const double b2 = dot(F_REAL.b, F_REAL.b) + dot(F_IMAG.b, F_IMAG.b);
    const double F_ref = b2 / (2.0 * HARMONIC_MU_0) * area_max;
    double3 ec = make_double3(0, 0, 0);
    for (int i = 0; i < e.n_nodes; ++i) ec = add(ec, scl(1.0 / e.n_nodes, e.nodes[i]));
    const double M_ref = F_ref * (dist(ec, com) + std::sqrt(area_max));

    double errF = 0.0, errM = 0.0;
    for (int f = 0; f < nf; ++f) {
        int lf[4];
        const int nfn = element_local_face_nodes(type, f, lf);
        Mesh::QuadFace q;
        for (int i = 0; i < 4; ++i) q.v[i] = (i < nfn) ? e.hex.v[lf[i]] : -1;
        q.parent_hex_id = 0;
        q.local_face_id = f;

        double fx = 0, fy = 0, fz = 0, mx = 0, my = 0, mz = 0;
        harmonic_maxwell_body(0, 1, &q, &e.hex, e.nodes.data(), e.edges.data(),
                              e.signs.data(), &etype, ar.data(), ai.data(), com,
                              &fx, &fy, &fz, &mx, &my, &mz);

        double3 c, n; double a;
        face_geometry(e, f, c, a, n);
        // Zeitmittel = halbe Summe der Zuege von Real- und Imaginaerteil
        const double3 F_ex = scl(0.5 * a, add(traction(F_REAL.b, n, HARMONIC_MU_0),
                                              traction(F_IMAG.b, n, HARMONIC_MU_0)));
        const double3 M_ex = crs(sub(c, com), F_ex);
        errF = std::fmax(errF, dist(make_double3(fx, fy, fz), F_ex) / F_ref);
        errM = std::fmax(errM, dist(make_double3(mx, my, mz), M_ex) / M_ref);
    }
    char buf[128];
    std::snprintf(buf, sizeof buf, "%s: harmonic_maxwell Kraft je Flaeche", type_name(type));
    check(buf, errF, 1e-12);
    std::snprintf(buf, sizeof buf, "%s: harmonic_maxwell Moment je Flaeche", type_name(type));
    check(buf, errM, 1e-12);
}

// -----------------------------------------------------------------------------
// 0. Vorbedingung: das Element ist unsymmetrisch genug
// -----------------------------------------------------------------------------
// Waere J (fast) symmetrisch, saehe der Kontrollfall die Transposition nicht,
// und alle Pruefungen oben waeren wertlos. Gemessen wird ||J - J^T|| / ||J||.
static void test_asymmetry(const Affine& G) {
    const OneElement e = make_element(ELEM_HEX, G);
    double J[3][3], invJ[3][3], detJ;
    compute_element_jacobian(ELEM_HEX, e.nodes.data(), 0.0, 0.0, 0.0, J, invJ, detJ);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            num += (J[i][j] - J[j][i]) * (J[i][j] - J[j][i]);
            den += J[i][j] * J[i][j];
        }
    const double asym = std::sqrt(num / den);
    std::printf("  Hexaeder: ||J - J^T|| / ||J|| = %.3f, detJ = %.3e\n", asym, detJ);
    if (!(asym > 0.1)) {
        std::printf("  FEHLGESCHLAGEN: Element zu symmetrisch fuer diesen Test\n");
        ++g_failures;
    }
}

int main() {
    bf::g_backend = bf::Backend::CPU;

    std::printf("Piola-Transformation in den produktiven Feldkoerpern\n");
    std::printf("(affines, gedrehtes und geschertes Element; A = 1/2 b x x + c)\n\n");

    const Affine G = make_affine();
    test_asymmetry(G);

    const int types[3] = {ELEM_HEX, ELEM_TET, ELEM_PRISM};

    std::printf("\n1. VtkExporter (compute_cell_centered_fields_kernel_body)\n");
    for (int t : types) test_cell_centered(t, G);

    std::printf("\n2. TransientSolver (compute_maxwell_loads_kernel_body), nur Hexaeder\n");
    test_maxwell_transient(G);

    std::printf("\n3. HarmonicLorentz (harmonic_lorentz_body)\n");
    for (int t : types) test_harmonic_lorentz(t, G);

    std::printf("\n4. HarmonicMaxwell (harmonic_maxwell_body)\n");
    for (int t : types) test_harmonic_maxwell(t, G);

    std::printf("\n%s (%d Fehler)\n", g_failures ? "FEHLGESCHLAGEN" : "BESTANDEN", g_failures);
    return g_failures ? 1 : 0;
}
