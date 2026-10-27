// =============================================================================
// Abnahmepruefung: typabhaengige Auswertung (AP7)
//
//   cmake --build build --config Release --target test_element_dispatch
//   ./build/test_element_dispatch
//
// Reiner Host-Code, keine GPU.
//
// WORUM ES GEHT
// -------------
// Die Auswertung von Basis, Rotation, Jacobi-Matrix, Formfunktionen und
// Quadratur haengt am Elementtyp und wird an mindestens fuenf Stellen
// gebraucht: Assemblierung, B-Feld-Ausgabe, Lorentzkraft,
// Maxwell-Spannungstensor und die Referenzloeser. Stand vor AP7 hatte jede
// Stelle ihre eigene, hexaederspezifische Fassung.
//
// Jetzt steht die Verzweigung nur noch in NedelecDispatch.cuh, und alle
// Aufrufer benutzen sie. Dieser Test prueft den Kopf selbst, denn ein Fehler
// darin schlaegt jetzt an allen fuenf Stellen gleichzeitig durch -- und zwar
// als falsche Kraft, nicht als Absturz.
//
// DIE ENTSCHEIDENDE PRUEFUNG
// --------------------------
// Nummer 4 ist genau das, was VtkExporter und HarmonicLorentz tun: aus
// Kantenkoeffizienten ein Feld rekonstruieren. Ein konstantes B muss dabei
// exakt herauskommen, fuer JEDEN Elementtyp, an den Quadraturpunkten UND im
// Mittelpunkt. Beim Tetraeder gilt das auf jeder Elementform, bei Hexaeder
// und Prisma nur auf affinen -- der Test benutzt deshalb einen Quader und ein
// gerades Prisma, siehe die Begruendung in tests/test_nedelec_prism.cu.
// =============================================================================

#include "NedelecDispatch.cuh"

#include <cstdio>
#include <cmath>

static int g_failures = 0;

static void check(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  %-52s %10.3e  (Grenze %8.1e)  %s\n",
                name, err, tol, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

static const char* type_name(int t) {
    return (t == ELEM_TET) ? "Tetraeder" : (t == ELEM_PRISM) ? "Prisma" : "Hexaeder";
}

// -----------------------------------------------------------------------------
// 1. Der Kopf muss dasselbe liefern wie die Elementheader direkt
// -----------------------------------------------------------------------------
static void test_dispatch_agrees() {
    double worst_basis = 0.0, worst_curl = 0.0, worst_shape = 0.0;

    for (int q = 0; q < TET_NUM_QUAD_POINTS; ++q) {
        double r, s, t, w;
        tet_quadrature_point(q, r, s, t, w);
        double3 a[12], b[12], ca[12], cb[12];
        double na[8], nb[8] = {0};
        evaluate_nedelec_basis_tet(r, s, t, a);
        evaluate_element_basis(ELEM_TET, r, s, t, b);
        evaluate_nedelec_curl_tet(r, s, t, ca);
        evaluate_element_curl(ELEM_TET, r, s, t, cb);
        tet_shape_functions(r, s, t, na);
        element_shape_functions(ELEM_TET, r, s, t, nb);
        for (int i = 0; i < 12; ++i) {
            worst_basis = std::fmax(worst_basis, std::fabs(a[i].x - b[i].x)
                                  + std::fabs(a[i].y - b[i].y) + std::fabs(a[i].z - b[i].z));
            worst_curl = std::fmax(worst_curl, std::fabs(ca[i].x - cb[i].x)
                                 + std::fabs(ca[i].y - cb[i].y) + std::fabs(ca[i].z - cb[i].z));
        }
        for (int i = 0; i < 4; ++i)
            worst_shape = std::fmax(worst_shape, std::fabs(na[i] - nb[i]));
    }

    for (int q = 0; q < PRISM_NUM_QUAD_POINTS; ++q) {
        double r, s, t, w;
        prism_quadrature_point(q, r, s, t, w);
        double3 a[12], b[12], ca[12], cb[12];
        evaluate_nedelec_basis_prism(r, s, t, a);
        evaluate_element_basis(ELEM_PRISM, r, s, t, b);
        evaluate_nedelec_curl_prism(r, s, t, ca);
        evaluate_element_curl(ELEM_PRISM, r, s, t, cb);
        for (int i = 0; i < 12; ++i) {
            worst_basis = std::fmax(worst_basis, std::fabs(a[i].x - b[i].x)
                                  + std::fabs(a[i].y - b[i].y) + std::fabs(a[i].z - b[i].z));
            worst_curl = std::fmax(worst_curl, std::fabs(ca[i].x - cb[i].x)
                                 + std::fabs(ca[i].y - cb[i].y) + std::fabs(ca[i].z - cb[i].z));
        }
    }

    // Hexaeder: die 8 Gausspunkte des Kopfes muessen dieselben sein wie die
    // frueher in fem.cu von Hand aufgeschriebenen.
    const double gp = 0.5773502691896257;
    const double ref[8][3] = {
        {-gp,-gp,-gp},{gp,-gp,-gp},{-gp,gp,-gp},{gp,gp,-gp},
        {-gp,-gp,gp},{gp,-gp,gp},{-gp,gp,gp},{gp,gp,gp}
    };
    double worst_pts = 0.0;
    for (int q = 0; q < 8; ++q) {
        double r, s, t, w;
        element_quadrature_point(ELEM_HEX, q, r, s, t, w);
        worst_pts = std::fmax(worst_pts, std::fabs(r - ref[q][0]));
        worst_pts = std::fmax(worst_pts, std::fabs(s - ref[q][1]));
        worst_pts = std::fmax(worst_pts, std::fabs(t - ref[q][2]));
        worst_pts = std::fmax(worst_pts, std::fabs(w - 1.0));
    }

    check("Basis: Kopf == Elementheader", worst_basis, 0.0);
    check("Rotation: Kopf == Elementheader", worst_curl, 0.0);
    check("Formfunktionen: Kopf == Elementheader", worst_shape, 0.0);
    check("Hexaeder-Gausspunkte unveraendert", worst_pts, 0.0);
}

// -----------------------------------------------------------------------------
// 2. Quadraturgewichte summieren auf das Referenzvolumen
// -----------------------------------------------------------------------------
static void test_quadrature_weights() {
    const int types[3] = {ELEM_TET, ELEM_PRISM, ELEM_HEX};
    const double expect[3] = {1.0 / 6.0, 1.0, 8.0};
    for (int k = 0; k < 3; ++k) {
        double sum = 0.0;
        for (int q = 0; q < element_num_quad_points(types[k]); ++q) {
            double r, s, t, w;
            element_quadrature_point(types[k], q, r, s, t, w);
            sum += w;
        }
        char buf[96];
        std::snprintf(buf, sizeof buf, "%s: Gewichtssumme = Referenzvolumen",
                      type_name(types[k]));
        check(buf, std::fabs(sum - expect[k]) / expect[k], 1e-15);
    }
}

// -----------------------------------------------------------------------------
// 3. Formfunktionen: Zerlegung der Eins
// -----------------------------------------------------------------------------
static void test_partition_of_unity() {
    const int types[3] = {ELEM_TET, ELEM_PRISM, ELEM_HEX};
    for (int k = 0; k < 3; ++k) {
        const int type = types[k];
        double worst = 0.0;
        for (int q = 0; q < element_num_quad_points(type); ++q) {
            double r, s, t, w;
            element_quadrature_point(type, q, r, s, t, w);
            double N[8];
            element_shape_functions(type, r, s, t, N);
            double sum = 0.0;
            for (int i = 0; i < element_num_nodes(type); ++i) sum += N[i];
            worst = std::fmax(worst, std::fabs(sum - 1.0));
        }
        // auch im Mittelpunkt
        double r, s, t;
        element_reference_center(type, r, s, t);
        double N[8];
        element_shape_functions(type, r, s, t, N);
        double sum = 0.0;
        for (int i = 0; i < element_num_nodes(type); ++i) sum += N[i];
        worst = std::fmax(worst, std::fabs(sum - 1.0));

        char buf[96];
        std::snprintf(buf, sizeof buf, "%s: Summe der Formfunktionen = 1", type_name(type));
        check(buf, worst, 1e-15);
    }
}

// -----------------------------------------------------------------------------
// 4. Konstantes B aus Kantenkoeffizienten -- was VtkExporter und
//    HarmonicLorentz tatsaechlich tun
// -----------------------------------------------------------------------------
static double field_reconstruction_error(int type, const double3 nodes[8]) {
    const double3 B = make_double3(0.37, -0.81, 0.52);
    const int n_edges = element_num_edges(type);

    // Kantenkoeffizienten aus A = 0.5 * (B x r), linear, also exakt
    // A(Kantenmitte) . (x_B - x_A)
    double a[12] = {0.0};
    for (int e = 0; e < n_edges; ++e) {
        int vA, vB;
        element_local_edge_nodes(type, e, vA, vB);
        const double3 xa = nodes[vA], xb = nodes[vB];
        const double mx = 0.5*(xa.x+xb.x), my = 0.5*(xa.y+xb.y), mz = 0.5*(xa.z+xb.z);
        const double Ax = 0.5*(B.y*mz - B.z*my);
        const double Ay = 0.5*(B.z*mx - B.x*mz);
        const double Az = 0.5*(B.x*my - B.y*mx);
        a[e] = Ax*(xb.x-xa.x) + Ay*(xb.y-xa.y) + Az*(xb.z-xa.z);
    }

    double worst = 0.0;

    // an allen Quadraturpunkten UND im Mittelpunkt
    for (int q = 0; q <= element_num_quad_points(type); ++q) {
        double r, s, t, w;
        if (q < element_num_quad_points(type)) {
            element_quadrature_point(type, q, r, s, t, w);
        } else {
            element_reference_center(type, r, s, t);
        }

        double J[3][3], invJ[3][3], detJ;
        compute_element_jacobian(type, nodes, r, s, t, J, invJ, detJ);
        double3 curlW[12];
        evaluate_element_curl(type, r, s, t, curlW);

        double3 Bh = make_double3(0, 0, 0);
        for (int i = 0; i < n_edges; ++i) {
            const double3 c = piola_contravariant(J, detJ, curlW[i]);
            Bh.x += a[i]*c.x; Bh.y += a[i]*c.y; Bh.z += a[i]*c.z;
        }
        worst = std::fmax(worst, std::fabs(Bh.x - B.x));
        worst = std::fmax(worst, std::fabs(Bh.y - B.y));
        worst = std::fmax(worst, std::fabs(Bh.z - B.z));
    }
    return worst;
}

static void test_field_reconstruction() {
    // Tetraeder: immer affin, jede Form ist erlaubt
    const double3 tet[8] = {
        {0,0,0},{4,0,0},{0.6,3,0},{0.3,0.9,2.5},
        {0,0,0},{0,0,0},{0,0,0},{0,0,0}
    };
    check("Tetraeder: konstantes B exakt", field_reconstruction_error(ELEM_TET, tet), 1e-13);

    // Prisma: gerade, also affin (Deckflaeche ist Parallelverschiebung)
    const double3 prism[8] = {
        {0,0,0},{4,0,0},{0,3,0},{0,0,2.5},{4,0,2.5},{0,3,2.5},{0,0,0},{0,0,0}
    };
    check("Prisma (affin): konstantes B exakt", field_reconstruction_error(ELEM_PRISM, prism), 1e-13);

    // Hexaeder: Parallelepiped, also affin
    const double3 hex[8] = {
        {0,0,0},{4,0,0},{4,3,0},{0,3,0},
        {1.1,0.4,2.5},{5.1,0.4,2.5},{5.1,3.4,2.5},{1.1,3.4,2.5}
    };
    check("Hexaeder (affin): konstantes B exakt", field_reconstruction_error(ELEM_HEX, hex), 1e-13);
}

// -----------------------------------------------------------------------------
// 5. Volumen aus der Quadratur
// -----------------------------------------------------------------------------
static void test_volume() {
    const double3 tet[8] = {{0,0,0},{4,0,0},{0,3,0},{0,0,2.5},{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
    const double3 prism[8] = {{0,0,0},{4,0,0},{0,3,0},{0,0,2.5},{4,0,2.5},{0,3,2.5},{0,0,0},{0,0,0}};
    const double3 hex[8] = {{0,0,0},{4,0,0},{4,3,0},{0,3,0},{0,0,2.5},{4,0,2.5},{4,3,2.5},{0,3,2.5}};

    struct Case { int type; const double3* n; double vol; const char* label; };
    const Case cases[3] = {
        {ELEM_TET,   tet,   4.0*3.0*2.5/6.0, "Tetraeder: Volumen"},
        {ELEM_PRISM, prism, 0.5*4.0*3.0*2.5,  "Prisma: Volumen"},
        {ELEM_HEX,   hex,   4.0*3.0*2.5,      "Hexaeder: Volumen"}
    };

    for (int k = 0; k < 3; ++k) {
        double v = 0.0;
        for (int q = 0; q < element_num_quad_points(cases[k].type); ++q) {
            double r, s, t, w;
            element_quadrature_point(cases[k].type, q, r, s, t, w);
            double J[3][3], invJ[3][3], detJ;
            compute_element_jacobian(cases[k].type, cases[k].n, r, s, t, J, invJ, detJ);
            v += w * detJ;
        }
        check(cases[k].label, std::fabs(v - cases[k].vol) / cases[k].vol, 1e-14);
    }
}

// -----------------------------------------------------------------------------
// 6. Mittelpunkt: das Bild muss der Schwerpunkt sein
// -----------------------------------------------------------------------------
static void test_center() {
    const double3 tet[8] = {{0,0,0},{4,0,0},{0,3,0},{0,0,2.5},{0,0,0},{0,0,0},{0,0,0},{0,0,0}};
    double r, s, t;
    element_reference_center(ELEM_TET, r, s, t);
    double N[8];
    element_shape_functions(ELEM_TET, r, s, t, N);
    double3 c = make_double3(0, 0, 0);
    for (int i = 0; i < 4; ++i) { c.x += N[i]*tet[i].x; c.y += N[i]*tet[i].y; c.z += N[i]*tet[i].z; }
    const double err = std::fabs(c.x - 1.0) + std::fabs(c.y - 0.75) + std::fabs(c.z - 0.625);
    check("Tetraedermittelpunkt = Schwerpunkt", err, 1e-15);
}

int main() {
    std::printf("\n== Kopf gegen Elementheader ==\n");
    test_dispatch_agrees();

    std::printf("\n== Quadratur ==\n");
    test_quadrature_weights();
    test_volume();

    std::printf("\n== Formfunktionen ==\n");
    test_partition_of_unity();
    test_center();

    std::printf("\n== Feldrekonstruktion (was VtkExporter und Lorentz tun) ==\n");
    test_field_reconstruction();

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
