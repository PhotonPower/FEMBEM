// =============================================================================
// Abnahmeprüfung: Nédélec-Kantenelement auf dem Prisma (penta6)
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_nedelec_prism
//   ./build/test_nedelec_prism
//
// Reiner Host-Code, keine GPU, kein Netz, in Sekundenbruchteilen durch.
//
// WARUM DIESE FÜNF PRÜFUNGEN
// --------------------------
// Ein falsches Vorzeichen oder eine falsche Rotation in einem Kantenelement
// erzeugt keinen Absturz und kein NaN. Es erzeugt eine plausibel aussehende,
// systematisch falsche Kraft -- und damit eine falsche Schwebehöhe. Genau
// dieser Fehler steckte im Hexaederelement: evaluate_nedelec_curl() war für
// 10 von 12 Kanten nicht die Rotation der Basis, und es fehlte für die Kanten
// 2, 3, 6, 7 das Minuszeichen der lokalen Kantenrichtung. Beides ist erst
// durch Nachrechnen aufgefallen, nicht durch einen Lauf.
//
// Deshalb hier, in dieser Reihenfolge:
//
//   1. Kantenzirkulation: Int_{Kante i} W_j . t dl = delta_ij. Das ist die
//      definierende Eigenschaft. Sie prüft Normierung UND Orientierung.
//   2. Rotation gegen zentrale Differenzen. Genau der Hexaederfehler.
//   3. Volumen aus der Quadratur. Prüft Jacobi-Determinante und Gewichte.
//   4. Gradienten im Nullraum: K * (Knotendifferenzen) = 0, exakt. Prüft, dass
//      der diskrete Gradient im Element liegt -- die Voraussetzung dafür, dass
//      Eichung und AMS überhaupt greifen können.
//   5. Patchtest mit konstantem B. Prüft die Konsistenz der gesamten Kette aus
//      Basis, Rotation und beiden Piola-Transformationen.
//
// ZUM PATCHTEST UND ZUR AFFINITÄT
// -------------------------------
// Konstantes B wird nur auf AFFIN abgebildeten Prismen exakt reproduziert.
// Auf einem verzerrten Prisma bleibt ein Fehler, der LINEAR mit der
// Verzerrungsamplitude t geht (gemessen: t = 1 -> 1,34e-01; jede Halbierung
// von t halbiert den Fehler, Verhältnis 2,01; t = 0 -> 2,2e-16).
//
// Das ist keine Eigenheit dieses Elements, sondern der Elementfamilie. Das
// vorhandene Hexaederelement verhält sich unter demselben Test identisch:
//
//   Quader (affin)              1,1e-16
//   Parallelepiped (affin)      3,3e-16
//   verzerrt (nicht affin)      1,0e-01
//
// Der Test verlangt deshalb Exaktheit nur im affinen Fall und prüft im nicht
// affinen Fall das, was tatsächlich gelten muss: dass der Fehler mit der
// Verzerrung verschwindet. Eine Prüfung, die hier Exaktheit fordert, wäre
// falsch kalibriert und würde bei jedem realen Netz anschlagen.
//
// FOLGE FÜR DAS NETZ
// ------------------
// Bei einem extrudierten Prismennetz ist die Abbildung genau dann affin, wenn
// die Deckfläche eine Parallelverschiebung der Bodenfläche ist, die
// Extrusionsrichtung also je Element konstant bleibt. Für TEAM 28 mit
// Schichtung in z ist das erfüllbar und sollte im Netzgenerator eingehalten
// werden. Wo es nicht geht, gilt der übliche Konsistenzfehler erster Ordnung.

#include "NedelecPrism.cuh"
#include "NedelecHexahedron.cuh"

#include <cstdio>
#include <cmath>

static int g_failures = 0;

static void check(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    std::printf("  %-52s %10.3e  (Grenze %8.1e)  %s\n",
                name, err, tol, ok ? "ok" : "FEHLGESCHLAGEN");
    if (!ok) ++g_failures;
}

// -----------------------------------------------------------------------------
// Knoten des Referenzprismas
// -----------------------------------------------------------------------------
static void reference_node(int i, double& r, double& s, double& z) {
    const double R[PRISM_NUM_NODES][3] = {
        {0,0,-1}, {1,0,-1}, {0,1,-1},
        {0,0, 1}, {1,0, 1}, {0,1, 1}
    };
    r = R[i][0]; s = R[i][1]; z = R[i][2];
}

// -----------------------------------------------------------------------------
// Piola-Transformation, wortgleich zu assemble_local_matrices_kernel_body
// -----------------------------------------------------------------------------
static void piola_prism(const double3 nodes[PRISM_NUM_NODES],
                        double r, double s, double z,
                        double3 Wp[12], double3 Cp[12], double& detJ)
{
    double J[3][3], invJ[3][3];
    compute_jacobian_prism(nodes, r, s, z, J, invJ, detJ);

    double3 W[12], C[12];
    evaluate_nedelec_basis_prism(r, s, z, W);
    evaluate_nedelec_curl_prism(r, s, z, C);

    for (int i = 0; i < 12; ++i) {
        // kovariant: W_phys = invJ * W_ref (zeilenweise)
        Wp[i].x = invJ[0][0]*W[i].x + invJ[0][1]*W[i].y + invJ[0][2]*W[i].z;
        Wp[i].y = invJ[1][0]*W[i].x + invJ[1][1]*W[i].y + invJ[1][2]*W[i].z;
        Wp[i].z = invJ[2][0]*W[i].x + invJ[2][1]*W[i].y + invJ[2][2]*W[i].z;

        // kontravariant: rot W_phys = (1/detJ) * J_std * rot W_ref (spaltenweise)
        Cp[i].x = (J[0][0]*C[i].x + J[1][0]*C[i].y + J[2][0]*C[i].z) / detJ;
        Cp[i].y = (J[0][1]*C[i].x + J[1][1]*C[i].y + J[2][1]*C[i].z) / detJ;
        Cp[i].z = (J[0][2]*C[i].x + J[1][2]*C[i].y + J[2][2]*C[i].z) / detJ;
    }
}

// -----------------------------------------------------------------------------
// 1. Kantenzirkulation im Referenzraum
// -----------------------------------------------------------------------------
static void test_circulation() {
    double maxerr = 0.0;
    for (int e = 0; e < PRISM_NUM_EDGES; ++e) {
        int vA, vB;
        prism_local_edge_nodes(e, vA, vB);
        double rA, sA, zA, rB, sB, zB;
        reference_node(vA, rA, sA, zA);
        reference_node(vB, rB, sB, zB);
        const double dr = rB - rA, ds = sB - sA, dz = zB - zA;

        double circ[12] = {0.0};
        const int NQ = 2000;
        for (int q = 0; q <= NQ; ++q) {
            const double t  = static_cast<double>(q) / NQ;
            const double wq = (q == 0 || q == NQ) ? 0.5 / NQ : 1.0 / NQ;
            double3 W[12];
            evaluate_nedelec_basis_prism(rA + t*dr, sA + t*ds, zA + t*dz, W);
            for (int j = 0; j < PRISM_NUM_EDGES; ++j)
                circ[j] += wq * (W[j].x*dr + W[j].y*ds + W[j].z*dz);
        }
        for (int j = 0; j < PRISM_NUM_EDGES; ++j) {
            const double target = (e == j) ? 1.0 : 0.0;
            maxerr = std::fmax(maxerr, std::fabs(circ[j] - target));
        }
    }
    check("Kantenzirkulation = delta_ij", maxerr, 1e-12);
}

// -----------------------------------------------------------------------------
// 2. Rotation gegen zentrale Differenzen
// -----------------------------------------------------------------------------
static void test_curl_is_curl() {
    const double h = 1e-5;
    const double pts[5][3] = {
        {0.20, 0.30, -0.40}, {0.10, 0.10, 0.70}, {0.50, 0.20, 0.00},
        {0.25, 0.25, -0.90}, {0.30, 0.40, 0.55}
    };
    double maxerr = 0.0;
    for (int p = 0; p < 5; ++p) {
        const double r = pts[p][0], s = pts[p][1], z = pts[p][2];
        double3 an[12];
        evaluate_nedelec_curl_prism(r, s, z, an);

        double3 Wrp[12], Wrm[12], Wsp[12], Wsm[12], Wzp[12], Wzm[12];
        evaluate_nedelec_basis_prism(r + h, s, z, Wrp);
        evaluate_nedelec_basis_prism(r - h, s, z, Wrm);
        evaluate_nedelec_basis_prism(r, s + h, z, Wsp);
        evaluate_nedelec_basis_prism(r, s - h, z, Wsm);
        evaluate_nedelec_basis_prism(r, s, z + h, Wzp);
        evaluate_nedelec_basis_prism(r, s, z - h, Wzm);

        for (int j = 0; j < PRISM_NUM_EDGES; ++j) {
            const double cx = (Wsp[j].z - Wsm[j].z)/(2*h) - (Wzp[j].y - Wzm[j].y)/(2*h);
            const double cy = (Wzp[j].x - Wzm[j].x)/(2*h) - (Wrp[j].z - Wrm[j].z)/(2*h);
            const double cz = (Wrp[j].y - Wrm[j].y)/(2*h) - (Wsp[j].x - Wsm[j].x)/(2*h);
            maxerr = std::fmax(maxerr, std::fabs(cx - an[j].x));
            maxerr = std::fmax(maxerr, std::fabs(cy - an[j].y));
            maxerr = std::fmax(maxerr, std::fabs(cz - an[j].z));
        }
    }
    check("Rotation gegen zentrale Differenzen", maxerr, 1e-8);
}

// -----------------------------------------------------------------------------
// 3. Volumen aus der Quadratur
// -----------------------------------------------------------------------------
static void test_volume(const double3 nodes[PRISM_NUM_NODES], double expected,
                        const char* label) {
    double vol = 0.0;
    for (int q = 0; q < PRISM_NUM_QUAD_POINTS; ++q) {
        double r, s, z, w;
        prism_quadrature_point(q, r, s, z, w);
        double J[3][3], invJ[3][3], detJ;
        compute_jacobian_prism(nodes, r, s, z, J, invJ, detJ);
        vol += w * detJ;
    }
    check(label, std::fabs(vol - expected) / expected, 1e-13);
}

// -----------------------------------------------------------------------------
// 4. Gradienten im Nullraum der lokalen Steifigkeitsmatrix
// -----------------------------------------------------------------------------
static void test_gradient_nullspace(const double3 nodes[PRISM_NUM_NODES],
                                    const char* label) {
    const double phi[PRISM_NUM_NODES] = {0.31, -1.27, 0.88, 2.04, -0.55, 1.63};
    double a[12] = {0.0};
    for (int e = 0; e < PRISM_NUM_EDGES; ++e) {
        int vA, vB;
        prism_local_edge_nodes(e, vA, vB);
        a[e] = phi[vB] - phi[vA];
    }

    double K[12][12] = {{0.0}};
    for (int q = 0; q < PRISM_NUM_QUAD_POINTS; ++q) {
        double r, s, z, w;
        prism_quadrature_point(q, r, s, z, w);
        double3 Wp[12], Cp[12];
        double detJ;
        piola_prism(nodes, r, s, z, Wp, Cp, detJ);
        const double weight = w * detJ;
        for (int i = 0; i < PRISM_NUM_EDGES; ++i)
            for (int j = 0; j < PRISM_NUM_EDGES; ++j)
                K[i][j] += (Cp[i].x*Cp[j].x + Cp[i].y*Cp[j].y + Cp[i].z*Cp[j].z) * weight;
    }

    // Relativ zum größten Matrixeintrag, nicht absolut. Eine absolute Schranke
    // wäre an die Elementgröße gekoppelt und damit keine Aussage.
    double scale = 0.0, maxres = 0.0;
    for (int i = 0; i < PRISM_NUM_EDGES; ++i)
        for (int j = 0; j < PRISM_NUM_EDGES; ++j)
            scale = std::fmax(scale, std::fabs(K[i][j]));
    for (int i = 0; i < PRISM_NUM_EDGES; ++i) {
        double row = 0.0;
        for (int j = 0; j < PRISM_NUM_EDGES; ++j) row += K[i][j] * a[j];
        maxres = std::fmax(maxres, std::fabs(row));
    }
    check(label, maxres / scale, 1e-13);
}

// -----------------------------------------------------------------------------
// 5. Patchtest: konstantes B
// -----------------------------------------------------------------------------
static double constant_field_error(const double3 nodes[PRISM_NUM_NODES]) {
    const double3 B = make_double3(0.37, -0.81, 0.52);

    // A = 0.5 * (B x r) ist linear, das Kantenintegral über die gerade Kante
    // also exakt A(Kantenmitte) . (x_B - x_A).
    double a[12] = {0.0};
    for (int e = 0; e < PRISM_NUM_EDGES; ++e) {
        int vA, vB;
        prism_local_edge_nodes(e, vA, vB);
        const double3 xa = nodes[vA], xb = nodes[vB];
        const double mx = 0.5*(xa.x+xb.x), my = 0.5*(xa.y+xb.y), mz = 0.5*(xa.z+xb.z);
        const double Ax = 0.5*(B.y*mz - B.z*my);
        const double Ay = 0.5*(B.z*mx - B.x*mz);
        const double Az = 0.5*(B.x*my - B.y*mx);
        a[e] = Ax*(xb.x-xa.x) + Ay*(xb.y-xa.y) + Az*(xb.z-xa.z);
    }

    double maxerr = 0.0;
    for (int q = 0; q < PRISM_NUM_QUAD_POINTS; ++q) {
        double r, s, z, w;
        prism_quadrature_point(q, r, s, z, w);
        double3 Wp[12], Cp[12];
        double detJ;
        piola_prism(nodes, r, s, z, Wp, Cp, detJ);
        double bx = 0.0, by = 0.0, bz = 0.0;
        for (int j = 0; j < PRISM_NUM_EDGES; ++j) {
            bx += a[j]*Cp[j].x; by += a[j]*Cp[j].y; bz += a[j]*Cp[j].z;
        }
        maxerr = std::fmax(maxerr, std::fabs(bx - B.x));
        maxerr = std::fmax(maxerr, std::fabs(by - B.y));
        maxerr = std::fmax(maxerr, std::fabs(bz - B.z));
    }
    return maxerr;
}

// Nicht affiner Fall: der Fehler muss mit der Verzerrungsamplitude
// verschwinden, und zwar mindestens linear. Geprüft wird das Verhältnis
// aufeinanderfolgender Halbierungen; erwartet ist 2, zugelassen 1.7 bis 2.3.
static void test_affinity_limit() {
    const double3 base[PRISM_NUM_NODES] = {
        {0,0,0}, {4,0,0}, {0,3,0}, {0,0,2.5}, {4,0,2.5}, {0,3,2.5}
    };
    const double3 pert[PRISM_NUM_NODES] = {
        {0,0,0}, {0,0,0}, {0,0,0}, {0.3,0.2,-0.1}, {0.4,0.9,0.4}, {-0.2,0.3,-0.3}
    };

    double prev = 0.0;
    double worst_ratio_dev = 0.0;
    for (int k = 0; k < 5; ++k) {
        const double t = 1.0 / (1 << k);
        double3 n[PRISM_NUM_NODES];
        for (int i = 0; i < PRISM_NUM_NODES; ++i)
            n[i] = make_double3(base[i].x + t*pert[i].x,
                                base[i].y + t*pert[i].y,
                                base[i].z + t*pert[i].z);
        const double err = constant_field_error(n);
        if (k > 0) {
            const double ratio = prev / err;
            std::printf("    t = %-8.4f Fehler = %9.3e  Verhaeltnis = %5.2f\n",
                        t, err, ratio);
            worst_ratio_dev = std::fmax(worst_ratio_dev, std::fabs(ratio - 2.0));
        } else {
            std::printf("    t = %-8.4f Fehler = %9.3e\n", t, err);
        }
        prev = err;
    }
    check("Patchtest nicht affin: Fehler linear in der Verzerrung",
          worst_ratio_dev, 0.3);
}

int main() {
    const double3 straight[PRISM_NUM_NODES] = {
        {0.0, 0.0, 0.0}, {4.0, 0.0, 0.0}, {0.0, 3.0, 0.0},
        {0.0, 0.0, 2.5}, {4.0, 0.0, 2.5}, {0.0, 3.0, 2.5}
    };
    // Affin: die Deckfläche ist eine reine Parallelverschiebung der Bodenfläche.
    const double3 sheared[PRISM_NUM_NODES] = {
        {0.0, 0.0, 0.0}, {4.0, 0.5, 0.0}, {0.7, 3.0, 0.0},
        {1.1, 0.4, 2.5}, {5.1, 0.9, 2.5}, {1.8, 3.4, 2.5}
    };

    std::printf("\n== Referenzelement ==\n");
    test_circulation();
    test_curl_is_curl();

    std::printf("\n== Gerades Prisma ==\n");
    test_volume(straight, 0.5 * 4.0 * 3.0 * 2.5, "Volumen aus der Quadratur");
    check("Patchtest: konstantes B exakt", constant_field_error(straight), 1e-13);
    test_gradient_nullspace(straight, "Gradient im Nullraum von K");

    std::printf("\n== Affin geschertes Prisma ==\n");
    check("Patchtest: konstantes B exakt", constant_field_error(sheared), 1e-13);
    test_gradient_nullspace(sheared, "Gradient im Nullraum von K");

    std::printf("\n== Nicht affines Prisma ==\n");
    test_affinity_limit();

    std::printf("\n");
    if (g_failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", g_failures);
    return 1;
}
