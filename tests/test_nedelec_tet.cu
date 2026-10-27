// =============================================================================
// Abnahmepruefung: Nedelec-Kantenelement auf dem Tetraeder (CTETRA mit 4 Knoten)
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_nedelec_tet
//   ./build/test_nedelec_tet
//
// Reiner Host-Code, keine GPU, kein Netz, in Sekundenbruchteilen durch.
//
// WARUM DIESES ELEMENT
// --------------------
// Das gelieferte Netz (team28_simple_air.bdf, HyperMesh) besteht
// ausschliesslich aus 3692 linearen Tetraedern. Das Prisma aus
// NedelecPrism.cuh wird darauf nicht gebraucht; es bleibt fuer den Fall, dass
// spaeter extrudierte Netzteile dazukommen. Tetraeder und Prismen mischen sich
// uebrigens konform, die gemeinsame Flaeche ist beidseitig ein Dreieck.
//
// DER ENTSCHEIDENDE UNTERSCHIED ZU HEXAEDER UND PRISMA
// ----------------------------------------------------
// Die Abbildung eines linearen Tetraeders ist IMMER affin. Damit entfaellt der
// Vorbehalt, der dort gilt: ein konstantes B wird auf JEDEM Tetraeder exakt
// reproduziert, unabhaengig von der Elementform. Beim Prisma bleibt auf einem
// verzerrten Element ein Fehler linear in der Verzerrung (gemessen 1,34e-01
// bei t = 1), beim Hexaeder ebenso (1,03e-01).
//
// Diese Aussage wird hier nicht behauptet, sondern geprueft: der Patchtest
// laeuft ueber 20000 ZUFAELLIGE Tetraeder, verworfen werden nur die fast
// entarteten (Formguete V/l^3 unter 1e-3). Schlaegt er dort an, ist entweder
// die Basis oder eine der beiden Piola-Transformationen falsch.
//
// Ein schlecht geformter Tetraeder verschlechtert die Kondition, nicht die
// Konsistenz. Fuer einen Direktloeser ist das der guenstige Fall.
//
// DIE PRUEFUNGEN
// --------------
//   1. Kantenzirkulation: Int_{Kante i} W_j . t dl = delta_ij. Definierende
//      Eigenschaft, prueft Normierung UND Orientierung.
//   2. Rotation gegen zentrale Differenzen. Genau der Fehler, der im
//      Hexaederelement fuer 10 von 12 Kanten steckte und erst durch
//      Nachrechnen aufgefallen ist.
//   3. Volumen aus der Quadratur gegen das Spatprodukt.
//   4. Gradienten im Nullraum: K * (Knotendifferenzen) = 0, exakt.
//   5. Patchtest mit konstantem B, auf 20000 Zufallselementen.
// =============================================================================

#include "NedelecTetrahedron.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

static int g_failures = 0;
static void report(const char* name, double err, double tol) {
    const bool ok = (err <= tol) && std::isfinite(err);
    printf("  %-50s %10.3e  (tol %8.1e)  %s\n", name, err, tol, ok ? "OK" : "FEHLER");
    if (!ok) ++g_failures;
}

static void ref_node(int i, double& r, double& s, double& t) {
    const double R[4][3] = {{0,0,0},{1,0,0},{0,1,0},{0,0,1}};
    r = R[i][0]; s = R[i][1]; t = R[i][2];
}

static void piola(const double3 n[4], double r, double s, double t,
                  double3 Wp[12], double3 Cp[12], double& detJ) {
    double J[3][3], invJ[3][3];
    compute_jacobian_tet(n, r, s, t, J, invJ, detJ);
    double3 W[12], C[12];
    evaluate_nedelec_basis_tet(r, s, t, W);
    evaluate_nedelec_curl_tet(r, s, t, C);
    for (int i = 0; i < 12; ++i) {
        Wp[i].x = invJ[0][0]*W[i].x + invJ[0][1]*W[i].y + invJ[0][2]*W[i].z;
        Wp[i].y = invJ[1][0]*W[i].x + invJ[1][1]*W[i].y + invJ[1][2]*W[i].z;
        Wp[i].z = invJ[2][0]*W[i].x + invJ[2][1]*W[i].y + invJ[2][2]*W[i].z;
        Cp[i].x = (J[0][0]*C[i].x + J[1][0]*C[i].y + J[2][0]*C[i].z)/detJ;
        Cp[i].y = (J[0][1]*C[i].x + J[1][1]*C[i].y + J[2][1]*C[i].z)/detJ;
        Cp[i].z = (J[0][2]*C[i].x + J[1][2]*C[i].y + J[2][2]*C[i].z)/detJ;
    }
}

static void test_circulation() {
    double maxerr = 0.0;
    for (int e = 0; e < TET_NUM_EDGES; ++e) {
        int vA, vB; tet_local_edge_nodes(e, vA, vB);
        double rA,sA,tA,rB,sB,tB;
        ref_node(vA, rA,sA,tA); ref_node(vB, rB,sB,tB);
        const double dr=rB-rA, ds=sB-sA, dt=tB-tA;
        double circ[12] = {0.0};
        const int NQ = 4000;
        for (int q = 0; q <= NQ; ++q) {
            const double u = double(q)/NQ;
            const double wq = (q==0||q==NQ) ? 0.5/NQ : 1.0/NQ;
            double3 W[12];
            evaluate_nedelec_basis_tet(rA+u*dr, sA+u*ds, tA+u*dt, W);
            for (int j = 0; j < TET_NUM_EDGES; ++j)
                circ[j] += wq*(W[j].x*dr + W[j].y*ds + W[j].z*dt);
        }
        for (int j = 0; j < TET_NUM_EDGES; ++j)
            maxerr = std::fmax(maxerr, std::fabs(circ[j] - (e==j?1.0:0.0)));
    }
    report("Kantenzirkulation = delta_ij", maxerr, 1e-12);
}

static void test_curl_fd() {
    const double h = 1e-5;
    double maxerr = 0.0;
    const double P[4][3] = {{0.2,0.3,0.1},{0.1,0.1,0.5},{0.3,0.2,0.2},{0.05,0.4,0.3}};
    for (int p = 0; p < 4; ++p) {
        const double r=P[p][0], s=P[p][1], t=P[p][2];
        double3 an[12]; evaluate_nedelec_curl_tet(r,s,t,an);
        double3 a1[12],a2[12],b1[12],b2[12],c1[12],c2[12];
        evaluate_nedelec_basis_tet(r+h,s,t,a1); evaluate_nedelec_basis_tet(r-h,s,t,a2);
        evaluate_nedelec_basis_tet(r,s+h,t,b1); evaluate_nedelec_basis_tet(r,s-h,t,b2);
        evaluate_nedelec_basis_tet(r,s,t+h,c1); evaluate_nedelec_basis_tet(r,s,t-h,c2);
        for (int j = 0; j < TET_NUM_EDGES; ++j) {
            const double cx = (b1[j].z-b2[j].z)/(2*h) - (c1[j].y-c2[j].y)/(2*h);
            const double cy = (c1[j].x-c2[j].x)/(2*h) - (a1[j].z-a2[j].z)/(2*h);
            const double cz = (a1[j].y-a2[j].y)/(2*h) - (b1[j].x-b2[j].x)/(2*h);
            maxerr = std::fmax(maxerr, std::fabs(cx-an[j].x));
            maxerr = std::fmax(maxerr, std::fabs(cy-an[j].y));
            maxerr = std::fmax(maxerr, std::fabs(cz-an[j].z));
        }
    }
    report("Rotation gegen zentrale Differenzen", maxerr, 1e-8);
}

static double patch_error(const double3 n[4]) {
    const double3 B = make_double3(0.37, -0.81, 0.52);
    double a[12] = {0.0};
    for (int e = 0; e < TET_NUM_EDGES; ++e) {
        int vA,vB; tet_local_edge_nodes(e,vA,vB);
        const double3 xa=n[vA], xb=n[vB];
        const double mx=0.5*(xa.x+xb.x), my=0.5*(xa.y+xb.y), mz=0.5*(xa.z+xb.z);
        const double Ax=0.5*(B.y*mz-B.z*my), Ay=0.5*(B.z*mx-B.x*mz), Az=0.5*(B.x*my-B.y*mx);
        a[e] = Ax*(xb.x-xa.x)+Ay*(xb.y-xa.y)+Az*(xb.z-xa.z);
    }
    double maxerr = 0.0;
    for (int q = 0; q < TET_NUM_QUAD_POINTS; ++q) {
        double r,s,t,w; tet_quadrature_point(q,r,s,t,w);
        double3 Wp[12],Cp[12]; double detJ;
        piola(n,r,s,t,Wp,Cp,detJ);
        double bx=0,by=0,bz=0;
        for (int j = 0; j < TET_NUM_EDGES; ++j) { bx+=a[j]*Cp[j].x; by+=a[j]*Cp[j].y; bz+=a[j]*Cp[j].z; }
        maxerr = std::fmax(maxerr, std::fabs(bx-B.x));
        maxerr = std::fmax(maxerr, std::fabs(by-B.y));
        maxerr = std::fmax(maxerr, std::fabs(bz-B.z));
    }
    return maxerr;
}

static double nullspace_error(const double3 n[4]) {
    const double phi[4] = {0.31,-1.27,0.88,2.04};
    double a[12]={0.0};
    for (int e=0;e<TET_NUM_EDGES;++e){int vA,vB;tet_local_edge_nodes(e,vA,vB);a[e]=phi[vB]-phi[vA];}
    double K[12][12]={{0.0}};
    for (int q=0;q<TET_NUM_QUAD_POINTS;++q){
        double r,s,t,w; tet_quadrature_point(q,r,s,t,w);
        double3 Wp[12],Cp[12]; double detJ; piola(n,r,s,t,Wp,Cp,detJ);
        for(int i=0;i<TET_NUM_EDGES;++i)for(int j=0;j<TET_NUM_EDGES;++j)
            K[i][j]+=(Cp[i].x*Cp[j].x+Cp[i].y*Cp[j].y+Cp[i].z*Cp[j].z)*w*detJ;
    }
    double scale=0,res=0;
    for(int i=0;i<TET_NUM_EDGES;++i)for(int j=0;j<TET_NUM_EDGES;++j)
        scale=std::fmax(scale,std::fabs(K[i][j]));
    for(int i=0;i<TET_NUM_EDGES;++i){double row=0;
        for(int j=0;j<TET_NUM_EDGES;++j) row+=K[i][j]*a[j];
        res=std::fmax(res,std::fabs(row));}
    return res/scale;
}

static double volume_error(const double3 n[4]) {
    double vol=0;
    for (int q=0;q<TET_NUM_QUAD_POINTS;++q){
        double r,s,t,w; tet_quadrature_point(q,r,s,t,w);
        double J[3][3],invJ[3][3],detJ; compute_jacobian_tet(n,r,s,t,J,invJ,detJ);
        vol += w*detJ;
    }
    // exaktes Volumen aus dem Spatprodukt
    const double3 e1={n[1].x-n[0].x,n[1].y-n[0].y,n[1].z-n[0].z};
    const double3 e2={n[2].x-n[0].x,n[2].y-n[0].y,n[2].z-n[0].z};
    const double3 e3={n[3].x-n[0].x,n[3].y-n[0].y,n[3].z-n[0].z};
    const double ex = (e1.y*e2.z-e1.z*e2.y)*e3.x + (e1.z*e2.x-e1.x*e2.z)*e3.y
                    + (e1.x*e2.y-e1.y*e2.x)*e3.z;
    return std::fabs(vol - ex/6.0)/std::fabs(ex/6.0);
}

int main() {
    printf("\n== Referenzelement ==\n");
    test_circulation();
    test_curl_fd();

    const double3 unit[4] = {{0,0,0},{4,0,0},{0,3,0},{0,0,2.5}};
    printf("\n== Gerader Tetraeder ==\n");
    report("Volumen aus der Quadratur", volume_error(unit), 1e-13);
    report("Patchtest: konstantes B", patch_error(unit), 1e-13);
    report("Gradient im Nullraum von K", nullspace_error(unit), 1e-13);

    // Der entscheidende Unterschied zum Prisma: der Tetraeder ist IMMER affin.
    // Also muss der Patchtest auch auf beliebig verzerrten Elementen exakt
    // sein. 20000 Zufallstetraeder, verworfen werden nur fast entartete.
    printf("\n== 20000 Zufallstetraeder ==\n");
    std::srand(20260909u);
    double worst_patch = 0.0, worst_null = 0.0, worst_vol = 0.0, worst_q = 1e30;
    int used = 0;
    auto rnd = []{ return 2.0*(double(std::rand())/RAND_MAX) - 1.0; };
    for (int k = 0; k < 20000; ++k) {
        double3 n[4];
        for (int i = 0; i < 4; ++i) n[i] = make_double3(10*rnd(), 10*rnd(), 10*rnd());
        double J[3][3],invJ[3][3],detJ;
        compute_jacobian_tet(n,0.25,0.25,0.25,J,invJ,detJ);
        // Formguete: Volumen gegen die dritte Potenz der laengsten Kante
        double lmax = 0.0;
        for (int a=0;a<4;++a) for (int b=a+1;b<4;++b) {
            const double dx=n[a].x-n[b].x, dy=n[a].y-n[b].y, dz=n[a].z-n[b].z;
            lmax = std::fmax(lmax, std::sqrt(dx*dx+dy*dy+dz*dz));
        }
        const double q = std::fabs(detJ/6.0)/(lmax*lmax*lmax);
        if (q < 1e-3) continue;            // fast entartet, uebersprungen
        worst_q = std::fmin(worst_q, q);
        worst_patch = std::fmax(worst_patch, patch_error(n));
        worst_null  = std::fmax(worst_null,  nullspace_error(n));
        worst_vol   = std::fmax(worst_vol,   volume_error(n));
        ++used;
    }
    printf("  (%d verwendet, schlechteste Formguete V/l^3 = %.2e)\n", used, worst_q);
    report("Patchtest: konstantes B, schlechtester Fall", worst_patch, 1e-12);
    report("Gradient im Nullraum, schlechtester Fall", worst_null, 1e-11);
    report("Volumen, schlechtester Fall", worst_vol, 1e-12);

    printf("\n%s\n\n", g_failures==0 ? "ALLE PRUEFUNGEN BESTANDEN" : "PRUEFUNGEN FEHLGESCHLAGEN");
    return g_failures ? 1 : 0;
}
