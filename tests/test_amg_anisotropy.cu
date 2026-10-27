// =============================================================================
// Abnahmeprüfung: verträgt das AMG anisotrope Gitter?
//
// Bauen und laufen lassen:
//   cmake --build build --config Release --target test_amg_anisotropy
//   ./build/test_amg_anisotropy
//
// HINTERGRUND
// -----------
// Der skalare AMS-Hilfsraum konvergierte am TEAM-28-Netz mit 0.72 je V-Zyklus
// gegen 0.30 auf dem 3D-Laplace. Ursache war nicht der Operator, sondern die
// Gitteranisotropie: die Aluminiumplatte ist 2 mm dick bei 126 mm Durchmesser
// und hat ein Zell-Seitenverhältnis von 4.5 im Median (max 6.1). Die Spulen
// liegen bei 1.4 bis 1.9.
//
// Der alte Stärkemaßstab war ABSOLUT:
//
//     |a_IJ| > theta * sqrt(|a_II * a_JJ|)
//
// Er kennt keine Richtungen. Bei gestreckten Zellen ist die Kopplung quer zur
// dünnen Richtung um das Quadrat des Seitenverhältnisses kleiner; mit einem
// festen theta gelten dann entweder alle Richtungen als stark (die Aggregate
// werden isotrop, das Verfahren verliert seine Mehrgittereigenschaft) oder
// keine (die Vergröberung bricht ab). Ein einziger Wert kann Platte und Spulen
// nicht zugleich bedienen.
//
// Der neue Maßstab ist ZEILENRELATIV (klassisch nach Ruge und Stüben):
//
//     |a_IJ| > theta * max_{K != I} |a_IK|
//
// Er normiert je Zeile auf die größte Kopplung und ist damit skaleninvariant:
// im isotropen Fall bestehen alle Kopplungen die Schwelle, im anisotropen nur
// die starke Richtung. Die Aggregate richten sich von selbst aus.
//
// GEMESSEN (Poisson 16x16x3, Faktor je V-Zyklus, Grobgitter in Klammern):
//
//   Zell-Verh.  alt: diagonal 0.08   neu: zeilenrel. 0.60
//      1.0      0.237 (42)           0.273 (21)
//      4.5      0.774 (40)           0.439 (36)
//     10.0      0.854 (40)           0.478 (36)
//     30.0      0.873 (41)           0.490 (36)
//
// Eine einzige Einstellung für alle Seitenverhältnisse. Der isotrope Fall wird
// dabei minimal schlechter (0.273 gegen 0.237) -- der Preis ist klein gegen
// den Gewinn bei 4.5.
//
// Zum Vergleich: der diagonale Maßstab erreicht bei 4.5 mit theta = 0.35
// denselben Faktor 0.439, kommt dann aber nur auf ein Grobgitter von 289 statt
// 36 -- also fast keine Vergröberung, und jeder Zyklus kostet entsprechend.
//
// EINE FALLE, DIE DIESER TEST AUSDRÜCKLICH MITPRÜFT
// -------------------------------------------------
// Ein zu großes theta liefert scheinbar traumhafte Faktoren -- 0.0009 beim
// diagonalen Maßstab mit theta = 0.35 auf isotropem Gitter. Dann gilt aber gar
// nichts mehr als stark, das Grobgitter ist so groß wie das feine, und das
// "AMG" ist eine teure direkte Lösung. Ohne die mitgemessene Grobgittergröße
// hätte ich das als Durchbruch gemeldet.
//
// EXIT-CODE 0, wenn alle Prüfungen bestehen, sonst 1.
// =============================================================================

#include "DiscreteGradient.cuh"
#include "AlgebraicMultigrid.cuh"
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static double measure(const AlgebraicMultigrid& amg, const HostCsr& A) {
    const int N = A.num_rows;
    std::mt19937 rng(3); std::uniform_real_distribution<double> u(-1,1);
    std::vector<double> x(N,0.0), b(N), r(N), e(N), Ax(N);
    for (int i=0;i<N;++i) b[i]=u(rng);
    auto rn=[&](){ apply_host_csr(A,x.data(),Ax.data()); double s=0;
        for(int i=0;i<N;++i){r[i]=b[i]-Ax[i]; s+=r[i]*r[i];} return std::sqrt(s); };
    const double r0=rn(); double rk=r0;
    for(int c=0;c<5;++c){ amg.apply(r.data(),e.data());
        for(int i=0;i<N;++i) x[i]+=e[i]; rk=rn(); }
    return std::pow(rk/r0, 1.0/5);
}

int main() {
    // Host-Test: ohne GPU lauffaehig. Ohne diese Zeile legt DeviceArray den
    // Speicher mit cudaMallocManaged an und bricht ohne Grafikkarte ab.
    bf::g_backend = bf::Backend::CPU;
    int failures = 0;
    double old_iso = 0, old_aniso = 0, new_iso = 0, new_aniso = 0, new_30 = 0;
    int new_coarse_aniso = 0, trap_coarse = 0, fine_rows = 0;

    std::printf("\n=== Staerkemaszstab der Aggregation auf anisotropen Gittern ===\n\n");
    std::printf("  %10s %-10s %5s %9s   %-16s\n",
                "Zell-Verh.", "Maszstab", "theta", "Faktor", "fein -> grob");
    const int nx = 16, ny = 16, nz = 3;      // wie die Platte: duenn in z
    // ar ist jetzt das ZELL-Seitenverhaeltnis (in der Ebene zu Dicke), nicht
    // die Gebietsstreckung. Der erste Anlauf hat das verwechselt: bei nz = 3
    // ueber Einheitshoehe sind die Zellen schon bei "1" um Faktor 5 gestreckt,
    // und die Tabelle war dadurch unbrauchbar.
    for (double ar : {1.0, 4.5, 10.0, 30.0}) {
        const int px=nx+1, py=ny+1, pz=nz+1;
        const double h_inplane = 1.0 / nx;          // Zellgroesse in der Ebene
        const double h_thick   = h_inplane / ar;    // Zelldicke
        const double zmax      = h_thick * nz;
        Mesh mesh;
        mesh.num_nodes=(size_t)px*py*pz; mesh.num_hexes=(size_t)nx*ny*nz;
        mesh.nodes_current.allocate(mesh.num_nodes);
        mesh.hex_nodes.allocate(mesh.num_hexes);
        auto nid=[&](int i,int j,int k){ return (k*py + j)*px + i; };
        for (int k=0;k<pz;++k) for(int j=0;j<py;++j) for(int i=0;i<px;++i)
            mesh.nodes_current.data()[nid(i,j,k)] =
                make_double3(i*1.0/nx, j*1.0/ny, k*zmax/nz);
        int h=0;
        for (int k=0;k<nz;++k) for(int j=0;j<ny;++j) for(int i=0;i<nx;++i){
            Hexahedron hx;
            hx.v[0]=nid(i,j,k);   hx.v[1]=nid(i+1,j,k);
            hx.v[2]=nid(i+1,j+1,k); hx.v[3]=nid(i,j+1,k);
            hx.v[4]=nid(i,j,k+1); hx.v[5]=nid(i+1,j,k+1);
            hx.v[6]=nid(i+1,j+1,k+1); hx.v[7]=nid(i,j+1,k+1);
            mesh.hex_nodes.data()[h++]=hx;
        }
        std::vector<double> nu((size_t)nx*ny*nz, 1.0);
        HostCsr A3 = build_alpha_poisson_matrix(mesh, nu.data(), nullptr);
        // Skalarer Anteil herausziehen (die Matrix ist blockdiagonal aus 3 Kopien)
        HostCsr A; A.num_rows=A.num_cols=(int)mesh.num_nodes;
        A.row_offsets.assign(A.num_rows+1,0);
        for (int i=0;i<A.num_rows;++i){
            A.row_offsets[i]=(int)A.col_indices.size();
            for (int k2=A3.row_offsets[3*i]; k2<A3.row_offsets[3*i+1]; ++k2){
                A.col_indices.push_back(A3.col_indices[k2]/3);
                A.values.push_back(A3.values[k2]);
            }
        }
        A.row_offsets[A.num_rows]=(int)A.col_indices.size();
        double dmax=0; for (int i=0;i<A.num_rows;++i)
            for (int k2=A.row_offsets[i];k2<A.row_offsets[i+1];++k2)
                if (A.col_indices[k2]==i) dmax=std::fmax(dmax,std::fabs(A.values[k2]));
        for (int i=0;i<A.num_rows;++i)
            for (int k2=A.row_offsets[i];k2<A.row_offsets[i+1];++k2)
                if (A.col_indices[k2]==i) A.values[k2]+=1e-3*dmax;
        // Ausgeschriebenes Feld statt einer Initialisierungsliste mit
        // gemischt getippten Elementen: nvcc leitet den Typ dort nicht her
        // ("cannot deduce type of range in range-based for loop"), waehrend
        // g++ es tut. Der Unterschied faellt erst beim echten Build auf.
        struct Cfg { bool row_relative; double theta; };
        static const Cfg cfgs[] = {
            {false, 0.08}, {false, 0.35}, {true, 0.40}, {true, 0.60}
        };
        for (const Cfg& cfg : cfgs) {
            AlgebraicMultigrid amg;
            AlgebraicMultigrid::Options o;
            o.row_relative_strength = cfg.row_relative;
            o.theta = cfg.theta;
            const double th = cfg.theta;
            amg.setup(A, 1, o);
            const int nlev = amg.num_levels();
            const int coarsest = amg.level_size(nlev - 1);
            // Ein kleiner Faktor bei kaum vergroebertem Grobgitter ist KEIN
            // Mehrgitterergebnis, sondern eine teure direkte Loesung. Ohne die
            // Grobgittergroesse ist die Faktorspalte irrefuehrend.
            const double f = measure(amg, A);
            std::printf("  %10.1f %-10s %5.2f %9.4f   %5d -> %-5d  %s\n",
                        ar, cfg.row_relative ? "zeilenrel." : "diagonal", th, f,
                        A.num_rows, coarsest,
                        (coarsest > A.num_rows / 4)
                            ? "<- kaum vergroebert, faktisch direkt" : "");
            fine_rows = A.num_rows;
            if (!cfg.row_relative && th == 0.08) { if (ar == 1.0) old_iso = f;
                                            if (ar == 4.5) old_aniso = f; }
            if (!cfg.row_relative && th == 0.35 && ar == 1.0) trap_coarse = coarsest;
            if (cfg.row_relative && th == 0.60) {
                if (ar == 1.0)  new_iso = f;
                if (ar == 4.5) { new_aniso = f; new_coarse_aniso = coarsest; }
                if (ar == 30.0) new_30 = f;
            }
        }
    }
    std::printf("\n");

    std::printf("  [%s] zeilenrelativ ist bei Seitenverhaeltnis 4.5 deutlich besser "
                "(%.4f gegen %.4f)\n",
                new_aniso < 0.7 * old_aniso ? " OK " : "FEHL", new_aniso, old_aniso);
    if (!(new_aniso < 0.7 * old_aniso)) ++failures;

    std::printf("  [%s] und dabei mit gesunder Vergroeberung (%d von %d)\n",
                new_coarse_aniso < fine_rows / 10 ? " OK " : "FEHL",
                new_coarse_aniso, fine_rows);
    if (!(new_coarse_aniso < fine_rows / 10)) ++failures;

    std::printf("  [%s] auch bei Seitenverhaeltnis 30 noch brauchbar (%.4f < 0.6)\n",
                new_30 < 0.6 ? " OK " : "FEHL", new_30);
    if (!(new_30 < 0.6)) ++failures;

    // Der isotrope Fall darf nicht wesentlich leiden -- die Spulen liegen dort.
    std::printf("  [%s] isotroper Fall bleibt gut (%.4f, alt %.4f)\n",
                new_iso < 1.3 * old_iso ? " OK " : "FEHL", new_iso, old_iso);
    if (!(new_iso < 1.3 * old_iso)) ++failures;

    // Die Falle: ein zu grosses theta sieht traumhaft aus, vergroebert aber nicht.
    std::printf("  [%s] Gegenprobe: diagonal mit theta = 0.35 zerstoert die "
                "Vergroeberung im isotropen Fall (%d von %d)\n",
                trap_coarse > fine_rows / 4 ? " OK " : "FEHL", trap_coarse, fine_rows);
    if (!(trap_coarse > fine_rows / 4)) ++failures;

    std::printf("\n");
    if (failures == 0) { std::printf("Alle Pruefungen bestanden.\n\n"); return 0; }
    std::printf("%d Pruefung(en) fehlgeschlagen.\n\n", failures);
    return 1;
}
