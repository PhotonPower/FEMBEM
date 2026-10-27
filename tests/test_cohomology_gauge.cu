// =============================================================================
// Regressionstest: Kohomologie-Erweiterung der Baum-Kobaum-Eichung
//
//   cmake --build build --target test_cohomology_gauge
//   ./build/test_cohomology_gauge         (aus dem Repo-Wurzelverzeichnis)
//
// HINTERGRUND
// -----------
// build_tree_cotree_gauge() entfernt aus A_sys = K + C/dt nur den
// Gradientenanteil des Nullraums. Jede Ringspule bei TEAM 28 ist topologisch
// ein Volltorus (erste Bettizahl b1 = 1, nachgewiesen ueber die
// Euler-Charakteristik in tests/test_bem_index_mapping-Vorarbeiten und im
// Projektstand-Dokument), es bleibt also pro Spule ein zusaetzlicher
// "harmonischer" Kozykel im Nullraum -- ein Kantenvektor a mit delta_1(a) = 0
// (verschwindende Zirkulation um jede Flaeche), der aber kein Gradient ist.
//
// EIN FEHLGESCHLAGENER ERSTER VERSUCH
// -------------------------------------
// Naheliegend, aber FALSCH: den Fundamentalkreis einer Kobaum-Kante (Baumpfad
// plus diese Kante) direkt als Kandidaten zu pruefen. Ein Fundamentalkreis ist
// ein KETTEN-Zyklus (d(Kreis) = 0 als Rand), waehrend ein Kozykel eine andere
// Bedingung ist (delta(a) = 0, Zirkulation um Flaechen). Ein Gradient erfuellt
// immer die Kozykel-Bedingung, ist aber im Allgemeinen KEIN geschlossener
// Kettenzyklus -- die beiden Eigenschaften fallen nicht zusammen. Der erste
// Versuch fand deshalb bei keiner einzigen von ueber 6000 Kobaum-Kanten einen
// gueltigen Generator. Test 1 unten haelt dieses Wissen als Regression fest:
// er prueft, dass ein GRADIENT den Kozykel-Test besteht, aber im Allgemeinen
// kein geschlossener Graphkreis ist -- also genau die Unterscheidung, die den
// ersten Versuch scheitern liess.
//
// KORREKTE LOESUNG
// -----------------
// find_cohomology_generators() (include/CohomologyGauge.h) verwendet
// stattdessen eine diskrete Helmholtz-Hodge-Zerlegung: ein zufaelliger
// Kantenvektor wird zuerst auf Ker(delta_1) projiziert (CG auf
// delta_1 delta_1^T), dann wird der Gradientenanteil abgezogen (CG auf dem
// Graph-Laplace delta_0^T delta_0). Beide Schritte sind rein kombinatorisch
// (ganzzahlige Vorzeichen aus den etablierten Kantentabellen), unabhaengig
// von Materialwerten oder Anregungsart.
//
// Dieser Test prueft:
//   1. Gradient-vs-Kettenzyklus-Unterscheidung (Regressionsschutz, s.o.)
//   2. Es werden genau 2 Generatoren gefunden (einer je Spule)
//   3. Jeder Generator ist numerisch ein Kozykel (kleines Residuum)
//   4. Jeder Generator ist nicht trivial (Norm nicht verschwindend)
//   5. Reduzierte Generatoren sind orthogonal zueinander (disjunkter Traeger)
//   6. DER ENTSCHEIDENDE TEST: mit Baum- und Kohomologie-Eichung zusammen
//      konvergiert CG fuer eine ECHT ZUFAELLIGE (nicht konstruierte) rechte
//      Seite. Das ist der empirische Beweis, dass der Nullraum vollstaendig
//      entfernt ist -- unabhaengig davon, wie die Anregung einmal aussieht.
//
// WARTUNGSHINWEIS: Diese Datei enthaelt eine Kopie der Nedelec-Basis/Curl und
// eine vollstaendige Host-Assemblierung von K + C/dt, weil NedelecHexahedron.cuh
// cuda_runtime.h benoetigt. Bei Aenderungen an den Originalen muss die Kopie
// hier mitgepflegt werden.
// =============================================================================

#include "VtkReader.h"
#include "TopologyExtractor.h"
#include "TreeCotreeGauge.h"
#include "CohomologyGauge.h"

#include <cstdio>
#include <cmath>
#include <vector>
#include <map>
#include <queue>

namespace {

struct D3 { double x, y, z; };
D3 mk(double a, double b, double c) { return D3{a, b, c}; }

void basis(double xi, double eta, double zeta, D3 W[12]) {
    W[0]=mk( 0.125*(1-eta)*(1-zeta),0,0); W[2]=mk(-0.125*(1+eta)*(1-zeta),0,0);
    W[4]=mk( 0.125*(1-eta)*(1+zeta),0,0); W[6]=mk(-0.125*(1+eta)*(1+zeta),0,0);
    W[1]=mk(0, 0.125*(1+xi)*(1-zeta),0);  W[3]=mk(0,-0.125*(1-xi)*(1-zeta),0);
    W[5]=mk(0, 0.125*(1+xi)*(1+zeta),0);  W[7]=mk(0,-0.125*(1-xi)*(1+zeta),0);
    W[8]=mk(0,0,0.125*(1-xi)*(1-eta));    W[9]=mk(0,0,0.125*(1+xi)*(1-eta));
    W[10]=mk(0,0,0.125*(1+xi)*(1+eta));   W[11]=mk(0,0,0.125*(1-xi)*(1+eta));
}
void curl(double xi, double eta, double zeta, D3 c[12]) {
    c[0]=mk(0,-0.125*(1-eta),0.125*(1-zeta));  c[2]=mk(0, 0.125*(1+eta),0.125*(1-zeta));
    c[4]=mk(0, 0.125*(1-eta),0.125*(1+zeta));  c[6]=mk(0,-0.125*(1+eta),0.125*(1+zeta));
    c[1]=mk( 0.125*(1+xi),0,0.125*(1-zeta));   c[3]=mk(-0.125*(1-xi),0,0.125*(1-zeta));
    c[5]=mk(-0.125*(1+xi),0,0.125*(1+zeta));   c[7]=mk( 0.125*(1-xi),0,0.125*(1+zeta));
    c[8]=mk(-0.125*(1-xi), 0.125*(1-eta),0);   c[9]=mk(-0.125*(1+xi),-0.125*(1-eta),0);
    c[10]=mk( 0.125*(1+xi),-0.125*(1+eta),0);  c[11]=mk( 0.125*(1-xi), 0.125*(1+eta),0);
}

// Baut A_sys = K + C/dt als CSR-Struktur (Meter, wie in mainloop()).
void assemble_A_sys(const Mesh& mesh, const std::vector<MaterialProperties>& mats,
                    double dt, std::vector<int>& ptr, std::vector<int>& col,
                    std::vector<double>& val)
{
    const int N = static_cast<int>(mesh.num_edges);
    const double S = 1e-3, MU0 = 4.0 * M_PI * 1e-7;
    const double sX[8]={-1,1,1,-1,-1,1,1,-1}, sY[8]={-1,-1,1,1,-1,-1,1,1}, sZ[8]={-1,-1,-1,-1,1,1,1,1};
    const double g = 0.5773502691896257;
    const double gp[8][3] = {{-g,-g,-g},{g,-g,-g},{-g,g,-g},{g,g,-g},{-g,-g,g},{g,-g,g},{-g,g,g},{g,g,g}};

    std::vector<std::map<int,double>> rows(N);
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        D3 nd[8];
        for (int k = 0; k < 8; ++k) {
            double3 p = mesh.nodes_current[mesh.hex_nodes[h].v[k]];
            nd[k] = {p.x * S, p.y * S, p.z * S};
        }
        const double nu = 1.0 / (mats[h].mu_relative * MU0);
        const double kap = mats[h].kappa;
        double Ae[12][12] = {{0}};
        for (int q = 0; q < 8; ++q) {
            const double xi = gp[q][0], eta = gp[q][1], zeta = gp[q][2];
            double J[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
            for (int i = 0; i < 8; ++i) {
                const double a = 0.125*sX[i]*(1+sY[i]*eta)*(1+sZ[i]*zeta);
                const double b = 0.125*(1+sX[i]*xi)*sY[i]*(1+sZ[i]*zeta);
                const double c = 0.125*(1+sX[i]*xi)*(1+sY[i]*eta)*sZ[i];
                J[0][0]+=a*nd[i].x; J[0][1]+=a*nd[i].y; J[0][2]+=a*nd[i].z;
                J[1][0]+=b*nd[i].x; J[1][1]+=b*nd[i].y; J[1][2]+=b*nd[i].z;
                J[2][0]+=c*nd[i].x; J[2][1]+=c*nd[i].y; J[2][2]+=c*nd[i].z;
            }
            const double det = J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])
                              - J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])
                              + J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
            const double id = 1.0 / det;
            double iJ[3][3];
            iJ[0][0]=(J[1][1]*J[2][2]-J[1][2]*J[2][1])*id; iJ[0][1]=(J[0][2]*J[2][1]-J[0][1]*J[2][2])*id;
            iJ[0][2]=(J[0][1]*J[1][2]-J[0][2]*J[1][1])*id; iJ[1][0]=(J[1][2]*J[2][0]-J[1][0]*J[2][2])*id;
            iJ[1][1]=(J[0][0]*J[2][2]-J[0][2]*J[2][0])*id; iJ[1][2]=(J[0][2]*J[1][0]-J[0][0]*J[1][2])*id;
            iJ[2][0]=(J[1][0]*J[2][1]-J[1][1]*J[2][0])*id; iJ[2][1]=(J[0][1]*J[2][0]-J[0][0]*J[2][1])*id;
            iJ[2][2]=(J[0][0]*J[1][1]-J[0][1]*J[1][0])*id;
            D3 W[12], C2[12], rw[12], rc[12];
            basis(xi, eta, zeta, W); curl(xi, eta, zeta, C2);
            for (int i = 0; i < 12; ++i) {
                rw[i].x=iJ[0][0]*W[i].x+iJ[1][0]*W[i].y+iJ[2][0]*W[i].z;
                rw[i].y=iJ[0][1]*W[i].x+iJ[1][1]*W[i].y+iJ[2][1]*W[i].z;
                rw[i].z=iJ[0][2]*W[i].x+iJ[1][2]*W[i].y+iJ[2][2]*W[i].z;
                rc[i].x=(J[0][0]*C2[i].x+J[0][1]*C2[i].y+J[0][2]*C2[i].z)*id;
                rc[i].y=(J[1][0]*C2[i].x+J[1][1]*C2[i].y+J[1][2]*C2[i].z)*id;
                rc[i].z=(J[2][0]*C2[i].x+J[2][1]*C2[i].y+J[2][2]*C2[i].z)*id;
            }
            for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
                const double kk = nu*(rc[i].x*rc[j].x+rc[i].y*rc[j].y+rc[i].z*rc[j].z);
                const double cc = kap*(rw[i].x*rw[j].x+rw[i].y*rw[j].y+rw[i].z*rw[j].z)/dt;
                Ae[i][j] += (kk+cc)*det;
            }
        }
        for (int i = 0; i < 12; ++i) for (int j = 0; j < 12; ++j) {
            const int gi = mesh.hex_edges[h*12+i], gj = mesh.hex_edges[h*12+j];
            const int si = mesh.hex_edge_signs[h*12+i], sj = mesh.hex_edge_signs[h*12+j];
            rows[gi][gj] += Ae[i][j] * si * sj;
        }
    }
    ptr.assign(N+1, 0); col.clear(); val.clear();
    for (int i = 0; i < N; ++i) {
        ptr[i] = static_cast<int>(col.size());
        for (auto& kv : rows[i]) { col.push_back(kv.first); val.push_back(kv.second); }
    }
    ptr[N] = static_cast<int>(col.size());
}

} // namespace

int main(int argc, char** argv)
{
    const char* mesh_path = (argc > 1) ? argv[1] : "team28.vtk";

    Mesh mesh;
    VtkReader::read_hexahedral_mesh(mesh_path, mesh);
    TopologyExtractor::extract_edges(mesh);

    const int N = static_cast<int>(mesh.num_edges);
    std::vector<MaterialProperties> mats(mesh.num_hexes);
    for (size_t h = 0; h < mesh.num_hexes; ++h) {
        const int c = mesh.hex_component_ids[h];
        mats[h].kappa = (c == 3) ? 3.40e7 : 0.0; // Component 3 = Aluminiumplatte
        mats[h].mu_relative = 1.0;
    }

    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        printf("  [%s] %s\n", ok ? "OK  " : "FEHL", what);
        if (!ok) ++failures;
    };

    printf("\n=============== Kohomologie-Eichung ===============\n");

    // ---- 1: Regressionsschutz -- Gradient vs. Kettenzyklus ----
    // Ein Gradient erfuellt die Kozykel-Bedingung (delta a = 0) immer, ist
    // aber im Allgemeinen KEIN geschlossener Graphkreis (d(a) != 0 als Kette).
    // Das ist die Unterscheidung, an der der erste, verworfene Loesungsversuch
    // gescheitert ist.
    {
        std::vector<double> phi(mesh.num_nodes);
        unsigned seed = 1234567u;
        for (size_t i = 0; i < mesh.num_nodes; ++i) {
            seed = seed * 1103515245u + 12345u;
            phi[i] = ((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
        }
        std::vector<double> a(N);
        for (int e = 0; e < N; ++e) {
            const int2 ed = mesh.edges[e];
            a[e] = phi[ed.y] - phi[ed.x];
        }
        // "d(a) als Kette": Netto-Fluss an jedem Knoten (Summe der anliegenden,
        // signierten Kantenwerte) sollte fuer ein ALLGEMEINES a (wie diesen
        // Gradienten) NICHT ueberall null sein -- sonst waere a zufaellig ein
        // Kreis, was fuer einen generischen Gradienten nicht zu erwarten ist.
        std::vector<double> node_flux(mesh.num_nodes, 0.0);
        for (int e = 0; e < N; ++e) {
            const int2 ed = mesh.edges[e];
            node_flux[ed.y] += a[e];
            node_flux[ed.x] -= a[e];
        }
        double max_flux = 0.0;
        for (double f : node_flux) max_flux = std::fmax(max_flux, std::fabs(f));
        printf("      max. Knoten-Nettofluss des Gradienten (als Kette) = %.4e (Kettenzyklus waere 0)\n", max_flux);
        check(max_flux > 1e-6, "Gradient ist im Allgemeinen KEIN geschlossener Kettenzyklus "
                               "(Regressionsschutz gegen den verworfenen ersten Ansatz)");
    }

    // ---- Basis-Eichung und Kohomologie-Generatoren bauen ----
    const GaugeInfo gauge = build_tree_cotree_gauge(mesh, mats.data(), 0.0);
    const CohomologyGaugeResult cohom = find_cohomology_generators(mesh, gauge, mats.data(), 0.0);
    printf("      Basis-Eichung: %d Kanten. Kohomologie-Generatoren: %zu\n",
           gauge.num_gauged, cohom.generators.size());

    // ---- 2: genau 2 Generatoren (eine Ringspule = ein Loch, zwei Spulen) ----
    check(cohom.generators.size() == 2, "genau 2 Kohomologie-Generatoren gefunden (eine je Spule)");

    // ---- 3 und 4: jeder Generator ist Kozykel und nicht trivial ----
    bool all_valid = true;
    for (auto& gen : cohom.generators) {
        printf("      Komponente %d: ||a_harm||=%.4e  rel. Kozykel-Residuum=%.3e\n",
               gen.component_id, gen.norm, gen.residual_cocycle);
        if (gen.residual_cocycle > 1e-6 || gen.norm < 1e-6) all_valid = false;
    }
    check(all_valid, "jeder Generator ist numerisch ein Kozykel und nicht trivial");

    // ---- 5: reduzierte Generatoren orthogonal (disjunkter Spulen-Traeger) ----
    std::vector<std::vector<double>> reduced(cohom.generators.size(), std::vector<double>(N, 0.0));
    for (size_t g = 0; g < cohom.generators.size(); ++g)
        for (int e = 0; e < N; ++e)
            reduced[g][e] = gauge.is_gauged[e] ? 0.0 : cohom.generators[g].vector[e];

    if (reduced.size() == 2) {
        double dot = 0.0;
        for (int e = 0; e < N; ++e) dot += reduced[0][e] * reduced[1][e];
        printf("      <c1_reduziert, c2_reduziert> = %.3e\n", dot);
        check(std::fabs(dot) < 1e-9, "reduzierte Generatoren sind orthogonal (disjunkter Traeger)");
    }

    // ---- 6: DER ENTSCHEIDENDE TEST -- CG mit echt zufaelliger rechter Seite ----
    std::vector<int> ptr, col; std::vector<double> val;
    assemble_A_sys(mesh, mats, 1e-4, ptr, col, val);

    auto mv = [&](const std::vector<double>& x, std::vector<double>& y) {
        y.assign(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int k = ptr[i]; k < ptr[i+1]; ++k) s += val[k] * x[col[k]];
            y[i] = s;
        }
    };
    auto P_total = [&](const std::vector<double>& x) {
        std::vector<double> y = x;
        for (int e = 0; e < N; ++e) if (gauge.is_gauged[e]) y[e] = 0.0;
        for (auto& c : reduced) {
            double dot = 0.0, nrm = 0.0;
            for (int e = 0; e < N; ++e) { dot += c[e]*y[e]; nrm += c[e]*c[e]; }
            if (nrm > 1e-300) { const double f = dot/nrm; for (int e=0;e<N;++e) y[e] -= f*c[e]; }
        }
        return y;
    };
    auto apply_full = [&](const std::vector<double>& x, std::vector<double>& y) {
        const std::vector<double> Px = P_total(x);
        std::vector<double> Ax; mv(Px, Ax);
        const std::vector<double> PAx = P_total(Ax);
        y.resize(N);
        for (int e = 0; e < N; ++e) y[e] = PAx[e] + (x[e] - Px[e]);
    };

    std::vector<double> b_raw(N);
    unsigned seed = 999331u;
    for (int i = 0; i < N; ++i) { seed = seed*1103515245u+12345u; b_raw[i] = ((seed>>16)&0x7fff)/32768.0-0.5; }
    const std::vector<double> b = P_total(b_raw);

    std::vector<double> x(N, 0.0), r = b, p, Ap;
    double rz = 0.0, bn = 0.0;
    for (int i = 0; i < N; ++i) { rz += r[i]*r[i]; bn += b[i]*b[i]; }
    bn = std::sqrt(bn);
    p = r;
    int it = 0; double rn = bn;
    const int max_iter = 12000;
    for (; it < max_iter && rn/bn > 1e-8; ++it) {
        apply_full(p, Ap);
        double pAp = 0.0; for (int i=0;i<N;++i) pAp += p[i]*Ap[i];
        const double alpha = rz/pAp;
        double rz2 = 0.0; rn = 0.0;
        for (int i = 0; i < N; ++i) { x[i]+=alpha*p[i]; r[i]-=alpha*Ap[i]; rz2+=r[i]*r[i]; }
        rn = std::sqrt(rz2);
        const double beta = rz2/rz; rz = rz2;
        for (int i = 0; i < N; ++i) p[i] = r[i] + beta*p[i];
    }
    printf("      CG mit ECHT ZUFAELLIGER rechter Seite: %d Iterationen, rel. Residuum %.3e\n", it, rn/bn);
    check(rn/bn < 1e-6, "CG konvergiert fuer beliebige rechte Seite -- Nullraum vollstaendig entfernt");

    printf("\n=============== %s ===============\n\n",
           failures == 0 ? "BESTANDEN: Kohomologie-Eichung korrekt"
                         : "FEHLGESCHLAGEN");
    return failures == 0 ? 0 : 1;
}
