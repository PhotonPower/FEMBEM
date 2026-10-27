// =============================================================================
// test_mesh_refinement.cu -- Abnahmepruefung der uniformen Netzverfeinerung
// =============================================================================
//
// Geprueft wird an einem kleinen 2x2x2-Blocknetz, einmal affin und einmal
// bewusst VERZERRT (die Platte in TEAM 28 ist ein mit Hexaedern vernetzter
// Zylinder, ihre Elemente sind also nicht achsparallel):
//
//   1. Elementzahl x8 je Stufe
//   2. Knotenzahl = Ecken + Kantenmitten + Flaechenmitten + Zellmitten
//   3. KONFORMITAET -- keine zwei Knoten an derselben Stelle. Das ist die
//      kritische Eigenschaft: legte jedes Element seine neuen Knoten selbst an,
//      entstuenden Duplikate an geteilten Kanten und Flaechen, das Netz waere
//      zerrissen (haengende Knoten ohne Kopplung), und FEM wie BEM wuerden
//      stillschweigend Unsinn rechnen.
//   4. Orientierung: alle Kinder rechtsorientiert (detJ > 0)
//   5. Volumen -- bei affinen Elementen exakt erhalten
//   6. Component-IDs werden an die Kinder vererbt (sonst bricht die
//      Materialzuweisung nach Component ID)
//
// Braucht keine GPU.
// =============================================================================

#include "MeshRefinement.cuh"
#include "NedelecDispatch.cuh"
#include <algorithm>
#include "NodalHexahedron.cuh"
#include <cstdio>
#include <cmath>
#include <map>
#include <set>
#include <vector>
#include <array>

static int fails=0;
static void ck(bool ok,const char*w){ printf("  [%s] %s\n", ok?"OK ":"FEHL", w); if(!ok) ++fails; }

// 2x2x2-Bloecke, absichtlich VERZERRT (obere Ebene verschoben)
static void build(Mesh& m,int n,bool distort){
    std::vector<double3> p; std::map<std::array<int,3>,int> lut;
    auto id=[&](int i,int j,int k){ std::array<int,3> a{i,j,k}; auto it=lut.find(a);
        if(it!=lut.end()) return it->second;
        double x=i, y=j, z=k;
        if(distort){ x += 0.18*k*std::sin(1.7*j); y += 0.13*k; z += 0.09*i*j; }
        int q=(int)p.size(); p.push_back(make_double3(x,y,z)); lut[a]=q; return q; };
    std::vector<Hexahedron> hx; std::vector<int> cid;
    for(int i=0;i<n;++i)for(int j=0;j<n;++j)for(int k=0;k<n;++k){
        Hexahedron h{};
        h.v[0]=id(i,j,k);     h.v[1]=id(i+1,j,k);   h.v[2]=id(i+1,j+1,k);   h.v[3]=id(i,j+1,k);
        h.v[4]=id(i,j,k+1);   h.v[5]=id(i+1,j,k+1); h.v[6]=id(i+1,j+1,k+1); h.v[7]=id(i,j+1,k+1);
        hx.push_back(h); cid.push_back((i+j+k)%3+1);
    }
    m.num_nodes=p.size(); m.num_hexes=hx.size();
    m.nodes_initial.allocate(m.num_nodes); m.nodes_current.allocate(m.num_nodes);
    m.hex_nodes.allocate(m.num_hexes); m.hex_component_ids.allocate(m.num_hexes);
    for(size_t q=0;q<p.size();++q){ m.nodes_initial[q]=p[q]; m.nodes_current[q]=p[q]; }
    for(size_t q=0;q<hx.size();++q){ m.hex_nodes[q]=hx[q]; m.hex_component_ids[q]=cid[q]; }
}
static double vol(const Mesh&m){ double v=0;
    for(size_t h=0;h<m.num_hexes;++h){ Hexahedron H=m.hex_nodes[h]; double3 nd[8];
        for(int n=0;n<8;++n) nd[n]=m.nodes_current[H.v[n]];
        for(int g=0;g<nodal_gauss_count_n(3);++g){ double xi,e,z,w;
            nodal_gauss_point_n(3,g,xi,e,z,w); double3 gr[8]; double dJ;
            evaluate_nodal_gradients(nd,xi,e,z,gr,dJ); v+=w*dJ; } }
    return v; }
static int bad_orient(const Mesh&m){ int b=0;
    for(size_t h=0;h<m.num_hexes;++h){ Hexahedron H=m.hex_nodes[h]; double3 nd[8];
        for(int n=0;n<8;++n) nd[n]=m.nodes_current[H.v[n]];
        double3 gr[8]; double dJ; evaluate_nodal_gradients(nd,0,0,0,gr,dJ);
        if(dJ<=0) ++b; } return b; }


// =============================================================================
// Teil 2: TETRAEDER
// =============================================================================
//
// Angehaengt statt danebengestellt: es gibt genau ein Ziel
// test_mesh_refinement, und ein zweites gleichen Namens laesst CMake mit
// CMP0002 abbrechen -- dann wird gar nichts mehr uebersetzt und man faehrt
// unbemerkt eine alte Binaerdatei weiter.
//
// Der Tetraeder teilt anders als der Hexaeder: vier Eckkinder plus ein
// inneres OKTAEDER, das laengs einer seiner drei Raumdiagonalen in vier
// zerlegt wird. Genommen wird die kuerzeste (Bey).
//
// Zwei Pruefungen sind hier schaerfer als im Hexaederteil:
//
//   Volumen. Der Tetraeder wird AFFIN abgebildet, immer -- die acht Kinder
//   muessen ihn also bis auf Rundung exakt ausfuellen, auch auf einem
//   verzerrten Netz. Beim Hexaeder gilt das nur im affinen Fall.
//
//   Flaechenzaehlung. Keine Flaeche darf mehr als zweimal vorkommen, und die
//   Randflaechen muessen sich genau vervierfachen. Das faengt Ueberlappung
//   UND Luecken, die eine blosse Knotenzaehlung nicht sieht.
//
// Beides war noetig: die Tabelle der Oktaeder-Aequatoren hatte anfangs bei
// zwei der drei Diagonalen einen Endpunkt der eigenen Diagonale im Ring. Die
// Innenkinder deckten dann nur die Haelfte des Oktaeders ab -- ein Viertel
// des Volumens verschwand und die Elemente ueberlappten sich zugleich. An
// einem Kuhn-Netz allein waere es nicht aufgefallen, weil dessen Elemente
// praktisch nie die eine Diagonale waehlen, die richtig war.
// =============================================================================

static void ck2(const char* w, bool ok, const char* extra = "") {
    printf("  [%s] %s%s\n", ok ? "OK " : "FEHL", w, extra);
    if (!ok) ++fails;
}

struct TestMesh {
    std::vector<double3> pts;
    std::vector<std::vector<int>> tets;
    double h = 0.0;   // nominale Netzweite
};

static double frand(unsigned& seed) {
    seed = seed * 1103515245u + 12345u;
    return static_cast<double>((seed >> 8) & 0xFFFFFF) / 16777216.0;  // [0,1)
}

static double tet_volume(const double3& a, const double3& b,
                         const double3& c, const double3& d) {
    const double3 e1 = {b.x - a.x, b.y - a.y, b.z - a.z};
    const double3 e2 = {c.x - a.x, c.y - a.y, c.z - a.z};
    const double3 e3 = {d.x - a.x, d.y - a.y, d.z - a.z};
    return ((e1.y * e2.z - e1.z * e2.y) * e3.x
          + (e1.z * e2.x - e1.x * e2.z) * e3.y
          + (e1.x * e2.y - e1.y * e2.x) * e3.z) / 6.0;
}

// Der Kasten liegt bewusst NEBEN der z-Achse (x von 1 bis 2). Ein azimutales
// J_s = J * phi_hat ist auf der Achse undefiniert; dort greift im
// Produktionskernel die Abfrage r < 1e-12, und der Test wuerde eine
// Sonderfallbehandlung messen statt der Quadratur.
static TestMesh build_kuhn_mesh(int N, double distort, unsigned seed) {
    TestMesh m;
    const double3 lo = {1.0, -0.5, 0.0};
    const double3 hi = {2.0,  0.5, 1.0};
    m.h = 1.0 / N;

    const auto nid = [N](int i, int j, int k) {
        return (k * (N + 1) + j) * (N + 1) + i;
    };

    for (int k = 0; k <= N; ++k)
    for (int j = 0; j <= N; ++j)
    for (int i = 0; i <= N; ++i) {
        double3 p;
        p.x = lo.x + (hi.x - lo.x) * i / N;
        p.y = lo.y + (hi.y - lo.y) * j / N;
        p.z = lo.z + (hi.z - lo.z) * k / N;
        // Nur INNERE Knoten verschieben. Der Rand bleibt eben, sonst waere
        // "innerer Knoten" nicht mehr sauber vom Gebietsrand zu trennen.
        const bool interior = (i > 0 && i < N) && (j > 0 && j < N) && (k > 0 && k < N);
        if (interior && distort > 0.0) {
            p.x += distort * m.h * (2.0 * frand(seed) - 1.0);
            p.y += distort * m.h * (2.0 * frand(seed) - 1.0);
            p.z += distort * m.h * (2.0 * frand(seed) - 1.0);
        }
        m.pts.push_back(p);
    }

    // Die sechs Kuhn-Tetraeder als Eckindizes 0..7 des Einheitswuerfels,
    // wobei v = i + 2j + 4k. Jeder laeuft von 0 ueber zwei Zwischenecken
    // nach 7, einmal je Achsenreihenfolge.
    const int KUHN[6][4] = {
        {0, 1, 3, 7},   // x, y, z
        {0, 1, 5, 7},   // x, z, y
        {0, 2, 3, 7},   // y, x, z
        {0, 2, 6, 7},   // y, z, x
        {0, 4, 5, 7},   // z, x, y
        {0, 4, 6, 7},   // z, y, x
    };

    for (int k = 0; k < N; ++k)
    for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i) {
        int corner[8];
        for (int c = 0; c < 8; ++c)
            corner[c] = nid(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));

        for (int t = 0; t < 6; ++t) {
            std::vector<int> e = {corner[KUHN[t][0]], corner[KUHN[t][1]],
                                  corner[KUHN[t][2]], corner[KUHN[t][3]]};
            // Positive Orientierung erzwingen. Fuer die kontravariante
            // Piola-Transformation waere ein negatives detJ zwar konsistent,
            // aber ein Netz mit gemischten Vorzeichen verdeckt echte Fehler.
            if (tet_volume(m.pts[e[0]], m.pts[e[1]], m.pts[e[2]], m.pts[e[3]]) < 0.0)
                std::swap(e[2], e[3]);
            m.tets.push_back(e);
        }
    }
    return m;
}

static void fill_mesh(Mesh& mesh, const TestMesh& tm) {
    mesh.num_nodes = tm.pts.size();
    mesh.nodes_initial.allocate(tm.pts.size());
    mesh.nodes_current.allocate(tm.pts.size());
    for (size_t i = 0; i < tm.pts.size(); ++i) {
        mesh.nodes_initial[i] = tm.pts[i];
        mesh.nodes_current[i] = tm.pts[i];
    }

    mesh.num_hexes = tm.tets.size();
    mesh.hex_nodes.allocate(tm.tets.size());
    mesh.hex_component_ids.allocate(tm.tets.size());
    mesh.elem_types.allocate(tm.tets.size());
    for (size_t i = 0; i < tm.tets.size(); ++i) {
        Hexahedron h;
        for (int k = 0; k < 8; ++k) h.v[k] = -1;
        for (int k = 0; k < 4; ++k) h.v[k] = tm.tets[i][k];
        mesh.hex_nodes[i] = h;
        mesh.hex_component_ids[i] = 0;
        mesh.elem_types[i] = ELEM_TET;
    }
}




/// Summe der Tetraedervolumina. Eigene Funktion, NICHT das vol() des
/// Hexaederteils: das wertet die trilineare Abbildung ueber acht Knoten aus
/// und liefert auf einem Tetraeder mit v[4..7] = -1 eine bedeutungslose Zahl.
/// Genau diese Verwechslung stand einmal auch in refine_tet_mesh und liess
/// die Volumenprobe -1.27 m^3 fuer ein Netz mit 1.0 m^3 melden.
static double tet_vol(const Mesh& m) {
    double v = 0.0;
    for (size_t h = 0; h < m.num_hexes; ++h) {
        const Hexahedron H = m.hex_nodes[h];
        const double3 a = m.nodes_current[H.v[0]], b = m.nodes_current[H.v[1]];
        const double3 c = m.nodes_current[H.v[2]], d = m.nodes_current[H.v[3]];
        const double3 e1 = {b.x-a.x, b.y-a.y, b.z-a.z};
        const double3 e2 = {c.x-a.x, c.y-a.y, c.z-a.z};
        const double3 e3 = {d.x-a.x, d.y-a.y, d.z-a.z};
        v += ((e1.y*e2.z - e1.z*e2.y)*e3.x + (e1.z*e2.x - e1.x*e2.z)*e3.y
            + (e1.x*e2.y - e1.y*e2.x)*e3.z) / 6.0;
    }
    return v;
}

static int tet_checks() {
    Mesh mesh;
    const TestMesh tm = build_kuhn_mesh(3, 0.25, 20260909u);
    fill_mesh(mesh, tm);
    const double v0 = tet_vol(mesh);
    const size_t h0 = mesh.num_hexes, n0 = mesh.num_nodes;
    std::map<int,int> cid0;
    for (size_t h=0; h<mesh.num_hexes; ++h) ++cid0[mesh.hex_component_ids[h]];

    printf("\n== Tetraeder-Verfeinerung ==\n");
    refine_mesh(mesh, 1);

    printf("\n");
    ck2("Elementzahl x8", mesh.num_hexes == h0*8);
    char buf[128];
    const double v1 = tet_vol(mesh);
    snprintf(buf,sizeof buf,"   (%.17g -> %.17g, rel %.2e)", v0, v1, std::fabs(v1-v0)/std::fabs(v0));
    ck2("Volumen exakt erhalten", std::fabs(v1-v0) <= 1e-12*std::fabs(v0), buf);

    bool pos = true;
    for (size_t h=0; h<mesh.num_hexes; ++h) {
        const Hexahedron H = mesh.hex_nodes[h];
        const double3 a=mesh.nodes_current[H.v[0]], b=mesh.nodes_current[H.v[1]];
        const double3 c=mesh.nodes_current[H.v[2]], d=mesh.nodes_current[H.v[3]];
        const double3 e1={b.x-a.x,b.y-a.y,b.z-a.z}, e2={c.x-a.x,c.y-a.y,c.z-a.z}, e3={d.x-a.x,d.y-a.y,d.z-a.z};
        if (((e1.y*e2.z-e1.z*e2.y)*e3.x + (e1.z*e2.x-e1.x*e2.z)*e3.y + (e1.x*e2.y-e1.y*e2.x)*e3.z) <= 0) pos=false;
    }
    ck2("alle Kinder positiv orientiert", pos);

    // Konformitaet: jede Flaeche kommt 1x (Rand) oder 2x (innen) vor. 3x oder
    // mehr hiesse ueberlappende Elemente, und ein haengender Knoten zeigte
    // sich als Flaeche, die nur einmal vorkommt OBWOHL sie innen liegt.
    std::map<std::vector<int>,int> fc;
    for (size_t h=0; h<mesh.num_hexes; ++h) {
        const Hexahedron H = mesh.hex_nodes[h];
        for (int f=0; f<4; ++f) {
            int loc[4]; int nn = element_local_face_nodes(ELEM_TET,f,loc);
            std::vector<int> k; for (int i=0;i<nn;++i) k.push_back(H.v[loc[i]]);
            std::sort(k.begin(),k.end()); ++fc[k];
        }
    }
    int once=0, twice=0, more=0;
    for (auto&kv:fc){ if(kv.second==1)++once; else if(kv.second==2)++twice; else ++more; }
    snprintf(buf,sizeof buf,"   (%d Rand, %d innen, %d mehrfach)", once, twice, more);
    ck2("keine Flaeche kommt mehr als zweimal vor", more==0, buf);

    // Randflaechen des verfeinerten Netzes muessen genau 4x so viele sein wie
    // vorher -- jede Randflaeche wird in vier geteilt. Stimmt das nicht, ist
    // die Huelle zerrissen.
    std::map<std::vector<int>,int> fc0;
    Mesh m0; fill_mesh(m0, tm);
    for (size_t h=0; h<m0.num_hexes; ++h) {
        const Hexahedron H = m0.hex_nodes[h];
        for (int f=0; f<4; ++f) {
            int loc[4]; int nn = element_local_face_nodes(ELEM_TET,f,loc);
            std::vector<int> k; for (int i=0;i<nn;++i) k.push_back(H.v[loc[i]]);
            std::sort(k.begin(),k.end()); ++fc0[k];
        }
    }
    int once0=0; for (auto&kv:fc0) if (kv.second==1) ++once0;
    snprintf(buf,sizeof buf,"   (%d -> %d, erwartet %d)", once0, once, once0*4);
    ck2("Randflaechen genau vervierfacht", once == once0*4, buf);

    std::map<int,int> cid1;
    for (size_t h=0; h<mesh.num_hexes; ++h) ++cid1[mesh.hex_component_ids[h]];
    bool cid_ok = cid1.size()==cid0.size();
    for (auto&kv:cid0) if (cid1[kv.first] != kv.second*8) cid_ok=false;
    ck2("Component-IDs an alle 8 Kinder vererbt", cid_ok);

    bool types_ok = mesh.elem_types.size()==mesh.num_hexes && mesh.is_pure_tet();
    ck2("elem_types mitgewachsen und rein Tetraeder", types_ok);

    // Gemischtes Netz muss abbrechen
    Mesh mixed; fill_mesh(mixed, tm);
    mixed.elem_types[0] = ELEM_HEX;
    bool threw=false;
    std::string msg;
    try { refine_mesh(mixed, 1); } catch (const std::exception& e) { threw=true; msg=e.what(); }
    ck2("gemischtes Netz wirft", threw);
    if (threw) printf("    -> %.150s\n", msg.c_str());

    return 0;
}

int main(){
    bf::g_backend = bf::Backend::CPU;
    for(int dis=0; dis<2; ++dis){
        printf("\n=== %s Netz, 2x2x2 Bloecke ===\n", dis?"VERZERRTES":"affines");
        Mesh m; build(m,2,dis==1);
        const size_t h0=m.num_hexes, n0=m.num_nodes; const double v0=vol(m);
        std::set<int> cid0; for(size_t h=0;h<h0;++h) cid0.insert(m.hex_component_ids[h]);

        refine_hex_mesh(m,1);

        ck(m.num_hexes==h0*8,"Elementzahl x8");
        // Erwartete Knotenzahl: Ecken + Kanten + Flaechen + Zellen des Ausgangsnetzes
        // 3x3x3 Knoten, 2x2x2 Zellen -> Kanten 54, Flaechen 36, Zellen 8
        ck(m.num_nodes==27+54+36+8,"Knotenzahl = Ecken+Kanten+Flaechen+Zellen");
        printf("      %zu -> %zu Knoten (erwartet %d)\n", n0, m.num_nodes, 27+54+36+8);

        // KONFORMITAET: keine zwei Knoten an derselben Stelle
        std::map<std::array<long long,3>,int> seen; int dup=0;
        for(size_t q=0;q<m.num_nodes;++q){ double3 p=m.nodes_current[q];
            auto r=[](double v){ return (long long)llround(v*1e9); };
            std::array<long long,3> k{r(p.x),r(p.y),r(p.z)};
            if(seen.count(k)) ++dup; else seen[k]=(int)q; }
        ck(dup==0,"konform: keine doppelten Knoten an derselben Stelle");
        if(dup) printf("      %d Duplikate -> haengende Knoten, Netz waere zerrissen\n",dup);

        ck(bad_orient(m)==0,"alle Kinder rechtsorientiert (detJ > 0)");

        const double v1=vol(m); const double rel=std::fabs(v1-v0)/std::fabs(v0);
        printf("      Volumen %.12f -> %.12f, rel. %.3e\n", v0, v1, rel);
        if(dis==0) ck(rel<1e-13,"affin: Volumen EXAKT erhalten");
        else       ck(rel<2e-2, "verzerrt: Volumen aendert sich nur wenig");

        std::set<int> cid1; for(size_t h=0;h<m.num_hexes;++h) cid1.insert(m.hex_component_ids[h]);
        ck(cid0==cid1,"Component-IDs unveraendert vererbt");
    }
    tet_checks();

    printf("\n%s\n", fails?"FEHLGESCHLAGEN":"ALLE PRUEFUNGEN BESTANDEN");
    return fails?1:0;
}
