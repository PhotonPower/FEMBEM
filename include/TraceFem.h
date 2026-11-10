#ifndef TRACE_FEM_H
#define TRACE_FEM_H

// =============================================================================
// FEM-Seite der Maxwell-Spur-Kopplung, dicht, Host (Phase 1b, Schritt 4)
// =============================================================================
// Liest ein Hexaedernetz (VtkReader), bildet Kantenstruktur und Rand (TopologyExtractor) und die
// Baum-Kobaum-Eichung (build_tree_cotree_gauge) des Bestands und assembliert K = int nu curl a . curl v,
// M_sigma = int kappa a . v und den Quellterm f = int J_s . W_e mit den Produktionsbasisfunktionen
// (NedelecHexahedron.cuh, 2x2x2 Gauss fuer K und M_sigma; 3x3x3 Gauss fuer f). Koordinaten wie in der Datei
// (kein Millimeter-Faktor), mu0 frei waehlbar (Tests: 1). Reiner C++-Header; die Implementierung steht in
// src/TraceFem.cu, weil Mesh.cuh / NedelecHexahedron.cuh CUDA-Header sind.
// =============================================================================

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "TraceSurface.h"

namespace trace {

struct FemMaterials {
    std::vector<double> mu_r;    // je Komponenten-ID (Index = hex_component_id)
    std::vector<double> kappa;   // Leitfaehigkeit je Komponenten-ID
    double mu_r_of(int c) const { return c < static_cast<int>(mu_r.size()) ? mu_r[c] : 1.0; }
    double kappa_of(int c) const { return c < static_cast<int>(kappa.size()) ? kappa[c] : 0.0; }
};

struct FemProblem {
    int num_nodes = 0, num_hexes = 0, num_edges = 0;
    double mu0 = 1.0;
    std::vector<Vec3> X;                       // alle Knoten
    std::vector<std::array<int, 8>> hex;
    std::vector<int> hex_comp;
    std::vector<std::array<int, 2>> edges;     // (vmin, vmax)
    std::vector<Triplet> K;                    // E x E, Tripletliste (doppelte Eintraege werden summiert): int nu curl a . curl v
    std::vector<Triplet> Msig;                 // E x E: int kappa a . v
    std::vector<double> f;                     // E: int J_s . W_e (nicht projiziert)
    std::vector<char> gauged;                  // E: Baum-Kobaum (nur Gebiete mit kappa <= 0)
    int num_gauged = 0;

    TraceSurface surface;                      // Rand: Quads aller Randflaechen, Knoten monoton umnumeriert
    std::vector<int> surf_edge_to_fem;         // Flaechenkante -> FEM-Kante
    std::vector<int> surf_node_to_fem;         // Flaechenknoten -> FEM-Knoten

    // Rotation von a (Kantenkoeffizienten) im Hexaedermittelpunkt
    Vec3 curl_at_center(int h, const std::vector<double>& a) const;
    Vec3 center(int h) const;
};

// Js(x, comp): Stromdichte am Punkt x im Hexaeder der Komponente comp (nur fuer comp mit Quelle von null verschieden)
// length_scale multipliziert alle Koordinaten (z. B. 1e-3 fuer Netze in Millimetern). comp_shift (optional, Index = Komponenten-ID,
// in den Einheiten NACH length_scale) verschiebt alle Knoten der Hexaeder dieser Komponente (z. B. die Platte auf eine andere Hoehe).
FemProblem build_fem_problem(const std::string& vtk_path, const FemMaterials& mats, double mu0,
                             const std::function<Vec3(const Vec3&, int)>& Js, double length_scale = 1.0,
                             const std::vector<Vec3>* comp_shift = nullptr);

// Baum-Kobaum-Eichung (Spannwald des Knotengraphen ueber alle Kanten, Kruskal) fuer ein Gebiet ohne Leiter. seed = 0: Kanten in
// aufsteigender Reihenfolge; sonst eine zufaellige Permutation der Kanten (anderer Baum). Gibt gauged (1 = Baumkante) zurueck.
std::vector<char> make_tree_gauge(const FemProblem& F, unsigned seed);

// Gradientenprojektion des Quellterms (fem_direct::project_out_gradient des Bestands), in place.
// Gibt ||G^T f|| / ||f|| vorher und nachher zurueck.
struct ProjectionResult { double rel_before = 0, rel_after = 0; };
// blocked (optional, E Eintraege): 1 = Kante nimmt nicht teil (b bleibt dort null). Noetig bei Netzen aus mehreren GETRENNTEN
// Koerpern: project_out_gradient bereinigt nur EINE globale Konstante; ohne Maske waechst ||f|| auf dem Netz Ring + Platte von 1,0 auf 4,2
// (gemessen, test_trace_system), mit Maske fuer die Platte ist das Ergebnis exakt.
ProjectionResult project_source(const std::string& vtk_path, std::vector<double>& f, const std::vector<char>* blocked = nullptr);

// ---- Quelle: Projektion und Renormierung des Nettostroms (Phase 2, Schritt 0b) ----
// Schnittschichten eines Rings um die z-Achse: die Knoten der Komponente liegen auf K Halbebenen phi = const (Gmsh-Revolve bzw. Testringe);
// Schicht k = alle Kanten zwischen Halbebene k und k+1 (zyklisch), Vorzeichen +1, wenn die globale Richtung (vmin -> vmax) in wachsendes phi zeigt.
// <f, v_cut(k)> = Summe der Vorzeichen mal f ueber die Schicht ist der Strom durch die Schnittflaeche (fuer diskret divergenzfreies f unabhaengig von k).
struct CutLayers {
    int K = 0;
    std::vector<std::vector<std::pair<int, double>>> layers;
    std::vector<char> in_comp_edge;   // E: Kante gehoert zur Komponente (beide Knoten)
    double net(const std::vector<double>& f, int k) const { double s = 0; for (const auto& le : layers[k]) s += le.second * f[le.first]; return s; }
};
CutLayers build_cut_layers(const FemProblem& F, int comp);

struct CoilSource { int comp; double target; };   // target = N I (signiert, in A)
struct SourceDiagnostics {
    int comp = 0; int K = 0;
    double rel_gt_before = 0, rel_gt_after = 0;           // ||G^T f|| / ||f|| auf den Kanten der Komponente
    double net_min_before = 0, net_max_before = 0;        // Nettostrom ueber die Schichten vor der Projektion
    double net_proj = 0, net_proj_spread = 0;             // nach der Projektion (Mittel, (max-min)/|mittel|)
    double scale = 1.0;                                   // Renormierungsfaktor target / net_proj
    double net_final = 0, net_final_spread = 0;
};
// Projiziert den Gradientenanteil je Spule (Maske: nur Kanten der Spule) und renormiert dann den Nettostrom je Spule auf target.
// project = false / renormalize = false schalten die Schritte fuer die Gegenproben ab.
std::vector<double> prepare_source(const FemProblem& F, const std::string& vtk_path, const std::vector<CoilSource>& coils, bool project,
                                   bool renormalize, std::vector<SourceDiagnostics>* diag = nullptr);

} // namespace trace

#endif
