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
    std::vector<double> K;                     // E x E, dicht, zeilenweise: int nu curl a . curl v
    std::vector<double> Msig;                  // E x E: int kappa a . v
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
// length_scale multipliziert alle Koordinaten (z. B. 1e-3 fuer Netze in Millimetern).
FemProblem build_fem_problem(const std::string& vtk_path, const FemMaterials& mats, double mu0,
                             const std::function<Vec3(const Vec3&, int)>& Js, double length_scale = 1.0);

// Gradientenprojektion des Quellterms (fem_direct::project_out_gradient des Bestands), in place.
// Gibt ||G^T f|| / ||f|| vorher und nachher zurueck.
struct ProjectionResult { double rel_before = 0, rel_after = 0; };
// blocked (optional, E Eintraege): 1 = Kante nimmt nicht teil (b bleibt dort null). Noetig bei Netzen aus mehreren GETRENNTEN
// Koerpern: project_out_gradient bereinigt nur EINE globale Konstante; ohne Maske waechst ||f|| auf dem Netz Ring + Platte von 1,0 auf 4,2
// (gemessen, test_trace_system), mit Maske fuer die Platte ist das Ergebnis exakt.
ProjectionResult project_source(const std::string& vtk_path, std::vector<double>& f, const std::vector<char>* blocked = nullptr);

} // namespace trace

#endif
