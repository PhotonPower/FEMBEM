#ifndef TRACE_SURFACE_H
#define TRACE_SURFACE_H

// =============================================================================
// TraceSurface -- Oberflaechenstrukturen der Maxwell-Spur-Kopplung (Phase 1b)
// =============================================================================
// Reiner Host-Code, Double, ohne CUDA. Theorie: docs/theory/maxwell-trace-coupling.md,
// Abschnitte 3.2 (Lambda-Raum), 4.1 (harmonische Felder). Bericht: docs/design/maxwell-trace-operators.md.
//
// Eingabe: Knotenkoordinaten und Randquads, die Knotenreihenfolge jedes Quads laeuft
// GEGEN den Uhrzeigersinn von aussen gesehen, d. h. n = dr/du x dr/dv (bilineares Quad mit
// Knoten 0..3 auf (-1,-1),(+1,-1),(+1,+1),(-1,+1)) zeigt aus dem Koerper in die Luft. Das ist
// die Konvention von evaluate_rooftop_basis / evaluate_quad_normal (BemIntegration.cuh).
//
// Globale Kante e = (vmin, vmax), Orientierung vmin -> vmax. Lokale Kante i eines Quads:
// {0,1},{1,2},{2,3},{3,0}. quad_sign[q][i] = +1, wenn die lokale Richtung der globalen entspricht
// (gleiche Konvention wie mesh.boundary_edge_signs).
//
//   G: Knoten -> Kante,  (G psi)_e = psi[vmax] - psi[vmin]
//   R: Kante  -> Quad,   (R c)_q  = sum_i quad_sign[q][i] c[quad_edges[q][i]]  (Zirkulation um das Quad)
//   R G = 0 exakt (Kombinatorik).
// =============================================================================

#include <array>
#include <string>
#include <vector>

namespace trace {

struct Vec3 { double x = 0, y = 0, z = 0; };

struct Triplet { int row, col; double val; };

struct TraceSurface {
    // ---- Eingabe ----
    std::vector<Vec3> X;                       // Knoten
    std::vector<std::array<int, 4>> quads;     // Randquads (nach aussen orientiert)

    // ---- Topologie ----
    std::vector<std::array<int, 2>> edges;     // (vmin, vmax)
    std::vector<std::array<int, 4>> quad_edges;
    std::vector<std::array<int, 4>> quad_sign;
    std::vector<int> quad_comp;                // Zusammenhangskomponente je Quad
    std::vector<int> node_comp;
    std::vector<int> edge_comp;
    int num_components = 0;
    std::vector<int> comp_V, comp_E, comp_F;   // Zaehlungen je Komponente
    std::vector<int> comp_genus;               // g_j = (2 - (V-E+F)) / 2
    bool manifold_oriented = false;            // jede Kante in genau 2 Quads, entgegengesetzt durchlaufen
    std::string diagnostics;                   // Text, falls manifold_oriented == false

    int num_nodes() const { return static_cast<int>(X.size()); }
    int num_quads() const { return static_cast<int>(quads.size()); }
    int num_edges() const { return static_cast<int>(edges.size()); }
    int total_genus() const { int s = 0; for (int g : comp_genus) s += g; return s; }

    // ---- dünne Matrizen ----
    std::vector<Triplet> G_triplets() const;   // E x V
    std::vector<Triplet> R_triplets() const;   // F x E
    std::vector<double> G_dense() const;       // E x V, zeilenweise
    std::vector<double> R_dense() const;       // F x E, zeilenweise

    // ---- harmonische Felder ----
    // Spalten H_k (E x 2*sum g), k geordnet nach Komponente. Jede Spalte ist durch Kantenkoeffizienten
    // c_e = int_e w . dl eines tangentialen Feldes w = sum c_e W_e gegeben; sie liegt in ker R, ist
    // orthogonal zu range G und traegt nur auf einer Komponente. Basis orthonormal (Euklid).
    std::vector<double> H;       // E x nH, zeilenweise
    std::vector<int> H_comp;     // Komponente je Spalte
    int nH = 0;
    double harmonic_residual_R = 0.0;   // max ||R H_k||
    double harmonic_residual_G = 0.0;   // max ||G^T H_k||
};

// Baut Kanten, Quadkanten, Zusammenhang, Geschlecht.
TraceSurface build_trace_surface(const std::vector<Vec3>& X,
                                 const std::vector<std::array<int, 4>>& quads);

// Berechnet die 2*sum g harmonischen Felder (CG-Projektion wie in CohomologyGauge.cpp,
// nur auf der Oberflaeche). Fuellt S.H, S.H_comp, S.nH. Wirft std::runtime_error, wenn die
// gefundene Dimension nicht 2*sum g ist.
void compute_harmonic_fields(TraceSurface& S, unsigned seed = 12345u);

// Zirkulation sum_i s_i c[e_i] eines Kantenzyklus. cycle[i] = (Kante, Vorzeichen).
double cycle_circulation(const std::vector<std::pair<int, int>>& cycle, const double* c);

// Normiert die Spalten von S.H so um, dass die Zirkulationsmatrix C_{ik} = circ(cycle_i, H_k) die
// Einheitsmatrix wird (Spalten der ersten Komponente, nur sinnvoll fuer genau 2g Zyklen je Komponente).
// Gibt den kleinsten Singulaerwert der Zirkulationsmatrix vor der Normierung zurueck.
double normalize_harmonic_to_cycles(TraceSurface& S, int comp,
                                    const std::vector<std::vector<std::pair<int, int>>>& cycles);

} // namespace trace

#endif
