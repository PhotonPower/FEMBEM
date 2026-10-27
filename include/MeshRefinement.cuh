#ifndef MESH_REFINEMENT_CUH
#define MESH_REFINEMENT_CUH

// =============================================================================
// MeshRefinement.cuh -- uniforme Verfeinerung des Hexaedernetzes
// =============================================================================
//
// Aufruf:  ./fembem --refine=1 --reference
//
// Jedes Hexaeder wird in 2x2x2 = 8 Kinder geteilt. Nicht 4: eine Teilung in nur
// zwei Richtungen (2x2x1) wuerde die Plattendicke unveraendert lassen, und
// genau die ist hier knapp -- drei Elemente ueber 3 mm.
//
// Neue Knoten entstehen an den 12 Kantenmitten, den 6 Flaechenmitten und der
// Zellmitte. Ihre Lage folgt der TRILINEAREN Geometrieabbildung des
// Elternelements, ausgewertet an den Referenzpunkten mit Koordinaten aus
// {-1, 0, 1} -- dieselbe Abbildung, die die Q1-Formfunktionen definieren. Damit
// ist die Verfeinerung mit der FEM-Diskretisierung konsistent.
//
// -----------------------------------------------------------------------------
// Konformitaet
// -----------------------------------------------------------------------------
// Kantenmitten und Flaechenmitten werden von benachbarten Elementen GETEILT.
// Sie werden deshalb ueber einen kanonischen Schluessel gesucht: das sortierte
// Tupel der beteiligten Elternknoten-IDs. Wuerde jedes Element seine neuen
// Knoten selbst anlegen, entstuenden doppelte Knoten an derselben Stelle, das
// Netz waere nicht mehr zusammenhaengend, und FEM wie BEM wuerden stillschweigend
// Unsinn rechnen (haengende Knoten ohne Kopplung).
//
// -----------------------------------------------------------------------------
// Was die Verfeinerung NICHT verbessert
// -----------------------------------------------------------------------------
// Die Geometrie bleibt exakt dieselbe. Die Mitte einer geraden Kante liegt auf
// dieser Kante; der polyedrische Rand wird also nicht runder. Die Facettierung
// des Plattenrands und der Spulenmaentel bleibt unveraendert -- verfeinert wird
// die Auflaesung des FELDES, nicht die der Form. Wer die Kruemmung besser
// treffen will, muss die neuen Randknoten auf die analytische Flaeche
// projizieren; das tut diese Funktion bewusst nicht, weil dazu bekannt sein
// muesste, welche Flaeche zu welchem Radius gehoert.
//
// -----------------------------------------------------------------------------
// Groessen je Verfeinerungsstufe (aus der Topologie des TEAM-28-Netzes)
// -----------------------------------------------------------------------------
//   Stufe 0:   4805 Elemente,   7013 Knoten,  4136 Randquads,  4138 Randknoten
//   Stufe 1:  38440 Elemente,  46991 Knoten, 16544 Randquads, 16542 Randknoten
//   Stufe 2: 307520 Elemente, ~361000 Knoten, 66176 Randquads, ~66000 Randknoten
//
// ACHTUNG bei der vollbesetzten BEM: der Speicher waechst mit N_bem^2.
//   Stufe 0: eine dichte Matrix 0.13 GiB, drei davon 0.38 GiB
//   Stufe 1: eine dichte Matrix 2.04 GiB, drei davon 6.1 GiB   -- passt in 16 GiB
//   Stufe 2: eine dichte Matrix  33 GiB                        -- passt NICHT
// Mit vollbesetzter BEM ist auf einer 16-GiB-Karte also genau EINE Stufe
// moeglich. build_reference_boundary() prueft das und bricht sonst mit einer
// Meldung ab, statt an einer Allokation zu scheitern.
// =============================================================================

#include "Mesh.cuh"

/// Verfeinert das Netz 'levels' mal uniform. Prueft VORHER den Elementtyp und
/// waehlt danach den Weg: Hexaeder 2x2x2, Tetraeder 4 Ecken + Oktaeder.
///
/// Wirft bei einem GEMISCHTEN Netz und bei Prismen oder Pyramiden. Grund: die
/// beiden Typen teilen ihre gemeinsame Flaeche unterschiedlich -- der Hexaeder
/// in vier Vierecke ueber eine Flaechenmitte, der Tetraeder in vier Dreiecke
/// ohne eine. Dort entstuenden haengende Knoten, das Netz saehe unauffaellig
/// aus, die Assemblierung liefe durch, und das Feld waere an der Grenze nicht
/// gekoppelt.
///
/// Erwartet ein Netz direkt aus VtkReader und VOR der Topologieextraktion.
void refine_mesh(Mesh& mesh, int levels);

/// Verfeinert ein reines Hexaedernetz (je Stufe 1 -> 8, 2x2x2).
/// levels <= 0 laesst das Netz unveraendert.
///
/// Erwartet ein Netz direkt aus VtkReader (Knoten, Hexaeder,
/// hex_component_ids) und VOR der Topologieextraktion -- Kanten, Randflaechen
/// und Materialzuweisung muessen danach auf dem verfeinerten Netz aufgebaut
/// werden. Component-IDs werden an die Kinder vererbt, die Materialzuweisung
/// nach Component ID funktioniert also unveraendert.
void refine_hex_mesh(Mesh& mesh, int levels);

/// Verfeinert ein reines Tetraedernetz (je Stufe 1 -> 8 nach Freudenthal/Bey:
/// vier Eckkinder plus ein Oktaeder, dieser laengs seiner KUERZESTEN Diagonale
/// in vier geteilt).
///
/// Anders als beim Hexaeder ist die Abbildung affin, die acht Kinder fuellen
/// das Elternelement also exakt aus -- die Volumenprobe ist deshalb eine
/// echte Pruefung und keine blosse Anzeige.
///
/// Gleiche Erwartungen an den Aufrufzeitpunkt wie refine_hex_mesh.
void refine_tet_mesh(Mesh& mesh, int levels);

/// Vorhersage der Groessen ohne das Netz anzufassen -- fuer eine Warnung, bevor
/// gerechnet wird. num_boundary_* darf nullptr sein.
void predict_refined_sizes(size_t num_nodes, size_t num_hexes, size_t num_edges,
                           size_t num_boundary_faces, size_t num_boundary_nodes,
                           int levels,
                           size_t* out_nodes, size_t* out_hexes,
                           size_t* out_boundary_faces, size_t* out_boundary_nodes);

#endif // MESH_REFINEMENT_CUH
