#ifndef TOPOLOGY_EXTRACTOR_H
#define TOPOLOGY_EXTRACTOR_H

#include "Mesh.cuh"

class TopologyExtractor {
public:
    static void extract_edges(Mesh& mesh);

    // Sucht alle äußeren Flächen und verknüpft sie mit den FEM-Kanten
    static void extract_boundary_faces(Mesh& mesh);

    // Extrahiert die Grenzflaeche EINER Materialkomponente (component_id,
    // z.B. der Aluminiumplatte) zu ihrer Umgebung -- eine geschlossene Flaeche,
    // die NUR diese Komponente umschliesst. Anders als extract_boundary_faces()
    // (aeussere BEM-Kopplungsflaeche des gesamten Netzes) betrachtet diese
    // Funktion auch INNERE Flaechen (zwischen zwei Hexaedern verschiedener
    // Component-ID) und waehlt jeweils die Seite der Komponente selbst, damit
    // die resultierende Normale konsistent von der Komponente weg zeigt. Setzt
    // mesh.hex_component_ids voraus (vom VtkReader befuellt) -- KEINE
    // Abhaengigkeit von assign_materials_by_component_id() noetig, das liest
    // dieselben rohen IDs nur fuer die PhysicalObject-Zuweisung aus. Ergebnis
    // in mesh.component_surface_faces / mesh.num_component_surface_faces.
    // Baut anders als extract_boundary_faces() KEIN BEM-to-FEM Edge-Mapping
    // (nicht Teil des BEM-Systems, nur eine geometrische Integrationsflaeche
    // fuer die Kraftberechnung, siehe compute_maxwell_loads_kernel).
    static void extract_component_surface_faces(Mesh& mesh, int component_id);
};

#endif // TOPOLOGY_EXTRACTOR_H