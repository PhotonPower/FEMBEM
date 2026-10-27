#ifndef VTK_EXPORTER_H
#define VTK_EXPORTER_H

#include <string>
#include "Mesh.cuh"
#include "MaterialTypes.cuh"

class VtkExporter {
public:
    // Interpoliert die Kantenfreiheitsgrade ins Elementzentrum und schreibt
    // eine Standard ASCII VTK Datei für ParaView (Unstructured Grid).
    static void export_hexahedral_cell_data(
        const std::string& filename,
        const Mesh& mesh,
        const MaterialProperties* d_materials,
        const double* d_A_current,
        const double* d_A_previous,
        double dt
    );
};

#endif // VTK_EXPORTER_H