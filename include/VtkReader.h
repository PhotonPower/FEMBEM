#ifndef VTK_READER_H
#define VTK_READER_H

#include <string>
#include "Mesh.cuh"

class VtkReader {
public:
    // Liest eine VTK-Datei (ASCII, Unstructured Grid) und füllt das übergebene Mesh.
    // Wirft std::runtime_error bei Dateifehlern oder Formatverletzungen.
    static void read_hexahedral_mesh(const std::string& filepath, Mesh& mesh);
};

#endif // VTK_READER_H