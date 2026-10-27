#include "VtkReader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

void VtkReader::read_hexahedral_mesh(const std::string& filepath, Mesh& mesh) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("VtkReader Error: Could not open file: " + filepath);
    }

    std::string token;
    bool has_points = false;
    bool has_cells = false;
    bool has_cell_types = false;

    // Wir iterieren Wort für Wort durch die Datei (ignoriert beliebige Whitespaces/Newlines)
    while (file >> token) {
        
        // 1. Prüfen auf den richtigen Datensatz-Typ
        if (token == "DATASET") {
            file >> token;
            if (token != "UNSTRUCTURED_GRID") {
                throw std::runtime_error("VtkReader Error: Only UNSTRUCTURED_GRID is supported. Found: " + token);
            }
        }
        
        // 2. Einlesen der Knotenkoordinaten (POINTS)
        else if (token == "POINTS") {
            size_t num_points;
            std::string data_type;
            
            if (!(file >> num_points >> data_type)) {
                throw std::runtime_error("VtkReader Error: Malformed POINTS definition.");
            }

            mesh.num_nodes = num_points;
            mesh.nodes_initial.allocate(num_points);
            mesh.nodes_current.allocate(num_points);
            
            // Prefetching für schnellen Schreibzugriff durch die CPU
            mesh.nodes_initial.prefetch_to_host();
            mesh.nodes_current.prefetch_to_host();

            for (size_t i = 0; i < num_points; ++i) {
                double x, y, z;
                if (!(file >> x >> y >> z)) {
                    throw std::runtime_error("VtkReader Error: Not enough coordinates provided for POINTS.");
                }
                
                mesh.nodes_initial[i] = make_double3(x, y, z);
                mesh.nodes_current[i] = make_double3(x, y, z); // Zu t=0 sind initial und current identisch
            }
            has_points = true;
            std::cout << "[VtkReader] Successfully read " << num_points << " points." << std::endl;
        }
        
        // 3. Einlesen der Element-Konnektivität (CELLS)
        else if (token == "CELLS") {
            size_t num_cells, total_values;
            
            if (!(file >> num_cells >> total_values)) {
                throw std::runtime_error("VtkReader Error: Malformed CELLS definition.");
            }

            mesh.num_hexes = num_cells;
            mesh.hex_nodes.allocate(num_cells);
            mesh.hex_nodes.prefetch_to_host();

            // Der Elementtyp steht erst im spaeteren Abschnitt CELL_TYPES. Die
            // Knotenzahl je Zelle legt ihn aber schon hier eindeutig fest
            // (4 = Tetraeder, 6 = Prisma, 8 = Hexaeder), also wird er hier
            // vorlaeufig gesetzt und in CELL_TYPES gegengeprueft. Weichen
            // beide voneinander ab, ist die Datei in sich widerspruechlich --
            // und das ist ein Abbruchgrund, kein Fall fuer eine Annahme.
            mesh.elem_types.allocate(num_cells);
            mesh.elem_types.prefetch_to_host();

            for (size_t i = 0; i < num_cells; ++i) {
                size_t num_nodes_in_cell;
                if (!(file >> num_nodes_in_cell)) {
                    throw std::runtime_error("VtkReader Error: EOF reached while reading CELLS.");
                }

                int type;
                switch (num_nodes_in_cell) {
                    case 4: type = ELEM_TET;   break;
                    case 6: type = ELEM_PRISM; break;
                    case 8: type = ELEM_HEX;   break;
                    default:
                        throw std::runtime_error(
                            "VtkReader Error: Encountered a cell with " + std::to_string(num_nodes_in_cell) +
                            " nodes at index " + std::to_string(i) +
                            ". Supported: 4 (tetrahedron), 6 (prism), 8 (hexahedron). "
                            "Second-order cells (10, 15, 20 nodes) are NOT supported -- the "
                            "solver is first order, and dropping the mid-side nodes would "
                            "change the geometry on curved boundaries."
                        );
                }
                mesh.elem_types[i] = type;

                // Ungenutzte Plaetze auf -1. DeviceArray legt seinen Speicher
                // per cudaMallocManaged ohne Konstruktoraufruf an; ohne diese
                // Zeile stuende dort Muell, den die Assemblierung als gueltige
                // Knotennummer laese.
                Hexahedron hex;
                for (int j = 0; j < 8; ++j) hex.v[j] = -1;

                for (size_t j = 0; j < num_nodes_in_cell; ++j) {
                    if (!(file >> hex.v[j])) {
                        throw std::runtime_error("VtkReader Error: EOF reached while reading node IDs for a cell.");
                    }
                    // Optional: Bounds Checking der Knoten-IDs
                    if (hex.v[j] < 0 || hex.v[j] >= static_cast<int>(mesh.num_nodes)) {
                        throw std::runtime_error("VtkReader Error: Node ID out of bounds in CELLS definition.");
                    }
                }
                mesh.hex_nodes[i] = hex;
            }
            has_cells = true;
            std::cout << "[VtkReader] Successfully read " << num_cells << " cells." << std::endl;
        }
        
        // 4. Einlesen und Verifizieren der Element-Typen (CELL_TYPES)
        else if (token == "CELL_TYPES") {
            size_t num_types;
            if (!(file >> num_types)) {
                throw std::runtime_error("VtkReader Error: Malformed CELL_TYPES definition.");
            }

            if (num_types != mesh.num_hexes) {
                throw std::runtime_error("VtkReader Error: Number of CELL_TYPES does not match number of CELLS.");
            }

            for (size_t i = 0; i < num_types; ++i) {
                int cell_type;
                if (!(file >> cell_type)) {
                    throw std::runtime_error("VtkReader Error: EOF reached while reading CELL_TYPES.");
                }

                // VTK-Zelltypen: 10 Tetraeder, 12 Hexaeder, 13 Prisma. Die
                // Konstanten ELEM_* in Mesh.cuh sind absichtlich genau diese
                // Zahlen, deshalb ist der Vergleich direkt moeglich.
                if (cell_type != ELEM_TET && cell_type != ELEM_HEX && cell_type != ELEM_PRISM) {
                    throw std::runtime_error(
                        "VtkReader Error: Invalid cell type " + std::to_string(cell_type) +
                        " found at index " + std::to_string(i) +
                        ". Supported: 10 (tetrahedron), 12 (hexahedron), 13 (prism). "
                        "Type 14 (pyramid) is deliberately not supported: first-order edge "
                        "elements on the pyramid need rational basis functions."
                    );
                }

                // Gegenprobe zur Knotenzahl aus dem CELLS-Abschnitt. Eine
                // Datei, die hier widerspruechlich ist, wuerde sonst still
                // falsch gerechnet.
                if (cell_type != mesh.elem_types[i]) {
                    throw std::runtime_error(
                        "VtkReader Error: Cell " + std::to_string(i) + " declares type " +
                        std::to_string(cell_type) + " in CELL_TYPES but has " +
                        std::to_string(element_num_nodes(mesh.elem_types[i])) +
                        " nodes in CELLS. The file contradicts itself."
                    );
                }
            }
            has_cell_types = true;

            size_t n_tet = 0, n_prism = 0, n_hex = 0;
            for (size_t i = 0; i < num_types; ++i) {
                const int t = mesh.elem_types[i];
                if (t == ELEM_TET) ++n_tet; else if (t == ELEM_PRISM) ++n_prism; else ++n_hex;
            }
            std::cout << "[VtkReader] Cell types: " << n_hex << " hexahedra, "
                      << n_prism << " prisms, " << n_tet << " tetrahedra." << std::endl;
        }
        // 5. Einlesen der Component IDs (Optionaler Block: CELL_DATA)
        else if (token == "CELL_DATA") {
            size_t num_cell_data;
            if (!(file >> num_cell_data)) {
                throw std::runtime_error("VtkReader Error: Malformed CELL_DATA definition.");
            }

            if (num_cell_data != mesh.num_hexes) {
                throw std::runtime_error("VtkReader Error: CELL_DATA count does not match number of cells.");
            }

            // Alloziere Speicher für die Component-IDs
            mesh.hex_component_ids.allocate(mesh.num_hexes);
            mesh.hex_component_ids.prefetch_to_host();

            // Standardmäßig alles mit 0 initialisieren
            for (size_t i = 0; i < mesh.num_hexes; ++i) {
                mesh.hex_component_ids[i] = 0;
            }

            bool found_scalars = false;

            // Suche nach dem Keyword SCALARS
            while (file >> token) {
                if (token == "SCALARS") {
                    found_scalars = true;

                    // Wir überspringen jetzt alle variablen Format-Wörter (Name, Typ, Dimension),
                    // bis wir exakt bei der LOOKUP_TABLE Definition ankommen.
                    while (file >> token) {
                        if (token == "LOOKUP_TABLE") {
                            file >> token; // Ignoriere den Tabellennamen (meist "default")
                            break;         // Die Schleife abbrechen, der Stream steht jetzt VOR der ersten Zahl!
                        }
                    }

                    // Jetzt die echten Werte rigoros einlesen:
                    for (size_t i = 0; i < mesh.num_hexes; ++i) {
                        double val;
                        if (!(file >> val)) {
                            throw std::runtime_error(
                                "VtkReader Error: Formatting error while reading component IDs. "
                                "Expected " + std::to_string(mesh.num_hexes) +
                                " values, but stream failed at index " + std::to_string(i)
                            );
                        }
                        mesh.hex_component_ids[i] = static_cast<int>(val);
                    }

                    std::cout << "[VtkReader] Successfully read component IDs for " << num_cell_data << " cells." << std::endl;
                    break; // CELL_DATA Block erfolgreich beendet
                }
                else if (token == "POINT_DATA" || token == "DATASET") {
                    // Falls ein neuer Hauptblock beginnt und keine SCALARS da waren, abbrechen
                    break;
                }
            }

            if (!found_scalars) {
                std::cout << "[VtkReader] WARNING: CELL_DATA block found, but no SCALARS. All Component IDs set to 0." << std::endl;
            }
            }

    }

    file.close();

    // Abschließende Validierung, dass alle Pflicht-Blöcke in der Datei vorhanden waren
    if (!has_points || !has_cells || !has_cell_types) {
        throw std::runtime_error(
            "VtkReader Error: File is missing required blocks. "
            "Need POINTS, CELLS, and CELL_TYPES."
        );
    }
}