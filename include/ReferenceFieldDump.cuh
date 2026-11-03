#ifndef REFERENCE_FIELD_DUMP_CUH
#define REFERENCE_FIELD_DUMP_CUH

// =============================================================================
// ReferenceFieldDump.cuh -- Felddump des harmonischen Referenzpfads
// =============================================================================
//
// --reference --harmonic --fixed-plate --harmonic-field-dump=DATEI schreibt
// je Hexaeder eine Zeile im Format des BEM-FEM-Dumps (fem.cu,
// write_harmonic_field_dump), damit tools/compare_fields.py und
// tools/team28_field_check.py beide Dateien lesen:
//
//   elem,body,x,y,z,Arx,Ary,Arz,Aix,Aiy,Aiz,Brx,Bry,Brz,Bix,Biy,Biz
//
// SI-Einheiten (m, Vs/m, T). body: 0 sonst, 1 Platte, 2 Innen-, 3 Aussenspule.
//
// Auswertung, anders als im Nedelec-Pfad: A ist hier das Q1-Knotenfeld
// A(x) = sum_i N_i(x) A_i. Am Schwerpunkt (Referenzmittelpunkt xi = 0, dort
// ist der Bildpunkt der Mittelwert der acht Knoten) gilt
//
//     A_c = (1/8) sum_i A_i,        B_c = rot A = sum_i grad N_i x A_i,
//
// mit grad N_i aus evaluate_nodal_gradients (NodalHexahedron.cuh). B ist
// also das ELEMENTWEISE rot A des Q1-Feldes im Schwerpunkt, nicht eine
// geglaettete Knotengroesse. Es ist die Auswertung, mit der auch die
// Lorentzkraft des Referenzpfads rechnet.
//
// Reiner Host-Code; kein Code aus dem Nedelec-Pfad (kein Piola, keine Kanten).
// =============================================================================

#include <string>

#include "Mesh.cuh"
#include "NodalHexahedron.cuh"

/// A und B = rot A am Schwerpunkt eines Q1-Hexaeders. nd: die acht Knoten,
/// Ah: Knotenwerte von A. out[0..2] = A, out[3..5] = B. Bei detJ <= 0 alles 0.
void reference_hex_center_fields(const double3 nd[8], const double3 Ah[8],
                                 double3& center, double out[6]);

/// Schreibt den Dump. nodes: Knotenkoordinaten (m, nodes_current), hexes:
/// num_hexes Hexaeder, body: Koerperkennung je Hexaeder, Ar/Ai: je 3*num_nodes
/// Knotenwerte (x,y,z je Knoten). Rueckgabe false, wenn die Datei nicht
/// schreibbar ist.
bool write_reference_field_dump(const std::string& path, const double3* nodes,
                                const Hexahedron* hexes, size_t num_hexes,
                                const int* body, const double* Ar, const double* Ai);

#endif // REFERENCE_FIELD_DUMP_CUH
