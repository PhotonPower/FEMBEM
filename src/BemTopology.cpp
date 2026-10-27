#include "Mesh.cuh"
#include "SingularPair.cuh"
#include <vector>
#include <algorithm>
#include <iostream>

// Hilfsfunktion: Teilt ein Quad in Dreieck A (Knoten 0,1,2) und B (Knoten 0,2,3)
inline void get_triangle_nodes(const Mesh::QuadFace& quad, int tri_type, int out_nodes[3]) {
    if (tri_type == 0) { // Dreieck A
        out_nodes[0] = quad.v[0];
        out_nodes[1] = quad.v[1];
        out_nodes[2] = quad.v[2];
    } else {             // Dreieck B
        out_nodes[0] = quad.v[0];
        out_nodes[1] = quad.v[2];
        out_nodes[2] = quad.v[3];
    }
}

// Hilfsfunktion: Zählt gemeinsame Knoten zweier Dreiecke und gibt deren Positionen zurück
inline int find_shared_nodes(const int tri_p[3], const int tri_q[3], 
                             int shared_p_idx[3], int shared_q_idx[3]) 
{
    int count = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (tri_p[i] == tri_q[j]) {
                shared_p_idx[count] = i;
                shared_q_idx[count] = j;
                count++;
            }
        }
    }
    return count;
}

// =================================================================================
// Hauptroutine zur Klassifizierung benachbarter BEM-Elemente
// =================================================================================
void classify_singular_pairs(const Mesh& mesh, std::vector<SingularPair>& out_edge_pairs, std::vector<SingularPair>& out_vertex_pairs) {
    
    // Iteriere über alle Quad-Paare (nur untere Dreiecksmatrix, da Symmetrie nutzbar)
    for (size_t p = 0; p < mesh.num_boundary_faces; ++p) {
        for (size_t q = p + 1; q < mesh.num_boundary_faces; ++q) { // q > p verhindert Self-Patch (q==p)
            
            const Mesh::QuadFace& quad_p = mesh.boundary_faces[p];
            const Mesh::QuadFace& quad_q = mesh.boundary_faces[q];

            // 1. Grober Pre-Check: Wie viele Knoten teilen sich die Quads insgesamt?
            int shared_quad_nodes = 0;
            for(int i=0; i<4; ++i) {
                for(int j=0; j<4; ++j) {
                    if(quad_p.v[i] == quad_q.v[j]) shared_quad_nodes++;
                }
            }

            // Wenn keine Knoten geteilt werden -> Fernfeld (Regular), wird von ACA übernommen
            if (shared_quad_nodes == 0) continue; 

            // Ein Quad wird in 2 Dreiecke gespalten. Wir müssen 4 Dreieckspaare prüfen:
            // (A,A), (A,B), (B,A), (B,B)
            for (int tri_p_type = 0; tri_p_type < 2; ++tri_p_type) {
                int tri_p_nodes[3];
                get_triangle_nodes(quad_p, tri_p_type, tri_p_nodes);

                for (int tri_q_type = 0; tri_q_type < 2; ++tri_q_type) {
                    int tri_q_nodes[3];
                    get_triangle_nodes(quad_q, tri_q_type, tri_q_nodes);

                    int shared_p[3], shared_q[3];
                    int num_shared = find_shared_nodes(tri_p_nodes, tri_q_nodes, shared_p, shared_q);

                    if (num_shared == 0) continue;

                    SingularPair pair;
                    pair.quad_p_idx = static_cast<int>(p);
                    pair.quad_q_idx = static_cast<int>(q);
                    pair.tri_p_type = tri_p_type;
                    pair.tri_q_type = tri_q_type;

                    // ---------------------------------------------------------
                    // Fall 1: Common Edge (Genau 2 Knoten geteilt)
                    // SS-Konvention: Die geteilten Knoten MÜSSEN auf Index 0 und 1 liegen.
                    // ---------------------------------------------------------
                    if (num_shared == 2) {
                        pair.type = SingularityType::COMMON_EDGE;
                        
                        // Permutation für P: Die beiden shared Knoten an Position 0 und 1 schieben
                        int unshared_p = 3 - (shared_p[0] + shared_p[1]); // 0+1=1->2, 0+2=2->1, 1+2=3->0
                        pair.p_nodes[0] = shared_p[0];
                        pair.p_nodes[1] = shared_p[1];
                        pair.p_nodes[2] = unshared_p;

                        // Permutation für Q: (Müssen identisch zu P aufgestellt werden!)
                        int unshared_q = 3 - (shared_q[0] + shared_q[1]);
                        pair.q_nodes[0] = shared_q[0];
                        pair.q_nodes[1] = shared_q[1];
                        pair.q_nodes[2] = unshared_q;

                        out_edge_pairs.push_back(pair);
                    }
                    // ---------------------------------------------------------
                    // Fall 2: Common Vertex (Genau 1 Knoten geteilt)
                    // SS-Konvention: Der geteilte Knoten MUSS auf Index 0 liegen.
                    // ---------------------------------------------------------
                    else if (num_shared == 1) {
                        pair.type = SingularityType::COMMON_VERTEX;
                        
                        // Permutation P: Shared Knoten auf 0
                        pair.p_nodes[0] = shared_p[0];
                        pair.p_nodes[1] = (shared_p[0] + 1) % 3;
                        pair.p_nodes[2] = (shared_p[0] + 2) % 3;

                        // Permutation Q: Shared Knoten auf 0
                        pair.q_nodes[0] = shared_q[0];
                        pair.q_nodes[1] = (shared_q[0] + 1) % 3;
                        pair.q_nodes[2] = (shared_q[0] + 2) % 3;

                        out_vertex_pairs.push_back(pair);
                    }
                }
            }
        }
    }
    
    std::cout << "[BEM Topology] Found " << out_edge_pairs.size() << " Common-Edge triangle pairs.\n";
    std::cout << "[BEM Topology] Found " << out_vertex_pairs.size() << " Common-Vertex triangle pairs.\n";
}