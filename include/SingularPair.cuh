#ifndef SINGULAR_PAIR_CUH
#define SINGULAR_PAIR_CUH

enum class SingularityType {
    COINCIDENT = 0,    // Identisches Quad (Self-Patch)
    COMMON_EDGE = 1,   // Quads teilen sich 2 Knoten (1 Kante)
    COMMON_VERTEX = 2  // Quads teilen sich 1 Knoten
};

struct SingularPair {
    int quad_p_idx;    // Globaler Index des Beobachtungs-Quads
    int quad_q_idx;    // Globaler Index des Quell-Quads
    SingularityType type;

    // Spezifisch für Common Edge / Common Vertex:
    // Speichert, ob das Quad in Dreieck A (0) oder B (1) gespalten wurde 
    // und wie die 3 Knoten rotieren müssen, um die Sauter-Schwab-Konvention zu erfüllen.
    int tri_p_type; // 0 für A, 1 für B
    int tri_q_type; // 0 für A, 1 für B
    
    // Lokale Knotenindizes (0,1,2), sortiert nach SS-Konvention
    int p_nodes[3]; 
    int q_nodes[3];
};

#endif