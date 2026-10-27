#ifndef CLUSTER_TREE_H
#define CLUSTER_TREE_H

#include <vector>
#include <memory>
#include <cmath>
#include "Mesh.cuh"

// Hilfsstruktur für die Bounding Box
struct BoundingBox {
    double3 min_pt;
    double3 max_pt;
    double3 center;
    double diameter;
};

// Ein Knoten im Cluster-Baum
struct ClusterNode {
    int id;                           // Eindeutige ID des Knotens
    int left_child;                   // Index des linken Kindes (-1 wenn Blatt)
    int right_child;                  // Index des rechten Kindes (-1 wenn Blatt)
    bool is_leaf;                     // True, wenn dieser Knoten nicht weiter unterteilt ist
    
    BoundingBox bbox;                 // Räumliche Ausdehnung dieses Clusters
    std::vector<int> quad_indices;    // Nur befüllt, wenn is_leaf == true
};

class ClusterTree {
private:
    int m_max_leaf_size;              // N_leaf (z.B. 30)
    std::vector<ClusterNode> m_nodes; // Flacher Speicher für alle Baumknoten (Cache-freundlich)
    int m_root_index;                 // Index des Wurzelknotens (meist 0)

    // Interne rekursive Methode zum Baumaufbau
    int build_recursive(Mesh& mesh, std::vector<int>& current_quads, const std::vector<double3>& quad_centroids);

    // Interne rekursive Methode zum Update
    void update_recursive(Mesh& mesh, int node_index);

public:
    ClusterTree(int max_leaf_size = 30);

    // Aufgabe 4.1 & 4.2: Berechnet Bounding Boxes und baut den BVH-Baum auf
    void build_tree(Mesh& mesh);

    // Aufgabe 4.3: Prüft, ob zwei Cluster weit genug voneinander entfernt sind
    bool is_admissible(int node_idx_X, int node_idx_Y, double eta) const;

    // Aufgabe 4.4: Aktualisiert die Bounding Boxes nach einer Bewegung der Knoten
    void update_tree_kinematics(Mesh& mesh);

    // Getter
    const ClusterNode& get_node(int index) const { return m_nodes[index]; }
    int get_root_index() const { return m_root_index; }
    size_t get_num_nodes() const { return m_nodes.size(); }
};

#endif // CLUSTER_TREE_H