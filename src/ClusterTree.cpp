#include "ClusterTree.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

// Hilfsfunktion: Berechnet den euklidischen Abstand zweier Punkte
inline double distance_3d(const double3& a, const double3& b) {
    return std::sqrt((a.x - b.x)*(a.x - b.x) + 
                     (a.y - b.y)*(a.y - b.y) + 
                     (a.z - b.z)*(a.z - b.z));
}

// Konstruktor
ClusterTree::ClusterTree(int max_leaf_size) : m_max_leaf_size(max_leaf_size), m_root_index(-1) {}

// -----------------------------------------------------------------------------
// Aufgabe 4.1 & 4.2: Initialer Baumaufbau
// -----------------------------------------------------------------------------
void ClusterTree::build_tree(Mesh& mesh) {
    if (mesh.num_boundary_faces == 0) {
        throw std::runtime_error("ClusterTree Error: Mesh has no boundary faces.");
    }

    // Stellen Sie sicher, dass die CPU die aktuellsten Knotendaten hat
    mesh.nodes_current.prefetch_to_host();

    std::vector<int> all_quads(mesh.num_boundary_faces);
    std::vector<double3> quad_centroids(mesh.num_boundary_faces);

    // Aufgabe 4.1: Vorberechnung der Schwerpunkte (Centroids) für alle Rand-Quads
    for (size_t i = 0; i < mesh.num_boundary_faces; ++i) {
        all_quads[i] = static_cast<int>(i);
        
        Mesh::QuadFace quad = mesh.boundary_faces[i];
        double3 p0 = mesh.nodes_current[quad.v[0]];
        double3 p1 = mesh.nodes_current[quad.v[1]];
        double3 p2 = mesh.nodes_current[quad.v[2]];
        double3 p3 = mesh.nodes_current[quad.v[3]];

        quad_centroids[i].x = 0.25 * (p0.x + p1.x + p2.x + p3.x);
        quad_centroids[i].y = 0.25 * (p0.y + p1.y + p2.y + p3.y);
        quad_centroids[i].z = 0.25 * (p0.z + p1.z + p2.z + p3.z);
    }

    m_nodes.clear();
    m_nodes.reserve(mesh.num_boundary_faces / (m_max_leaf_size / 2)); // Grobe Reservierung

    // Start der Rekursion
    m_root_index = build_recursive(mesh, all_quads, quad_centroids);
    
    std::cout << "[ClusterTree] Built binary cluster tree with " 
              << m_nodes.size() << " nodes for " 
              << mesh.num_boundary_faces << " BEM faces." << std::endl;
}

int ClusterTree::build_recursive(Mesh& mesh, std::vector<int>& current_quads, const std::vector<double3>& quad_centroids) {
    
    // Neuen Knoten in der flachen Liste anlegen
    int current_node_idx = static_cast<int>(m_nodes.size());
    m_nodes.push_back(ClusterNode());
    ClusterNode& node = m_nodes[current_node_idx]; // Referenz auf den soeben erstellten Knoten
    
    node.id = current_node_idx;
    node.left_child = -1;
    node.right_child = -1;

    // 1. Bounding Box für diesen Cluster berechnen
    double3 bb_min = { 1e30,  1e30,  1e30};
    double3 bb_max = {-1e30, -1e30, -1e30};

    for (int quad_idx : current_quads) {
        Mesh::QuadFace quad = mesh.boundary_faces[quad_idx];
        for (int i = 0; i < 4; ++i) {
            double3 pt = mesh.nodes_current[quad.v[i]];
            if (pt.x < bb_min.x) bb_min.x = pt.x;
            if (pt.y < bb_min.y) bb_min.y = pt.y;
            if (pt.z < bb_min.z) bb_min.z = pt.z;
            
            if (pt.x > bb_max.x) bb_max.x = pt.x;
            if (pt.y > bb_max.y) bb_max.y = pt.y;
            if (pt.z > bb_max.z) bb_max.z = pt.z;
        }
    }

    node.bbox.min_pt = bb_min;
    node.bbox.max_pt = bb_max;
    node.bbox.center.x = 0.5 * (bb_min.x + bb_max.x);
    node.bbox.center.y = 0.5 * (bb_min.y + bb_max.y);
    node.bbox.center.z = 0.5 * (bb_min.z + bb_max.z);
    node.bbox.diameter = distance_3d(bb_min, bb_max);

    // 2. Abbruchbedingung prüfen (Blattknoten)
    if (current_quads.size() <= static_cast<size_t>(m_max_leaf_size)) {
        node.is_leaf = true;
        node.quad_indices = current_quads; // Kopie der Indizes speichern
        return current_node_idx;
    }

    // 3. Unterteilung (Median Split entlang der längsten Achse)
    node.is_leaf = false;
    double dx = bb_max.x - bb_min.x;
    double dy = bb_max.y - bb_min.y;
    double dz = bb_max.z - bb_min.z;

    // Finde die längste Achse (0=x, 1=y, 2=z)
    int split_axis = 0;
    if (dy > dx && dy > dz) split_axis = 1;
    else if (dz > dx && dz > dy) split_axis = 2;

    // Sortiere die Quads basierend auf ihrem Schwerpunkt entlang der längsten Achse.
    //
    // Der Quad-Index dient als Tiebreak. Ohne ihn vergleicht der Komparator nur
    // eine Koordinate, und bei TEAM 28 gibt es davon massenhaft exakte
    // Gleichstände: die Spulen sind axialsymmetrische Extrusionen, die Platte
    // ist ein reguläres Gitter, ganze Ringe bzw. Reihen von Randflächen teilen
    // denselben Schwerpunkt entlang der Splitachse. std::sort ist nicht stabil,
    // die Reihenfolge gleicher Elemente also implementierungsabhängig. Folge
    // war ein plattformabhängiger Cluster-Baum: identische Eingabe ergab unter
    // libstdc++ und der MSVC-STL verschiedene Blattpartitionen und damit eine
    // verschiedene ACA/dense-Aufteilung (61466/4070 gegen 61456/4080), also
    // einen anderen komprimierten Operator und andere Rechenergebnisse.
    //
    // Mit dem Index als zweitem Kriterium ist die Ordnung total, das Ergebnis
    // damit eindeutig und unabhängig vom Sortieralgorithmus.
    std::sort(current_quads.begin(), current_quads.end(), 
        [&quad_centroids, split_axis](int a, int b) {
            double ka, kb;
            if      (split_axis == 0) { ka = quad_centroids[a].x; kb = quad_centroids[b].x; }
            else if (split_axis == 1) { ka = quad_centroids[a].y; kb = quad_centroids[b].y; }
            else                      { ka = quad_centroids[a].z; kb = quad_centroids[b].z; }
            if (ka != kb) return ka < kb;
            return a < b;
        });

    // Halbiere die Liste
    size_t mid = current_quads.size() / 2;
    std::vector<int> left_quads(current_quads.begin(), current_quads.begin() + mid);
    std::vector<int> right_quads(current_quads.begin() + mid, current_quads.end());

    // Rekursion für die Kinder
    // WICHTIG: node-Referenz könnte ungültig werden, wenn m_nodes realloziert wird!
    // Daher weisen wir die Kinder über den Index current_node_idx zu.
    int left_idx = build_recursive(mesh, left_quads, quad_centroids);
    int right_idx = build_recursive(mesh, right_quads, quad_centroids);

    m_nodes[current_node_idx].left_child = left_idx;
    m_nodes[current_node_idx].right_child = right_idx;

    return current_node_idx;
}

// -----------------------------------------------------------------------------
// Aufgabe 4.3: Zulässigkeitsprüfung (Admissibility) für ACA
// -----------------------------------------------------------------------------
bool ClusterTree::is_admissible(int node_idx_X, int node_idx_Y, double eta) const {
    const ClusterNode& X = m_nodes[node_idx_X];
    const ClusterNode& Y = m_nodes[node_idx_Y];

    double min_diam = std::min(X.bbox.diameter, Y.bbox.diameter);
    double dist = distance_3d(X.bbox.center, Y.bbox.center);

    // Kriterium: min(diam(X), diam(Y)) <= eta * distance(center_X, center_Y)
    return min_diam <= (eta * dist);
}

// -----------------------------------------------------------------------------
// Aufgabe 4.4: Schnelles Update nach Objektbewegung
// -----------------------------------------------------------------------------
void ClusterTree::update_tree_kinematics(Mesh& mesh) {
    if (m_root_index == -1) return;

    // Nach dem Ausführen des GPU-Kinematik-Kernels liegen die neuen Koordinaten
    // im VRAM. Wir müssen sie für das Baum-Update zwingend in den CPU-RAM pre-fetchen.
    mesh.nodes_current.prefetch_to_host();

    update_recursive(mesh, m_root_index);
}

void ClusterTree::update_recursive(Mesh& mesh, int node_index) {
    ClusterNode& node = m_nodes[node_index];

    if (node.is_leaf) {
        // Für Blätter: Bounding Box aus den exakten Knotenpunkten neu berechnen
        double3 bb_min = { 1e30,  1e30,  1e30};
        double3 bb_max = {-1e30, -1e30, -1e30};

        for (int quad_idx : node.quad_indices) {
            Mesh::QuadFace quad = mesh.boundary_faces[quad_idx];
            for (int i = 0; i < 4; ++i) {
                double3 pt = mesh.nodes_current[quad.v[i]];
                if (pt.x < bb_min.x) bb_min.x = pt.x;
                if (pt.y < bb_min.y) bb_min.y = pt.y;
                if (pt.z < bb_min.z) bb_min.z = pt.z;
                
                if (pt.x > bb_max.x) bb_max.x = pt.x;
                if (pt.y > bb_max.y) bb_max.y = pt.y;
                if (pt.z > bb_max.z) bb_max.z = pt.z;
            }
        }
        node.bbox.min_pt = bb_min;
        node.bbox.max_pt = bb_max;
    } 
    else {
        // Für innere Knoten: Erst Kinder aktualisieren (Bottom-Up)...
        update_recursive(mesh, node.left_child);
        update_recursive(mesh, node.right_child);

        // ...dann die eigene Bounding Box aus den Boxen der Kinder zusammensetzen
        const ClusterNode& left = m_nodes[node.left_child];
        const ClusterNode& right = m_nodes[node.right_child];

        node.bbox.min_pt.x = std::min(left.bbox.min_pt.x, right.bbox.min_pt.x);
        node.bbox.min_pt.y = std::min(left.bbox.min_pt.y, right.bbox.min_pt.y);
        node.bbox.min_pt.z = std::min(left.bbox.min_pt.z, right.bbox.min_pt.z);

        node.bbox.max_pt.x = std::max(left.bbox.max_pt.x, right.bbox.max_pt.x);
        node.bbox.max_pt.y = std::max(left.bbox.max_pt.y, right.bbox.max_pt.y);
        node.bbox.max_pt.z = std::max(left.bbox.max_pt.z, right.bbox.max_pt.z);
    }

    // Zentrum und Durchmesser neu berechnen (gilt für Blätter und innere Knoten)
    node.bbox.center.x = 0.5 * (node.bbox.min_pt.x + node.bbox.max_pt.x);
    node.bbox.center.y = 0.5 * (node.bbox.min_pt.y + node.bbox.max_pt.y);
    node.bbox.center.z = 0.5 * (node.bbox.min_pt.z + node.bbox.max_pt.z);
    node.bbox.diameter = distance_3d(node.bbox.min_pt, node.bbox.max_pt);
}