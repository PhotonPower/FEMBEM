#include "BemCoupling.h"

#include <stdexcept>
#include <string>

BemCouplingMap build_bem_coupling_map(const Mesh& mesh, const ClusterTree& tree)
{
    BemCouplingMap out;

    const int num_faces = static_cast<int>(mesh.num_boundary_faces);
    const int num_edges = static_cast<int>(mesh.num_edges);

    out.num_bem_dofs = num_faces * 4;
    out.perm.reserve(num_faces);
    out.leaf_offset.assign(tree.get_num_nodes(), -1);

    // ---------------------------------------------------------------------
    // Blätter in aufsteigender Knotenindex-Reihenfolge durchlaufen. Genau so
    // iteriert auch die Blockschleife in mainloop(), damit die Offsets passen.
    // ---------------------------------------------------------------------
    for (size_t node = 0; node < tree.get_num_nodes(); ++node) {
        const ClusterNode& nd = tree.get_node(node);
        if (!nd.is_leaf) continue;

        out.leaf_offset[node] = static_cast<int>(out.perm.size());
        for (int quad_id : nd.quad_indices) {
            out.perm.push_back(quad_id);
        }
    }

    // ---------------------------------------------------------------------
    // Konsistenzprüfung: perm muss eine Bijektion über alle Randquads sein.
    // Fehlt ein Quad oder kommt eines doppelt vor, ist der Baum kaputt und
    // jede weitere Rechnung wäre still falsch.
    // ---------------------------------------------------------------------
    if (static_cast<int>(out.perm.size()) != num_faces) {
        throw std::runtime_error(
            "build_bem_coupling_map: Die Blaetter des ClusterTree decken "
            + std::to_string(out.perm.size()) + " Quads ab, das Netz hat aber "
            + std::to_string(num_faces) + ".");
    }
    {
        std::vector<char> seen(num_faces, 0);
        for (int q : out.perm) {
            if (q < 0 || q >= num_faces) {
                throw std::runtime_error(
                    "build_bem_coupling_map: Quad-Index " + std::to_string(q)
                    + " liegt ausserhalb von [0," + std::to_string(num_faces) + ").");
            }
            if (seen[q]) {
                throw std::runtime_error(
                    "build_bem_coupling_map: Quad " + std::to_string(q)
                    + " kommt in mehr als einem Blatt vor.");
            }
            seen[q] = 1;
        }
    }

    // ---------------------------------------------------------------------
    // Abbildung und Vorzeichen in permutierter Reihenfolge auslegen
    // ---------------------------------------------------------------------
    out.bem_to_fem.resize(out.num_bem_dofs);
    out.bem_sign.resize(out.num_bem_dofs);

    for (int pos = 0; pos < num_faces; ++pos) {
        const int quad_id = out.perm[pos];
        for (int e = 0; e < 4; ++e) {
            const int flat_src = quad_id * 4 + e;   // Reihenfolge im Mesh
            const int flat_dst = pos     * 4 + e;   // permutierte Reihenfolge

            const int fem_edge = mesh.boundary_edges_to_fem_edges[flat_src];
            const int sign     = mesh.boundary_edge_signs[flat_src];

            if (fem_edge < 0 || fem_edge >= num_edges) {
                throw std::runtime_error(
                    "build_bem_coupling_map: FEM-Kanten-ID " + std::to_string(fem_edge)
                    + " fuer Randquad " + std::to_string(quad_id) + " lokale Kante "
                    + std::to_string(e) + " liegt ausserhalb von [0,"
                    + std::to_string(num_edges) + ").");
            }
            if (sign != 1 && sign != -1) {
                throw std::runtime_error(
                    "build_bem_coupling_map: Vorzeichen " + std::to_string(sign)
                    + " ist weder +1 noch -1 (Randquad " + std::to_string(quad_id)
                    + ", lokale Kante " + std::to_string(e) + ").");
            }

            out.bem_to_fem[flat_dst] = fem_edge;
            out.bem_sign[flat_dst]   = sign;
        }
    }

    // ---------------------------------------------------------------------
    // Offsets und Groessen fuer ALLE Knoten, nicht nur Blaetter.
    // Gueltig, weil die Blaetter eines Teilbaums in der Permutation
    // zusammenhaengend liegen; der Teilbaum belegt daher [offset, offset+size).
    // ---------------------------------------------------------------------
    out.node_offset.assign(tree.get_num_nodes(), -1);
    out.node_size.assign(tree.get_num_nodes(), 0);

    // Iterativ statt rekursiv: der Baum ist flach in m_nodes abgelegt und Kinder
    // haben stets einen groesseren Index als ihr Elternknoten (build_recursive
    // legt den Elternknoten vor der Rekursion an). Ein Rueckwaerttslauf sieht
    // also beide Kinder bereits fertig.
    for (int node = static_cast<int>(tree.get_num_nodes()) - 1; node >= 0; --node) {
        const ClusterNode& nd = tree.get_node(node);
        if (nd.is_leaf) {
            out.node_offset[node] = out.leaf_offset[node];
            out.node_size[node]   = static_cast<int>(nd.quad_indices.size());
        } else {
            const int l = nd.left_child, r = nd.right_child;
            if (l < 0 || r < 0) {
                throw std::runtime_error("build_bem_coupling_map: innerer Knoten "
                    + std::to_string(node) + " hat kein Kindpaar.");
            }
            out.node_offset[node] = out.node_offset[l];
            out.node_size[node]   = out.node_size[l] + out.node_size[r];
            if (out.node_offset[r] != out.node_offset[l] + out.node_size[l]) {
                throw std::runtime_error("build_bem_coupling_map: Teilbaum von Knoten "
                    + std::to_string(node) + " ist im permutierten Raum nicht "
                      "zusammenhaengend. Die Blattreihenfolge passt nicht zur "
                      "Baumstruktur.");
            }
        }
    }

    return out;
}

std::vector<BemBlockPair> build_block_pairs(const ClusterTree& tree, double eta)
{
    std::vector<BemBlockPair> out;
    out.reserve(16384);

    // Explizite Arbeitsliste statt Rekursion: die Tiefe ist bei 4136 Quads
    // unkritisch, aber so bleibt der Speicherbedarf sichtbar und begrenzt.
    std::vector<std::pair<int,int>> stack;
    stack.push_back({tree.get_root_index(), tree.get_root_index()});

    while (!stack.empty()) {
        const int i = stack.back().first;
        const int j = stack.back().second;
        stack.pop_back();

        if (tree.is_admissible(i, j, eta)) {
            out.push_back({i, j, true});
            continue;
        }

        const ClusterNode& a = tree.get_node(i);
        const ClusterNode& b = tree.get_node(j);

        if (a.is_leaf && b.is_leaf) {
            out.push_back({i, j, false});   // Nahfeld, dense mit Sauter-Schwab
            continue;
        }
        if (a.is_leaf) {
            stack.push_back({i, b.left_child});
            stack.push_back({i, b.right_child});
            continue;
        }
        if (b.is_leaf) {
            stack.push_back({a.left_child, j});
            stack.push_back({a.right_child, j});
            continue;
        }
        stack.push_back({a.left_child,  b.left_child});
        stack.push_back({a.left_child,  b.right_child});
        stack.push_back({a.right_child, b.left_child});
        stack.push_back({a.right_child, b.right_child});
    }
    return out;
}
