#include "PhysicalObject.h"
#include <iostream>
#include <cmath>
#include <vector>

PhysicalObject::PhysicalObject(const std::string& obj_name)
    : name(obj_name), mass(0.0), velocity({ 0,0,0 }), angular_velocity({ 0,0,0 })
{
    inertia[0] = 0.0; inertia[1] = 0.0; inertia[2] = 0.0;

    props.is_moving = false;
    props.translation = make_double3(0.0, 0.0, 0.0);
    update_rotation_matrix();
}

void PhysicalObject::update_rotation_matrix() {
    // Wandelt das Einheitsquaternion (w, x, y, z) in eine 3x3 Rotationsmatrix um.
    // Diese Matrix dreht Vektoren vom körperfesten ins weltfeste System.
    double q0 = orientation.w, q1 = orientation.x, q2 = orientation.y, q3 = orientation.z;

    props.rot_matrix[0][0] = 1.0 - 2.0 * (q2 * q2 + q3 * q3);
    props.rot_matrix[0][1] = 2.0 * (q1 * q2 - q0 * q3);
    props.rot_matrix[0][2] = 2.0 * (q1 * q3 + q0 * q2);

    props.rot_matrix[1][0] = 2.0 * (q1 * q2 + q0 * q3);
    props.rot_matrix[1][1] = 1.0 - 2.0 * (q1 * q1 + q3 * q3);
    props.rot_matrix[1][2] = 2.0 * (q2 * q3 - q0 * q1);

    props.rot_matrix[2][0] = 2.0 * (q1 * q3 - q0 * q2);
    props.rot_matrix[2][1] = 2.0 * (q2 * q3 + q0 * q1);
    props.rot_matrix[2][2] = 1.0 - 2.0 * (q1 * q1 + q2 * q2);
}

void PhysicalObject::apply_newtons_law(double3 force, double3 torque, double dt) {
    if (!props.is_moving || mass <= 0.0) return;

    // =========================================================================
    // 1. Translation (Weltfestes Koordinatensystem)
    // =========================================================================
    // a) Beschleunigung berechnen (inkl. Gravitation in z-Richtung: -9.81 m/s^2)
    double a_x = force.x / mass;
    double a_y = force.y / mass;
    double a_z = (force.z / mass) - 9.81;

    // b) Symplectic Euler (Erst Geschwindigkeit, dann Ort updaten)
    velocity.x += a_x * dt;
    velocity.y += a_y * dt;
    velocity.z += a_z * dt;

    props.translation.x += velocity.x * dt;
    props.translation.y += velocity.y * dt;
    props.translation.z += velocity.z * dt;

    // Auflage: die Platte kann nicht durch die Wickelform sinken. Beim Auftreffen
    // wird die Abwaertsgeschwindigkeit auf null gesetzt (voll unelastischer
    // Stoss) -- die Platte liegt dann auf, bis die Kraft sie wieder anhebt.
    if (has_lower_stop && props.translation.z < lower_stop_z) {
        props.translation.z = lower_stop_z;
        if (velocity.z < 0.0) velocity.z = 0.0;
    }

    // =========================================================================
    // 2. Rotation (Starre Körper Dynamik)
    // =========================================================================
    if (inertia[0] > 0.0 && inertia[1] > 0.0 && inertia[2] > 0.0) {

        // a) Drehmoment vom Welt- ins Körper-System transformieren.
        // Dafür multiplizieren wir mit der Transponierten der Rotationsmatrix.
        double M_body_x = props.rot_matrix[0][0] * torque.x + props.rot_matrix[1][0] * torque.y + props.rot_matrix[2][0] * torque.z;
        double M_body_y = props.rot_matrix[0][1] * torque.x + props.rot_matrix[1][1] * torque.y + props.rot_matrix[2][1] * torque.z;
        double M_body_z = props.rot_matrix[0][2] * torque.x + props.rot_matrix[1][2] * torque.y + props.rot_matrix[2][2] * torque.z;

        // b) Euler'sche Kreiselgleichungen lösen (Expliziter Euler)
        // I_xx * dw_x/dt + (I_zz - I_yy) * w_y * w_z = M_x
        double dw_x = (M_body_x - (inertia[2] - inertia[1]) * angular_velocity.y * angular_velocity.z) / inertia[0];
        double dw_y = (M_body_y - (inertia[0] - inertia[2]) * angular_velocity.z * angular_velocity.x) / inertia[1];
        double dw_z = (M_body_z - (inertia[1] - inertia[0]) * angular_velocity.x * angular_velocity.y) / inertia[2];

        // Update der körperfesten Winkelgeschwindigkeiten
        angular_velocity.x += dw_x * dt;
        angular_velocity.y += dw_y * dt;
        angular_velocity.z += dw_z * dt;

        // c) Orientierung (Quaternion) integrieren
        // dq/dt = 0.5 * q \otimes (0, omega_x, omega_y, omega_z)
        double q0 = orientation.w, q1 = orientation.x, q2 = orientation.y, q3 = orientation.z;
        double wx = angular_velocity.x, wy = angular_velocity.y, wz = angular_velocity.z;

        double dq_w = -0.5 * (q1 * wx + q2 * wy + q3 * wz);
        double dq_x = 0.5 * (q0 * wx - q3 * wy + q2 * wz);
        double dq_y = 0.5 * (q3 * wx + q0 * wy - q1 * wz);
        double dq_z = -0.5 * (q2 * wx + q1 * wy + q0 * wz);

        // Explizites Update
        orientation.w += dq_w * dt;
        orientation.x += dq_x * dt;
        orientation.y += dq_y * dt;
        orientation.z += dq_z * dt;

        // d) Normalisieren, um numerischen Drift der orthogonalen Matrix zu verhindern
        orientation.normalize();

        // e) Neue Rotationsmatrix für den CUDA-Kernel (Aufgabe 3.3) berechnen
        update_rotation_matrix();
    }
}
void PhysicalObject::compute_center_of_mass(const Mesh& mesh) {
    if (hex_indices.empty()) return;

    double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
    int total_nodes = 0;

    // Wir iterieren über alle Hexaeder des Objekts und addieren deren Knoten.
    // (Achtung: Innere Knoten werden mehrfach addiert, was für einen groben 
    // geometrischen Schwerpunkt ausreichend ist. Für exakte Massenträgheit 
    // müssten Hexaeder-Volumina integriert werden).
    for (int hex_id : hex_indices) {
        Hexahedron hex = mesh.hex_nodes[hex_id];
        // NUR die tatsaechlich belegten Knoten. Bei einem Tetraeder stehen
        // v[4..7] auf -1; mesh.nodes_initial[-1] ist ein Zugriff ausserhalb
        // des Arrays und beendet das Programm ohne jede Meldung. Genau daran
        // ist der erste Lauf von --fem-direct gescheitert.
        const int nn = element_num_nodes(mesh.elem_type(hex_id));
        for (int i = 0; i < nn; ++i) {
            double3 pt = mesh.nodes_initial[hex.v[i]];
            sum_x += pt.x;
            sum_y += pt.y;
            sum_z += pt.z;
            total_nodes++;
        }
    }

    props.center_of_mass.x = sum_x / total_nodes;
    props.center_of_mass.y = sum_y / total_nodes;
    props.center_of_mass.z = sum_z / total_nodes;
}

// -----------------------------------------------------------------------------
// Hilfsfunktion: Geometrie-basierte Zuweisung von Hexaedern zu Objekten
// -----------------------------------------------------------------------------
void assign_materials_by_bounding_box(Mesh& mesh, PhysicalObject* obj, 
                                      double x_min, double x_max, 
                                      double y_min, double y_max, 
                                      double z_min, double z_max) {
    int assigned_count = 0;
    
    for (size_t i = 0; i < mesh.num_hexes; ++i) {
        Hexahedron hex = mesh.hex_nodes[i];
        
        // Berechne das Zentrum des Hexaeders
        double cx = 0, cy = 0, cz = 0;
        for (int j = 0; j < 8; ++j) {
            double3 pt = mesh.nodes_initial[hex.v[j]];
            cx += pt.x; cy += pt.y; cz += pt.z;
        }
        cx /= 8.0; cy /= 8.0; cz /= 8.0;

        // Liegt das Hex-Zentrum in der Box?
        if (cx >= x_min && cx <= x_max &&
            cy >= y_min && cy <= y_max &&
            cz >= z_min && cz <= z_max) {
            
            obj->add_hex(static_cast<int>(i));
            assigned_count++;
        }
    }
    
    obj->compute_center_of_mass(mesh);
    std::cout << "[Material Zuweisung] " << assigned_count << " Hexaeder dem Objekt zugewiesen. "
              << "Schwerpunkt: (" << obj->get_properties().center_of_mass.x << ", " 
              << obj->get_properties().center_of_mass.y << ", " 
              << obj->get_properties().center_of_mass.z << ")\n";
}

// =================================================================================
// Zuweisung von Hexaedern zu Objekten basierend auf VTK Component IDs
// =================================================================================
void assign_materials_by_component_id(Mesh& mesh, PhysicalObject* obj, int target_component_id) {
    if (mesh.hex_component_ids.size() == 0) {
        throw std::runtime_error(
            "Material Assignment Error: No Component IDs found in the VTK mesh. "
            "Make sure your mesh contains CELL_DATA SCALARS."
        );
    }

    int assigned_count = 0;

    // CPU muss Zugriff auf die IDs haben
    mesh.hex_component_ids.prefetch_to_host();

    // Iteriere über alle Hexaeder und prüfe die ID
    for (size_t i = 0; i < mesh.num_hexes; ++i) {
        if (mesh.hex_component_ids[i] == target_component_id) {
            obj->add_hex(static_cast<int>(i));
            assigned_count++;
        }
    }

    // Aktualisiere den Massenschwerpunkt (für Mechanik & Maxwell-Tensor)
    if (assigned_count > 0) {
        obj->compute_center_of_mass(mesh);
    }

    std::cout << "[Material Zuweisung] " << assigned_count << " Hexaeder der Component ID "
        << target_component_id << " dem Objekt '" << obj->get_name() << "' zugewiesen.\n";

    if (assigned_count > 0) {
        std::cout << "  -> Schwerpunkt: (" << obj->get_properties().center_of_mass.x << ", "
            << obj->get_properties().center_of_mass.y << ", "
            << obj->get_properties().center_of_mass.z << ")\n";
    }
}



void PhysicalObject::build_node_cache(const Mesh& mesh) {
    if (hex_indices.empty()) return;

    // Ein bool-Array über das gesamte Netz, um bereits gefundene Knoten zu markieren
    std::vector<bool> is_node_added(mesh.num_nodes, false);
    std::vector<int> unique_nodes_temp;

    // Grobe Schätzung für die Reservierung (vermeidet Reallokationen)
    unique_nodes_temp.reserve(hex_indices.size() * 4);

    // Iteriere über alle dem Objekt zugewiesenen Hexaeder
    for (int hex_id : hex_indices) {
        Hexahedron hex = mesh.hex_nodes[hex_id];

        // Nur die belegten Knoten. Bei einem Tetraeder stehen v[4..7] auf -1,
        // und is_node_added[-1] ist ein SCHREIBZUGRIFF ausserhalb des Arrays.
        const int nn = element_num_nodes(mesh.elem_type(hex_id));
        for (int i = 0; i < nn; ++i) {
            int node_id = hex.v[i];

            // Wenn der Knoten noch nicht in der Liste ist -> hinzufügen
            if (!is_node_added[node_id]) {
                is_node_added[node_id] = true;
                unique_nodes_temp.push_back(node_id);
            }
        }
    }

    // Speicher auf dem Device (Unified Memory) allozieren
    cached_unique_nodes.allocate(unique_nodes_temp.size());
    cached_unique_nodes.prefetch_to_host(); // Wir schreiben gleich von der CPU hinein

    // Daten aus dem temporären std::vector übertragen
    for (size_t i = 0; i < unique_nodes_temp.size(); ++i) {
        cached_unique_nodes[i] = unique_nodes_temp[i];
    }

    // Asynchron zur GPU schieben, damit der Kernel später sofort zugreifen kann
    cached_unique_nodes.prefetch_to_device();

    std::cout << "[PhysicalObject] Built node cache for '" << name << "'. "
        << "Contains " << cached_unique_nodes.size() << " unique nodes.\n";
}
