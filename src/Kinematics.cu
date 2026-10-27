#include "Kinematics.cuh"
#include "Backend.cuh"

// -----------------------------------------------------------------------------
// Der GPU-Kernel zur Aktualisierung der Koordinaten
// -----------------------------------------------------------------------------
// Rumpf: laeuft unveraendert auf GPU (aus dem Kernel unten) und CPU (aus
// BF_LAUNCH_1D_PAR). Jeder Index schreibt ausschliesslich in seinen eigenen
// Knoten -- die Knotenliste des Objekts ist per Konstruktion eindeutig
// (build_node_cache), also keine Ueberschneidung und damit rennfrei.
BF_HD inline void update_kinematics_kernel_body(
    int idx,
    int num_nodes_in_object,
    const int* object_node_indices,
    const double3* initial_nodes,
    double3* current_nodes,
    const MaterialProperties& props)
{
    (void)num_nodes_in_object;
    int global_node_id = object_node_indices[idx];
    
    double3 p0 = initial_nodes[global_node_id];
    double3 c = props.center_of_mass;
    
    // 1. Verschiebe Knoten in den Ursprung (relativ zum Schwerpunkt)
    double dx = p0.x - c.x;
    double dy = p0.y - c.y;
    double dz = p0.z - c.z;
    
    // 2. Rotiere den Knoten (Matrix-Vektor-Multiplikation)
    double rx = props.rot_matrix[0][0] * dx + props.rot_matrix[0][1] * dy + props.rot_matrix[0][2] * dz;
    double ry = props.rot_matrix[1][0] * dx + props.rot_matrix[1][1] * dy + props.rot_matrix[1][2] * dz;
    double rz = props.rot_matrix[2][0] * dx + props.rot_matrix[2][1] * dy + props.rot_matrix[2][2] * dz;
    
    // 3. Zurückverschieben zum Schwerpunkt und Translation addieren
    current_nodes[global_node_id].x = rx + c.x + props.translation.x;
    current_nodes[global_node_id].y = ry + c.y + props.translation.y;
    current_nodes[global_node_id].z = rz + c.z + props.translation.z;
}

__global__ void update_kinematics_kernel(
    int num_nodes_in_object,
    const int* object_node_indices, // Liste der eindeutigen Knoten-IDs dieses Objekts
    const double3* initial_nodes,
    double3* current_nodes,
    MaterialProperties props)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes_in_object) return;
    update_kinematics_kernel_body(idx, num_nodes_in_object, object_node_indices,
                                  initial_nodes, current_nodes, props);
}

// -----------------------------------------------------------------------------
// C++ Host-Wrapper zum Aufrufen des Kernels
// -----------------------------------------------------------------------------
void apply_kinematics(Mesh& mesh, const PhysicalObject& obj) {
    // Wenn das Objekt sich nicht bewegt oder keine Knoten hat, direkt abbrechen
    if (!obj.is_moving() || obj.get_num_unique_nodes() == 0) return;

    int num_nodes = obj.get_num_unique_nodes();
    const int* d_unique_nodes = obj.get_unique_nodes_ptr();

    // props muss ein lvalue sein: der CPU-Pfad nimmt eine const-Referenz
    // entgegen, der GPU-Pfad kopiert den Wert wie bisher in den Kernel.
    const MaterialProperties props = obj.get_properties();

    BF_LAUNCH_1D_PAR(update_kinematics_kernel, num_nodes,
        num_nodes,
        d_unique_nodes,
        mesh.nodes_initial.data(),
        mesh.nodes_current.data(),
        props);

    // Synchronisieren, da Folgeoperationen (wie Baum-Updates)
    // zwingend auf die neuen Koordinaten warten müssen. Im CPU-Modus ein No-Op.
    bf::sync();
}