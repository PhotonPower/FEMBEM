#pragma once

#include "Mesh.cuh"
#include "MaterialTypes.cuh"
#include "PhysicalObject.h"

__global__ void update_kinematics_kernel(
    int num_nodes_in_object,
    const int* object_node_indices, // Liste der eindeutigen Knoten-IDs dieses Objekts
    const double3* initial_nodes,
    double3* current_nodes,
    MaterialProperties props) ;

void apply_kinematics(Mesh& mesh, const PhysicalObject& obj); 
