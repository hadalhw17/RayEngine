#pragma once
#include "../RayEngine/RayEngine.h"
#include "../KDThreeGPU.h" 


////////////////////////////////////////////////////
// Checks if point is to the left of split plane
////////////////////////////////////////////////////
HOST_DEVICE_FUNCTION
bool is_point_to_the_left_of_split(RKDTreeNodeGPU node, const float3& p)
{
	if (node.axis == X_Axis) {
		return (p.x < node.split_val);
	}
	else if (node.axis == Y_Axis) {
		return (p.y < node.split_val);
	}
	else if (node.axis == Z_Axis) {
		return (p.z < node.split_val);
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else {
		return false;
	}
}


////////////////////////////////////////////////////
// Returns a node index roped to a corresponding face
////////////////////////////////////////////////////
HOST_DEVICE_FUNCTION
size_t get_neighboring_node_index(RKDTreeNodeGPU node, float3 p)
{
	// Check left face.
	if (fabs(p.x - node.box.Min.x) < K_EPSILON) {
		return node.neighbor_node_indices[LEFT];
	}
	// Check front face.
	else if (fabs(p.z - node.box.Max.z) < K_EPSILON) {
		return node.neighbor_node_indices[FRONT];
	}
	// Check right face.
	else if (fabs(p.x - node.box.Max.x) < K_EPSILON) {
		return node.neighbor_node_indices[RIGHT];
	}
	// Check back face.
	else if (fabs(p.z - node.box.Min.z) < K_EPSILON) {
		return node.neighbor_node_indices[BACK];
	}
	// Check top face.
	else if (fabs(p.y - node.box.Max.y) < K_EPSILON) {
		return node.neighbor_node_indices[TOP];
	}
	// Check bottom face.
	else if (fabs(p.y - node.box.Min.y) < K_EPSILON) {
		return node.neighbor_node_indices[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else {
		return -1;
	}
}
