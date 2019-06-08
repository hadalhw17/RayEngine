#pragma once
#include "GPUBoundingBox.h"

class RKDTreeCPU;



class RKDTreeNodeGPU
{
public:
	
	HOST_DEVICE_FUNCTION
	RKDTreeNodeGPU();

	// Getters.
	
	GPUBoundingBox GetBox() const;

	
	Axis GetAxis() const;

	
	size_t GetLeftIndex() const;
	
	size_t GetRightIndex() const;
	
	size_t GetIndexOfFirstObject() const;
	
	size_t GetNumObjs() const;

	
	bool IsLeaf() const; 

	GPUBoundingBox box;

	Axis axis;
	float split_val;

	size_t left_index, right_index;
	size_t index_of_first_object;

	size_t num_objs, num_nodes;

	bool is_leaf;

	int neighbor_node_indices[6];

	void PrintDebugString();

private:



};

