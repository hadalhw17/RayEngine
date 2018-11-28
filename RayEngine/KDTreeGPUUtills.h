#pragma once
#include "GPUBoundingBox.h"

class RKDTreeCPU;



class RKDTreeNodeGPU
{
public:
	
	RKDTreeNodeGPU();

	// Getters.
	
	GPUBoundingBox GetBox() const;

	
	Axis GetAxis() const;

	
	int GetLeftIndex() const;
	
	int GetRightIndex() const;
	
	int GetIndexOfFirstObject() const;
	
	int GetNumObjs() const;

	
	bool IsLeaf() const; 

	GPUBoundingBox box;

	Axis axis;
	float split_val;

	int left_index, right_index;
	int index_of_first_object;

	int num_objs, num_nodes;

	bool is_leaf;

	int neighbor_node_indices[6];

	void PrintDebugString();

private:



};

