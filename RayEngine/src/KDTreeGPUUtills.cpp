#include "KDTreeGPUUtills.h"

#include "KDTree.h"

#include <iostream>



	RKDTreeNodeGPU::RKDTreeNodeGPU()
	{
		left_index = -1;
		right_index = -1;
		index_of_first_object = -1;
		num_objs = 0;

		is_leaf = false;

		for (int i = 0; i < 6; ++i) {
			neighbor_node_indices[i] = -1;
		}
	}


	GPUBoundingBox RKDTreeNodeGPU::GetBox() const
	{
		return box;
	}

	Axis RKDTreeNodeGPU::GetAxis() const
	{
		return axis;
	}

	size_t RKDTreeNodeGPU::GetLeftIndex() const
	{
		return left_index;
	}

	size_t RKDTreeNodeGPU::GetRightIndex() const
	{
		return right_index;
	}

	size_t RKDTreeNodeGPU::GetIndexOfFirstObject() const
	{
		return index_of_first_object;
	}

	size_t RKDTreeNodeGPU::GetNumObjs() const
	{
		return num_objs;
	}


	bool RKDTreeNodeGPU::IsLeaf() const
	{
		return is_leaf;
	}

	void RKDTreeNodeGPU::PrintDebugString()
	{
		std::cout << "bounding box min: ( " << box.Min.x << ", " << box.Min.y << ", " << box.Min.z << " )" << std::endl;
		std::cout << "bounding box max: ( " << box.Max.x << ", " << box.Max.y << ", " << box.Max.z << " )" << std::endl;
		std::cout << "num_tris: " << num_objs << std::endl;
		std::cout << "first_tri_index: " << index_of_first_object << std::endl;

		// Print split plane axis.
		if (axis == X_Axis) {
			std::cout << "split plane axis: X_AXIS" << std::endl;
		}
		else if (axis == Y_Axis) {
			std::cout << "split plane axis: Y_AXIS" << std::endl;
		}
		else if (axis == Z_Axis) {
			std::cout << "split plane axis: Z_AXIS" << std::endl;
		}
		else {
			std::cout << "split plane axis: invalid" << std::endl;
		}

		// Print whether or not node is a leaf node.
		if (is_leaf) {
			std::cout << "is leaf node: YES" << std::endl;
		}
		else {
			std::cout << "is leaf node: NO" << std::endl;
		}

		// Print children indices.
		std::cout << "left child index: " << left_index << std::endl;
		std::cout << "right child index: " << right_index << std::endl;


		// Print empty line.
		std::cout << std::endl;
	}

