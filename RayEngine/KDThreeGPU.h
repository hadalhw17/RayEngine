#pragma once

#include "KDTreeGPUUtills.h"


class RTriangle;


class RKDThreeGPU
{
public:
	RKDThreeGPU(RKDTreeCPU *CPUNode);

	~RKDThreeGPU();

	// Getters.
	RKDTreeNodeGPU *GetNodes() const;
	std::vector<int> GetIndexList() const;
	float3 *get_verts() const;
	float3 *get_faces() const;
	int GetNumNodes() const;
	int get_num_verts() const;
	int get_num_faces() const;
	int get_root_index() const;
	

	void PrintCPUAndGPUTrees(class KDNodeCPU *CPUNode, bool PauseOnEachNode = false);


private:
	RKDTreeNodeGPU *nodes;

	std::vector<int> obj_index_list;

	float3 *verts, *faces;

	int num_nodes;
	int root_index;
	int num_faces;
	int num_verts;

	void buildTree(class KDNodeCPU *CPUNode);
};

