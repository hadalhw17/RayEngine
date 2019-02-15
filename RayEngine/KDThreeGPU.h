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
	float3 *get_normals() const;
	int GetNumNodes() const;
	int get_num_verts() const;
	int get_num_faces() const;
	int get_root_index() const;
	

	void PrintCPUAndGPUTrees(class KDNodeCPU *CPUNode, bool PauseOnEachNode = false);

	std::vector<int> obj_index_list;
	int root_index;

private:
	RKDTreeNodeGPU *nodes;


	float3 *verts, *faces, *normals;

	int num_nodes;
	int num_faces;
	int num_verts;

	void buildTree(class KDNodeCPU *CPUNode);
};

