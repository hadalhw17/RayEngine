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
	float2 *get_uvs() const;
	size_t GetNumNodes() const;
	size_t get_num_verts() const;
	size_t get_num_faces() const;
	size_t get_num_norms() const;
	size_t get_num_uvs() const;
	size_t get_root_index() const;
	

	void PrintCPUAndGPUTrees(class KDNodeCPU *CPUNode, bool PauseOnEachNode = false);

	std::vector<int> obj_index_list;
	size_t root_index;
	float2 *uvs;
	size_t num_uvs;

private:
	RKDTreeNodeGPU *nodes;


	float3 *verts, *faces, *normals;


	size_t num_nodes;
	size_t num_faces;
	size_t num_verts;
	size_t num_norms;


	void buildTree(class KDNodeCPU *CPUNode);
};

