#include "repch.h"


#include "KDThreeGPU.h"

#include "Object.h"


RKDThreeGPU::RKDThreeGPU(RKDTreeCPU* CPUNode)
{
	num_nodes = CPUNode->numNodes;
	root_index = CPUNode->root->vid;

	num_verts = CPUNode->num_verts;
	num_norms = CPUNode->num_norms;
	num_faces = CPUNode->num_faces;
	num_uvs = CPUNode->num_uvs;

	float3* tmp_verts = CPUNode->verts;
	verts = new float3[num_verts];
	for (int i = 0; i < num_verts; ++i)
	{
		verts[i] = tmp_verts[i];
	}

	uint3* tmp_faces = CPUNode->faces;
	faces = new uint3[num_faces];
	for (int i = 0; i < num_faces; ++i)
	{
		faces[i] = tmp_faces[i];
	}

	float3* tmp_norms = CPUNode->norms;
	normals = new float3[num_norms];
	for (int i = 0; i < num_norms; ++i)
	{
		normals[i] = tmp_norms[i];
	}

	float2* tmp_uvs = CPUNode->uvs;
	uvs = new float2[num_uvs];
	for (int i = 0; i < num_uvs; ++i)
	{
		uvs[i] = tmp_uvs[i];
	}

	nodes = new RKDTreeNodeGPU[num_nodes];

	obj_index_list = {};

	buildTree(CPUNode->root);

	//PrintCPUAndGPUTrees(CPUNode->root);
}


RKDThreeGPU::~RKDThreeGPU()
{
	if (num_verts > 0) {
		delete[] verts;
		delete[] normals;
	}

	if (num_faces > 0) {
		delete[] faces;
	}

	if (num_uvs > 0)
	{
		delete[] uvs;
	}

	delete[] nodes;
}

RKDTreeNodeGPU* RKDThreeGPU::GetNodes() const
{
	return nodes;
}

std::vector<int> RKDThreeGPU::GetIndexList() const
{
	return obj_index_list;
}

float3* RKDThreeGPU::get_verts() const
{
	return verts;
}

uint3* RKDThreeGPU::get_faces() const
{
	return faces;
}

float3* RKDThreeGPU::get_normals() const
{
	return normals;
}

inline float2* RKDThreeGPU::get_uvs() const
{
	return uvs;
}

size_t RKDThreeGPU::GetNumNodes() const
{
	return num_nodes;
}

size_t RKDThreeGPU::get_num_verts() const
{
	return num_verts;
}

size_t RKDThreeGPU::get_num_faces() const
{
	return num_faces;
}

size_t RKDThreeGPU::get_num_norms() const
{
	return num_norms;
}

inline size_t RKDThreeGPU::get_num_uvs() const
{
	return num_uvs;
}

size_t RKDThreeGPU::get_root_index() const
{
	return root_index;
}


void RKDThreeGPU::PrintCPUAndGPUTrees(KDNodeCPU* CPUNode, bool PauseOnEachNode)
{
	std::cout << "root index " << root_index << std::endl;
	CPUNode->PrintDebugString();
	nodes[CPUNode->vid].PrintDebugString();

	if (PauseOnEachNode) {
		std::cin.ignore();
	}

	if (CPUNode->LeftNode) {
		PrintCPUAndGPUTrees(CPUNode->LeftNode, PauseOnEachNode);
	}
	if (CPUNode->RightNode) {
		PrintCPUAndGPUTrees(CPUNode->RightNode, PauseOnEachNode);
	}
}

void RKDThreeGPU::buildTree(KDNodeCPU* CPUNode)
{
	int index = CPUNode->vid;

	nodes[index].is_leaf = CPUNode->isLeaf;
	nodes[index].axis = CPUNode->axis;
	nodes[index].split_val = CPUNode->split_val;
	nodes[index].box = GPUBoundingBox(&CPUNode->box);
	nodes[index].num_objs = CPUNode->numFaces;

	if (CPUNode->isLeaf)
	{
		nodes[index].index_of_first_object = obj_index_list.size(); // tri_index_list initially contains 0 elements.

		// Add triangles to tri_index_list as each leaf node is processed.
		for (int i = 0; i < CPUNode->numFaces; ++i) {
			obj_index_list.push_back(CPUNode->objIndeces[i]);
		}

		for (int i = 0; i < 6; ++i) {
			if (CPUNode->ropes[i]) {
				nodes[index].neighbor_node_indices[i] = CPUNode->ropes[i]->vid;
			}
			else
				nodes[index].neighbor_node_indices[i] = -1;
		}
	}
	else {
		if (CPUNode->LeftNode) {
			// Set child node index for current node and recurse.
			nodes[index].left_index = CPUNode->LeftNode->vid;
			buildTree(CPUNode->LeftNode);
		}
		if (CPUNode->RightNode) {
			// Set child node index for current node and recurse.
			nodes[index].right_index = CPUNode->RightNode->vid;
			buildTree(CPUNode->RightNode);
		}
	}
}
