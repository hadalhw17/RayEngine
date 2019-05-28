#pragma once

#include "helper_math.h"

#include <vector>
#include <string>

class RMeshAdjacencyTable;
class RKDTreeCPU;
class GPUBoundingBox;
class RStaticMesh;

struct GridNode
{
	float3 point;
	float distance;
	size_t face_index;
	float3 hit_point;
	GridNode()
	{
		distance = 9999999999;
	}

	GridNode(float3 _point, float _distance)
	{
		point = _point;
		distance = _distance;
	}


};

class Grid
{
public:
	Grid();
	Grid(std::string file_name);
	~Grid();

	inline GridNode *get_nodes() { return &voxels[0]; }
	inline float3 get_steps();
	inline int3 get_dim();
	inline float3 get_box_max() { return box_max; }

	std::vector<GridNode> voxels;
	float3 spacing;
	int3 sdf_dim;
	float3 box_max;
private:
	void load_volume_floam_file(const std::string filename);



	GPUBoundingBox* volume;
	RStaticMesh* mesh;

	float2 min_max_distance;


};

