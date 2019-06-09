#pragma once

#include "helper_math.h"

#include <vector>
#include <string>
#include "RayEngine/RayEngine.h"

class RMeshAdjacencyTable;
class RKDTreeCPU;
class GPUBoundingBox;
class RStaticMesh;

struct RAY_ENGINE_API GridNode
{
	float3 point;
	float distance;
	size_t face_index;
	float3 hit_point;
	GridNode()
	{
		distance = 9999999999.f;
	}

	GridNode(float3 _point, float _distance)
	{
		point = _point;
		distance = _distance;
	}


};

class RAY_ENGINE_API Grid
{
public:
	Grid();
	Grid(std::string file_name);
	~Grid();

	inline GridNode *get_nodes() { return &voxels[0]; }
	inline float3 get_steps();
	inline uint3 get_dim();
	inline float3 get_box_max() { return box_max; }

	std::vector<GridNode> voxels;
	float3 spacing;
	uint3 sdf_dim;
	float3 box_max;
private:
	void load_volume_floam_file(const std::string filename);



	GPUBoundingBox* volume;
	RStaticMesh* mesh;
	float2 min_max_distance;


};

