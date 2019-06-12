#pragma once


#include <vector>
#include <string>
#include "RayEngine/RayEngine.h"

class RMeshAdjacencyTable;
class RKDTreeCPU;
struct GPUBoundingBox;
class RStaticMesh;

struct RAY_ENGINE_API GridNode
{
	float3 point;
	float distance;
	float material;
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

#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<GridNode>()
	{
		return members(
			member("point", &GridNode::point),
			member("distance", &GridNode::distance),
			member("face_index", &GridNode::face_index),
			member("hit_point", &GridNode::hit_point)
		);
	}

} // end of namespace meta


class RAY_ENGINE_API Grid
{
public:
	Grid();
	Grid(std::string file_name);
	~Grid();

	inline GridNode* get_nodes() { return &voxels[0]; }
	inline float3 get_steps();
	inline uint3 get_dim();
	inline float3 get_box_max() { return box_max; }

	std::vector<GridNode> voxels;
	float3 spacing;
	uint3 sdf_dim;
	float3 box_max;
	GPUBoundingBox* volume;
	RStaticMesh* mesh;
	float2 min_max_distance;
private:
	void load_volume_floam_file(const std::string filename);






};


namespace meta {

	template <>
	inline auto registerMembers<Grid>()
	{
		return members(
			//member("voxels", &Grid::voxels),
			member("spacing", &Grid::spacing),
			member("sdf_dim", &Grid::sdf_dim),
			//member("mesh", &Grid::mesh),
			member("min_max_distance", &Grid::min_max_distance)
		);
	}

} // end of namespace meta
