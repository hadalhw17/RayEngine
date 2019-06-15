#include "repch.h"
#include "Chunk.h"
namespace RayEngine
{
	RChunk::RChunk()
	{
		REE_LOG("SDF should be provided");
	}
	RChunk::RChunk(Grid& _distance_field)
		:distance_field(_distance_field)
	{
	}
	RChunk::~RChunk()
	{
	}
	void RChunk::update_sdf_box(const float3& new_box_max)
	{
		distance_field.box_max = new_box_max;
	}
}