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
}