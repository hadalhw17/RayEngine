#pragma once

#include "TextureObject.h"

struct VoxelMaterial
{
	uint2 texture_resolution;
	std::vector<RTextureObject> texture_aray;
};

class RMaterial
{
public:
	RMaterial(const RTextureObject& texture1, const RTextureObject& texture2, const RTextureObject& texture3);
	RMaterial(const RTextureObject& texture1);
	//std::vector<float4> read_ppm(char* filename);

	VoxelMaterial material;
};

