#pragma once

#include "helper_math.h"
#include <vector>

struct VoxelMaterial
{
	uint2 texture_resolution;
	std::vector<float4> texture_aray[3];
};

class RMaterial
{
public:
	RMaterial(char* texture1, char* texture2, char* texture3);
	std::vector<float4> read_ppm(char* filename);

	VoxelMaterial material;
};

