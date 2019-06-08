#pragma once
#include "helper_math.h"
#include <vector>


class RTextureObject
{
public:
	RTextureObject(char* filename)
	{
		texels = read_ppm(filename);
	}
	std::vector<struct float4> read_ppm(char* filename);

	std::vector<struct float4> texels;
	uint2 resolution;
};

