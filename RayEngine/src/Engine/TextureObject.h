#pragma once
#include <vector>
#include "RayEngine/RayEngine.h"
#include <ostream>

class RAY_ENGINE_API RTextureObject
{
public:
	RTextureObject(char* filename)
	{
		texture_path = filename;
		texels = read_ppm(filename);
	}
	std::vector<struct float4> read_ppm(char* filename);

	std::vector<struct float4> texels;
	uint2 resolution;
	std::string texture_path;
};
