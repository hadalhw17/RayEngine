#include "Material.h"

#include <iostream>
#include "helper_math.h"
#include <vector>
#include <fstream>
#include <filesystem/resolver.h>

RMaterial::RMaterial(char* texture1, char* texture2, char* texture3)
{
	material.texture_aray[0] = std::vector<float4>(read_ppm(texture1));
	material.texture_aray[1] = std::vector<float4>(read_ppm(texture2));
	material.texture_aray[2] = std::vector<float4>(read_ppm(texture3));
}

std::vector<float4> RMaterial::read_ppm(char* filename)
{
	std::ifstream is(filename);
	std::vector<float4>imag;
	std::string line_str;
	std::getline(is, line_str);
	if (line_str != "P3")
		return imag;
	std::getline(is, line_str); // Comment.
	std::getline(is, line_str);
	std::istringstream line(line_str);
	int width, height;
	line >> width >> height;
	material.texture_resolution = make_uint2(width, height);

	std::getline(is, line_str); // Color.

	int i = 0;
	while (std::getline(is, line_str))
	{
		float4 img;
		line = std::istringstream(line_str);
		line >> img.x;
		std::getline(is, line_str);
		line = std::istringstream(line_str);
		line >> img.y;
		std::getline(is, line_str);
		line = std::istringstream(line_str);
		line >> img.z;
		img = make_float4(img.x / 255.f, img.y / 255.f, img.z / 255.f, 0);
		imag.push_back(img);
		++i;
	}

	return imag;
}
