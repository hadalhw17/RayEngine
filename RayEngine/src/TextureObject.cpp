#include "repch.h"


#include "TextureObject.h"
#include <filesystem/resolver.h>

std::vector<float4> RTextureObject::read_ppm(char* filename)
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
	resolution = make_uint2(width, height);
	std::getline(is, line_str); // Colour.

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