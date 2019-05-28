#include "Grid.h"
#include <algorithm>
#include <iostream>
#include "KDTree.h"
#include "GPUBoundingBox.h"
#include <fstream>
#include <filesystem/resolver.h>
#include "MeshAdjacencyTable.h"
#include "RStaticMesh.h"
#include<string>


Grid::Grid()
{
}

Grid::Grid(std::string file_name)
{
	load_volume_floam_file(file_name);

}


Grid::~Grid()
{
}



void Grid::load_volume_floam_file(const std::string filename)
{
	// Open binary file for reading at the end.
	std::ifstream volume_stream(filename.c_str(), std::ios::in | std::ios::ate | std::ios::binary);
	float* memblock;
	if (volume_stream.is_open())
	{
		// Get the size of the binary file.
		size_t size = volume_stream.tellg();

		// Initialize a block of memory for that file.
		memblock = new float[size];

		// Nove reader to the begining of the file.
		volume_stream.seekg(0, std::ios::beg);

		// Read all floats of the file.
		volume_stream.read((char*)memblock, size);

		// Move readed to the begining again.
		volume_stream.seekg(0, std::ios::beg);

		// Read volume dimentions from the file.
		unsigned char in = 0;
		volume_stream.read((char*)& in, sizeof(int));
		sdf_dim.x = (int)in;

		volume_stream.read((char*)& in, sizeof(int));
		sdf_dim.y = (int)in;

		volume_stream.read((char*)& in, sizeof(int));
		sdf_dim.z = (int)in;

		// Write spacings of the volume to the memory.
		spacing.x = memblock[3];
		spacing.y = memblock[4];
		spacing.z = memblock[5];
		memblock += 6;

		// Write distances to the memory.
		voxels = std::vector<GridNode>(sdf_dim.x * sdf_dim.y * sdf_dim.z);
		for (unsigned int iz = 0; iz < sdf_dim.z; iz++)
		{
			for (unsigned int iy = 0; iy < sdf_dim.y; iy++)
			{
				for (unsigned int ix = 0; ix < sdf_dim.x; ix++)
				{
					voxels[ix + sdf_dim.y * (iy + sdf_dim.x * iz)].distance = *memblock;
					voxels[ix + sdf_dim.y * (iy + sdf_dim.x * iz)].point = make_float3(ix * spacing.x, iy * spacing.y, iz * spacing.z);
					memblock++;
				}
			}
		}
		box_max = voxels.back().point;
		volume_stream.close();
	}
}

float3 Grid::get_steps()
{
	return spacing;
}

int3 Grid::get_dim()
{
	return sdf_dim;
}
