#include "repch.h"

#include "Grid.h"
#include <filesystem/resolver.h>
#include "Primitives/MeshAdjacencyTable.h"


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
	std::ifstream volume_stream(filename.c_str(), std::ios::binary);
	RAY_ENGINE_ASSERT(volume_stream.is_open(), "Volume file not found!");
	if (volume_stream.is_open())
	{
		// Nove reader to the begining of the file.
		volume_stream.seekg(0, std::ios::beg);
		// Read volume dimentions from the file.
		volume_stream.read(reinterpret_cast<char*>(&sdf_dim.x), sizeof(int));
		volume_stream.read(reinterpret_cast<char*>(&sdf_dim.y), sizeof(int));
		volume_stream.read(reinterpret_cast<char*>(&sdf_dim.z), sizeof(int));

		// Write spacings of the volume to the memory.
		volume_stream.read(reinterpret_cast<char*>(&spacing.x), sizeof(float));
		volume_stream.read(reinterpret_cast<char*>(&spacing.y), sizeof(float));
		volume_stream.read(reinterpret_cast<char*>(&spacing.z), sizeof(float));


		// Write distances to the memory.
		voxels = std::vector<GridNode>(sdf_dim.x * sdf_dim.y * sdf_dim.z);
		for (size_t iz = 0; iz < sdf_dim.z; iz++)
		{
			for (size_t iy = 0; iy < sdf_dim.y; iy++)
			{
				for (size_t ix = 0; ix < sdf_dim.x; ix++)
				{
					float2 in;
					volume_stream.read(reinterpret_cast<char*>(&in), sizeof(float2));
					voxels[ix + sdf_dim.y * (iy + sdf_dim.x * iz)].distance = in.x;
					voxels[ix + sdf_dim.y * (iy + sdf_dim.x * iz)].material = in.y;
					voxels[ix + sdf_dim.y * (iy + sdf_dim.x * iz)].point = make_float3(ix * spacing.x, iy * spacing.y, iz * spacing.z);
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

uint3 Grid::get_dim()
{
	return sdf_dim;
}
