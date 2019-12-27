#pragma once
#include "Scene.h"
#include "RayEngine/RayEngine.h"
#include "World/Chunk.h"
#include "PerlinNoise.h"
#include "Objects/Character.h"
#include "Engine/TextureObject.h"
#include "Engine/Material.h"
#include <vector>
#include <string>


class RAY_ENGINE_API SDFScene :
	public RScene
{

public:
	SDFScene();
	SDFScene(RCharacter& character);

	inline RayEngine::RPerlinNoise& get_noise() { return noise; }
	inline std::vector<VoxelMaterial>& get_materials() { return materials; }
	inline RayEngine::RChunk& get_world_chunk() { return world_chunk; }
	void move_chunk(float3 chunk_location);

	void update_chunk();
	void generate_chunk();
	void load_chunk_from_file(std::string filename);

	RayEngine::RPerlinNoise noise;
	std::vector<VoxelMaterial> materials;
	virtual void write_to_file();
	
	RayEngine::RChunk world_chunk;
protected:
	virtual void build_scene() override;
	void init_cuda_res();

protected:

protected:
};


