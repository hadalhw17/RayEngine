#pragma once
#include <World/SDFScene.h>
#include <Objects/Character.h>
#include <Engine/TextureObject.h>
#include <Engine/Material.h>
#include <World/PerlinNoise.h>
#include <RayEngine/Application.h>
#include <World/Grid.h>


class  TestScene :
	public SDFScene
{
public:
	TestScene(RCharacter &character)
	:SDFScene(character)
	{
		Grid* distance_field = new Grid(std::string("SDFs/Edited.rsdf"));
		world_chunk = RayEngine::RChunk(*distance_field);
		//world_chunk.set_location(make_float3(10, 0, 100));
		//distance_field[1] = Grid(std::string(PATH_TO_VOLUMES) + std::string("cat250.rsdf"));
		scene_settings.volume_resolution = make_uint3(1);
		scene_settings.world_size = make_float3(1.f);
		RTextureObject snow_texture = RTextureObject((char*) "Meshes/snow.ppm");
		RTextureObject cobblestone_texture = RTextureObject((char*) "Meshes/1.ppm");
		RTextureObject grass_texture = RTextureObject((char*) "Meshes/2.ppm");
		RTextureObject rock_texture = RTextureObject((char*) "Meshes/3.ppm");
		RMaterial default_material(rock_texture, grass_texture, rock_texture);
		RMaterial cobblestone_material(cobblestone_texture);
		RMaterial snow_material(snow_texture);
		materials.push_back(default_material.material);
		materials.push_back(cobblestone_material.material);
		materials.push_back(snow_material.material);
		RayEngine::Application& app = RayEngine::Application::get();
		init_cuda_res();
	}
};

