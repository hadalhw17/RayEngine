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
		//Grid* distance_field = new Grid(std::string("SDFs/Edited.rsdf"));
		//world_chunk.set_location(make_float3(10, 0, 100));
		//distance_field[1] = Grid(std::string(PATH_TO_VOLUMES) + std::string("cat250.rsdf"));
		RTextureObject snow_texture = RTextureObject((char*) "Meshes/snow.ppm");
		RTextureObject cobblestone_texture = RTextureObject((char*) "Meshes/1.ppm");
		RTextureObject grass_texture = RTextureObject((char*) "Meshes/2.ppm");
		RTextureObject rock_texture = RTextureObject((char*) "Meshes/3.ppm");
		RMaterial *default_material = new RMaterial(rock_texture, grass_texture, rock_texture,"Ground");
		RMaterial *cobblestone_material = new RMaterial(cobblestone_texture, "Cobblestone");
		RMaterial *snow_material = new RMaterial(snow_texture, "Snow");
		materials.push_back(default_material);
		materials.push_back(cobblestone_material);
		//materials.push_back(snow_material);
		init_cuda_res();
		generate_chunk();
	}

};


//namespace meta {
//
//	template <>
//	inline auto registerMembers<TestScene>()
//	{
//		return members(
//			member("noise", &TestScene::noise),
//			member("materials", &TestScene::materials),
//			member("world_chunk", &TestScene::world_chunk),
//			member("scene_camera", &TestScene::scene_camera),
//			//member("scene_objects", &TestScene::scene_objects),
//			member("scene_settings", &TestScene::scene_settings)
//		);
//	}
//
//} // end of namespace meta