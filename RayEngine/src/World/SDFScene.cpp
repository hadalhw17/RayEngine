#include "repch.h"


#include "SDFScene.h"
#include "RayEngine/Application.h"
#include "World/Chunk.h"

extern "C" void initialize_volume_render(RCamera& sceneCam, const RayEngine::RChunk& world_chunk, const int& num_sdf, const std::vector<VoxelMaterial>& materials, const RenderingSettings& render_settings,
	const SceneSettings& scene_settings, const RayEngine::RPerlinNoise& nosie);
extern "C"
void cuda_update_chunk(const RayEngine::RChunk& world_chunk, const RayEngine::RPerlinNoise& noise);
extern "C"
void cuda_update_chunk_gen(const RayEngine::RChunk& world_chunk, const RayEngine::RPerlinNoise& noise);

extern
void load_map(std::string filename);

SDFScene::SDFScene()
{

}

SDFScene::SDFScene(RCharacter& character)
	:RScene(character)
{
}

void SDFScene::move_chunk(float3 chunk_location)
{
	float3 new_location = chunk_location - world_chunk.get_sdf().box_max / 2;
	world_chunk.set_location(world_chunk.get_location() + chunk_location);
}

void SDFScene::build_scene()
{
	setup_camera();
	RE_LOG("Building scene...");
	update_camera();
	RE_LOG("Done building scene...");
}

void SDFScene::init_cuda_res()
{
	RayEngine::Application& app = RayEngine::Application::get();
	initialize_volume_render(scene_camera, world_chunk, 1, materials, app.render_settings, scene_settings, noise);
}

void SDFScene::update_chunk()
{

	cuda_update_chunk(world_chunk, noise);
	
}

void SDFScene::generate_chunk()
{
	RE_LOG("Generating chunk");
	cuda_update_chunk_gen(world_chunk, noise);
}

void SDFScene::load_chunk_from_file(std::string filename)
{
	std::thread(load_map, filename).detach();
	
}

void SDFScene::write_to_file()
{
	//std::ofstream os("scene.json");
	//json scene = &*this;
	//RE_LOG(std::setw(4) << scene);
}

