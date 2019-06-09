#include "SDFScene.h"
#include "Camera.h"
#include "RayEngine/Application.h"

extern "C" void initialize_volume_render(RCamera& sceneCam, const Grid& sdf, const int& num_sdf, const std::vector<VoxelMaterial>& materials, const RenderingSettings& render_settings,
	const SceneSettings& scene_settings, const RayEngine::RPerlinNoise& nosie);

SDFScene::SDFScene()
{

}

SDFScene::SDFScene(RCharacter& character)
	:RScene(character)
{
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
	initialize_volume_render(scene_camera, distance_field, 1, materials, app.render_settings, scene_settings, noise);
}