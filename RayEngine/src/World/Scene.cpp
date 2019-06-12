#include "repch.h"


#include "Scene.h"


#include "Objects/Character.h"

#include <filesystem/resolver.h>

RScene::RScene()
	:main_character(*(new RCharacter()))
{
}

RScene::RScene(RCharacter& _character)
	:main_character(_character)
{
}

extern "C"
void update_objects(std::vector<GPUSceneObject> objs);

RScene::~RScene()
{
	clear_memory();
}
void RScene::on_attach()
{
	build_scene();
}
void RScene::on_detach()
{
}
void RScene::on_update()
{
	rebuild_scene();
	update_camera();
}
void RScene::on_event(RayEngine::Event& event)
{
	main_character.on_event(event);
}
bool RScene::on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e)
{
	return false;
}
bool RScene::on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e)
{
	return false;
}
bool RScene::on_mouse_moved(RayEngine::MouseMovedEvent& e)
{
	return false;
}
bool RScene::on_mouse_scrolled(RayEngine::MouseScrolledEvent& e)
{
	return false;
}
bool RScene::on_window_reseized(RayEngine::WindowResizedEvent& e)
{
	return false;
}
bool RScene::on_key_released(RayEngine::KeyReleaseEvent& e)
{
	return false;
}
bool RScene::on_key_pressed(RayEngine::KeyPressedEvent& e)
{
	return false;
}
bool RScene::on_key_typed(RayEngine::KeyTypedEvent& e)
{
	return false;
}
float RScene::moveCounter = 0.f;


void RScene::Tick(float delta_time)
{
	main_character.tick(delta_time);

	rebuild_scene();
	std::vector<GPUSceneObject> tmp_objs;
	for (auto obj : scene_objects)
	{
		obj->tick(delta_time);
		tmp_objs.push_back(obj->object_properties);
	}
	update_objects(tmp_objs);
}

void RScene::update_camera()
{
	main_character.camera.build_camera(scene_camera);
}


std::vector<float4> RScene::read_ppm(char* filename)
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
	std::cout << width << height << std::endl;
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

void RScene::setup_camera()
{
	std::cout << "Camera initial setup." << std::endl;
	scene_camera = RCamera();
	main_character.camera.build_camera(scene_camera);
}
