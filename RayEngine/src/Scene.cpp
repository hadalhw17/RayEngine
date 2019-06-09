#include "Scene.h"
#include "Plane.h"
#include "Object.h"
#include "Triangle.h"
#include "Sphere.h"
#include "KDTree.h"
#include "RStaticMesh.h"

#include "Character.h"

#include <iostream>
#include <vector>
#include <fstream>
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
float RScene::moveCounter = 0.f;


void RScene::Tick(float delta_time)
{
	rebuild_scene();
	std::vector<GPUSceneObject> tmp_objs;
	for (auto obj : sceneObjects)
	{
		obj->tick(delta_time);
		tmp_objs.push_back(obj->object_properties);
	}
	update_objects(tmp_objs);
}

void RScene::update_camera()
{
	//main_character->camera = &camera;
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
