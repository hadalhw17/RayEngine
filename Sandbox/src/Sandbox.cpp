#include <RayCore.h>
#include "TestScene.h"
#include"TextCharacter.h"
#include "TestSceneLayer.h"
#include <Meta.h>
#include <SDFScene.h>
#include <json.hpp>
#include <JsonCast.h>

template<typename T>
void write_to_file(const T& obj)
{
	std::ofstream os("scene.json");
	json scene;
	to_json(scene, obj);
	RE_LOG(std::setw(4) << scene);
}

class Sandbox : public RayEngine::Application
{
public:
	Sandbox() 
		: Application()
	{
		TextCharacter *main_chacter =  new TextCharacter();
		TestScene *scene = new  TestScene(*main_chacter);
		TestSceneLayer *scene_layer = new TestSceneLayer(*scene);

		//std::cout << "Serializing scene:" << '\n';
		//json root;
		//root = meta::serialize(scene);
		//std::cout << std::setw(4) << root << std::endl;
		push_layer(scene_layer);
		push_overlay(new RayEngine::RUILayer());
		std::cout << "Serializing character:" << '\n';
		json root;
		root = meta::serialize(*main_chacter);
		std::cout << std::setw(4) << root << std::endl;
	}

	~Sandbox()
	{

	}

};



RayEngine::Application* create_application()
{
	return new Sandbox();
}

