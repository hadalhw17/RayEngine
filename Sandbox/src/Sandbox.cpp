#include <RayCore.h>
#include "TestScene.h"
#include"TextCharacter.h"
#include "TestSceneLayer.h"


class Sandbox : public RayEngine::Application
{
public:
	Sandbox() 
		: Application()
	{
		TextCharacter* main_chacter= new TextCharacter();
		TestScene scene = TestScene(*main_chacter);
		TestSceneLayer* scene_layer = new TestSceneLayer(scene);
		push_layer(scene_layer);
		push_overlay(new RayEngine::RUILayer());
	}

	~Sandbox()
	{

	}

};


RayEngine::Application* create_application()
{
	return new Sandbox();
}