#pragma once
#include <C:\dev\RayEngine\RayEngine\src\Layers\SceneLayer.h>
#include "TestScene.h"
class TestSceneLayer :
	public RayEngine::RSceneLayer
{
public:
	TestSceneLayer(TestScene& scene)
		:RayEngine::RSceneLayer(scene) {}
};

