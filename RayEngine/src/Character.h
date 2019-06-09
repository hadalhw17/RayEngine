#pragma once
#include "SceneObject.h"
#include "MovableCamera.h"


class RAY_ENGINE_API RCharacter : public RayEngine::RSceneObject
{
public:
	RCharacter();
	~RCharacter();

	virtual void tick(float delta_time);

	RMovableCamera camera;
};


