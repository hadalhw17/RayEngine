#pragma once
#include "SceneObject.h"
#include "MovableCamera.h"

class RCharacter : public RSceneObject
{
public:
	RCharacter();
	~RCharacter();

	virtual void tick(float delta_time);

	RMovableCamera *camera;
};

