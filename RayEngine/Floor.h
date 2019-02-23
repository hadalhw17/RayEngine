#pragma once

#include "SceneObject.h"
class AFloor :
	public RSceneObject
{
public:
	AFloor();
	~AFloor();

	virtual void tick(float delta_time);
};

