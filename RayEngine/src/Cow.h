#pragma once
#include "SceneObject.h"
class ACow :
	public RSceneObject
{
public:
	ACow();
	~ACow();

	virtual void tick(float delta_time);
};

