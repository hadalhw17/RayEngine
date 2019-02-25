#pragma once
#include "SceneObject.h"
class AGlass :
	public RSceneObject
{
public:
	AGlass();
	~AGlass();

	virtual void tick(float delta_time);
};

