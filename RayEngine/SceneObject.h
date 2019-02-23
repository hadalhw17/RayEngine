#pragma once

#include <vector>
#include "RayEngine.h"
#include "RStaticMesh.h"


class RSceneObject
{
public:
	RSceneObject(const char *file_name);
	~RSceneObject();

	virtual void tick(float delta_time);

	std::vector<class RObjectComponent *> components;
	RStaticMesh *root_component;

	GPUSceneObject object_properties;
};

