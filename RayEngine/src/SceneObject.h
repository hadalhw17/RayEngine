#pragma once

#include <vector>
#include "RayEngine/RayEngine.h"
#include "RStaticMesh.h"
#include "GPUBoundingBox.h"


class RSceneObject
{
public:
	RSceneObject(const char *file_name);
	~RSceneObject();

	virtual void tick(float delta_time);

	std::vector<class RObjectComponent *> components;
	RStaticMesh *root_component;

	GPUSceneObject object_properties;
	GPUBoundingBox collision_box;
};

