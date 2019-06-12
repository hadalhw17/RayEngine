#pragma once
#include "SceneObject.h"
#include "RStaticMesh.h"


class RMeshObject :
	public RayEngine::RSceneObject
{
public:
	RMeshObject(const char* file_name);

	RStaticMesh* root_component;
};

