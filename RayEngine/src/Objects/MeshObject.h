#pragma once
#include "SceneObject.h"
#include "RStaticMesh.h"
#include "Attachable.h"


class RMeshObject :
	public RSceneObject
{
public:
	RMeshObject(const char* file_name);

	RStaticMesh* root_component;

	// Inherited via Attachable
	virtual void on_attach() override;
	virtual void on_detach() override;
	virtual void on_update() override;
	virtual void on_event(RayEngine::Event& event) override;

	// Inherited via Attachable
};

