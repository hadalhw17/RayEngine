#include "repch.h"


#include "SceneObject.h"

#include "ObjectComponent.h"

#include "Primitives/MeshAdjacencyTable.h"



RSceneObject::RSceneObject()
{
}

RSceneObject::~RSceneObject()
{
}

void RSceneObject::tick(float delta_time)
{
	for (auto component : components)
	{
		component->tick(delta_time);
	}
}

void RSceneObject::on_attach()
{
	for (auto component : components)
	{
		component->on_attach(this);
	}
}
void RSceneObject::on_detach()
{
	for (auto component : components)
	{
		component->on_detach();
	}
}
void RSceneObject::on_update()
{
	for (auto component : components)
	{
		component->on_update();
	}
}
void RSceneObject::on_event(RayEngine::Event& event)
{
	for (auto component : components)
	{
		component->on_event(event);
	}
}