#include "repch.h"

#include "ObjectComponent.h"


RSceneObject* RObjectComponent::s_owner = nullptr;

RObjectComponent::RObjectComponent()
{
}


RObjectComponent::~RObjectComponent()
{
}

void RObjectComponent::on_attach(RSceneObject* owner)
{
	s_owner = owner;
}

void RObjectComponent::on_detach()
{
}

void RObjectComponent::on_update()
{
}

void RObjectComponent::on_event(RayEngine::Event& event)
{
}

void RObjectComponent::tick(float delta_time)
{
}
