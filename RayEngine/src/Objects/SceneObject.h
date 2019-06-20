#pragma once

#include "RStaticMesh.h"
#include "Primitives/GPUBoundingBox.h"
#include "Events/Event.h"
#include "cuda-src/gpu_structs.h"

class RAY_ENGINE_API RSceneObject
{
public:
	RSceneObject();

	virtual ~RSceneObject();

	virtual void tick(float delta_time);

	RStaticMesh* root_component;

	GPUSceneObject object_properties;
	GPUBoundingBox collision_box;

	inline void attach_component(RObjectComponent* compoent) { components.push_back(compoent); }

	// Inherited via Attachable
	virtual void on_attach();
	virtual void on_detach();
	virtual void on_update();
	virtual void on_event(RayEngine::Event& event);

protected:
	std::vector<class RObjectComponent*> components;
};

//
//#include <Meta.h>
//namespace meta {
//
//	template <>
//	inline auto registerMembers<RSceneObject>()
//	{
//		return members(
//			member("components", &RSceneObject::components),
//			member("object_properties", &RSceneObject::object_properties),
//			member("collision_box", &RSceneObject::collision_box)
//		);
//	}
//
//} // end of namespace meta