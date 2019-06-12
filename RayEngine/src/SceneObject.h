#pragma once

#include "RStaticMesh.h"
#include "GPUBoundingBox.h"
#include "Events/Event.h"
#include "cuda-src/gpu_structs.h"

namespace RayEngine
{
	class RAY_ENGINE_API RSceneObject
	{
	public:
		RSceneObject();

		virtual ~RSceneObject();

		virtual void tick(float delta_time);

		std::vector<class RObjectComponent*> components;
		//RStaticMesh* root_component;

		GPUSceneObject object_properties;
		GPUBoundingBox collision_box;
	};

}


#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<RayEngine::RSceneObject>()
	{
		return members(
			member("components", &RayEngine::RSceneObject::components),
			member("object_properties", &RayEngine::RSceneObject::object_properties),
			member("collision_box", &RayEngine::RSceneObject::collision_box)
		);
	}

} // end of namespace meta