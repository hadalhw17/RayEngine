#pragma once

#include "RStaticMesh.h"
#include "Primitives/GPUBoundingBox.h"
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
