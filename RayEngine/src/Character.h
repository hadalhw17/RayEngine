#pragma once
#include "SceneObject.h"
#include "MovableCamera.h"


class RAY_ENGINE_API RCharacter : public RayEngine::RSceneObject
{
public:
	RCharacter();
	~RCharacter();

	virtual void tick(float delta_time);

	RMovableCamera camera;
};


#include <Meta.h>

namespace meta {

	template <>
	inline auto registerMembers<RCharacter>()
	{
		return members(
			member("components", &RCharacter::components),
			member("object_properties", &RCharacter::object_properties),
			member("collision_box", &RCharacter::collision_box),
			member("camera", &RCharacter::camera)
		);
	}

} // end of namespace meta