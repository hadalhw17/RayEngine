#include "Character.h"

#include "helper_math.h"




	RCharacter::RCharacter()
	{
		camera = RMovableCamera();
		Material material;
		material.type = TILE;
		material.color = make_float3(0.f, 0.5f, 0.2f);
		material.uvs = true;
		float3 location = make_float3(0);
		object_properties.material = material;
		object_properties.is_character = true;
	}


	RCharacter::~RCharacter()
	{
	}

	void RCharacter::tick(float delta_time)
	{
		RSceneObject::tick(delta_time);

		if (&camera)
		{
			float xDirection = sinf(camera.yaw) * cosf(camera.pitch);
			float yDirection = sinf(camera.pitch);
			float zDirection = cosf(camera.yaw) * cosf(camera.pitch);
			float3 directionToCamera = make_float3(xDirection, yDirection, zDirection);
			float3 view_direction = directionToCamera * (-1.0f);
			float3 eyePosition = camera.position;

			object_properties.location = eyePosition;
			object_properties.rotation = make_float3(0, -camera.yaw, 0.f);
		}
	}

