#include "Character.h"

#include "cutil_math.h"



RCharacter::RCharacter()
	:RSceneObject((char *)"Meshes/character.obj")
{
	Material material;
	material.type = PHONG;
	material.color = make_float4(0, 0.5, 0.2, 0);
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

	if (camera)
	{
		float xDirection = sin(camera->yaw) * cos(camera->pitch);
		float yDirection = sin(camera->pitch);
		float zDirection = cos(camera->yaw) * cos(camera->pitch);
		float3 directionToCamera = make_float3(xDirection, yDirection, zDirection);
		float3 view_direction = directionToCamera * (-1.0f);
		float3 eyePosition = camera->position;

		object_properties.location = eyePosition;
		object_properties.rotation = make_float3(0, -camera->yaw, 0.f);
	}
}
