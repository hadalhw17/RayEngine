#include "repch.h"


#include "Character.h"





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

void RCharacter::on_attach()
{
	RSceneObject::on_attach();
}

void RCharacter::on_detach()
{
}

void RCharacter::on_update()
{
}

void RCharacter::on_event(RayEngine::Event& event)
{
}

bool RCharacter::on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e)
{
	return false;
}

bool RCharacter::on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e)
{
	return false;
}

bool RCharacter::on_mouse_moved(RayEngine::MouseMovedEvent& e)
{
	return false;
}

bool RCharacter::on_mouse_scrolled(RayEngine::MouseScrolledEvent& e)
{
	return false;
}

bool RCharacter::on_window_reseized(RayEngine::WindowResizedEvent& e)
{
	return false;
}

bool RCharacter::on_key_released(RayEngine::KeyReleaseEvent& e)
{
	return false;
}

bool RCharacter::on_key_pressed(RayEngine::KeyPressedEvent& e)
{
	return false;
}

bool RCharacter::on_key_typed(RayEngine::KeyTypedEvent& e)
{
	return false;
}

void RCharacter::tick(float delta_time)
{
	RSceneObject::tick(delta_time);
	on_update();
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



