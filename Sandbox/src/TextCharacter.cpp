#include <Engine/KeyCodes.h>
#include <Engine/MouseButtonCodes.h>
#include <Primitives/Camera.h>

#include "TextCharacter.h"
#include "TestSceneLayer.h"
#include <iostream>
#include "ResourceManager.h"

TextCharacter::TextCharacter()
{
	ResourceManager* res_manager = new ResourceManager();
	this->attach_component(res_manager);
}

void TextCharacter::on_attach()
{
	RCharacter::on_attach();
}

void TextCharacter::on_detach()
{
	RCharacter::on_detach();
}

bool exited_chunk(const GPUBoundingBox& tBox, const float3& vecPoint)
{
	return
		(vecPoint.x < tBox.Min.x || vecPoint.x > tBox.Max.x) ||
		(vecPoint.z < tBox.Min.z || vecPoint.z > tBox.Max.z);

}

float3 position_chunk(const GPUBoundingBox& tBox, const float3& vecPoint)
{
	RayEngine::Application& app = RayEngine::Application::get();
	TestSceneLayer& scene_layer = dynamic_cast<TestSceneLayer&>(app.get_scene_layer());
	SDFScene& scene = static_cast<SDFScene&>(scene_layer.get_scene());

	GPUBoundingBox world_box = GPUBoundingBox(make_float3(0.f) + scene.get_world_chunk().get_location(),
		scene.get_world_chunk().get_sdf().box_max + scene.get_world_chunk().get_location());

	float3 new_pos = make_float3(0);
	if (vecPoint.x < tBox.Min.x) new_pos.x -= scene.scene_settings.world_size.x;
	else if (vecPoint.x > tBox.Max.x) new_pos.x += scene.scene_settings.world_size.x;
	if (vecPoint.z < tBox.Min.z) new_pos.z -= scene.scene_settings.world_size.z;
	else if (vecPoint.z > tBox.Max.z) new_pos.z += scene.scene_settings.world_size.z;
	
	return new_pos;
}

void TextCharacter::on_update()
{
	RayEngine::Application& app = RayEngine::Application::get();

	TestSceneLayer& scene_layer = dynamic_cast<TestSceneLayer&>(app.get_scene_layer());
	//------------------------Spawn object on click---------------------------------------------------------
	if (app.should_spawn && app.edit_mode && !app.ctrl)
	{
		scene_layer.brush.brush_type = scene_layer.brush_type;
		app.app_spawn_obj(scene_layer.m_scene->get_camera(), scene_layer.brush, last_x, app.get_window().get_heigth() - last_y);
	}
	else if(app.should_spawn && !app.edit_mode && !app.ctrl)
	{
		
	}
	//-----------------------------------------------------------------------------------------------------
}



// Bind events for the character.
void TextCharacter::on_event(RayEngine::Event& event)
{
	RayEngine::EventDispatcher dispatcher(event);

	dispatcher.dipatch<RayEngine::MouseButtonPresedEvent>(BIND_EVENT_FN(TextCharacter::on_mouse_button_pressed));
	dispatcher.dipatch<RayEngine::MouseButtonReleasedEvent>(BIND_EVENT_FN(TextCharacter::on_mouse_button_relseased));
	dispatcher.dipatch<RayEngine::MouseMovedEvent>(BIND_EVENT_FN(TextCharacter::on_mouse_moved));
	dispatcher.dipatch<RayEngine::MouseScrolledEvent>(BIND_EVENT_FN(TextCharacter::on_mouse_scrolled));
	dispatcher.dipatch<RayEngine::WindowResizedEvent>(BIND_EVENT_FN(TextCharacter::on_window_reseized));
	dispatcher.dipatch<RayEngine::KeyReleaseEvent>(BIND_EVENT_FN(TextCharacter::on_key_released));
	dispatcher.dipatch<RayEngine::KeyPressedEvent>(BIND_EVENT_FN(TextCharacter::on_key_pressed));
	dispatcher.dipatch<RayEngine::KeyTypedEvent>(BIND_EVENT_FN(TextCharacter::on_key_typed));
}

bool TextCharacter::on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e)
{
	RayEngine::Application& app = RayEngine::Application::get();
	TestSceneLayer &scene_layer = dynamic_cast<TestSceneLayer&>(app.get_scene_layer());

	switch (e.get_button())
	{
	case RE_MOUSE_BUTTON_1:
		if (app.shift)
		{
			scene_layer.brush_type = TerrainBrushType::SPHERE_SUBTRACT;
			HitResult res = app.cast_single_ray(scene_layer.m_scene->get_camera(), last_x, app.get_window().get_heigth() - last_y);
			SDFScene& sdf_scene = dynamic_cast<SDFScene&>(scene_layer.get_scene());
			if (&sdf_scene)
			{
				if (res.hits)
				{
					for (auto component : components)
					{
						ResourceManager* resource_manager = dynamic_cast<ResourceManager*>(component);
						if (resource_manager)
						{
							resource_manager->add_resource(Resource(sdf_scene.get_materials()[res.material_index]->get_name()), 10);
						}

					}
				}
			}

		}
		else
		{
			scene_layer.brush_type = TerrainBrushType::SPHERE_ADD;
		}
		break;
	case RE_MOUSE_BUTTON_2:
		if (app.shift)
		{
			scene_layer.brush_type = TerrainBrushType::CUBE_SUBTRACT;
		}
		else
		{
			scene_layer.brush_type = TerrainBrushType::CUBE_ADD;
		}
		break;
	default:
		break;
	}
	app.should_spawn = true;
	return true;
}

bool TextCharacter::on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e)
{
	RayEngine::Application& app = RayEngine::Application::get();

	app.mouse_right = false;
	app.should_spawn = false;
	return true;
}

bool TextCharacter::on_mouse_moved(RayEngine::MouseMovedEvent& e)
{
	RayEngine::Application& app = RayEngine::Application::get();
	float x_pos = e.get_x();
	float y_pos = e.get_y();

	double deltaX = last_x - x_pos;
	double deltaY = last_y - y_pos;
	if (app.edit_mode && !app.ctrl)
	{
		deltaX = 0;
		deltaY = 0;
	}
	else if (app.edit_mode && app.ctrl && app.mouse_right)
	{
		deltaX = -deltaX;
		deltaY = -deltaY;
	}
	else if (app.edit_mode && app.ctrl && !app.mouse_right)
	{
		deltaX = 0;
		deltaY = 0;
	}


	if (deltaX != 0.f || deltaY != 0.f)
	{
		camera.change_yaw(-deltaX * 0.005f);
		camera.change_pitch(-deltaY * 0.005f);
	}

	last_x = x_pos;
	last_y = y_pos;
	return false;
}

bool TextCharacter::on_mouse_scrolled(RayEngine::MouseScrolledEvent& e)
{
	RayEngine::Application& app = RayEngine::Application::get();
	TestSceneLayer& scene_layer = dynamic_cast<TestSceneLayer&>(app.get_scene_layer());
	scene_layer.brush_radius += e.get_y_offset() * 0.01f;
	return false;
}

bool TextCharacter::on_window_reseized(RayEngine::WindowResizedEvent& e)
{
	return false;
}

bool TextCharacter::on_key_released(RayEngine::KeyReleaseEvent& e)
{
	return false;
}

bool TextCharacter::on_key_pressed(RayEngine::KeyPressedEvent& e)
{
	RayEngine::Application& app = RayEngine::Application::get();

	RMovableCamera tmp_cam = camera;
	switch (e.get_key_code())
	{
	case RE_KEY_W:
		tmp_cam.move_forward(app.character_speed);
		break;
	case RE_KEY_S:
		tmp_cam.move_forward(-app.character_speed);
		break;
	case RE_KEY_A:
		tmp_cam.strafe(app.character_speed);
		break;
	case RE_KEY_LEFT_SHIFT:
		app.shift = true;
		break;
	case RE_KEY_D:
		tmp_cam.strafe(-app.character_speed);
		break;
	case RE_KEY_Q:
		app.app_toggle_shadow();
		break;
	default:
		break;
	}
	RCamera tmp_ca;
	tmp_cam.build_camera(tmp_ca);
	if (app.render_settings.gravity)
	{
		bool overlaps = app.app_sdf_collision(tmp_ca);
		if (!overlaps && app.render_settings.gravity)
			camera = tmp_cam;
		tmp_cam.position.y -= 0.5;
		tmp_cam.build_camera(tmp_ca);
		overlaps = app.app_sdf_collision(tmp_ca);
		if (!overlaps && app.render_settings.gravity)
		{
			tmp_cam.position.y += 0.4f;
			camera = tmp_cam;
		}
	}
	else
	{
		camera = tmp_cam;
	}
	TestSceneLayer& scene_layer = dynamic_cast<TestSceneLayer&>(app.get_scene_layer());
	SDFScene& scene = static_cast<SDFScene&>(scene_layer.get_scene());
	GPUBoundingBox world_box = GPUBoundingBox(make_float3(0.f) + scene.get_world_chunk().get_location(),
		scene.scene_settings.world_size + scene.get_world_chunk().get_location());

	if (exited_chunk(world_box, camera.position))
	{
		float3 rounded_pos = { floorf(camera.position.x * scene.scene_settings.world_size.x) / scene.scene_settings.world_size.x, 0, 
			floorf(camera.position.z * scene.scene_settings.world_size.z) / scene.scene_settings.world_size.z };

		float3 pos = position_chunk(world_box, rounded_pos);
		scene.move_chunk({ floorf(pos.x * scene.scene_settings.world_size.x) / scene.scene_settings.world_size.x, 0,
			floorf(pos.z * scene.scene_settings.world_size.z) / scene.scene_settings.world_size.z });

		std::ostringstream file_name;

		file_name << "SDFs/" << scene.get_world_chunk().get_location().x  << "_" << scene.get_world_chunk().get_location().z << ".rsdf";
		std::ifstream volume_file_stream(file_name.str(), std::ios::binary);
		if (volume_file_stream.is_open())
		{
			REE_LOG("Loading chunk from file");
			scene.load_chunk_from_file(file_name.str());
		}
		else
			scene.generate_chunk();
	}
	switch (e.get_key_code())
	{
	case RE_KEY_TAB:
		app.show_mouse = true;
		break;
	case RE_KEY_LEFT_CONTROL:
		app.ctrl = true;
		break;
	case RE_KEY_LEFT_ALT:
		app.edit_mode = !app.edit_mode;
		break;
	}

	
	return true;
}

bool TextCharacter::on_key_typed(RayEngine::KeyTypedEvent& e)
{
	return false;
}
