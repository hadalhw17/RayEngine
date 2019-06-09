#include "SceneLayer.h"
#include "Camera.h"

#include "Scene.h"
#include "MovableCamera.h"
#include <iostream>
#include "Input.h"
#include "KeyCodes.h"
#include "MouseButtonCodes.h"
#include "Events/Event.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"
#include "Events/ApplicationEvent.h"
#include "RayEngine/Application.h"


extern void toggle_shadow();
extern bool sdf_collision(RCamera cam);
extern void spawn_obj(RCamera pos, TerrainBrush brush, int x, int y);

namespace RayEngine
{
	RSceneLayer::RSceneLayer(RScene& scene)
		: RLayer("Scene Layer")
	{
		m_scene = &scene;
	}

	void RSceneLayer::mouse_motion(double xPos, double yPos)
	{

	}

	void RSceneLayer::on_attach()
	{

		m_scene->build_scene();
	}

	void RSceneLayer::on_detach()
	{
	}

	void RSceneLayer::on_update()
	{
		RayEngine::Application& app = RayEngine::Application::get();

		m_scene->update_camera();
		if (app.should_spawn && app.edit_mode && !app.ctrl && app.click_timer > 10.f)
		{
			brush.brush_type = brush_type;
			spawn_obj(m_scene->get_camera(), brush, lastX, SCR_HEIGHT - lastY);
			app.click_timer = 0.f;
		}


		m_scene->Tick(app.deltaTime);
	}

	void RSceneLayer::on_event(RayEngine::Event& event)
	{
		RayEngine::EventDispatcher dispatcher(event);

		dispatcher.dipatch<RayEngine::MouseButtonPresedEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_button_pressed));
		dispatcher.dipatch<RayEngine::MouseButtonReleasedEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_button_relseased));
		dispatcher.dipatch<RayEngine::MouseMovedEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_moved));
		dispatcher.dipatch<RayEngine::MouseScrolledEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_scrolled));
		dispatcher.dipatch<RayEngine::WindowResizedEvent>(BIND_EVENT_FN(RSceneLayer::on_window_reseized));
		dispatcher.dipatch<RayEngine::KeyReleaseEvent>(BIND_EVENT_FN(RSceneLayer::on_key_released));
		dispatcher.dipatch<RayEngine::KeyPressedEvent>(BIND_EVENT_FN(RSceneLayer::on_key_pressed));
		dispatcher.dipatch<RayEngine::KeyTypedEvent>(BIND_EVENT_FN(RSceneLayer::on_key_typed));
	}

	bool RSceneLayer::on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e)
	{
		RayEngine::Application& app = RayEngine::Application::get();

		switch (e.get_button())
		{
		case RE_MOUSE_BUTTON_1:
			if (app.shift)
			{
				brush_type = TerrainBrushType::SPHERE_SUBTRACT;
			}
			else
			{
				brush_type = TerrainBrushType::SPHERE_ADD;
			}
			break;
		case RE_MOUSE_BUTTON_2:
			if (app.shift)
			{
				brush_type = TerrainBrushType::CUBE_SUBTRACT;
			}
			else
			{
				brush_type = TerrainBrushType::CUBE_ADD;
			}
			break;
		default:
			break;
		}
		app.should_spawn = true;
		return true;
	}

	bool RSceneLayer::on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e)
	{
		RayEngine::Application& app = RayEngine::Application::get();

		app.mouse_right = false;
		app.should_spawn = false;
		app.click_timer = 100;
		return true;
	}

	bool RSceneLayer::on_mouse_moved(RayEngine::MouseMovedEvent& e)
	{
		RayEngine::Application& app = RayEngine::Application::get();
		float x_pos = e.get_x();
		float y_pos = e.get_y();

		double deltaX = lastX - x_pos;
		double deltaY = lastY - y_pos;
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
			m_scene->get_smart_camera().change_yaw(-deltaX * 0.005f);
			m_scene->get_smart_camera().change_pitch(-deltaY * 0.005f);
		}

		lastX = x_pos;
		lastY = y_pos;
		return false;
	}

	bool RSceneLayer::on_mouse_scrolled(RayEngine::MouseScrolledEvent& e)
	{
		brush_radius += e.get_y_offset() * 0.01f;
		return false;
	}

	bool RSceneLayer::on_window_reseized(RayEngine::WindowResizedEvent& e)
	{
		return false;
	}

	bool RSceneLayer::on_key_released(RayEngine::KeyReleaseEvent& e)
	{
		return false;
	}

	bool RSceneLayer::on_key_pressed(RayEngine::KeyPressedEvent& e)
	{
		RayEngine::Application& app = RayEngine::Application::get();

		RMovableCamera tmp_cam = m_scene->get_smart_camera();
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
			app.click_timer = 0.f;
			toggle_shadow();
			break;
		default:
			break;
		}
		RCamera tmp_ca;
		tmp_cam.build_camera(tmp_ca);
		if (app.render_settings.gravity)
		{
			bool overlaps = sdf_collision(tmp_ca);
			if (!overlaps && app.render_settings.gravity)
				m_scene->get_smart_camera() = tmp_cam;
			tmp_cam.position.y -= 0.5;
			tmp_cam.build_camera(tmp_ca);
			overlaps = sdf_collision(tmp_ca);
			if (!overlaps && app.render_settings.gravity)
			{
				tmp_cam.position.y += 0.4f;
				m_scene->get_smart_camera() = tmp_cam;
			}
		}
		else
		{
			m_scene->get_smart_camera() = tmp_cam;
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
			app.click_timer = 0.f;
			break;
		}
		return false;
	}

	bool RSceneLayer::on_key_typed(RayEngine::KeyTypedEvent& e)
	{
		return false;
	}

	void RSceneLayer::init_triangles()
	{
	}


}