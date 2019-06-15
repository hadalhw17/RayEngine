#define NOMINMAX
#include "repch.h"
#include "Application.h"


#include "../World/Scene.h"
#include "../Objects/SceneObject.h"


#include "../World/Grid.h"
#include "../Engine/Material.h"
#include "../World/PerlinNoise.h"
#include "Events/ApplicationEvent.h"
#include "Layers/SceneLayer.h"
#include "Layers/UILayer.h"

extern void free_memory();
extern void toggle_shadow();
extern bool sdf_collision(RCamera cam);
extern void spawn_obj(RCamera pos, TerrainBrush brush, int x, int y);


namespace RayEngine
{

	Application* Application::s_instance = nullptr;

	Application::Application()
	{
		RE_LOG("HIIIII");
		s_instance = this;
		application_window = new Window();
		application_window->set_event_callback(BIND_EVENT_FN(Application::on_event));

	}

	Application::~Application()
	{
		//delete Scene, SceneCam;

		//if(distance_field) delete distance_field;
		free_memory();

		//if (pbo)
		//{
		//	cudaGraphicsUnregisterResource(cuda_pbo_resource);
		//	glDeleteBuffers(1, &pbo);
		//	glDeleteTextures(1, &tex);
		//}
	}

	RSceneLayer& Application::get_scene_layer()
	{
		RSceneLayer* res = nullptr;
		for (auto layers : m_layer_stack)
		{
			RSceneLayer* sceneLayer = dynamic_cast<RSceneLayer*>(layers);
			if (sceneLayer)
			{
				return *sceneLayer;
			}
		}
		return *res;
	}



	// Register different events.
	void Application::on_event(Event& e)
	{
		EventDispatcher dispatcher(e);
		dispatcher.dipatch<MouseButtonPresedEvent>(BIND_EVENT_FN(Application::mouse_button_presssed));
		dispatcher.dipatch<MouseMovedEvent>(BIND_EVENT_FN(Application::on_mouse_move));
		dispatcher.dipatch<MouseScrolledEvent>(BIND_EVENT_FN(Application::on_mouse_scroll));
		dispatcher.dipatch<KeyPressedEvent>(BIND_EVENT_FN(Application::on_key_pressed));
		dispatcher.dipatch<KeyReleaseEvent>(BIND_EVENT_FN(Application::on_key_released));
		dispatcher.dipatch<WindowClosedEvent>(BIND_EVENT_FN(Application::on_window_closed));

		for (auto it = m_layer_stack.end(); it != m_layer_stack.begin();)
		{
			(*--it)->on_event(e);
			if (e.get_handled())
				break;
		}
	}


	void Application::app_toggle_shadow()
	{
		toggle_shadow();
	}

	bool Application::app_sdf_collision(RCamera cam)
	{
		return sdf_collision(cam);
	}

	void Application::app_spawn_obj(RCamera pos, TerrainBrush brush, int x, int y)
	{
		spawn_obj(pos, brush, x, y);
	}

	void Application::push_layer(RLayer* layer)
	{
		m_layer_stack.push_layer(layer);
		layer->on_attach();
	}

	void Application::push_overlay(RLayer* overlay)
	{
		m_layer_stack.push_overlay(overlay);
		overlay->on_attach();
	}

	void Application::Run()
	{
		while (!application_window->should_close())
		{
			application_window->updateGL();

			for (auto layer : m_layer_stack)
			{
				layer->on_update();
			}


			
			application_window->RenderFrame();


		}

		// Cleanup

		// glfw: terminate, clearing all previously allocated GLFW resources.
		// ------------------------------------------------------------------
		application_window->destroy_window();

	}

	bool Application::mouse_button_presssed(MouseButtonPresedEvent& e)
	{
		return false;
	}

	bool Application::on_mouse_released_callback(MouseButtonReleasedEvent& e)
	{
		return false;
	}

	bool Application::on_mouse_move(MouseMovedEvent& e)
	{
		return false;
	}

	bool Application::on_key_pressed(KeyPressedEvent& e)
	{
		if (e.get_key_code() == RE_KEY_ESCAPE) application_window->close_window();
		return true;
	}

	bool Application::on_key_released(KeyReleaseEvent& e)
	{
		if (!show_mouse)
		{
			switch (e.get_key_code())
			{
			case RE_KEY_LEFT_SHIFT:
				shift = false;
				break;
			default:
				break;
			}
		}
		switch (e.get_key_code())
		{
		case RE_KEY_TAB:
			show_mouse = false;
			break;
		case RE_KEY_LEFT_CONTROL:
			ctrl = false;
			break;
		default:
			break;
		}

		return true;
	}

	bool Application::on_mouse_scroll(MouseScrolledEvent& e)
	{
		return false;
	}

	bool Application::on_window_closed(WindowClosedEvent& e)
	{
		return false;
	}



}