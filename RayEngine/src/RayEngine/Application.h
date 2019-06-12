#pragma once

#include "RayEngine.h"
#include "../Layers/LayerStack.h"
#include "Engine/Window.h"


namespace RayEngine
{
	class RAY_ENGINE_API Application
	{

	public:
		Application();
		virtual ~Application();

		void Run();

		inline const float delta_time() const { return application_window->delta_time(); }

		void on_event(class Event& e);
		virtual bool mouse_button_presssed(class MouseButtonPresedEvent& e);
		virtual bool on_mouse_released_callback(class MouseButtonReleasedEvent& e);
		virtual bool on_mouse_move(class MouseMovedEvent& e);
		virtual bool on_key_pressed(class KeyPressedEvent& e);
		virtual bool on_key_released(class KeyReleaseEvent& e);
		virtual bool on_mouse_scroll(class MouseScrolledEvent& e);
		virtual bool on_window_closed(class WindowClosedEvent& e);

		//-----------------------------------
		bool should_spawn = false;
		float character_speed = 150;
		bool shift = false;
		bool ctrl = false;
		bool mouse_right = false;
		bool show_mouse = true;
		bool edit_mode = false;
		RenderingSettings render_settings;


		void app_toggle_shadow();
		bool app_sdf_collision(class RCamera cam);
		void app_spawn_obj(class RCamera pos, TerrainBrush brush, int x, int y);

		//------------------------Layer functions--------------------------
		void push_layer(RLayer* layer);
		void push_overlay(RLayer* overlay);
		//-----------------------------------------------------------------


		//------------------------Getters----------------------------------
		inline static Application& get() { return *s_instance; }
		inline Window& get_window() { return *application_window; }
		inline RLayerStack& get_layer_stack() { return m_layer_stack; }
		class RSceneLayer& get_scene_layer();
		//-----------------------------------------------------------------

		RLayerStack m_layer_stack;
		//---------------------Should be defined in the !client!------------------------
		Application* create_application();
		//------------------------------------------------------------------------------
	private:
		static Application* s_instance;
		Window* application_window;
	};

}


#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<RayEngine::Application>()
	{
		return members(
			member("m_layer_stack", &RayEngine::Application::m_layer_stack)
		);
	}
} // end of namespace meta