#pragma once

#include "RayEngine.h"
#include "../Layers/LayerStack.h"
#include <vector>


class RRayTracer;
class RScene;
class RKDTreeCPU;
class RKDThreeGPU;
struct RCamera;
struct float4;
struct GLFWwindow;
class TextRenderer;
class RMovableCamera;

namespace RayEngine
{
	class RAY_ENGINE_API Application
	{

	public:
		Application();
		virtual ~Application();

		void Run();
		void RenderFrame();

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
		float click_timer;
		double currentFrame;
		double lastFrame;
		double deltaTime;
		RenderingSettings render_settings;
		WindowData window_data;

		//------------------------Layer functions--------------------------
		void push_layer(RLayer* layer);
		void push_overlay(RLayer* overlay);
		//-----------------------------------------------------------------


		//------------------------Getters----------------------------------
		inline static Application& get() { return *s_instance; }
		inline GLFWwindow& get_window() { return *window; }
		inline RLayerStack& get_layer_stack() { return m_layer_stack; }
		class RSceneLayer& get_scene_layer();
		//-----------------------------------------------------------------
		void set_event_callback(std::function<void(Event&)> e) { window_data.function_callback = e; }


		//---------------------Should be defined in the !client!------------------------
		Application* create_application();
		//------------------------------------------------------------------------------
	private:
		//----------------------Window functions. Wille be moved to a separate class----
		void initPixelBuffer();
		int initGL();
		int create_window();
		bool check_shader_compile_status(unsigned int obj);
		bool check_program_link_status(unsigned int obj);
		//------------------------------------------------------------------------------

	private:
		static Application* s_instance;
		GLFWwindow* window;
		// vao and vbo handle
		unsigned int vao, vbo, ibo;
		// texture handle
		unsigned int texture;
		unsigned int shader_program, vertex_shader, fragment_shader;
		int texture_location;
		unsigned int pbo = 0;     // OpenGL pixel buffer object
		unsigned int tex = 0;     // OpenGL texture object
		struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
		// map PBO to get CUDA device pointer
		unsigned int* d_output;
		RLayerStack m_layer_stack;
	};

}