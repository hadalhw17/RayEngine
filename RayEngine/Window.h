#pragma once

#include <string>
#include "RayEngine/RayEngine.h"
#include <functional>
#include "Events/Event.h"

class RCamera;
namespace RayEngine
{
	struct WindowData
	{
		std::string m_title;
		size_t m_width, m_heigth;
		bool vsync;

		WindowData(const std::string& title = "RayEngine",
			size_t width = 1920,
			size_t heigth = 1080) 
		: m_title(title), m_width(width), m_heigth(heigth) {}
	};

	class RAY_ENGINE_API Window
	{
	public:
		using event_callback_function = std::function<void(Event&)>;
		Window();
		~Window();
		inline size_t get_width() const { return m_data.m_width; }
		inline size_t get_heigth() const { return m_data.m_heigth; }

		inline bool set_vsync(bool vsync) { m_data.vsync = vsync; }
		inline bool get_vsync() const { return m_data.vsync; }

		void on_update();
		void set_event_callback(const event_callback_function& callback);

		static Window* create(const const WindowData& data = WindowData());
		void RenderFrame();
	private:
		void initPixelBuffer();
		int  initGL();

	private:
		class GLFWwindow* window;
		WindowData m_data;
	};
}