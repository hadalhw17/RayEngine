#include "WindowsInput.h"
#include "RayEngine/Application.h"
#include <iostream>


#include <GLFW/glfw3.h>

namespace RayEngine
{

	Input* Input::s_instance = new WindowsInput();

	bool WindowsInput::is_key_pressed_impl(int keycode)
	{
		auto& window = Application::get().get_window();
		auto state = glfwGetKey(&window, keycode);
		return state == GLFW_PRESS || state == GLFW_REPEAT;
	}

	bool WindowsInput::is_mouse_button_pressed_impl(int button)
	{
		auto& window = Application::get().get_window();
		auto state = glfwGetMouseButton(&window, button);
		return state == GLFW_PRESS;
	}

	float WindowsInput::get_mouse_x_impl()
	{
		auto pos= get_mouse_pos_impl();

		return pos.first;
	}
	
	float WindowsInput::get_mouse_y_impl()
	{
		auto pos = get_mouse_pos_impl();

		return pos.second;
	}

	std::pair<float, float> WindowsInput::get_mouse_pos_impl()
	{
		auto& window = Application::get().get_window();
		double x_pos, y_pos;
		glfwGetCursorPos(&window, &x_pos, &y_pos);
		return { x_pos, y_pos };
	}
}