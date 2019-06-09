#pragma once
#include "Input.h"

namespace RayEngine
{
	class RAY_ENGINE_API WindowsInput :
		public Input
	{
	protected:
		virtual bool is_key_pressed_impl(int keycode) override;
		virtual bool is_mouse_button_pressed_impl(int button);
		virtual float get_mouse_x_impl();
		virtual float get_mouse_y_impl();
		virtual std::pair<float, float> get_mouse_pos_impl();
	};

}