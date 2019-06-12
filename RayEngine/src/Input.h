#pragma once



namespace RayEngine
{
	class RAY_ENGINE_API Input
	{
	public:
		inline static bool is_key_pressed(int keycode) { return s_instance->is_key_pressed_impl(keycode); }
		inline static bool is_mouse_button_pressed(int keycode) { return s_instance->is_mouse_button_pressed_impl(keycode); }
		inline static float get_mouse_x() { return s_instance->get_mouse_x_impl(); }
		inline static float get_mouse_y() { return s_instance->get_mouse_y_impl(); }
		inline static std::pair<float, float> get_mouse_pos() { return s_instance->get_mouse_pos_impl(); }

	protected:
		virtual bool is_key_pressed_impl(int keycode) = 0;
		virtual bool is_mouse_button_pressed_impl(int button) = 0;
		virtual float get_mouse_x_impl() = 0;
		virtual float get_mouse_y_impl() = 0;
		virtual std::pair<float, float> get_mouse_pos_impl() = 0;
	private:
		static Input* s_instance;
	};



}
