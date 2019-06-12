#pragma once
#include "Event.h"
#include <sstream>

namespace RayEngine
{
	class RAY_ENGINE_API KeyEvent : public Event
	{
	public:
		inline int get_key_code() const { return m_keycode; }

		EVENT_CLASS_CATEGORY(EventCategoryKeybord | EventCategoryInput)
	protected:
		KeyEvent(int keycode)
			:m_keycode(keycode) {}

		int m_keycode;
	};

	class RAY_ENGINE_API KeyPressedEvent : public KeyEvent
	{
	public:
		KeyPressedEvent(int keycode, int repeat_count)
			:KeyEvent(keycode), m_repeat_count(repeat_count) {}

		inline int get_repeat_cunt() const { return m_repeat_count; }
		std::string to_string() const override
		{
			std::stringstream ss;
			ss << "KeyPressedEvent: " << m_keycode << " (" << m_repeat_count << ") repeats";
			return ss.str();
		}
		EVENT_CLASS_TYPE(KeyPressed);
	private:
		int m_repeat_count;
	};


	class RAY_ENGINE_API KeyReleaseEvent : public KeyEvent
	{
	public:
		KeyReleaseEvent(int keycode)
			:KeyEvent(keycode) {}

		std::string to_string() const override
		{
			std::stringstream ss;
			ss << "KeyReleaseEvent: " << m_keycode;
			return ss.str();
		}
		EVENT_CLASS_TYPE(KeyReleased)
	};

	class RAY_ENGINE_API KeyTypedEvent : public KeyEvent
	{
	public:
		KeyTypedEvent(int keycode)
			:KeyEvent(keycode) {}

		std::string to_string() const override
		{
			std::stringstream ss;
			ss << "KeyTypedEvent: " << m_keycode;
			return ss.str();
		}
		EVENT_CLASS_TYPE(KeyTyped);
	private:

	};
}