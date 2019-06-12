#pragma once

#include "../RayEngine/RayEngine.h"




namespace RayEngine
{
	enum class EventType
	{
		None = 0,
		WindowClosed, WindowResize, WindowFocus, WindowMoved,		// Window events.
		AppTick, AppUpdate, AppRender,								// Application events.
		KeyPressed, KeyReleased, KeyTyped,									// Keybord events.
		MouseButtonPressed, MouseButtonReleased, MouseMoved, MouseScrolled		// Mouse events.
	};

	enum EventCategory
	{
		None = 0,
		EventCategoryApplication	= BIT(0),
		EventCategoryInput			= BIT(1),
		EventCategoryKeybord		= BIT(2),
		EventCategoryMouse			= BIT(3),
		EventCategoryMouseButton	= BIT(4)
	};

#define EVENT_CLASS_TYPE(type) static EventType get_static_type() {return EventType::##type; } \
						virtual EventType get_event_type() const override {return get_static_type(); } \
						virtual const char *get_name() const override { return #type; }

#define EVENT_CLASS_CATEGORY(category) virtual int get_category_flags() const override { return category; }

	class RAY_ENGINE_API Event
	{
		friend class EventDispatcher;
	public:
		virtual EventType get_event_type() const = 0;
		virtual const char* get_name() const = 0;
		virtual int get_category_flags() const = 0;
		virtual std::string to_string() const { return get_name(); }


		inline bool is_in_category(EventCategory category)
		{
			return get_category_flags() & category;
		}

		inline const bool get_handled() const { return m_handled; }

	protected:
		bool m_handled = false;
	};

	class EventDispatcher
	{
		template<typename T>
		using event_fun = std::function<bool(T&)>;

	public:
		EventDispatcher(Event& event)
			: m_event(event) {}

		template<typename T>
		bool dipatch(event_fun<T> func)
		{
			if (m_event.get_event_type() == T::get_static_type())
			{
				m_event.m_handled = func(*(T*)&m_event);
				return true;
			}
			return false;
		}

		Event& m_event;
	};

	inline std::ostream& operator<<(std::ostream& os, const Event& e)
	{
		return os << e.to_string();
	}
}
