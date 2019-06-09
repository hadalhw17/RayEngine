#pragma once

#include "Event.h"


namespace RayEngine
{
	class RAY_ENGINE_API WindowClosedEvent : public Event
	{
	public:
		WindowClosedEvent() {}
		EVENT_CLASS_TYPE(WindowClosed)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};

	class RAY_ENGINE_API AppTickEvent : public Event
	{
	public:
		AppTickEvent() {}
		EVENT_CLASS_TYPE(AppTick)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};

	class RAY_ENGINE_API AppUpdateEvent : public Event
	{
	public:
		AppUpdateEvent() {}
		EVENT_CLASS_TYPE(AppUpdate)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};

	class RAY_ENGINE_API AppRenderEvent : public Event
	{
	public:
		AppRenderEvent() {}
		EVENT_CLASS_TYPE(AppRender)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)
	};

	class RAY_ENGINE_API WindowResizedEvent : public Event
	{
	public:
		WindowResizedEvent(size_t width, size_t heigth) :
			m_width(width), m_heigth(heigth) {}

		inline size_t get_width() const { return m_width; }
		inline size_t get_height() const { return m_heigth; }
		std::string to_string() const override
		{
			std::stringstream ss;
			ss << "WidndowReseizedEvent: " << m_width << " x " << m_heigth;
			return ss.str();
		}
		EVENT_CLASS_TYPE(WindowResize)
		EVENT_CLASS_CATEGORY(EventCategoryApplication)

	private:
		size_t m_width, m_heigth;
	};
}