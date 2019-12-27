#pragma once


#include "Events/Event.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"
#include "Events/ApplicationEvent.h"

namespace RayEngine
{
	class RAY_ENGINE_API RLayer
	{
	public:
		RLayer(const std::string& name = "Name a layer");
		virtual ~RLayer();

		virtual void on_attach() {}
		virtual void on_detach() {}
		virtual void on_update() {}
		virtual void on_event(Event& event);

		inline const std::string& get_name() const { return m_debug_name; }
		inline void set_name(std::string& name) { m_debug_name = name; }

		std::string m_debug_name;
	private:
		virtual bool on_mouse_button_pressed(MouseButtonPresedEvent& e) = 0;
		virtual bool on_mouse_button_relseased(MouseButtonReleasedEvent& e) = 0;
		virtual bool on_mouse_moved(MouseMovedEvent& e) = 0;
		virtual bool on_mouse_scrolled(MouseScrolledEvent& e) = 0;
		virtual bool on_window_reseized(WindowResizedEvent& e) = 0;
		virtual bool on_key_released(KeyReleaseEvent& e) = 0;
		virtual bool on_key_pressed(KeyPressedEvent& e) = 0;
		virtual bool on_key_typed(KeyTypedEvent& e) = 0;

	protected:

	};

}

//#include <Meta.h>
//namespace meta {
//
//	template <>
//	inline auto registerMembers<RayEngine::RLayer>()
//	{
//		return members(
//			member("m_debug_name", &RayEngine::RLayer::m_debug_name)
//		);
//	}
//} // end of na