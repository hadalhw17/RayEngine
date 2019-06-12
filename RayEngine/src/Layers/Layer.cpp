#include "repch.h"


#include "Layer.h"

#include "Events/ApplicationEvent.h"

RayEngine::RLayer::RLayer(const std::string& name)
	:
	m_debug_name(name)
{
}

RayEngine::RLayer::~RLayer()
{
}

void RayEngine::RLayer::on_event(Event& event)
{
	EventDispatcher dispatcher(event);

	dispatcher.dipatch<MouseButtonPresedEvent>(BIND_EVENT_FN(RLayer::on_mouse_button_pressed));
	dispatcher.dipatch<MouseButtonReleasedEvent>(BIND_EVENT_FN(RLayer::on_mouse_button_relseased));
	dispatcher.dipatch<MouseMovedEvent>(BIND_EVENT_FN(RLayer::on_mouse_moved));
	dispatcher.dipatch<MouseScrolledEvent>(BIND_EVENT_FN(RLayer::on_mouse_scrolled));
	dispatcher.dipatch<WindowResizedEvent>(BIND_EVENT_FN(RLayer::on_window_reseized));
	dispatcher.dipatch<KeyReleaseEvent>(BIND_EVENT_FN(RLayer::on_key_released));
	dispatcher.dipatch<KeyPressedEvent>(BIND_EVENT_FN(RLayer::on_key_pressed));
	dispatcher.dipatch<KeyTypedEvent>(BIND_EVENT_FN(RLayer::on_key_typed));
}
