#pragma once
#include <Objects/Character.h>

#include <Events/KeyEvent.h>
#include <Events/MouseEvent.h>
#include <Events/ApplicationEvent.h>
#include <RayEngine/Application.h>
class TextCharacter :
	public RCharacter
{
public:
	TextCharacter() {}

	virtual void on_attach() override;
	virtual void on_detach() override;
	virtual void on_update() override;
	virtual void on_event(RayEngine::Event& event) override;

	virtual bool on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e) override;
	virtual bool on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e) override;
	virtual bool on_mouse_moved(RayEngine::MouseMovedEvent& e) override;
	virtual bool on_mouse_scrolled(RayEngine::MouseScrolledEvent& e) override;
	virtual bool on_window_reseized(RayEngine::WindowResizedEvent& e) override;
	virtual bool on_key_released(RayEngine::KeyReleaseEvent& e) override;
	virtual bool on_key_pressed(RayEngine::KeyPressedEvent& e) override;
	virtual bool on_key_typed(RayEngine::KeyTypedEvent& e) override;
};
