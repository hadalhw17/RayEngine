#pragma once
#include "SceneObject.h"
#include "MovableCamera.h"
#include "Events/Event.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"
#include "Events/ApplicationEvent.h"
#include "RayEngine/Application.h"
#include "cuda-src/gpu_structs.h"


class RAY_ENGINE_API RCharacter : public RayEngine::RSceneObject
{
public:
	RCharacter();
	~RCharacter();

	virtual void on_attach();
	virtual void on_detach();
	virtual void on_update();
	virtual void on_event(RayEngine::Event& event);

	virtual bool on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e);
	virtual bool on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e);
	virtual bool on_mouse_moved(RayEngine::MouseMovedEvent& e);
	virtual bool on_mouse_scrolled(RayEngine::MouseScrolledEvent& e);
	virtual bool on_window_reseized(RayEngine::WindowResizedEvent& e);
	virtual bool on_key_released(RayEngine::KeyReleaseEvent& e);
	virtual bool on_key_pressed(RayEngine::KeyPressedEvent& e);
	virtual bool on_key_typed(RayEngine::KeyTypedEvent& e);

	virtual void tick(float delta_time) override;

	RMovableCamera camera;

protected:
	double last_x = 0, last_y = 0;
};


#include <Meta.h>

namespace meta {

	template <>
	inline auto registerMembers<RCharacter>()
	{
		return members(
			member("components", &RCharacter::components),
			member("object_properties", &RCharacter::object_properties),
			member("collision_box", &RCharacter::collision_box),
			member("camera", &RCharacter::camera)
		);
	}

} // end of namespace meta