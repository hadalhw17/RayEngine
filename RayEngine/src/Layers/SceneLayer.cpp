#include "repch.h"


#include "SceneLayer.h"

#include "Scene.h"
#include "Input.h"
#include "Events/ApplicationEvent.h"
#include "RayEngine/Application.h"


namespace RayEngine
{
	RSceneLayer::RSceneLayer(SDFScene& scene)
		: RLayer("Scene Layer")
	{
		m_scene = &scene;
	}

	void RSceneLayer::mouse_motion(double xPos, double yPos)
	{

	}

	void RSceneLayer::on_attach()
	{
		m_scene->on_attach();
	}

	void RSceneLayer::on_detach()
	{
	}

	void RSceneLayer::on_update()
	{
		m_scene->on_update();
	}

	void RSceneLayer::on_event(RayEngine::Event& event)
	{
		Application& app = Application::get();
		m_scene->Tick(app.delta_time());
		m_scene->on_event(event);
		RayEngine::EventDispatcher dispatcher(event);

		dispatcher.dipatch<RayEngine::MouseButtonPresedEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_button_pressed));
		dispatcher.dipatch<RayEngine::MouseButtonReleasedEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_button_relseased));
		dispatcher.dipatch<RayEngine::MouseMovedEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_moved));
		dispatcher.dipatch<RayEngine::MouseScrolledEvent>(BIND_EVENT_FN(RSceneLayer::on_mouse_scrolled));
		dispatcher.dipatch<RayEngine::WindowResizedEvent>(BIND_EVENT_FN(RSceneLayer::on_window_reseized));
		dispatcher.dipatch<RayEngine::KeyReleaseEvent>(BIND_EVENT_FN(RSceneLayer::on_key_released));
		dispatcher.dipatch<RayEngine::KeyPressedEvent>(BIND_EVENT_FN(RSceneLayer::on_key_pressed));
		dispatcher.dipatch<RayEngine::KeyTypedEvent>(BIND_EVENT_FN(RSceneLayer::on_key_typed));
	}

	bool RSceneLayer::on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e)
	{

		return true;
	}

	bool RSceneLayer::on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e)
	{
		return true;
	}

	bool RSceneLayer::on_mouse_moved(RayEngine::MouseMovedEvent& e)
	{
		return true;

	}

	bool RSceneLayer::on_mouse_scrolled(RayEngine::MouseScrolledEvent& e)
	{
		return true;

	}

	bool RSceneLayer::on_window_reseized(RayEngine::WindowResizedEvent& e)
	{
		return true;
	}

	bool RSceneLayer::on_key_released(RayEngine::KeyReleaseEvent& e)
	{
		return true;
	}

	bool RSceneLayer::on_key_pressed(RayEngine::KeyPressedEvent& e)
	{
		return true;
	}

	bool RSceneLayer::on_key_typed(RayEngine::KeyTypedEvent& e)
	{
		return true;
	}

	void RSceneLayer::init_triangles()
	{
	}


}