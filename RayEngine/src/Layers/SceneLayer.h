#pragma once
#include "Layers/Layer.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"
#include "Events/ApplicationEvent.h"
#include "Primitives/KDTree.h"
#include "Primitives/KDThreeGPU.h"
#include "Primitives/Camera.h"

#include "World/SDFScene.h"

namespace RayEngine
{
	class RAY_ENGINE_API RSceneLayer :
		public RayEngine::RLayer
	{
	public:
		RSceneLayer(SDFScene& scene);
		virtual ~RSceneLayer() {}

		virtual void on_attach() override;
		virtual void on_detach() override;
		virtual void on_update() override;
		virtual void on_event(RayEngine::Event& event) override;

		inline RScene& get_scene() const { return *m_scene; }
		float brush_radius = 1;
		TerrainBrushType brush_type;
		TerrainBrush brush;
		SDFScene* m_scene;

		float m_time;
		
	private:
		virtual bool on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e) override;
		virtual bool on_mouse_button_relseased(RayEngine::MouseButtonReleasedEvent& e) override;
		virtual bool on_mouse_moved(RayEngine::MouseMovedEvent& e) override;
		virtual bool on_mouse_scrolled(RayEngine::MouseScrolledEvent& e) override;
		virtual bool on_window_reseized(RayEngine::WindowResizedEvent& e) override;
		virtual bool on_key_released(RayEngine::KeyReleaseEvent& e) override;
		virtual bool on_key_pressed(RayEngine::KeyPressedEvent& e) override;
		virtual bool on_key_typed(RayEngine::KeyTypedEvent& e) override;

		void mouse_motion(double xPos, double yPos);

	private:
		void init_triangles();

	private:

	};
}
