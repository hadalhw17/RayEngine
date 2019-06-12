#pragma once


#include "Layers/Layer.h"
#include "UserInterface.h"

#include "RayEngine/Application.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"
#include "Events/ApplicationEvent.h"

namespace RayEngine
{
	class RAY_ENGINE_API RUILayer : public RLayer
	{
	public:
		RUILayer();
		~RUILayer();

		virtual void on_attach() override;
		virtual void on_detach() override;
		virtual void on_update() override;
		virtual void on_event(Event& event) override;
		inline bool is_visible() const { return m_is_visible; }
	private:
		virtual bool on_mouse_button_pressed(MouseButtonPresedEvent &e) override;
		virtual bool on_mouse_button_relseased(MouseButtonReleasedEvent &e) override;
		virtual bool on_mouse_moved(MouseMovedEvent &e) override;
		virtual bool on_mouse_scrolled(MouseScrolledEvent &e) override;
		virtual bool on_window_reseized(WindowResizedEvent &e) override;
		virtual bool on_key_released(KeyReleaseEvent &e) override;
		virtual bool on_key_pressed(KeyPressedEvent &e) override;
		virtual bool on_key_typed(KeyTypedEvent &e) override;

		void render_menu(TerrainBrush& brush, float& character_speed, struct RenderingSettings& render_settings, struct SceneSettings& scene_settings, float3 curr_pos);

	private:
		float m_time;
		bool m_is_visible = false;
	};
}
