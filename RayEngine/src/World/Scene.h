#pragma once




#include "Primitives/Camera.h"
#include "Objects/SceneObject.h"
#include "Objects/MovableCamera.h"
#include "Objects/Character.h"
#include "Events/Event.h"
#include "Events/KeyEvent.h"
#include "Events/MouseEvent.h"
#include "Events/ApplicationEvent.h"
#include "RayEngine/Application.h"


class RKDTree;
class RKDTreeCPU;
class RObject;
class RSphere;
class RTriangle;
class RTriMesh;
class RPlane;
class RStaticMesh;

class RAY_ENGINE_API RScene
{
public:
	RScene();
	RScene(RCharacter& character);
	~RScene();

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

	virtual void rebuild_scene() {}
	virtual void build_scene() = 0;

	void update_camera();
	void Tick(float delta_time);

	inline RCharacter& get_character() { return main_character; }
	inline RCamera& get_camera() { return scene_camera; }
	inline RMovableCamera& get_smart_camera() { return main_character.camera; }
	
public:
	static float moveCounter;
	RCharacter& main_character;
	SceneSettings scene_settings;
	RCamera scene_camera;
	std::vector<RayEngine::RSceneObject*> scene_objects;
protected:
	std::vector<float4> read_ppm(char* filename);
	virtual void clear_memory() {}
	void setup_camera();
};

#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<RScene>()
	{
		return members(
			member("should not use this one!", &RScene::scene_settings)
		);
	}
} // end of namespace meta