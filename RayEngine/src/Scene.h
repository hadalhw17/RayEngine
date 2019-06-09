#pragma once
#include <vector>
#include <memory>

#include "RayEngine/RayEngine.h"
#include "Camera.h"
#include "SceneObject.h"
#include "MovableCamera.h"
#include "Character.h"


class RKDTree;
class RKDTreeCPU;
class RObject;
class RSphere;
class RTriangle;
class RTriMesh;
class RPlane;
class RStaticMesh;
class RayEngine::RSceneObject;
class RAY_ENGINE_API RScene
{
public:
	static float moveCounter;
	HOST_DEVICE_FUNCTION RScene();
	RScene(RCharacter& character);

	HOST_DEVICE_FUNCTION ~RScene();

	HOST_DEVICE_FUNCTION
	virtual void rebuild_scene() {}

	class RCharacter& main_character;

	void Tick(float delta_time);

	void update_camera();
	inline RCamera& get_camera() { return scene_camera; }
	virtual void build_scene() = 0;
	inline RMovableCamera& get_smart_camera() { return main_character.camera; }
	SceneSettings scene_settings;
protected:
	RCamera scene_camera;
	std::vector<float4> read_ppm(char* filename);
	std::vector<RayEngine::RSceneObject*> sceneObjects;
	virtual void clear_memory() {}
	void setup_camera();
};
