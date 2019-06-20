#pragma once
#include <C:\dev\RayEngine\RayEngine\src\Layers\SceneLayer.h>
#include "TestScene.h"
class TestSceneLayer :
	public RayEngine::RSceneLayer
{
public:
	TestSceneLayer(TestScene& scene)
		:RayEngine::RSceneLayer(scene) {}

	virtual bool on_mouse_button_pressed(RayEngine::MouseButtonPresedEvent& e) override;
};

//#include <Meta.h>
//namespace meta {
//
//	template <>
//	inline auto registerMembers<TestSceneLayer>()
//	{
//		return members(
//			member("brush_radius", &TestSceneLayer::brush_radius),
//			member("brush", &TestSceneLayer::brush),
//			member("m_scene", &TestSceneLayer::m_scene)
//		);
//	}
//} // end of na