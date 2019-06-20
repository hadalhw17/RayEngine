#pragma once
#include "Events/Event.h"


class RAY_ENGINE_API RObjectComponent
{
public:
	RObjectComponent();
	~RObjectComponent();

	virtual void on_attach(class RSceneObject* owner);
	virtual void on_detach();
	virtual void on_update();
	virtual void on_event(RayEngine::Event& event);

	virtual void tick(float delta_time);

	inline static RSceneObject& get_owner() { return *s_owner; }

private:
	static class  RSceneObject* s_owner;
};

//#include <Meta.h>
//namespace meta {
//
//	template <>
//	inline auto registerMembers<RObjectComponent>()
//	{
//		return members(
//		);
//	}
//
//} // end of namespace meta