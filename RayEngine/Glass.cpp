#include "Glass.h"
#include "helper_math.h"



AGlass::AGlass()
	:RSceneObject((char *)"Meshes/glass.obj")
{
	Material material;
	material.type = REFLECT;
	material.color = make_float3(0, 0, 0);
	float3 location = make_float3(0);
	object_properties.material = material;
	object_properties.location = location;
}



AGlass::~AGlass()
{
}

void AGlass::tick(float delta_time)
{
	RSceneObject::tick(delta_time);

	//object_properties.location.y -= 0.01f;
}
