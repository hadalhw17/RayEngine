#include "Cow.h"
#include "cutil_math.h"



ACow::ACow()
	:RSceneObject((char *)"Meshes/predator.obj")
{
	Material material;
	material.type = PHONG;
	material.color = make_float4(0, 0, 0.2, 0);
	float3 location = make_float3(0);
	object_properties.material = material;
	object_properties.location = location;
	object_properties.is_character = false;
}


ACow::~ACow()
{
}

void ACow::tick(float delta_time)
{
	RSceneObject::tick(delta_time);

	//object_properties.location.x += 0.01f;
	//object_properties.rotation.y += 10;
}
