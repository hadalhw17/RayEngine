#include "Floor.h"
#include "RayEngine.h"
#include "cutil_math.h"


AFloor::AFloor()
	:RSceneObject((char *)"Meshes/floor.obj")
{
	Material material;
	material.type = TILE;
	material.color = make_float4(0.f);
	float3 location = make_float3(0);
	object_properties.material = material;
	object_properties.location = location;
}


AFloor::~AFloor()
{
}

void AFloor::tick(float delta_time)
{
	RSceneObject::tick(delta_time);
}
