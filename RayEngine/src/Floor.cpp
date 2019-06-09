#include "Floor.h"
#include "RayEngine/RayEngine.h"
#include "helper_math.h"

namespace RayEngine
{

	AFloor::AFloor()
		:RMeshObject((char*)"Meshes/cat.obj")
	{
		Material material;
		material.uvs = true;
		material.type = TILE;
		material.color = make_float3(0.f, 0.2f, 0.5f);
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
}