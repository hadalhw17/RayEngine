#include "repch.h"


#include "MeshObject.h"
#include "ObjFileReader.h"

RMeshObject::RMeshObject(const char* file_name)
{
	RayEngine::WavefrontOBJ* reader = new RayEngine::WavefrontOBJ;
	this->root_component = reader->loadObjFromFile(file_name);
	this->root_component->generate_face_normals();


	object_properties = GPUSceneObject();
	components.push_back(this->root_component);
}
