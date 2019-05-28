#include "SceneObject.h"

#include "ObjectComponent.h"
#include "ObjFileReader.h"
#include "MeshAdjacencyTable.h"



RSceneObject::RSceneObject(const char *file_name)
{
	WavefrontOBJ *reader = new WavefrontOBJ;
	this->root_component = reader->loadObjFromFile(file_name);
	this->root_component->generate_face_normals();


	object_properties = GPUSceneObject();
	components.push_back(this->root_component);
}


RSceneObject::~RSceneObject()
{
}

void RSceneObject::tick(float delta_time)
{
}
