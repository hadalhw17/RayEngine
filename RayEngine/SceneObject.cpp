#include "SceneObject.h"

#include "ObjectComponent.h"
#include "ObjFileReader.h"



RSceneObject::RSceneObject(const char *file_name)
{
	WavefrontOBJ *reader = new WavefrontOBJ;
	this->root_component = reader->loadObjFromFile(file_name);
	object_properties = GPUSceneObject();
	components.push_back(this->root_component);
}


RSceneObject::~RSceneObject()
{
}

void RSceneObject::tick(float delta_time)
{
}
