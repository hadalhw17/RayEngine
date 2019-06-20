#include "Resource.h"
#include <Objects\SceneObject.cpp>


Resource::Resource()
{
}

Resource::Resource(std::string name)
{
	m_resource_data.m_name = name;
}

void Resource::on_attach()
{
	//RSceneObject::on_attach();
}