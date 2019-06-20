#pragma once
//#include <Objects/SceneObject.h>
#include <string>
//
struct ResourceData
{
public:
	ResourceData()
	{
		m_name = "No name";
	}

	std::string m_name;
};
//
class Resource
{
public:
	// Sets default values for this actor's properties
	Resource();
	Resource(std::string name);

	bool operator == (const Resource& rhs) { return this->m_resource_data.m_name == rhs.m_resource_data.m_name; }
	// Called when the game starts or when spawned
	virtual void on_attach();

public:

	ResourceData m_resource_data;
};
//
