#pragma once

#include <vector>
#include "Resource.h"
#include <Objects/ObjectComponent.h>

class ResourceManager : public RObjectComponent
{
public:
	// Sets default values for this component's properties
	ResourceManager();

	// Called when the game starts
	virtual void on_attach(RSceneObject* owner) override;

	std::vector<Resource> Resources;

	std::vector<int> ResourceAmount;

public:
	// Called every frame
	virtual void tick(float delta_time) override;

	int get_resource(const Resource &ResToFind);


	void get_resource_type(std::vector<Resource>& OwningResources);

	void update_resource_display();

	void add_resource(const Resource& Resource, int32_t Amount);

	void remove_resource(const Resource& Resource, int Amount);

};

