#include "ResourceManager.h"
#include <iostream>
#include "TextCharacter.h"

ResourceManager::ResourceManager()
{
}

void ResourceManager::on_attach(RSceneObject* owner)
{
	RObjectComponent::on_attach(owner);
}

void ResourceManager::tick(float delta_time)
{
	RObjectComponent::tick(delta_time);
}

template<class InputIterator, class T>
bool find(InputIterator first, InputIterator last, const T& val, int &index)
{
	while (first != last) {
		if (*first == val) return true;
		++first;
		++index;
	}
	return false;
}
int ResourceManager::get_resource(const Resource& ResToFind)
{
	int Amount = -100;
	int Index;

	if (find(Resources.begin(), Resources.end(), ResToFind, Index))
	{
		Amount = ResourceAmount[Index];
	}

	return Amount;
}

void ResourceManager::get_resource_type(std::vector<Resource>& OwningResources)
{
	OwningResources = Resources;
}

void ResourceManager::update_resource_display()
{

}

void ResourceManager::add_resource(const Resource& Resource, int32_t Amount)
{
	if (&Resource && Amount > 0)
	{
		int Index;
		if (!find(Resources.begin(), Resources.end(), Resource, Index))
		{
			Resources.push_back(Resource);
			ResourceAmount.push_back(Amount);
			RE_LOG("Hey, well done, you have found some " << Resource.m_resource_data.m_name);

			TextCharacter& owning_character = dynamic_cast<TextCharacter&>(get_owner());
			if(&owning_character)
			{
				update_resource_display();
			}
		}
		else
		{
			ResourceAmount[Index] += Amount;

			RE_LOG("You have now " << ResourceAmount[Index] << " " << Resource.m_resource_data.m_name);
			TextCharacter& owning_character = dynamic_cast<TextCharacter&>(get_owner());
			if (&owning_character)
			{
				update_resource_display();
			}
		}
	}
	return;
}

void ResourceManager::remove_resource(const Resource& Resource, int Amount)
{
	if (&Resource && Amount > 0)
	{
		int Index;
		if (find(Resources.begin(), Resources.end(), Resource, Index))
		{
			ResourceAmount[Index] -= Amount;
			if (ResourceAmount[Index] < 0)
			{
				ResourceAmount[Index] = 0;
				TextCharacter& owning_character = dynamic_cast<TextCharacter&>(get_owner());

				if (&owning_character)
				{
					update_resource_display();
				}
			}
			TextCharacter& owning_character = dynamic_cast<TextCharacter&>(get_owner());
			if (&owning_character)
			{
				update_resource_display();
			}
		}
	}
	return;
}
