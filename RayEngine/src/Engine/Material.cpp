#include "repch.h"


#include "Material.h"

#include "TextureObject.h"

RMaterial::RMaterial(const RTextureObject& texture1, const RTextureObject& texture2, const RTextureObject& texture3, std::string name)
{
	material.texture_aray.push_back(texture1);
	material.texture_aray.push_back(texture2);
	material.texture_aray.push_back(texture3);
	material_name = name;
}

RMaterial::RMaterial(const RTextureObject& texture1, std::string name)
{
	material.texture_aray.push_back(texture1);
	material.texture_aray.push_back(texture1);
	material.texture_aray.push_back(texture1);
	material_name = name;
}


