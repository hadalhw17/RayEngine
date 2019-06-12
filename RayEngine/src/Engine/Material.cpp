#include "repch.h"


#include "Material.h"

#include "TextureObject.h"

RMaterial::RMaterial(const RTextureObject& texture1, const RTextureObject& texture2, const RTextureObject& texture3)
{
	material.texture_aray.push_back(texture1);
	material.texture_aray.push_back(texture2);
	material.texture_aray.push_back(texture3);
}

RMaterial::RMaterial(const RTextureObject& texture1)
{
	material.texture_aray.push_back(texture1);
	material.texture_aray.push_back(texture1);
	material.texture_aray.push_back(texture1);
}


