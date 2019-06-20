#pragma once

#include "TextureObject.h"

struct RAY_ENGINE_API VoxelMaterial
{
	uint2 texture_resolution;
	std::vector<RTextureObject> texture_aray;
};

#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<VoxelMaterial>()
	{
		return members(
			member("texture_resolution", &VoxelMaterial::texture_resolution),
			member("texture_aray", &VoxelMaterial::texture_aray)
		);
	}

} // end of namespace meta


class RAY_ENGINE_API RMaterial
{
public:
	RMaterial(const RTextureObject& texture1, const RTextureObject& texture2, const RTextureObject& texture3, std::string name = "Default name");
	RMaterial(const RTextureObject& texture1, std::string name = "Default name");

	inline void change_name(std::string new_name) { material_name = new_name; }
	inline const std::string get_name() const { return material_name; }

	std::string material_name;
	VoxelMaterial material;
};

#include <Meta.h>
namespace meta {

	template <>
	inline auto registerMembers<RMaterial>()
	{
		return members(
			member("material", &RMaterial::material)
		);
	}

} // end of namespace meta

