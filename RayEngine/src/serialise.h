#pragma once

#include <Meta.h>
#include "RayEngine/RayEngine.h"

namespace meta {

	template <>
	inline auto registerMembers<float2>()
	{
		return members(
			member("x", &float2::x),
			member("y", &float2::y)
		);
	}

	template <>
	inline auto registerMembers<float3>()
	{
		return members(
			member("x", &float3::x),
			member("y", &float3::y),
			member("z", &float3::z)
		);
	}
	template <>
	inline auto registerMembers<float4>()
	{
		return members(
			member("x", &float4::x),
			member("y", &float4::y),
			member("z", &float4::z),
			member("w", &float4::w)
		);
	}

	template <>
	inline auto registerMembers<uint2>()
	{
		return members(
			member("x", &uint2::x),
			member("y", &uint2::y)
		);
	}

	template <>
	inline auto registerMembers<uint3>()
	{
		return members(
			member("x", &uint3::x),
			member("y", &uint3::y),
			member("z", &uint3::z)
		);
	}
	template <>
	inline auto registerMembers<uint4>()
	{
		return members(
			member("x", &uint4::x),
			member("y", &uint4::y),
			member("z", &uint4::z),
			member("w", &uint4::w)
		);
	}

	template <>
	inline auto registerMembers<SceneSettings>()
	{
		return members(
			member("x", &SceneSettings::light_pos),
			member("y", &SceneSettings::soft_shadow_k),
			member("z", &SceneSettings::light_intensity),
			member("w", &SceneSettings::enable_fog),
			member("w", &SceneSettings::fog_deisity),
			member("w", &SceneSettings::noise_freuency),
			member("w", &SceneSettings::noise_amplitude),
			member("w", &SceneSettings::noise_redistrebution),
			member("w", &SceneSettings::terracing),
			member("w", &SceneSettings::volume_resolution),
			member("w", &SceneSettings::world_size)
		);
	}

	template <>
	inline auto registerMembers<TerrainBrush>()
	{
		return members(
			//member("brush_type", &TerrainBrush::brush_type),
			member("brush_radius", &TerrainBrush::brush_radius),
			member("material_index", &TerrainBrush::material_index)
		);
	}

	template <>
	inline auto registerMembers<GPUSceneObject>()
	{
		return members(
			member("index_of_first_prim", &GPUSceneObject::index_of_first_prim),
			member("num_prims", &GPUSceneObject::num_prims),
			member("is_character", &GPUSceneObject::is_character),
			member("offset", &GPUSceneObject::offset),
			member("num_nodes", &GPUSceneObject::num_nodes),
			member("rotation", &GPUSceneObject::rotation),
			member("material", &GPUSceneObject::material),
			member("location", &GPUSceneObject::location)
		);
	}

	template <>
	inline auto registerMembers<Material>()
	{
		return members(
			member("index_of_first_prim", &Material::uvs),
			member("num_prims", &Material::normals),
			member("is_character", &Material::color)
		);
	}

} // end of namespace meta

