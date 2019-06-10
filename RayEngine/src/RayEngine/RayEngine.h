#pragma once
#include "cuda_runtime_api.h"
#include "stdio.h"
#include <string>
#include <functional>
#include <vector_types.h>

#ifdef RE_PLATFORM_WINDOWS
#ifdef RE_BUILD_DLL
#define RAY_ENGINE_API __declspec(dllexport)
#include "device_launch_parameters.h"
#else
#define	RAY_ENGINE_API __declspec(dllimport)
#endif	
#else
#error Ray Engine currently only supports Windows!
#endif

#define BIND_EVENT_FN(x) std::bind(&x, this, std::placeholders::_1)

#define BIT(x) (1 << x)
#define sphere_tracing
#include <utility>

#define K_INFINITY	1e20f					// Mathematical infinity
#define K_EPSILON	1e-4f					// Error value
#define M_PI		3.14159265358979323846  // pi
#define M_PI_2		1.57079632679489661923  // pi/2

#define HOST_DEVICE_FUNCTION __device__ __host__
#define NOMINMAX

// settings
#define SCR_WIDTH 1920
#define SCR_HEIGHT 1080

#define PATH_TO_VOLUMES "C://dev/SDFGenerator/SDFGenerator/SDFs/"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


#ifndef RAY_ENGINE_ASSERT
#include <cassert>
#define RAY_ENGINE_ASSERT(condition, msg) {assert((msg, condition));}
#endif // !RAY_ENGINE_ASSERT

#define RAY_LOG
#ifdef RAY_LOG
#define RE_LOG(msg) std::cout << msg << std::endl;
#define REE_LOG(msg) std::cout << "!!!" << msg << "!!!" << std::endl;
#else 
#define RE_LOG(msg)
#define REE_LOG(msg)
#endif



inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


typedef struct
{
	float4 m[3];
} float3x4;

typedef struct
{
	float2 m[2];
} float2x2;

using edge = std::pair<int, int>;

struct texturess
{
	uint2 resolution[3];
	cudaTextureObject_t texture[3];
};

namespace RayEngine
{
	struct WindowData
	{
		size_t width, heigth;
		std::string title;
		std::function<void(class Event&)> function_callback;
	};
}


enum BiomeTypes
{
	BIOME_NONE = 0,
	BIOME_OCEAN = 1,
	BOIME_SAND = 2,
	BOIME_GRASS = 3,
	BOIME_ROCK = 4,
	BIOME_SNOW = 5
};


enum TerrainBrushType
{
	SPHERE_ADD = 0,
	SPHERE_SUBTRACT = 1,
	CUBE_ADD = 2,
	CUBE_SUBTRACT = 3
};

struct TerrainBrush
{
	TerrainBrushType brush_type;
	float brush_radius;
	int material_index;
};

enum RenderQuality
{
	LOW = 0,
	MEDIUM = 1,
	HIGH = 2
};

struct RenderingSettings
{
	float texture_scale = 1;
	RenderQuality quality = HIGH;
	bool gamma;
	bool vignetting;
	float vignetting_k;
	bool gravity;
};

struct SceneSettings
{
	float3 light_pos;
	int soft_shadow_k = 32;
	float light_intensity;
	bool enable_fog = false;
	float fog_deisity;
	float noise_freuency;
	float noise_amplitude;
	float noise_redistrebution;
	int terracing = 1;
	uint3 volume_resolution;
	float3 world_size;
};

struct GPUVolumeObjectInstance
{
	int index;
	float3 location;
	float3 rotation;

	HOST_DEVICE_FUNCTION
		GPUVolumeObjectInstance()
	{
		index = -1;
		location = float3();
		rotation = float3();
	}

	HOST_DEVICE_FUNCTION
		GPUVolumeObjectInstance(int _index, float3 _location, float3 _roatation)
	{
		index = _index;
		location = _location;
		rotation = _roatation;
	}
};


enum DistanceType
{
	FACE = 0,		// Hit in the face.
	EDGE1 = 1,		// Hit on the first edge.
	EDGE2 = 2,		// Hit on the second edge.
	EDGE3 = 3,		// Hit on the third edge.
	VERT1 = 4,		// Hit on the first vertex.
	VERT2 = 5,		// Hit on the second vertex.
	VERT3 = 6		// Hit on the third vertex.
};

struct TriangleDistanceResult
{
	float distance;
	DistanceType hit_type;
	float3 hit_point;
};

enum Axis
{
	X_Axis = 0,		// X Axis.
	Y_Axis = 1,		// Y Axis.
	Z_Axis = 2		// Z Axis.
};

enum BoxFace
{
	LEFT = 0,		// Left.
	FRONT = 1,		// Front.
	RIGHT = 2,		// Right.
	BACK = 3,		// Back.
	TOP = 4,		// Top.
	BOTTOM = 5		// Bottom.
};

struct RGBType
{
	float r, g, b;
};

enum MaterialType
{
	COLOR = 0,		// Default material.
	TILE = 1,		// Tile pattern.
	PHONG = 2,		// Phong model.
	REFLECT = 3,	// Just reflection.
	REFRACT = 4		// Both - reflection and refration.
};

struct Material
{
	bool uvs = false;
	bool normals = true;
	float3 color;
	MaterialType type = COLOR;
};

struct GPUSceneObject
{
	size_t index_of_first_prim;
	size_t num_prims;
	float3 location;
	float3 rotation;
	size_t num_nodes;
	size_t index_list_size;
	size_t offset;
	Material material;
	bool is_character;

	GPUSceneObject()
	{
		index_of_first_prim = 0;
		num_prims = 0;
		num_nodes = 0;
		index_list_size = 0;
		offset = 0;
		is_character = false;
	}
};

struct HitResult
{
	float t;
	float3 normal;
	float3 hit_point;
	float3 ray_dir;
	float3 ray_o;
	float2 uv;
	bool hits;
	int obj_index;
	float3 hit_color;

	HOST_DEVICE_FUNCTION
		HitResult()
	{
		t = K_INFINITY;
		hits = false;
		obj_index = -1;
	}
};


#include <Meta.h>
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

