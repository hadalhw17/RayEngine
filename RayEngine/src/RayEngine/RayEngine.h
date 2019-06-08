#pragma once
#include "cuda_runtime_api.h"
#include "stdio.h"

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
#define sphere_tracing
#include <utility>

#define K_INFINITY	1e20f					// Mathematical infinity
#define K_EPSILON	1e-4f					// Error value
#define M_PI		3.14159265358979323846  // pi
#define M_PI_2		1.57079632679489661923  // pi/2

#define HOST_DEVICE_FUNCTION __device__ __host__


// settings
#define SCR_WIDTH 1920
#define SCR_HEIGHT 1080

#define PATH_TO_VOLUMES "C://dev/SDFGenerator/SDFGenerator/SDFs/"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


#ifndef RAY_ENGINE_ASSERT
#include <cassert>
#define RAY_ENGINE_ASSERT(condition, msg) {assert((msg, condition));}
#endif // !RAY_ENGINE_ASSERT



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
	cudaTextureObject_t texture[3];
};

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