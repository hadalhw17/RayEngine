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

#define K_INFINITY	1e20f					// Mathematical infinity
#define K_EPSILON	1e-4f					// Error value
#define M_PI		3.14159265358979323846  // pi
#define M_PI_2		1.57079632679489661923  // pi/2

#define HOST_DEVICE_FUNCTION __device__ __host__
#define NOMINMAX

// settings
#define SCR_WIDTH 300
#define SCR_HEIGHT 300

#define PATH_TO_VOLUMES "C://dev/SDFGenerator/SDFGenerator/SDFs/"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


#ifndef RAY_ENGINE_ASSERT
#include <cassert>
#define RAY_ENGINE_ASSERT(condition, msg) {assert((msg, condition));}
#endif // !RAY_ENGINE_ASSERT


#ifdef RE_DEVELOPMENT
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



using edge = std::pair<int, int>;

struct texturess
{
	uint2 resolution[3];
	cudaTextureObject_t texture[3];
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
	TerrainBrushType brush_type = SPHERE_ADD;
	int3 brush_extent = { 10, 10, 10 };
	int material_index = 0;
	bool snap_to_grid = false;
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
	RenderQuality quality = MEDIUM;
	bool gamma;
	bool vignetting;
	float vignetting_k;
	bool gravity = false;
	bool god_mode = true;
};

struct SceneSettings
{
	float3 light_pos = {0.f, 1.f, 0.f};
	int soft_shadow_k = 32;
	float light_intensity = 2500.f;
	bool enable_fog = false;
	float fog_deisity = 0.f;
	float noise_freuency = 2.f;
	float noise_amplitude = 200.f;
	float noise_redistrebution = 1.f;
	int terracing = 1;
	bool should_terrace = false;
	uint3 volume_resolution = { 100, 100, 100};
	float3 world_size = { 300, 300, 300 };
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
