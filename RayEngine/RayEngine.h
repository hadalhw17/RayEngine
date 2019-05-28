#pragma once
#include "device_launch_parameters.h"
#define sphere_tracing
#include <utility>

#define K_INFINITY	1e20f					// Mathematical infinity
#define K_EPSILON	1e-4f					// Error value
#define M_PI		3.14159265358979323846  // pi
#define M_PI_2		1.57079632679489661923  // pi/2

#define HOST_DEVICE_FUNCTION __device__ __host__


// settings
#define SCR_WIDTH 1280
#define SCR_HEIGHT 720

#define PATH_TO_VOLUMES "C://dev/SDFGenerator/SDFGenerator/SDFs/"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


using edge = std::pair<int, int>;

enum TerrainBrushType
{
	ADD = 0,
	SUBTRACT = 1,
	INTERSECT = 2
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
	float4 color;
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
	float4 hit_color;

	HOST_DEVICE_FUNCTION
	HitResult()
	{
		t = K_INFINITY;
		hits = false;
		obj_index = -1;
	}
};