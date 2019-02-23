#pragma once
#include "device_launch_parameters.h"


#define kInfinity 1e20f
#define kEpsilon 1e-4f

#define HOST_DEVICE_FUNCTION __device__ __host__


// settings
const unsigned int SCR_WIDTH = 1000;
const unsigned int SCR_HEIGHT = 500;

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
	float4 color;
	MaterialType type = COLOR;
};

struct GPUSceneObject
{
	size_t index_of_first_prim;
	size_t num_prims;
	float3 location;
	size_t num_nodes;
	size_t index_list_size;
	size_t offset;
	Material material;

	GPUSceneObject()
	{
		index_of_first_prim = 0;
		num_prims = 0;
		num_nodes = 0;
		index_list_size = 0;
		offset = 0;
	}
};

struct HitResult
{
	float t;
	float3 normal;
	float3 hit_point;
	float3 ray_dir;
	float3 ray_o;
	bool hits;
	int obj_index;
	float4 hit_color;
	HOST_DEVICE_FUNCTION
	HitResult()
	{
		t = kInfinity;
		hits = false;
		obj_index;
	}
};