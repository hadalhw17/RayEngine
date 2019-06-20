#pragma once
#include "RayEngine/RayEngine.h"


typedef struct
{
	float4 m[3];
} float3x4;

typedef struct
{
	float2 m[2];
} float2x2;


enum BiomeTypes
{
	BIOME_NONE = 0,
	BIOME_OCEAN = 1,
	BOIME_SAND = 2,
	BOIME_GRASS = 3,
	BOIME_ROCK = 4,
	BIOME_SNOW = 5
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

struct RayData
{
	float3 direction;		// Ray Direction,
	float3 origin;			// Ray Origin.

	// The folowing params should be set only in bounding volume intersection test.
	float min_distance = K_EPSILON;		// Ray nearest clipping.
	float max_distance = K_INFINITY;	// Ray far clipping.
};


struct HitResult
{
	RayData ray;
	float t;
	float3 normal;
	float2 uv;
	bool hits = false;
	int obj_index;
	float3 hit_color;
	uint material_index = -1;
	float prel;

	HOST_DEVICE_FUNCTION
	HitResult()
	{
		t = K_INFINITY;
		hits = false;
		obj_index = -1;
		material_index = -1;
	}
};
