#pragma once

#include "cuda_runtime.h"
#include "BoundingVolume.h"


struct GPUBoundingBox
{
public:
	HOST_DEVICE_FUNCTION
	GPUBoundingBox() { Min = make_float3(1e20, 1e20, 1e20), Max = make_float3(-1e20, -1e20, -1e20); }
	GPUBoundingBox(RBoundingVolume *box)
	{
		Min = make_float3(box->bounds[0].x, box->bounds[0].y, box->bounds[0].z);
		Max = make_float3(box->bounds[1].x, box->bounds[1].y, box->bounds[1].z);
	}

	float dx() const
	{
		return Max.x - Min.x;
	}

	float dy() const
	{
		return Max.y - Min.y;
	}

	float dz() const
	{
		return Max.z - Min.z;
	}

	inline float surface_area()
	{
		return 2 * dx()*dy() + 2 * dx()*dz() + 2 * dy()*dz();
	}

	float3 Min, Max;
};
