#pragma once

#include "cuda_runtime.h"
#include "BoundingVolume.h"


struct GPUBoundingBox
{
public:
	GPUBoundingBox() { Min = make_float3(1e20, 1e20, 1e20), Max = make_float3(-1e20, -1e20, -1e20); }
	GPUBoundingBox(RBoundingVolume *box)
	{
		Min = make_float3(box->bounds[0].x, box->bounds[0].y, box->bounds[0].z);
		Max = make_float3(box->bounds[1].x, box->bounds[1].y, box->bounds[1].z);
	}
	float3 Min, Max;
};
