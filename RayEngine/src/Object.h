#pragma once

#include "RayEngine/RayEngine.h"

class RRay;

extern float3 make_float3(float);
extern float3 make_float3(float, float, float);
extern float4 make_float4(float);

class RObject
{
public:
	float4 Color;
	HOST_DEVICE_FUNCTION
	RObject();

	HOST_DEVICE_FUNCTION
	virtual ~RObject();

	// method functions
	HOST_DEVICE_FUNCTION
	inline virtual float4 GetColor() { return make_float4(0); }

	HOST_DEVICE_FUNCTION
	virtual float3 GetNormalAt(float3) { return make_float3(0); }

	__device__
	virtual bool FindIntersection(RRay *, float &, float &, float &) { return true; };

	HOST_DEVICE_FUNCTION
	virtual float3 GetMax() { return make_float3(0); }

	HOST_DEVICE_FUNCTION
	inline virtual float3 GetMin() { return make_float3(0); }

	HOST_DEVICE_FUNCTION
	inline virtual float3 Centroid() { return make_float3(0); }
};

