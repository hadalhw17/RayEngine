#pragma once

#include "RayEngine/RayEngine.h"


class RRay
{
	float3 origin, direction;

public:

	HOST_DEVICE_FUNCTION RRay();

	HOST_DEVICE_FUNCTION
	RRay(float3 o, float3 d)
	{
		origin = o;
		direction = d;
	}

	// method functions
	HOST_DEVICE_FUNCTION
	float3 getRayOrigin() { return origin; }

	HOST_DEVICE_FUNCTION
	float3 getRayDirection() { return direction; }

};

