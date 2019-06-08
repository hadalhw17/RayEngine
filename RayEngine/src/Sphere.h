#pragma once
#include "Object.h"

struct float4;
class RRay;


class RSphere :
	public RObject
{

	float3 center;
	float radius;
	float4 color;
public:

	HOST_DEVICE_FUNCTION RSphere();
	HOST_DEVICE_FUNCTION RSphere(float3 centerValue, float radiusValue, float4 colorValue);

	// method functions
	HOST_DEVICE_FUNCTION inline float3 GetSphereCenter() { return center; }
	HOST_DEVICE_FUNCTION inline float GetSphereRadius() { return radius; }


	HOST_DEVICE_FUNCTION bool SolveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1);

	HOST_DEVICE_FUNCTION bool SolveQuadratic(float &a, float &b, float &c, float  &x0, float &x1);

	HOST_DEVICE_FUNCTION inline virtual float4 GetColor() { return color; }
	HOST_DEVICE_FUNCTION virtual float3 GetNormalAt(float3 point);
	__device__ virtual bool FindIntersection(RRay *ray, float &t, float &u, float &v);
	HOST_DEVICE_FUNCTION inline virtual float3 GetMax() { return make_float3(center.x + radius, center.y + radius, center.z + radius); }
	HOST_DEVICE_FUNCTION inline virtual float3 GetMin() { return make_float3(center.x - radius, center.y - radius, center.z - radius); }
	HOST_DEVICE_FUNCTION inline virtual float3 Centroid() { return center; }
};

