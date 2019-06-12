#pragma once
#include "Object.h"
#include "helper_math.h"




class RRay;
struct float4;
class RPlane :
	public RayEngine::RObject
{
	float3 normal;
	float distance;


public:


	HOST_DEVICE_FUNCTION RPlane();
	HOST_DEVICE_FUNCTION RPlane(float3 normalValue, float distanceValue, float4 ColorValue);


	// method functions
	HOST_DEVICE_FUNCTION float3 GetPlaneNormal() { return normal; }
	HOST_DEVICE_FUNCTION float GetPlaneDistance() { return distance; }

	//Inherited methods
	__device__ virtual bool FindIntersection(RRay* ray, float& t, float& u, float& v);
	HOST_DEVICE_FUNCTION inline virtual float4 GetColor() { return Color; }
	HOST_DEVICE_FUNCTION virtual float3 GetNormalAt(float3 point) { return normal; }
	HOST_DEVICE_FUNCTION
		virtual float3 GetMax()
	{
		if (this->normal.x == 0 && this->normal.z == 0)
		{
			return make_float3(K_INFINITY, this->distance, K_INFINITY);
		}
		else if (this->normal.x == 0 && this->normal.y == 0)
		{
			return make_float3(K_INFINITY, K_INFINITY, this->distance);
		}
		else if (this->normal.y == 0 && this->normal.z == 0)
		{
			return make_float3(this->distance, K_INFINITY, K_INFINITY);
		}
		return make_float3(0, 0, 0);
	}
	HOST_DEVICE_FUNCTION
		inline virtual float3 GetMin()
	{
		if (this->normal.x == 0 && this->normal.z == 0)
		{
			return make_float3(-K_INFINITY, this->distance, -K_INFINITY);
		}
		else if (this->normal.x == 0 && this->normal.y == 0)
		{
			return make_float3(-K_INFINITY, -K_INFINITY, this->distance);
		}
		else if (this->normal.y == 0 && this->normal.z == 0)
		{
			return make_float3(this->distance, -K_INFINITY, -K_INFINITY);
		}
		return make_float3(0, 0, 0);
	}
	HOST_DEVICE_FUNCTION
		inline virtual float3 Centroid() { return this->normal * (this->distance); }
};
