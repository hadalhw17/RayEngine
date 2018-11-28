#pragma once
#include "Object.h"
#include "Vector.h"

class RColor;
class RRay;

using RVectorF = RVector<float>;
class RSphere :
	public RObject
{

	RVectorF center;
	float radius;
	RColor color;
public:

	HOST_DEVICE_FUNCTION RSphere();
	HOST_DEVICE_FUNCTION RSphere(RVectorF centerValue, float radiusValue, RColor colorValue);

	// method functions
	HOST_DEVICE_FUNCTION inline RVectorF GetSphereCenter() { return center; }
	HOST_DEVICE_FUNCTION inline float GetSphereRadius() { return radius; }


	HOST_DEVICE_FUNCTION bool SolveQuadratic(const float &a, const float &b, const float &c, float &x0, float &x1);

	HOST_DEVICE_FUNCTION bool SolveQuadratic(float &a, float &b, float &c, float  &x0, float &x1);

	HOST_DEVICE_FUNCTION inline virtual RColor GetColor() { return color; }
	HOST_DEVICE_FUNCTION virtual RVectorF GetNormalAt(RVectorF point);
	__device__ virtual bool FindIntersection(RRay *ray, float &t, float &u, float &v);
	HOST_DEVICE_FUNCTION inline virtual RVectorF GetMax() { return RVectorF(center.x + radius, center.y + radius, center.z + radius); }
	HOST_DEVICE_FUNCTION inline virtual RVectorF GetMin() { return RVectorF(center.x - radius, center.y - radius, center.z - radius); }
	HOST_DEVICE_FUNCTION inline virtual RVectorF Centroid() { return center; }
};

