#pragma once

#include "Vector.h"
#include "Color.h"

class RRay;
using RVectorF = RVector<float>;
class RObject
{
public:
	RColor Color;
	HOST_DEVICE_FUNCTION
	RObject();

	HOST_DEVICE_FUNCTION
	virtual ~RObject();

	// method functions
	HOST_DEVICE_FUNCTION
	inline virtual RColor GetColor() { return RColor(1,0,1,2); }

	HOST_DEVICE_FUNCTION
	virtual RVectorF GetNormalAt(RVectorF) { return RVectorF(0, 0, 0); }

	__device__
	virtual bool FindIntersection(RRay *, float &, float &, float &) { return true; };

	HOST_DEVICE_FUNCTION
	virtual RVectorF GetMax() { return RVectorF(0, 0, 0); }

	HOST_DEVICE_FUNCTION
	inline virtual RVectorF GetMin() { return RVectorF(0, 0, 0); }

	HOST_DEVICE_FUNCTION
	inline virtual RVectorF Centroid() { return RVectorF(0, 0, 0); }
};

