#pragma once
#include "Object.h"
#include "Vector.h"
#include "Color.h"

class RRay;
class RColor;

using RVectorF = RVector<float>;

class RPlane :
	public RObject
{
	RVectorF normal;
	float distance;
	

public:


	HOST_DEVICE_FUNCTION RPlane();
	HOST_DEVICE_FUNCTION RPlane(RVectorF normalValue, float distanceValue, RColor ColorValue);


	// method functions
	HOST_DEVICE_FUNCTION RVectorF GetPlaneNormal() { return normal; }
	HOST_DEVICE_FUNCTION float GetPlaneDistance() { return distance; }

	//Inherited methods
	__device__ virtual bool FindIntersection(RRay *ray, float &t, float &u, float &v);
	HOST_DEVICE_FUNCTION inline virtual RColor GetColor() { return Color; }
	HOST_DEVICE_FUNCTION virtual RVectorF GetNormalAt(RVectorF point) { return normal; }
	HOST_DEVICE_FUNCTION 
	virtual RVectorF GetMax()
	{
		if (this->normal.getVecX() == 0 && this->normal.getVecZ() == 0)
		{
			return RVectorF(1e20, this->distance, 1e20);
		}
		else if (this->normal.getVecX() == 0 && this->normal.getVecY() == 0)
		{
			return RVectorF(1e20, 1e20, this->distance);
		}
		else if (this->normal.getVecY() == 0 && this->normal.getVecZ() == 0)
		{
			return RVectorF(this->distance, 1e20, 1e20);
		}
		return RVectorF(0, 0, 0);
	}
	HOST_DEVICE_FUNCTION
	inline virtual RVectorF GetMin()
	{
		if (this->normal.getVecX() == 0 && this->normal.getVecZ() == 0)
		{
			return RVectorF(-1e20, this->distance, -1e20);
		}
		else if (this->normal.getVecX() == 0 && this->normal.getVecY() == 0)
		{
			return RVectorF(-1e20, -1e20, this->distance);
		}
		else if (this->normal.getVecY() == 0 && this->normal.getVecZ() == 0)
		{
			return RVectorF(this->distance, -1e20, -1e20);
		}
		return RVectorF(0, 0, 0);
	}
	HOST_DEVICE_FUNCTION
	inline virtual RVectorF Centroid() { return this->normal.VectorMult(this->distance); }
};
