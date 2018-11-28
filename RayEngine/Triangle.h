#pragma once
#include "Object.h"
#include "Vector.h"

#include "RayEngine.h"

using RVectorF = RVector<float>;

constexpr float kEpsilon = 1e-4;


class RTriangle :
	public RObject
{

	RVectorF A;
	RVectorF B;

public:

	RVectorF v0, v1, v2;

	RTriangle() {}

	HOST_DEVICE_FUNCTION
	RTriangle(RVectorF c0, RVectorF c1, RVectorF c2, RColor c);

	HOST_DEVICE_FUNCTION
	RTriangle(const RTriangle&) = delete;



	HOST_DEVICE_FUNCTION
	virtual ~RTriangle();

	HOST_DEVICE_FUNCTION
	bool operator == (RTriangle *t) { return this->v0 == t->v0 && this->v1 == t->v1 && this->v2 == t->v2 ? true : false; }

	HOST_DEVICE_FUNCTION
	inline virtual RVectorF Centroid();

	__device__
	virtual bool FindIntersection(RRay *ray, float &t, float &u, float &v);

	HOST_DEVICE_FUNCTION
	inline virtual RColor GetColor() { return Color; }

	HOST_DEVICE_FUNCTION
	virtual RVectorF GetNormalAt(RVectorF intersection_position);

	HOST_DEVICE_FUNCTION
	virtual RVectorF GetMax();

	HOST_DEVICE_FUNCTION
	inline virtual RVectorF GetMin();

	HOST_DEVICE_FUNCTION 
	void ToString();

};

