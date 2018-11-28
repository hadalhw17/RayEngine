#pragma once

#include <limits>
#include <stdio.h>
#include "RayEngine.h"
#include "device_functions.h"

__constant__
const float kInfinity = 1e20;

template<typename T>
class RVector
{
public:
	T x, y, z;

	HOST_DEVICE_FUNCTION RVector() : x(0), y(0), z(0) {}
	HOST_DEVICE_FUNCTION RVector(T xx) : x(xx), y(xx), z(xx) {}
	HOST_DEVICE_FUNCTION RVector(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}


	// method functions
	HOST_DEVICE_FUNCTION inline T getVecX() { return x; }
	HOST_DEVICE_FUNCTION inline T getVecY() { return y; }
	HOST_DEVICE_FUNCTION inline T getVecZ() { return z; }

	//comparison operators
	HOST_DEVICE_FUNCTION inline bool operator >=(RVector<T> a) { return x >= a.x && y >= a.y && z >= a.z ? true : false; }
	HOST_DEVICE_FUNCTION inline bool operator == (RVector<T> a) { return this->Magnitude() == a.Magnitude() ? true : false; }


	HOST_DEVICE_FUNCTION inline T& operator [] (size_t i) { return (&x)[i]; }
	HOST_DEVICE_FUNCTION inline RVector<bool>& operator [](int i) { return RVector<bool>(x > 0, y > 0, z < 0); }

	HOST_DEVICE_FUNCTION inline friend RVector operator /(RVector<T> vec, T s) { return RVector<T>(s / vec.getVecX(), s / vec.getVecY(), s / vec.getVecZ()); }

	HOST_DEVICE_FUNCTION inline T Magnitude()
	{
		return sqrt((x*x) + (y*y) + (z*z));
	}

	HOST_DEVICE_FUNCTION inline T Distance(RVector<T> end)
	{
		return sqrt(pow(end.x - x, 2) + pow(end.y - y, 2) + pow(end.z - z, 2));
	}

	HOST_DEVICE_FUNCTION inline RVector<T> Normalize()
	{
		float magnitude = sqrt((x*x) + (y*y) + (z*z));
		return RVector(x / magnitude, y / magnitude, z / magnitude);
	}

	HOST_DEVICE_FUNCTION inline RVector<T> Negative()
	{
		return RVector(-x, -y, -z);
	}

	HOST_DEVICE_FUNCTION inline T DotProduct(RVector<T> v)
	{
		return x * v.getVecX() + y * v.getVecY() + z * v.getVecZ();
	}

	HOST_DEVICE_FUNCTION inline RVector<T> CrossProduct(RVector<T> v)
	{
		return RVector(y*v.getVecZ() - z * v.getVecY(), z*v.getVecX() - x * v.getVecZ(), x*v.getVecY() - y * v.getVecX());
	}

	HOST_DEVICE_FUNCTION inline RVector<T> VectorAdd(RVector<T> v)
	{
		return RVector<T>(x + v.getVecX(), y + v.getVecY(), z + v.getVecZ());
	}

	HOST_DEVICE_FUNCTION inline RVector<T> VectorMult(T scalar)
	{
		return RVector<T>(x*scalar, y*scalar, z*scalar);
	}

	HOST_DEVICE_FUNCTION inline void ToString()
	{
		printf("x = %f, y = %f, z = %f \n", x, y, z);
	}

	HOST_DEVICE_FUNCTION inline T Norm()
	{
		return x * x + y * y + z * z;
	}
};

