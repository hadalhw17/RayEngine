#pragma once

#include <atomic>
#include <vector>
#include <memory>
#include <chrono>
#include <limits>
#include <algorithm>
#include "device_functions.h"

#include "Vector.h"
#include "Ray.h"
#include "Object.h"
#include "RayEngine.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/swap.h>

#define DOUBLE_EPS  1e-9


using RVectorF = RVector<double>;
using RVectorB = RVector<bool>;

template<typename T = double>

class CUDABoundingVolume
{
	enum AXIS { XY, XZ, YZ, NONE };
	thrust::device_vector<thrust::device_ptr<RObject>> Objects;
public:
	int triangleAmount = 0;
	int numRayRBoundingVolumeTests = 0;
	RVectorF splitPos;
	RVector<double> bounds[2] = { RVectorF(1e5), RVectorF(-1e5) };

	HOST_DEVICE_FUNCTION
	CUDABoundingVolume() {}

	HOST_DEVICE_FUNCTION
	CUDABoundingVolume(RVector<T> min_, RVector<T> max_)
	{
		bounds[0] = min_;
		bounds[1] = max_;
	}

	HOST_DEVICE_FUNCTION
	void extendBy(const RVector<T>& p)
	{
		if (p.x < bounds[0].x) bounds[0].x = p.x;
		if (p.y < bounds[0].y) bounds[0].y = p.y;
		if (p.z < bounds[0].z) bounds[0].z = p.z;
		if (p.x > bounds[1].x) bounds[1].x = p.x;
		if (p.y > bounds[1].y) bounds[1].y = p.y;
		if (p.z > bounds[1].z) bounds[1].z = p.z;
	}


	HOST_DEVICE_FUNCTION
	double getSA()
	{
		const double xLen = this->bounds[1].x - this->bounds[0].x;
		const double yLen = this->bounds[1].y - this->bounds[0].y;
		const double zLen = this->bounds[1].z - this->bounds[0].z;

		double xyArea = xLen * yLen;
		double xzArea = xLen * zLen;
		double yzArea = yLen * zLen;

		double SA = (2 * xyArea) + (2 * xzArea) + (2 * yzArea);
		return SA;
	}

	HOST_DEVICE_FUNCTION
		/*inline */ RVector<T> centroid() const { return (bounds[0] + bounds[1]) * 0.5; }

	HOST_DEVICE_FUNCTION
		RVector<T>& operator [] (bool i) { return bounds[i]; }

	HOST_DEVICE_FUNCTION
		const RVector<T> operator [] (bool i) const { return bounds[i]; }

	HOST_DEVICE_FUNCTION
		bool intersect(RRay *r, double &tMin, double &tMax)
	{
		//auto start = std::chrono::system_clock::now();

		RVector<double> invdir = r->getRayDirection() / 1;
		RVector<double> boxMin = bounds[0];
		RVector<double> boxMax = bounds[1];

		//numRayRBoundingVolumeTests++;

		double lo = (boxMin.x - r->getRayOrigin().x) * invdir.x;
		double hi = (boxMax.x - r->getRayOrigin().x) * invdir.x;
		if (lo > hi) thrust::swap(lo, hi);
		tMin = lo;
		tMax = hi;

		double lo1 = (boxMin.y - r->getRayOrigin().y) * invdir.y;
		double hi1 = (boxMax.y - r->getRayOrigin().y) * invdir.y;
		if (lo1 > hi1) thrust::swap(lo1, hi1);

		if ((tMin > hi1) || (lo1 > tMax))
			return false;
		tMin = lo1 < tMin ? lo1:tMin;
		tMax = hi1 > tMax ? hi1:tMax;

		double lo2 = (boxMin.z - r->getRayOrigin().z) * invdir.z;
		double hi2 = (boxMax.z - r->getRayOrigin().z) * invdir.z;
		if (lo2 > hi2) thrust::swap(lo2, hi2);

		if ((tMin > hi2) || (lo2 > tMax))
			return false;

		tMin = lo2 < tMin ? lo2 : tMin;
		tMax = hi2 > tMax ? hi2 : tMax;

		printf("+");

		return (tMin <= tMax) && (tMax > 0.0);
	}

	HOST_DEVICE_FUNCTION
		void resize()
	{
		bounds[0] = RVectorF(1e20);
		bounds[1] = RVectorF(-1e20);

		for (int i = 0; i < Objects.size(); i++)
		{
			this->extendBy(thrust::raw_pointer_cast(Objects[i])->GetMax());
			this->extendBy(thrust::raw_pointer_cast(Objects[i])->GetMin());
		}
	}


		bool addObjectToBox(thrust::device_ptr<RObject> Object)
	{
		Objects.push_back(Object);
		triangleAmount++;
		RObject *variable = thrust::raw_pointer_cast(&Object[0]);
		//if (variable)
		//{
			this->extendBy(variable->GetMax());
			this->extendBy(variable->GetMin());
		//}

		return true;
	}

	HOST_DEVICE_FUNCTION
		int getLongestAxis()
	{
		const double xLen = bounds[1].x - bounds[0].x;
		const double yLen = bounds[1].y - bounds[0].y;
		const double zLen = bounds[1].z - bounds[0].z;

		float maxLen = xLen;
		int retAxis = 0;

		if (yLen > maxLen)
		{
			retAxis = 1;
			maxLen = yLen;
		}
		if (zLen > maxLen)
		{
			retAxis = 2;
			maxLen = zLen;
		}
		return retAxis;
	}

	HOST_DEVICE_FUNCTION
	thrust::device_vector<thrust::device_ptr<RObject>> getObjects()
	{
		return Objects;
	}

	HOST_DEVICE_FUNCTION
	bool enclose(RVectorF point) {
		if (bounds[1].x > point.x &&
			bounds[0].x < point.x &&
			bounds[1].y > point.y &&
			bounds[0].y < point.y &&
			bounds[1].z > point.z &&
			bounds[0].z < point.z) {
			return true;
		}
		else {
			return false;
		}
		return false;
	}

	HOST_DEVICE_FUNCTION
		bool inBox(RVectorF point)
	{
		if (point.x < bounds[1].x &&
			point.x > bounds[0].x &&
			point.x < bounds[1].x &&
			point.x > bounds[0].x &&
			point.x < bounds[1].x &&
			point.x > bounds[0].x)
		{
			return true;
		}
		else
		{
			return false;
		}
	}

	//int ObjectsInVoxel(std::vector<std::shared_ptr<RObject>> objs)
	//{
	//	int count = 0;
	//	for (int i = 0; i < objs.size(); i++)
	//	{
	//		if (this->enclose(objs[i]))
	//			count++;
	//	}
	//	return count;
	//}

	HOST_DEVICE_FUNCTION
	void splitVoxel(AXIS a, RVectorF coord, CUDABoundingVolume<> &lBox, CUDABoundingVolume<> &rBox)
	{
		lBox = this;
		rBox = this;

	}

	HOST_DEVICE_FUNCTION
	inline T clamp(const T &v, const T &lo, const T &hi)
	{
		return fmaxf(lo, fminf(v, hi));
	}

private:

};

