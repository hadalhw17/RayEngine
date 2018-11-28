#pragma once

#include <atomic>
#include <vector>
#include <memory>
#include <chrono>
#include <limits>
#include <algorithm>

#include "Vector.h"
#include "Ray.h"
#include "Object.h"
#include "Triangle.h"
#include "RayEngine.h"

#include "cuda_runtime.h"

#define DOUBLE_EPS  1e-9

enum AXIS { XY, XZ, YZ, NONE };


class RBoundingVolume
{

public:
	int triangleAmount = 0;
	int numRayRBoundingVolumeTests = 0;
	RVectorF splitPos;
	RVectorF bounds[2] = { 1e20, -1e20 };

	HOST_DEVICE_FUNCTION
	RBoundingVolume() {}

	HOST_DEVICE_FUNCTION
	RBoundingVolume(RVectorF min_, RVectorF max_)
	{
		bounds[0] = min_;
		bounds[1] = max_;
	}

	HOST_DEVICE_FUNCTION
	void extendBy(RVector<float> p)
	{
		if (p.x < bounds[0].x) bounds[0].x = p.x;
		if (p.y < bounds[0].y) bounds[0].y = p.y;
		if (p.z < bounds[0].z) bounds[0].z = p.z;
		if (p.x > bounds[1].x) bounds[1].x = p.x;
		if (p.y > bounds[1].y) bounds[1].y = p.y;
		if (p.z > bounds[1].z) bounds[1].z = p.z;
	}


	HOST_DEVICE_FUNCTION
	float getSA()
	{
		const float xLen = this->bounds[1].x - this->bounds[0].x;
		const float yLen = this->bounds[1].y - this->bounds[0].y;
		const float zLen = this->bounds[1].z - this->bounds[0].z;

		float xyArea = xLen * yLen;
		float xzArea = xLen * zLen;
		float yzArea = yLen * zLen;

		float SA = (2 * xyArea) + (2 * xzArea) + (2 * yzArea);
		return SA;
	}

	HOST_DEVICE_FUNCTION
	///*inline */ RVector<float> centroid() const { return (bounds[0] + bounds[1]) * 0.5; }

	HOST_DEVICE_FUNCTION
	RVectorF& operator [] (bool i) { return bounds[i]; }

	HOST_DEVICE_FUNCTION
	const RVectorF operator [] (bool i) const { return bounds[i]; }
	
	HOST_DEVICE_FUNCTION
	bool intersect(RRay *r, float &tMin, float &tMax)
	{
		float3 invdir = make_float3(r->getRayDirection().x / 1, r->getRayDirection().y / 1, r->getRayDirection().z / 1);
		float3 boxMin = make_float3(bounds[0].x, bounds[0].y, bounds[0].z);
		float3 boxMax = make_float3(bounds[1].x, bounds[1].y, bounds[1].z);

		float lo = (boxMin.x - r->getRayOrigin().x) * invdir.x;
		float hi = (boxMax.x - r->getRayOrigin().x) * invdir.x;
		if (lo > hi) std::swap(lo, hi);
		tMin = lo;
		tMax = hi;

		float lo1 = (boxMin.y - r->getRayOrigin().y) * invdir.y;
		float hi1 = (boxMax.y - r->getRayOrigin().y) * invdir.y;
		if (lo1 > hi1) std::swap(lo1, hi1);

		if ((tMin > hi1) || (lo1 > tMax))
			return false;
		tMin = std::max(lo1, tMin);
		tMax = std::min(hi1, tMax);

		float lo2 = (boxMin.z - r->getRayOrigin().z) * invdir.z;
		float hi2 = (boxMax.z - r->getRayOrigin().z) * invdir.z;
		if (lo2 > hi2) std::swap(lo2, hi2);

		if ((tMin > hi2) || (lo2 > tMax))
			return false;

		tMin = std::max(lo2, tMin);
		tMax = std::min(hi2, tMax);

		return (tMin <= tMax) && (tMax > 0.0);
	}

	RBoundingVolume *addObjectToBox(RTriangle *Object, int numObjs, int *objInd)
	{
		RBoundingVolume *box = new RBoundingVolume();
		for (int i = 0; i < numObjs; i++)
		{
			triangleAmount++;
			box->extendBy(Object[objInd[i]].GetMax());
			box->extendBy(Object[objInd[i]].GetMin());
		}

		return box;
	}

	HOST_DEVICE_FUNCTION
	int getLongestAxis()
	{
		const float xLen = bounds[1].x - bounds[0].x;
		const float yLen = bounds[1].y - bounds[0].y;
		const float zLen = bounds[1].z - bounds[0].z;

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

	//HOST_DEVICE_FUNCTION
	//std::vector<std::shared_ptr<RObject>> getObjects()
	//{
	//	return Objects;
	//}

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

	HOST_DEVICE_FUNCTION
	void splitVoxel(AXIS a, RVectorF coord, RBoundingVolume &lBox, RBoundingVolume &rBox)
	{
		lBox = *this;
		rBox = *this;

	}

	HOST_DEVICE_FUNCTION
	inline float clamp(const float &v, const float &lo, const float &hi)
	{
		return std::max(lo, std::min(v, hi));
	}

private:

};

