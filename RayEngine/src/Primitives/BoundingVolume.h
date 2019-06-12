#pragma once

#include <atomic>
#include <vector>
#include <memory>
#include <chrono>
#include <limits>
#include <algorithm>

#include "Ray.h"
#include "Object.h"
#include "Triangle.h"
#include "RayEngine/RayEngine.h"
#include <vector_types.h>

#include <helper_math.h>

#define DOUBLE_EPS  1e-9



	enum AXIS { XY, XZ, YZ, NONE };
	class RBoundingVolume
	{

	public:
		int triangleAmount = 0;
		int numRayRBoundingVolumeTests = 0;
		float3 splitPos;
		float3 bounds[2] = { K_INFINITY, -K_INFINITY };


		RBoundingVolume() {}


		RBoundingVolume(float3 min_, float3 max_)
		{
			bounds[0] = min_;
			bounds[1] = max_;
		}


		void extendBy(float3 p)
		{
			if (p.x < bounds[0].x) bounds[0].x = p.x;
			if (p.y < bounds[0].y) bounds[0].y = p.y;
			if (p.z < bounds[0].z) bounds[0].z = p.z;
			if (p.x > bounds[1].x) bounds[1].x = p.x;
			if (p.y > bounds[1].y) bounds[1].y = p.y;
			if (p.z > bounds[1].z) bounds[1].z = p.z;
		}



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


		///*inline */ float3 centroid() const { return (bounds[0] + bounds[1]) * 0.5; }


		float3& operator [] (bool i) { return bounds[i]; }


		const float3 operator [] (bool i) const { return bounds[i]; }


		bool intersect(RRay r, float& tMin, float& tMax)
		{
			float3 ray_dir = r.getRayDirection();
			float3 ray_o = r.getRayOrigin();
			float3 dirfrac = make_float3(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);

			float t1 = (bounds[0].x - ray_o.x) * dirfrac.x;
			float t2 = (bounds[1].x - ray_o.x) * dirfrac.x;
			float t3 = (bounds[0].y - ray_o.y) * dirfrac.y;
			float t4 = (bounds[1].y - ray_o.y) * dirfrac.y;
			float t5 = (bounds[0].z - ray_o.z) * dirfrac.z;
			float t6 = (bounds[1].z - ray_o.z) * dirfrac.z;

			float tmin = std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
			float tmax = std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

			// If tmax < 0, ray intersects AABB, but entire AABB is behind ray, so reject.
			if (tmax < .0f) {
				return false;
			}

			// If tmin > tmax, ray does not intersect AABB.
			if (tmin > tmax) {
				return false;
			}

			tMin = tmin;
			tMax = tmax;

			return true;
		}

		RBoundingVolume* addObjectToBox(RTriangle* Object, int numObjs, int* objInd)
		{
			RBoundingVolume* box = new RBoundingVolume();
			for (int i = 0; i < numObjs; i++)
			{
				triangleAmount++;
				box->extendBy(Object[objInd[i]].GetMax());
				box->extendBy(Object[objInd[i]].GetMin());
			}

			return box;
		}


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


		bool enclose(float3 point) {
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
		}


		bool inBox(float3 point)
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


		void splitVoxel(AXIS a, float3 coord, RBoundingVolume& lBox, RBoundingVolume& rBox)
		{
			lBox = *this;
			rBox = *this;

		}


		inline float clamp(const float& v, const float& lo, const float& hi)
		{
			return std::max(lo, std::min(v, hi));
		}

	private:

	};
