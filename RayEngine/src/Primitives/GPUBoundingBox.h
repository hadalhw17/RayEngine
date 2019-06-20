#pragma once

#include <vector_types.h>
#include "BoundingVolume.h"


	struct RAY_ENGINE_API GPUBoundingBox
	{
	public:
		HOST_DEVICE_FUNCTION
			GPUBoundingBox() { Min = make_float3(1e20f, 1e20f, 1e20f), Max = make_float3(-1e20f, -1e20f, -1e20f); }

		HOST_DEVICE_FUNCTION
			GPUBoundingBox(float3 min, float3 max) { Min = min, Max = max; }

		HOST_DEVICE_FUNCTION
			GPUBoundingBox(RBoundingVolume* box)
		{
			Min = make_float3(box->bounds[0].x, box->bounds[0].y, box->bounds[0].z);
			Max = make_float3(box->bounds[1].x, box->bounds[1].y, box->bounds[1].z);
		}

		HOST_DEVICE_FUNCTION
			float dx() const
		{
			return Max.x - Min.x;
		}

		HOST_DEVICE_FUNCTION
			float dy() const
		{
			return Max.y - Min.y;
		}

		HOST_DEVICE_FUNCTION
			float dz() const
		{
			return Max.z - Min.z;
		}

		HOST_DEVICE_FUNCTION
			float3 center() const
		{
			return make_float3(Min.x + dx() / 2.f, Min.y + dy() / 2.f, Min.z + dz() / 2.f);
		}

		inline float surface_area()
		{
			return 2 * dx() * dy() + 2 * dx() * dz() + 2 * dy() * dz();
		}

		float3 Min, Max;
	};

	//namespace meta {

	//	template <>
	//	inline auto registerMembers<GPUBoundingBox>()
	//	{
	//		return members(
	//			member("Min", &GPUBoundingBox::Min),
	//			member("Max", &GPUBoundingBox::Max)
	//		);
	//	}

	//} // end of namespace meta