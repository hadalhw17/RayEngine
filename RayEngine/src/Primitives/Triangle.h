#pragma once
#include "Object.h"



#include <vector_types.h>
class RRay;
	class RTriangle :
		public RayEngine::RObject
	{

		float3 A;
		float3 B;

	public:

		float3 v0, v1, v2;

		RTriangle() {}

		HOST_DEVICE_FUNCTION
			RTriangle(float3 c0, float3 c1, float3 c2, float4 c);

		HOST_DEVICE_FUNCTION
			RTriangle(const RTriangle&) = delete;

		HOST_DEVICE_FUNCTION
			virtual ~RTriangle();

		HOST_DEVICE_FUNCTION
			bool operator == (RTriangle* t)
		{
			bool ret =
				this->v0.x == t->v0.x && this->v1.x == t->v1.x && this->v2.x == t->v2.x &&
				this->v0.y == t->v0.y && this->v1.y == t->v1.y && this->v2.y == t->v2.y &&
				this->v0.z == t->v0.z && this->v1.z == t->v1.z && this->v2.z == t->v2.z;
			return  ret;
		}

		HOST_DEVICE_FUNCTION
			inline virtual float3 Centroid();

		__device__
			virtual bool FindIntersection(RRay* ray, float& t, float& u, float& v);

		HOST_DEVICE_FUNCTION
			inline virtual float4 GetColor() { return Color; }

		HOST_DEVICE_FUNCTION
			virtual float3 GetNormalAt(float3 intersection_position);

		HOST_DEVICE_FUNCTION
			virtual float3 GetMax();

		HOST_DEVICE_FUNCTION
			inline virtual float3 GetMin();
	};
