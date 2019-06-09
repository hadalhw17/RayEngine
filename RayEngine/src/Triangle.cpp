#include "Triangle.h"
#include "BoundingVolume.h"
#include "helper_math.h"
#include "cuda_runtime_api.h"


	RTriangle::RTriangle(float3 c0, float3 c1, float3 c2, float4 c)
	{
		v0 = c0;
		v1 = c1;
		v2 = c2;
		Color = c;
		A = v1 - v0;
		B = v2 - v0;
	}

	RTriangle::~RTriangle()
	{

	}

	float3 RTriangle::GetMax()
	{
		float x = (v0.x > v1.x && v0.x > v2.x) ? v0.x : (v1.x > v2.x ? v1.x : v2.x);
		float y = (v0.y > v1.y && v0.y > v2.y) ? v0.y : (v1.y > v2.y ? v1.y : v2.y);
		float z = (v0.z > v1.z && v0.z > v2.z) ? v0.z : (v1.z > v2.z ? v1.z : v2.z);

		return make_float3(x, y, z);
	}

	float3 RTriangle::GetMin()
	{
		float x = (v0.x < v1.x && v0.x < v2.x) ? v0.x : (v1.x < v2.x ? v1.x : v2.x);
		float y = (v0.y < v1.y && v0.y < v2.y) ? v0.y : (v1.y < v2.y ? v1.y : v2.y);
		float z = (v0.z < v1.z && v0.z < v2.z) ? v0.z : (v1.z < v2.z ? v1.z : v2.z);

		return make_float3(x, y, z);
	}



	float3 RTriangle::Centroid()
	{
		const float a = (v0.x + v1.x + v2.x) / 3;
		const float b = (v0.y + v1.y + v2.y) / 3;
		const float c = (v0.z + v1.z + v2.z) / 3;

		return make_float3(a, b, c);
	}

	__device__
		bool RTriangle::FindIntersection(RRay* ray, float& t, float& u, float& v)
	{
		//
		//float3 orig = ray->getRayOrigin();
		//float3 dir = ray->getRayDirection();
		//float3 v0v1 = A;
		//float3 v0v2 = B;
		//float3 pvec = dir.CrossProduct(v0v2);
		//float det = v0v1.DotProduct(pvec);

		//// ray and triangle are parallel if det is close to 0
		//if (fabs(det) < -K_EPSILON) return false;

		//float invDet = 1 / det;

		//float3 tvec = orig.VectorAdd(v0.Negative());
		//u = tvec.DotProduct(pvec) * invDet;
		//if (u < 0 || u > 1) return false;

		//float3 qvec = tvec.CrossProduct(v0v1);
		//v = dir.DotProduct(qvec) * invDet;
		//if (v < 0 || u + v > 1) return false;

		//t = v0v2.DotProduct(qvec) * invDet;

		return (t > K_EPSILON) ? true : false;
	}

	float3 RTriangle::GetNormalAt(float3 intersection_position)
	{
		return normalize(cross(A, B));
	}