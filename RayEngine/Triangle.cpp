#include "Triangle.h"
#include "BoundingVolume.h"

RTriangle::RTriangle(RVectorF c0, RVectorF c1, RVectorF c2, RColor c)
{
	v0 = c0;
	v1 = c1;
	v2 = c2;
	Color = c;
	A = v1.VectorAdd(v0.Negative());
	B = v2.VectorAdd(v0.Negative());
}

RTriangle::~RTriangle()
{

}

RVectorF RTriangle::GetMax()
{
	float x = (v0.x > v1.x && v0.x > v2.x) ? v0.x : (v1.x > v2.x ? v1.x : v2.x);
	float y = (v0.y > v1.y && v0.y > v2.y) ? v0.y : (v1.y > v2.y ? v1.y : v2.y);
	float z = (v0.z > v1.z && v0.z > v2.z) ? v0.z : (v1.z > v2.z ? v1.z : v2.z);

	return RVectorF(x, y, z);
}

RVectorF RTriangle::GetMin()
{
	float x = (v0.x < v1.x && v0.x < v2.x) ? v0.x : (v1.x < v2.x ? v1.x : v2.x);
	float y = (v0.y < v1.y && v0.y < v2.y) ? v0.y : (v1.y < v2.y ? v1.y : v2.y);
	float z = (v0.z < v1.z && v0.z < v2.z) ? v0.z : (v1.z < v2.z ? v1.z : v2.z);

	return RVectorF(x, y, z);
}


void RTriangle::ToString()
{
	v0.ToString();
	v1.ToString();
	v2.ToString();
	printf("\n");
}

RVectorF RTriangle::Centroid()
{
	const float a = (v0.x + v1.x + v2.x) / 3;
	const float b = (v0.y + v1.y + v2.y) / 3;
	const float c = (v0.z + v1.z + v2.z) / 3;

	return RVectorF(a, b, c);
}

__device__
bool RTriangle::FindIntersection(RRay * ray, float & t, float & u, float & v)
{
	//
	//RVectorF orig = ray->getRayOrigin();
	//RVectorF dir = ray->getRayDirection();
	//RVectorF v0v1 = A;
	//RVectorF v0v2 = B;
	//RVectorF pvec = dir.CrossProduct(v0v2);
	//float det = v0v1.DotProduct(pvec);

	//// ray and triangle are parallel if det is close to 0
	//if (fabs(det) < -kEpsilon) return false;

	//float invDet = 1 / det;

	//RVectorF tvec = orig.VectorAdd(v0.Negative());
	//u = tvec.DotProduct(pvec) * invDet;
	//if (u < 0 || u > 1) return false;

	//RVectorF qvec = tvec.CrossProduct(v0v1);
	//v = dir.DotProduct(qvec) * invDet;
	//if (v < 0 || u + v > 1) return false;

	//t = v0v2.DotProduct(qvec) * invDet;

	return (t > kEpsilon) ? true : false;
}

RVectorF RTriangle::GetNormalAt(RVectorF intersection_position)
{
	return A.CrossProduct(B).Normalize();
}
