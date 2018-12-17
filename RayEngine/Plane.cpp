#include "Plane.h"
#include "Ray.h"
#include <cmath>



RPlane::RPlane()
{
	normal = make_float3(1, 0, 0);
	distance = 0;
	Color = make_float4(0.5, 0.5, 0.5, 0);
}

RPlane::RPlane(float3 normalValue, float distanceValue, float4 ColorValue)
{
	normal = normalValue;
	distance = distanceValue;
	Color = ColorValue;
}

bool RPlane::FindIntersection(RRay * ray, float & t, float & u, float & v)
{
	//float3 ray_direction = ray->getRayDirection();
	//float a = normal.DotProduct(ray_direction);

	//if (fabs(a) == 1e-6) {
	//	// ray is parallel to the plane
	//	return false;
	//}
	//else {
	//	float b = ray->getRayOrigin().VectorAdd(normal.VectorMult(distance).Negative()).DotProduct(normal);
	//	t = -1 * b / a;
	//	return (t >= 1e-6);
	//}
	return false;
}
