#include "Plane.h"
#include "Color.h"
#include "Ray.h"
#include <cmath>



RPlane::RPlane()
{
	normal = RVectorF(1, 0, 0);
	distance = 0;
	Color = RColor(0.5, 0.5, 0.5, 0);
}

RPlane::RPlane(RVectorF normalValue, float distanceValue, RColor ColorValue)
{
	normal = normalValue;
	distance = distanceValue;
	Color = ColorValue;
}

bool RPlane::FindIntersection(RRay * ray, float & t, float & u, float & v)
{
	//RVectorF ray_direction = ray->getRayDirection();
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
