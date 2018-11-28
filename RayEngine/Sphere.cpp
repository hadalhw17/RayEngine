#include "Sphere.h"
#include "Color.h"
#include "Ray.h"

#include <memory>
#include <thrust/swap.h>



RSphere::RSphere()
{
	center = RVectorF(0, 0, 0);
	radius = 1.0;
	color = RColor(0.5, 0.5, 0.5, 0);
}

RSphere::RSphere(RVectorF centerValue, float radiusValue, RColor colorValue)
{
	center = centerValue;
	radius = radiusValue;
	color = colorValue;
}

bool RSphere::SolveQuadratic(const float & a, const float & b, const float & c, float & x0, float & x1)
{
	float discr = b * b - 4 * a * c;
	if (discr < 0) return false;
	else if (discr == 0) {
		x0 = x1 = -0.5 * b / a;
	}
	else {
		float q = (b > 0) ?
			-0.5 * (b + sqrt(discr)) :
			-0.5 * (b - sqrt(discr));
		x0 = q / a;
		x1 = c / q;
	}

	return true;
}

bool RSphere::SolveQuadratic(float & a, float & b, float & c, float & x0, float & x1)
{
	float discr = b * b - 4 * a * c;
	if (discr < 0) return false;
	else if (discr == 0) {
		x0 = x1 = -0.5 * b / a;
	}
	else {
		float q = (b > 0) ?
			-0.5 * (b + sqrt(discr)) :
			-0.5 * (b - sqrt(discr));
		x0 = q / a;
		x1 = c / q;
	}

	return true;
}

RVectorF RSphere::GetNormalAt(RVectorF point)
{
	// normal always points away from the center of a sphere
	RVectorF sphereNormal = point.VectorAdd(center.Negative());
	return sphereNormal.Normalize();
}

bool RSphere::FindIntersection(RRay * ray, float & t, float & u, float & v)
{
	//float3 orig = ray->getRayOrigin();
	//float3 dir = ray->getRayDirection();
	//float t0, t1; // solutions for t if the ray intersects
	//// analytic solution
	//float3 L = orig.VectorAdd(center.Negative());
	//float a = dir.DotProduct(dir);
	//float b = 2 * dir.DotProduct(L);
	//float radius2 = pow(radius, 2);
	//float c = L.DotProduct(L) - radius2;
	//if (!SolveQuadratic(a, b, c, t0, t1)) return false;

	//if (t0 > t1) thrust::swap(t0, t1);

	//if (t0 < -1e-5) {
	//	t0 = t1; // if t0 is negative, let's use t1 instead
	//	if (t0 < -1e-5) return false; // both t0 and t1 are negative
	//}

	//t = t0;

	return true;
}
