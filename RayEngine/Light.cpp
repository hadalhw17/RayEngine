#include "Light.h"

#define M_PI       3.14159265358979323846   // pi

RLight::RLight()
{
	position = RVectorF(0, 0, 0);
	color = RColor(1, 1, 1, 0);
}

RLight::RLight(RVectorF p, RColor c)
{
	position = p;
	color = c;
}

void RLight::Illuminate(RVectorF & P, RVectorF & lightDir, RColor & lightIntensity, float & distance)
{
	lightDir = (P.VectorAdd(position.Negative()));
	float r2 = lightDir.Norm();
	distance = sqrt(r2);
	lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
	// avoid division by 0
	lightIntensity = color.ColorScalar(intensity / (4 * M_PI * r2));
}
