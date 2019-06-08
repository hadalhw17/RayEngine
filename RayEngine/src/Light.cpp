#include "Light.h"
#include "helper_math.h"


RLight::RLight()
{
	position = make_float3(0, 0, 0);
	color = make_float4(1, 1, 1, 0);
}

RLight::RLight(float3 p, float4 c)
{
	position = p;
	color = c;
}

void RLight::Illuminate(float3 & P, float3 & lightDir, float4 & lightIntensity, float & distance)
{
	lightDir = P - position;
	float r2 = position.x * position.x + position.y * position.y + position.z * position.z;
	distance = sqrt(r2);
	lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
	// avoid division by 0
	lightIntensity = color * (intensity / (4 * M_PI * r2));
}
