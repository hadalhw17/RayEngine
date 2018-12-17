#pragma once
#include "Source.h"


class RLight : public RSource
{
	float3 position;
	float4 color;

public:
	float intensity = 500;

	HOST_DEVICE_FUNCTION RLight();
	
	HOST_DEVICE_FUNCTION
	RLight(float3 p, float4 c);

	// method functions
	HOST_DEVICE_FUNCTION
	inline virtual float3 getLightPosition() { return position; }

	HOST_DEVICE_FUNCTION
	inline virtual float4 getLightColor() { return color; }

	HOST_DEVICE_FUNCTION
	virtual void Illuminate(float3 &P, float3 &lightDir, float4 &lightIntensity, float &distance);

};

