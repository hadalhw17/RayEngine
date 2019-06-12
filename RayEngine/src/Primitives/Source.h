#pragma once


extern float3 make_float3(float);
extern float4 make_float4(float);
class RSource
{
public:
	float intensity;
	HOST_DEVICE_FUNCTION RSource();

	HOST_DEVICE_FUNCTION
	inline virtual float3 getLightPosition() { return make_float3(0); }

	HOST_DEVICE_FUNCTION
	inline virtual float4 getLightColor() { return make_float4(0); }

	HOST_DEVICE_FUNCTION
	inline virtual void Illuminate(float3 &P, float3 &lightDir, float4 &lightIntensity, float &distance) {}
};

