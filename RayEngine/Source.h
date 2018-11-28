#pragma once
#include "Vector.h"
#include "Color.h"

using RVectorF = RVector<float>;

class RSource
{
public:
	float intensity;
	HOST_DEVICE_FUNCTION RSource();

	HOST_DEVICE_FUNCTION
	inline virtual RVectorF getLightPosition() { return RVectorF(0.f, 0.f, 0.f); }

	HOST_DEVICE_FUNCTION
	inline virtual RColor getLightColor() { return RColor(1.f, 1.f, 1.f, 0.f); }

	HOST_DEVICE_FUNCTION
	inline virtual void Illuminate(RVectorF &P, RVectorF &lightDir, RColor &lightIntensity, float &distance) {}
};

