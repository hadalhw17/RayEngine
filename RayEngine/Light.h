#pragma once
#include "Source.h"
#include "Vector.h"
#include "Color.h"


using RVectorF = RVector<float>;

class RLight : public RSource
{
	RVectorF position;
	RColor color;

public:
	float intensity = 500;

	HOST_DEVICE_FUNCTION RLight();
	
	HOST_DEVICE_FUNCTION
	RLight(RVectorF p, RColor c);

	// method functions
	HOST_DEVICE_FUNCTION
	inline virtual RVectorF getLightPosition() { return position; }

	HOST_DEVICE_FUNCTION
	inline virtual RColor getLightColor() { return color; }

	HOST_DEVICE_FUNCTION
	virtual void Illuminate(RVectorF &P, RVectorF &lightDir, RColor &lightIntensity, float &distance);

};

