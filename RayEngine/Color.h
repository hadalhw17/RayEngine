#pragma once
#include "RayEngine.h"

class RColor
{
public:
	//Constructors
	HOST_DEVICE_FUNCTION RColor();

	
	RColor(float r, float g, float b, float s);

	// method functions
	 //Getters
	HOST_DEVICE_FUNCTION
	float GetColorRed() { return Red; }

	HOST_DEVICE_FUNCTION
	float GetColorGreen() { return Green; }

	HOST_DEVICE_FUNCTION
	float GetColorBlue() { return Blue; }

	HOST_DEVICE_FUNCTION
	float GetColorSpecial() { return Special; }

	HOST_DEVICE_FUNCTION
	float Brightness();

	//Setters
	HOST_DEVICE_FUNCTION
	void SetColorRed(float RedValue) { Red = RedValue; }

	HOST_DEVICE_FUNCTION
	void SetColorGreen(float greenValue) { Green = greenValue; }

	HOST_DEVICE_FUNCTION
	void SetColorBlue(float blueValue) { Blue = blueValue; }

	HOST_DEVICE_FUNCTION
	void SetColorSpecial(float specialValue) { Special = specialValue; }

	//Operations
	HOST_DEVICE_FUNCTION
	RColor ColorScalar(float scalar);

	HOST_DEVICE_FUNCTION
	RColor ColorAdd(RColor Color);

	HOST_DEVICE_FUNCTION
	RColor ColorMultiply(RColor Color);

	HOST_DEVICE_FUNCTION
	RColor ColorAverage(RColor Color);

	HOST_DEVICE_FUNCTION
	RColor Clip();

	float Red, Green, Blue, Special;
private:
	
};

