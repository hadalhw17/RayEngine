#pragma once
#include "Vector.h"

using RVectorF = RVector<float>;

class RCamera
{
	const float MOVEMENT_SPEED = 1;

public:

	float3 campos, camdir, camright, camdown;
	HOST_DEVICE_FUNCTION RCamera();

	HOST_DEVICE_FUNCTION
	RCamera(float3 pos, float3 dir, float3 right, float3 down);

	// method functions
	HOST_DEVICE_FUNCTION
	inline float3 getCameraPosition() { return campos; }

	HOST_DEVICE_FUNCTION
	inline float3 getCameraDirection() { return camdir; }

	HOST_DEVICE_FUNCTION
	inline float3 getCameraRight() { return camright; }

	HOST_DEVICE_FUNCTION
	inline float3 getCameraDown() { return camdown; }

	void lookAt(float3 target, float roll);

	void move_forward();
	void move_backward();
	void strafe_right();
	void strafe_left();

};

