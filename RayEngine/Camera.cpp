#include "Camera.h"

#include "cutil_math.h"
#include "cuda_runtime.h"



RCamera::RCamera()
{
	campos = make_float3(0, 0, 0);
	camdir = make_float3(0, 0, 1);
	camright = make_float3(0, 0, 0);
	camdown = make_float3(0, -1, 0);
	
}

RCamera::RCamera(float3 pos, float3 dir, float3 right, float3 down)
{
	campos = make_float3(0, 0, 0);
	camdir = make_float3(0.2, 0, 0);
	camdown = make_float3(0, 1, 0);
	camright = cross(camdir, camdown);
	
}

void RCamera::move_forward()
{
	campos += camdir * MOVEMENT_SPEED;
}

void RCamera::move_backward()
{
	campos -= camdir * MOVEMENT_SPEED;
}

void RCamera::strafe_right()
{
	campos += camright * MOVEMENT_SPEED;
}

void RCamera::strafe_left()
{
	campos -= camright * MOVEMENT_SPEED;
}

void RCamera::lookAt(float3 target, float roll)
{
	camdown.x = sinf(roll * 3.1415926535f / 180.0f);
	camdown.y = cosf(roll * 3.1415926535f / 180.0f);
	camdown.z = 0.0f;

	camdir = (target - campos) * (1.0f / (target - campos));
	camright = cross(camdown, camdir) * (1.0f / cross(camdown, camdir));
	camdown = cross(camdir, camright) * (1.0f / cross(camdir, camright));
}
