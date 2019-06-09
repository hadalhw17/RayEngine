#include "Camera.h"

#include "helper_math.h"

	RCamera::RCamera() :
		campos(make_float3(0)),
		camdir(make_float3(0)),
		camright(make_float3(0)),
		camdown(make_float3(0)),
		lookat(make_float3(0)),
		view(make_float3(0))
	{
	}
