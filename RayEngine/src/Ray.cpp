#include "repch.h"

#include "Ray.h"
#include "cuda_runtime.h"



	RRay::RRay()
	{
		origin = make_float3(0, 0, 0);
		direction = make_float3(1, 0, 0);
	}
