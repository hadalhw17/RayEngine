#pragma once

#include "cuda_runtime_api.h"


	struct RCamera
	{
		RCamera();
		RCamera(float3 pos, float3 dir, float3 right, float3 down) :
			campos(pos), camdir(dir), camright(right), camdown(down) {}
		float3 campos, camdir, camright, camdown, lookat, view;
		float2 fov;
		float focial_distance, apertude_radius;
	};

