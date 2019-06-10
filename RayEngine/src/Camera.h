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


#include <Meta.h>
	namespace meta {

		template <>
		inline auto registerMembers<RCamera>()
		{
			return members(
				member("campos", &RCamera::campos),
				member("camdir", &RCamera::camdir),
				member("camright", &RCamera::camright),
				member("camdown", &RCamera::camdown),
				member("lookat", &RCamera::lookat),
				member("fov", &RCamera::fov),
				member("focial_distance", &RCamera::focial_distance),
				member("apertude_radius", &RCamera::apertude_radius),
				member("view", &RCamera::view)
			);
		}

	} // end of namespace meta
