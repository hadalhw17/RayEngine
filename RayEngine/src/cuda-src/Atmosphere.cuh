#pragma once
#include "helper_math.h"


class Atmosphere
{
public:
	__device__ __host__
		Atmosphere(
			float3 sd = make_float3(0, -1, 0),
			float er = 6360e3, float ar = 6420e3,
			float hr = 79945, float hm = 1200) :
		sunDirection(sd),
		earthRadius(er),
		atmosphereRadius(ar),
		Hr(hr),
		Hm(hm),
		betaM(make_float3(21e-6f)),
		betaR(make_float3(3.8e-6f, 13.5e-6f, 33.1e-6f))
	{}

	//__device__ __host__
	//float3 compute_incident_light(const float3 orig, const float3 dir, float tmin, float tmax);


	//__device__ __host__
	//bool solve_quadratic(const float &a, const float & b, const float & c, float & x0, float & x1);

	//__device__ __host__
	//bool ray_sphere_intersect(const float3 ray_o, const float3 ray_dir, float &t0, float &t1);

	float3 sunDirection;     // The sun direction (normalized) 
	float earthRadius;      // In the paper this is usually Rg or Re (radius ground, eart) 
	float atmosphereRadius; // In the paper this is usually R or Ra (radius atmosphere) 
	float Hr;               // Thickness of the atmosphere if density was uniform (Hr) 
	float Hm;               // Same as above but for Mie scattering (Hm) 


	const float3 betaR;
	const float3 betaM;

};
