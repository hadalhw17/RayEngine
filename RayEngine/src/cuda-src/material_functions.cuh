#pragma once

#include "Primitives/KDTreeGPUUtills.h"
#include "cuda_helper_functions.h"
#include "ray_functions.cuh"
#include "kd_tree_functions.cuh"


	////////////////////////////////////////////////////
	// Tile material
	////////////////////////////////////////////////////
	__device__
		void tile_pattern(float3& color, const int& square)
	{
		if ((square % 2) == 0) {
			// black tile
			color.x += 0;
			color.y += 0;
			color.z += 0;
		}
		else {
			// white tile
			color.x += 1;
			color.y += 1;
			color.z += 1;
		}
	}

	////////////////////////////////////////////////////
	// Normal visualisation material
	////////////////////////////////////////////////////
	__device__
		void narmals_mat(float3& color, const float3& normal)
	{
		color.x = (normal.x < 0.0f) ? (normal.x * -1.0f) : normal.x;
		color.y = (normal.y < 0.0f) ? (normal.y * -1.0f) : normal.y;
		color.z = (normal.z < 0.0f) ? (normal.z * -1.0f) : normal.z;
	}

	////////////////////////////////////////////////////
	// Ambient light
	////////////////////////////////////////////////////
	__device__
		void ambient_light(float3& color)
	{
		color += color * 0.2;
	}

	////////////////////////////////////////////////////
	// Fog
	////////////////////////////////////////////////////
	__device__
		void apply_fog(float3& color, const float& distance, const float& d, const float3& ray_o, const float3& ray_dir)
	{
		float a = 0.0375f;
		float fogAmount = __saturatef(a / d * __expf(-ray_o.y * d) * (1.0 - __expf(-distance * ray_dir.y * d)) / ray_dir.y);
		//fogAmount = __saturatef(fogAmount + (__expf(-(100.f - distance) * d)) - 0.1);
		//float fogAmount = 1.0 - __expf(-distance * d);
		//float fo = 1.0 - exp(-pow(0.001 * distance / d, 1.5));
		float3  fogColor = 0.65 * make_float3(0.4, 0.65, 1.0);
		color = mix(color, fogColor, fogAmount);

	}

	////////////////////////////////////////////////////
	// Phong light
	////////////////////////////////////////////////////
	__device__
		void phong_light(float3* lights, size_t num_lights, float3& finalColor, RKDTreeNodeGPU* tree,
			GPUSceneObject* scene_objs, int num_objs, int* root_index, int* index_list, HitResult& hit_result, HitResult& shadow_hit_result)
	{
		float3 bias = hit_result.normal * make_float3(-K_EPSILON);
		for (int i = 0; i < num_lights; ++i)
		{
			float3 diffuse = make_float3(0), specular = make_float3(0);
			float3 lightpos = lights[i], lightDir;
			float3 lightInt;
			float t = K_INFINITY;
			illuminate(hit_result.hit_point, lightpos, lightDir, lightInt, t, 20000);

			shadow_hit_result.ray_o = hit_result.hit_point + bias;
			shadow_hit_result.ray_dir = -lightDir;

			trace_shadow(tree, scene_objs, num_objs, root_index, index_list, shadow_hit_result);

			diffuse += lightInt * make_float3(shadow_hit_result.hits * 0.18) * fmaxf(0.0, dot(hit_result.normal, -lightDir));

			float3 R = reflect(lightDir, hit_result.normal);
			specular += lightInt * make_float3(shadow_hit_result.hits * powf(fmaxf(0.0, dot(R, -hit_result.ray_dir)), 10));

			finalColor += diffuse * 0.8 + specular * 0.2;
			shadow_hit_result = HitResult();
		}
	}

	////////////////////////////////////////////////////
	// Shade
	////////////////////////////////////////////////////
	__device__
		void shade(float3* lights, size_t num_lights, float3& finalColor, RKDTreeNodeGPU* tree, GPUSceneObject* scene_objs, int num_objs,
			int* root_index, int* index_list, HitResult& hit_result, HitResult& shading_hit_result)
	{
		float3 bias = hit_result.normal * make_float3(K_EPSILON);
		for (int i = 0; i < num_lights; ++i)
		{
			float3 lightpos = lights[i], lightDir;
			float3 lightInt;
			float3 lightColor = make_float3(1, 0, 0);
			float t = K_INFINITY;
			illuminate(hit_result.hit_point, lightpos, lightDir, lightInt, t, 20000);

			shading_hit_result.ray_o = hit_result.hit_point + bias;
			shading_hit_result.ray_dir = -lightDir;

			trace_shadow(tree, scene_objs, num_objs, root_index, index_list, shading_hit_result);
			finalColor += !shading_hit_result.hits * lightInt * fmaxf(.0f, dot(hit_result.normal, (-lightDir)));
		}
		shading_hit_result.hit_color = finalColor;
	}

	////////////////////////////////////////////////////
	// Compute fresnel equation
	////////////////////////////////////////////////////
	__device__
		void fresnel(float3& I, float3& N, float& ior, float& kr)
	{
		const float dot_p = dot(I, N);
		float cosi = clamp(-1.f, 1.f, dot_p);
		float etai = 1, etat = ior;
		if (cosi > 0) { swap(etai, etat); }
		// Compute sini using Snell's law.
		const float sint = etai / etat * sqrtf(fmaxf(0.f, 1 - cosi * cosi));
		// Total internal reflection.
		if (sint >= 1) {
			kr = 1;
		}
		else {
			const float cost = sqrtf(fmaxf(0.f, 1 - sint * sint));
			cosi = fabsf(cosi);
			const float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
			const float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
			kr = (Rs * Rs + Rp * Rp) / 2;
		}
	}


	////////////////////////////////////////////////////
	// Refract
	////////////////////////////////////////////////////
	__device__
		float3 refract(float3& I, float3& N, float& ior)
	{
		float dot_p = dot(I, N);
		float cosi = clamp(-1.f, 1.f, dot_p);
		float etai = 1, etat = ior;
		float3 n = N;
		if (cosi < 0) { cosi = -cosi; }
		else { swap(etai, etat); n = -N; }
		float eta = etai / etat;
		float k = 1 - eta * eta * (1 - cosi * cosi);
		return k < 0 ? make_float3(0) : eta * I * (eta * cosi - sqrtf(k)) * n;
	}

	__device__
		void reflect_refract(float3& finalColor, RKDTreeNodeGPU* tree, GPUSceneObject* scene_objs, int num_objs,
			int* root_index, int* index_list, HitResult& hit_result, HitResult& shading_hit_result)
	{
		float3 refractionRColor = make_float3(0), reflectionRColor = make_float3(0);
		float kr = 0, ior = 1.3;
		float3 direct = hit_result.ray_dir;
		bool outside = dot(direct, hit_result.normal) < 0;
		fresnel(direct, hit_result.normal, ior, kr);

		if (kr < 1)
		{
			float3 refractionDirection = refract(direct, hit_result.normal, ior);
			float3 refractionRayOrig = outside ? hit_result.hit_point - hit_result.normal * make_float3(K_EPSILON) : hit_result.hit_point + hit_result.normal * make_float3(K_EPSILON);
			HitResult refraction_result;
			refraction_result.ray_dir = refractionDirection;
			refraction_result.ray_o = refractionRayOrig;
			trace_shadow(tree, scene_objs, num_objs, root_index, index_list, refraction_result);

			if (refraction_result.hits)
			{
				refractionRColor += scene_objs[refraction_result.obj_index].material.color;
			}

			shading_hit_result = refraction_result;

		}
		HitResult reflection_result;
		float3 reflectionDirection = reflect(direct, hit_result.normal);
		float3 reflectionRayOrig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(K_EPSILON) : hit_result.hit_point - hit_result.normal * make_float3(K_EPSILON);
		reflection_result.ray_dir = reflectionDirection;
		reflection_result.ray_o = reflectionRayOrig;
		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_result);

		if (reflection_result.hits)
		{
			reflectionRColor += scene_objs[reflection_result.obj_index].material.color;
		}
		// mix the two
		finalColor += reflectionRColor * (kr)+refractionRColor * (1 - kr);
		//finalColor = clip(finalColor);
	}

	__device__
		void refract_light(float3& finalColor, RKDTreeNodeGPU* tree, GPUSceneObject* scene_objs, int num_objs,
			int* root_index, int* index_list, HitResult& hit_result, HitResult& shading_hit_result)
	{
		float3 refractionRColor = make_float3(0), reflectionRColor = make_float3(0);
		float kr = 0, ior = 1.3;
		float3 direct = hit_result.ray_dir;
		bool outside = dot(direct, hit_result.normal) < 0;
		fresnel(direct, hit_result.normal, ior, kr);

		if (kr < 1)
		{
			float3 refractionDirection = refract(direct, hit_result.normal, ior);
			float3 refractionRayOrig = outside ? hit_result.hit_point - hit_result.normal * make_float3(K_EPSILON) : hit_result.hit_point + hit_result.normal * make_float3(K_EPSILON);
			HitResult refraction_result;
			refraction_result.ray_dir = refractionDirection;
			refraction_result.ray_o = refractionRayOrig;
			trace_shadow(tree, scene_objs, num_objs, root_index, index_list, refraction_result);

			if (refraction_result.hits)
			{
				refractionRColor += refraction_result.hits * 2 * fmaxf(.0f, dot(hit_result.normal, (-refractionDirection)));
			}

			shading_hit_result = refraction_result;

		}
		HitResult reflection_result;
		float3 reflectionDirection = reflect(direct, hit_result.normal);
		float3 reflectionRayOrig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(K_EPSILON) : hit_result.hit_point - hit_result.normal * make_float3(K_EPSILON);
		reflection_result.ray_dir = reflectionDirection;
		reflection_result.ray_o = reflectionRayOrig;
		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_result);

		if (reflection_result.hits)
		{
			reflectionRColor += reflection_result.hits * 2 * fmaxf(.0f, dot(hit_result.normal, (-reflectionDirection)));
		}
		// mix the two
		finalColor = reflectionRColor * (kr)+refractionRColor * (1 - kr);
		//finalColor = clip(finalColor);
	}

	__device__
		void reflect(float3& finalColor, RKDTreeNodeGPU* tree, GPUSceneObject* scene_objs, int num_objs,
			int* root_index, int* index_list, HitResult& hit_result, HitResult& shading_hit_result)
	{
		float3 direct = hit_result.ray_dir;
		bool outside = dot(direct, hit_result.normal) < 0;
		float3 dir = reflect(direct, hit_result.normal);
		float3 orig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(K_EPSILON) : hit_result.hit_point - hit_result.normal * make_float3(K_EPSILON);
		HitResult reflection_hit_result;
		reflection_hit_result.ray_dir = dir;
		reflection_hit_result.ray_o = orig;
		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_hit_result);
		shading_hit_result = reflection_hit_result;
		if (reflection_hit_result.hits)
		{
			finalColor += 0.8 * scene_objs[reflection_hit_result.obj_index].material.color;
		}
	}

	__device__
		void reflect_light(float3& finalColor, RKDTreeNodeGPU* tree, GPUSceneObject* scene_objs, int num_objs,
			int* root_index, int* index_list, HitResult& hit_result, HitResult& shading_hit_result)
	{
		float3 direct = hit_result.ray_dir;
		bool outside = dot(direct, hit_result.normal) < 0;
		float3 dir = reflect(direct, hit_result.normal);
		float3 orig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(K_EPSILON) : hit_result.hit_point - hit_result.normal * make_float3(K_EPSILON);
		HitResult reflection_hit_result;
		reflection_hit_result.ray_dir = dir;
		reflection_hit_result.ray_o = orig;
		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_hit_result);
		shading_hit_result = reflection_hit_result;
		if (reflection_hit_result.hits)
		{
			finalColor += reflection_hit_result.hits * 10 * fmaxf(.0f, dot(hit_result.normal, (-dir)));
			//finalColor = make_float4((finalColor.x + finalColor.y + finalColor.z) / 3);
		}
	}


	////////////////////////////////////////////////////
	// Algorithm for computing color of sky sphere
	// Code is taken from:
	// https://www.scratchapixel.com/code.php?id=52&origin=/lessons/procedural-generation-virtual-worlds/simulating-sky
	////////////////////////////////////////////////////
	__device__
		float3 compute_incident_light(Atmosphere* armosphere, const float3 orig, const float3 dir, float tmin, float tmax)
	{
		float t0, t1;
		if (!gpu_ray_sphere_intersect(orig, dir, armosphere->atmosphereRadius, t0, t1) || t1 < 0) return make_float3(1);
		if (t0 > tmin && t0 > 0) tmin = t0;
		if (t1 < tmax) tmax = t1;
		size_t numSamples = 32;
		size_t numSamplesLight = 8;
		float segmentLength = (tmax - tmin) / numSamples;
		float tCurrent = tmin;
		float3 sumR = make_float3(0), sumM = make_float3(0); // mie and rayleigh contribution 
		float opticalDepthR = 0, opticalDepthM = 0;
		float mu = dot(dir, armosphere->sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction 
		float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
		float g = 0.76f;
		float phaseM = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));
		for (size_t i = 0; i < numSamples; ++i) {
			float3 samplePosition = orig + (tCurrent + segmentLength * 0.5f) * dir;
			float height = length(samplePosition) - armosphere->earthRadius;
			// compute optical depth for light
			float hr = __expf(-height / armosphere->Hr) * segmentLength;
			float hm = __expf(-height / armosphere->Hm) * segmentLength;
			opticalDepthR += hr;
			opticalDepthM += hm;
			// light optical depth
			float t0Light, t1Light;
			gpu_ray_sphere_intersect(samplePosition, armosphere->sunDirection, armosphere->atmosphereRadius, t0Light, t1Light);
			float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
			float opticalDepthLightR = 0, opticalDepthLightM = 0;
			size_t j;
			for (j = 0; j < numSamplesLight; ++j) {
				float3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * armosphere->sunDirection;
				float heightLight = length(samplePositionLight) - armosphere->earthRadius;
				if (heightLight < 0) break;
				opticalDepthLightR += __expf(-heightLight / armosphere->Hr) * segmentLengthLight;
				opticalDepthLightM += __expf(-heightLight / armosphere->Hm) * segmentLengthLight;
				tCurrentLight += segmentLengthLight;
			}
			if (j == numSamplesLight) {
				float3 tau = armosphere->betaR * (opticalDepthR + opticalDepthLightR) + armosphere->betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
				float3 attenuation = make_float3(__expf(-tau.x), __expf(-tau.y), __expf(-tau.z));
				sumR += attenuation * hr;
				sumM += attenuation * hm;
			}
			tCurrent += segmentLength;
		}

		return (sumR * armosphere->betaR * phaseR + sumM + 100 * armosphere->betaM * phaseM) * 20;
	}
