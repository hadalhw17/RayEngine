#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cuda_runtime_api.h"

#include "../Camera.h"
#include "../RayEngine/RayEngine.h"
#include "helper_math.h"
#include "../GPUBoundingBox.h"
#include "Atmosphere.cuh"

// transform vector by matrix (no translation)
__device__
float2 mul(const float2x2& M, const float2& v)
{
	float2 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	return r;
}


// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4& M, const float3& v)
{
	float3 r;
	r.x = dot(v, make_float3(M.m[0]));
	r.y = dot(v, make_float3(M.m[1]));
	r.z = dot(v, make_float3(M.m[2]));
	return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4& M, const float4& v)
{
	float4 r;
	r.x = dot(v, M.m[0]);
	r.y = dot(v, M.m[1]);
	r.z = dot(v, M.m[2]);
	r.w = 1.0f;
	return r;
}


inline __device__ 
uint rgbaFloatToInt(float3 &rgba)
{
	rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
	rgba.y = __saturatef(rgba.y);
	rgba.z = __saturatef(rgba.z);
	float a = 1.f;
	return (uint(a * 255) << 24) | (uint(rgba.z * 255) << 16) | (uint(rgba.y * 255) << 8) | uint(rgba.x * 255);
}

inline __device__
float3 mix(const float3 &target, const float3 &second, const float &t)
{
	return make_float3(__saturatef(target.x + (t) * second.x),
		__saturatef(target.y + (t) * second.y),
		__saturatef(target.z + (t) * second.z));
}

inline __device__
float3 sqrtf(const float3 &target)
{
	return make_float3(sqrtf(target.x),
		sqrtf(target.y),
		sqrtf(target.z));
}

////////////////////////////////////////////////////
// Compute light intensity
////////////////////////////////////////////////////
HOST_DEVICE_FUNCTION
void illuminate(const float3& P, const float3 &light_pos, float3& lightDir, float3& lightIntensity, float& distance,const float &intensity)
{
	// Return not to devide by zero.
	if (distance == 0)
		distance = K_EPSILON;

	lightDir = P - light_pos;

	float r2 = light_pos.x * light_pos.x + light_pos.y * light_pos.y + light_pos.z * light_pos.z;
	distance = sqrtf(r2);
	lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
	lightIntensity = make_float3(0.86, 0.80, 0.45) * intensity / (4 * M_PI * r2);
}


////////////////////////////////////////////////////
// Clip color
////////////////////////////////////////////////////
__device__
float3 clip(const float3 &color)
{
	//float Red = color.x, Green = color.y, Blue = color.z, special = color.w;
	//float alllight = color.x + color.y + color.z;
	//float excesslight = alllight - 3;
	//if (excesslight > 0) {
	//	Red = Red + excesslight * (Red / alllight);
	//	Green = Green + excesslight * (Green / alllight);
	//	Blue = Blue + excesslight * (Blue / alllight);
	//}
	//if (Red > 1) { Red = 1; }
	//if (Green > 1) { Green = 1; }
	//if (Blue > 1) { Blue = 1; }
	//if (Red < 0) { Red = 0; }
	//if (Green < 0) { Green = 0; }
	//if (Blue < 0) { Blue = 0; }

	//return make_float4(Red, Green, Blue, special);
	return make_float3(__saturatef(color.x),
		__saturatef(color.y),
		__saturatef(color.z));
}


////////////////////////////////////////////////////
// Normal visualisation material
////////////////////////////////////////////////////
__forceinline__ HOST_DEVICE_FUNCTION
void simple_shade(float3& color, const float3 &normal, float3 &ray_dir)
{
	color += make_float3(fmaxf(0.f, dot(normal, -ray_dir) / 2)); // facing ratio 
}

////////////////////////////////////////////////////
// Sky material represent ray directions
////////////////////////////////////////////////////
HOST_DEVICE_FUNCTION
void sky_mat(float3& color, const float3 &ray_dir)
{
	//// Visualise ray directions on the sky.
	//color = make_float4(ray_dir, 0);
	//color.x = (color.x < 0.0f) ? (color.x * -1.0f) : color.x;
	//color.y = (color.y < 0.0f) ? (color.y * -1.0f) : color.y;
	//color.z = (color.z < 0.0f) ? (color.z * -1.0f) : color.z;

	float t = 0.5f * (ray_dir.y + 1.f);
	color = make_float3(1.f) - t * make_float3(1.f) = t * make_float3(0.5f, 0.7f, 1.f);
}

////////////////////////////////////////////////////
// std::swap()
////////////////////////////////////////////////////
template <typename T>
__forceinline__ HOST_DEVICE_FUNCTION
void swap(T& a, T& b)
{
	T c(a); a = b; b = c;
}


__forceinline__ HOST_DEVICE_FUNCTION
float vmax(const float3 &v)
{
	return fmaxf(fmaxf(v.x, v.y), v.z);
}

__forceinline__ HOST_DEVICE_FUNCTION
float3 max(const float3 &a, const float3 &b)
{
	return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

__forceinline__ HOST_DEVICE_FUNCTION
float3 min(const float3 &a, const float3 &b)
{
	return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}


__forceinline__ HOST_DEVICE_FUNCTION
void gray_scale(float3& color)
{
	color = make_float3((0.3 * color.x) + (0.59 * color.y) + (0.11 * color.z));

}


HOST_DEVICE_FUNCTION
void createCoordinateSystem(const float3& N, float3& Nt, float3& Nb)
{
	if (fabs(N.x) > fabs(N.y))
		Nt = make_float3(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
	else
		Nt = make_float3(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
	Nb = cross(N, Nt);
}

HOST_DEVICE_FUNCTION
float3 uniformSampleHemisphere(const float& r1, const float& r2)
{
	// cos(theta) = u1 = y
	// cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
	float sinTheta = sqrtf(1 - r1 * r1);
	float phi = 2 * M_PI * r2;
	float x = sinTheta * cosf(phi);
	float z = sinTheta * sinf(phi);
	return make_float3(x, r1, z);
}

// NVIDIA's
// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ 
void selection_sort(float2* data, int left, int right)
{
	for (int i = left; i <= right; ++i) {
		float2 min_val = data[i];
		int min_idx = i;

		// Find the smallest value in the range [left, right].
		for (int j = i + 1; j <= right; ++j) {
			float2 val_j = data[j];
			if (val_j.y < min_val.y) {
				min_idx = j;
				min_val = val_j;
			}
		}

		// Swap the values.
		if (i != min_idx) {
			data[min_idx] = data[i];
			data[i] = min_val;
		}
	}
}