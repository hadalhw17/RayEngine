#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cuda_runtime_api.h"

#include "Camera.h"
#include "RayEngine.h"
#include "helper_math.h"
#include "GPUBoundingBox.h"
#include "Atmosphere.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

////////////////////////////////////////////////////
// Compute light intensity
////////////////////////////////////////////////////
HOST_DEVICE_FUNCTION
void illuminate(float3& P, float3 light_pos, float3& lightDir, float4& lightIntensity, float& distance)
{
	// Return not to devide by zero.
	if (distance == 0)
		return;

	lightDir = P - light_pos;

	float r2 = light_pos.x * light_pos.x + light_pos.y * light_pos.y + light_pos.z * light_pos.z;
	distance = sqrtf(r2);
	lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
	lightIntensity = make_float4(0.86, 0.80, 0.45, 1) * 2500 / (4 * M_PI * r2);
}


////////////////////////////////////////////////////
// Clip color
////////////////////////////////////////////////////
HOST_DEVICE_FUNCTION
float4 clip(float4 color)
{
	float Red = color.x, Green = color.y, Blue = color.z, special = color.w;
	float alllight = color.x + color.y + color.z;
	float excesslight = alllight - 3;
	if (excesslight > 0) {
		Red = Red + excesslight * (Red / alllight);
		Green = Green + excesslight * (Green / alllight);
		Blue = Blue + excesslight * (Blue / alllight);
	}
	if (Red > 1) { Red = 1; }
	if (Green > 1) { Green = 1; }
	if (Blue > 1) { Blue = 1; }
	if (Red < 0) { Red = 0; }
	if (Green < 0) { Green = 0; }
	if (Blue < 0) { Blue = 0; }

	return make_float4(Red, Green, Blue, special);
}


////////////////////////////////////////////////////
// Normal visualisation material
////////////////////////////////////////////////////
__forceinline__ HOST_DEVICE_FUNCTION
void simple_shade(float4& color, float3 normal, float3 ray_dir)
{
	color += make_float4(fmaxf(0.f, dot(normal, -ray_dir) / 2)); // facing ratio 
}

////////////////////////////////////////////////////
// Sky material represent ray directions
////////////////////////////////////////////////////
HOST_DEVICE_FUNCTION
void sky_mat(float4& color, float3 ray_dir)
{
	//// Visualise ray directions on the sky.
	//color = make_float4(ray_dir, 0);
	//color.x = (color.x < 0.0f) ? (color.x * -1.0f) : color.x;
	//color.y = (color.y < 0.0f) ? (color.y * -1.0f) : color.y;
	//color.z = (color.z < 0.0f) ? (color.z * -1.0f) : color.z;

	float t = 0.5f * (ray_dir.y + 1.f);
	color = make_float4(1.f) - t * make_float4(1.f) = t * make_float4(0.5f, 0.7f, 1.f, 0.f);
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
float vmax(float3 v)
{
	return fmaxf(fmaxf(v.x, v.y), v.z);
}

__forceinline__ HOST_DEVICE_FUNCTION
float3 max(float3 a, float3 b)
{
	return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

__forceinline__ HOST_DEVICE_FUNCTION
float3 min(float3 a, float3 b)
{
	return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}


__forceinline__ HOST_DEVICE_FUNCTION
void gray_scale(float4& color)
{
	color = make_float4((0.3 * color.x) + (0.59 * color.y) + (0.11 * color.z));

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