#pragma once
#include "RayEngine.h"
#include "helper_math.h"
#include "cuda_helper_functions.h"
#include "cuda_memory_functions.cuh"
#include "ray_functions.cuh"
#include "sphere_tracing.cuh"



struct GPUScene
{
	GPUBoundingBox volume;
	float3 spacing;
	int3 dim;

	GPUScene()
	{

	}
};

template<typename T>
__device__
T bilinear(
	const float& tx,
	const float& ty,
	const T& c00,
	const T& c10,
	const T& c01,
	const T& c11)
{
#if 1 
	T a = c00 * (1.f - tx) + c10 * tx;
	T b = c01 * (1.f - tx) + c11 * tx;
	return a * (1.f - ty) + b * ty;
#else 
	return (1 - tx) * (1 - ty) * c00 +
		tx * (1 - ty) * c10 +
		(1.f - tx) * ty * c01 +
		tx * ty * c11;
#endif 
}


template<typename T>
__device__
T interpolate(cudaTextureObject_t tex, const float3& location)
{
	float gx, gy, gz, tx, ty, tz;
	unsigned gxi, gyi, gzi;
	// remap point coordinates to grid coordinates
	gx = location.x; gxi = int(gx); tx = gx - gxi;
	gy = location.y; gyi = int(gy); ty = gy - gyi;
	gz = location.z; gzi = int(gz); tz = gz - gzi;
	const T c000 = tex3D<T>(tex, gxi, gyi, gzi);
	const T c100 = tex3D<T>(tex, gxi + 1, gyi, gzi);
	const T c010 = tex3D<T>(tex, gxi, gyi + 1, gzi);
	const T c110 = tex3D<T>(tex, gxi + 1, gyi + 1, gzi);
	const T c001 = tex3D<T>(tex, gxi, gyi, gzi + 1);
	const T c101 = tex3D<T>(tex, gxi + 1, gyi, gzi + 1);
	const T c011 = tex3D<T>(tex, gxi, gyi + 1, gzi + 1);
	const T c111 = tex3D<T>(tex, gxi + 1, gyi + 1, gzi + 1);
#if 1
	T e = bilinear<T>(tx, ty, c000, c100, c010, c110);
	T f = bilinear<T>(tx, ty, c001, c101, c011, c111);
	return e * (1 - tz) + f * tz;
#else 
	return
		(T(1) - tx) * (T(1) - ty) * (T(1) - tz) * c000 +
		tx * (T(1) - ty) * (T(1) - tz) * c100 +
		(T(1) - tx) * ty * (T(1) - tz) * c010 +
		tx * ty * (T(1) - tz) * c110 +
		(T(1) - tx) * (T(1) - ty) * tz * c001 +
		tx * (T(1) - ty) * tz * c101 +
		(T(1) - tx) * ty * tz * c011 +
		tx * ty * tz * c111;
#endif 
}


__device__
GPUScene* d_scene;

GPUScene scene;

__forceinline__ HOST_DEVICE_FUNCTION
float sdf_smin(float a, float b, float k = 32)
{
	float res = exp(-k * a) + exp(-k * b);
	return -log(max(0.0001, res)) / k;
}

// Tricubic interpolated texture lookup, using unnormalized coordinates.
// Straight forward implementation, using 64 nearest neighbour lookups.
// @param tex  3D texture
// @param coord  unnormalized 3D texture coordinate

inline __host__ __device__
float bspline(float t)
{
	t = fabs(t);
	const float a = 2.0f - t;

	if (t < 1.0f) return 2.0f / 3.0f - 0.5f * t * t * a;
	else if (t < 2.0f) return a * a* a / 6.0f;
	else return 0.0f;
}



__device__
float cubicTex3DSimple(cudaTextureObject_t	tex, float3 coord)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	float3 index = floorf(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0f;
	for (float z = -1; z < 2.5f; z++)  //range [-1, 2]
	{
		float bsplineZ = bspline(z - fraction.z);
		float w = index.z + z;
		for (float y = -1; y < 2.5f; y++)
		{
			float bsplineYZ = bspline(y - fraction.y) * bsplineZ;
			float v = index.y + y;
			for (float x = -1; x < 2.5f; x++)
			{
				float bsplineXYZ = bspline(x - fraction.x) * bsplineYZ;
				float u = index.x + x;
				result += bsplineXYZ * tex3D<float>(tex, u, v, w);
			}
		}
	}
	return result;
}



template<class T> inline
__device__
void bspline_weights(T fraction, T & w0, T & w1, T & w2, T & w3)
{
	const T one_frac = 1.0f - fraction;
	const T squared = fraction * fraction;
	const T one_sqd = one_frac * one_frac;

	w0 = 1.0f / 6.0f * one_sqd * one_frac;
	w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
	w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
	w3 = 1.0f / 6.0f * squared * fraction;
}


__device__
float CUBICTEX3D(cudaTextureObject_t tex, float3 coord)
{
	// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	const float3 index = floorf(coord_grid);
	const float3 fraction = coord_grid - index;
	float3 w0, w1, w2, w3;
	bspline_weights<float3>(fraction, w0, w1, w2, w3);

	const float3 g0 = w0 + w1;
	const float3 g1 = w2 + w3;
	const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
	const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

	// fetch the eight linear interpolations
	// weighting and fetching is interleaved for performance and stability reasons
	float tex000 = tex3D<float>(tex, h0.x, h0.y, h0.z);
	float tex100 = tex3D<float>(tex, h1.x, h0.y, h0.z);
	tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
	float tex010 = tex3D<float>(tex, h0.x, h1.y, h0.z);
	float tex110 = tex3D<float>(tex, h1.x, h1.y, h0.z);
	tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
	tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
	float tex001 = tex3D<float>(tex, h0.x, h0.y, h1.z);
	float tex101 = tex3D<float>(tex, h1.x, h0.y, h1.z);
	tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
	float tex011 = tex3D<float>(tex, h0.x, h1.y, h1.z);
	float tex111 = tex3D<float>(tex, h1.x, h1.y, h1.z);
	tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
	tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

	return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
}


__device__
__forceinline__
float get_distance(RenderingSettings render_settings, cudaTextureObject_t tex,  float3 r_orig, float3 step, GPUVolumeObjectInstance instance)
{
	float dist;
	switch (render_settings.quality)
	{
	case LOW:
		dist = tex3D<float>(tex, r_orig.x / step.x, r_orig.y / step.y, r_orig.z / step.z);
		break;
	case MEDIUM:
		dist = interpolate<float>(tex, r_orig / step);
		break;
	case HIGH:
		dist = cubicTex3DSimple(tex, r_orig / step);
		break;
	default:
		break;
	}
	
	return dist;
}


HOST_DEVICE_FUNCTION
__forceinline__
float sphere_distance(float3 p, float radius)
{
	return length(p) - radius;
}

HOST_DEVICE_FUNCTION
__forceinline__
float aabb_distance(float3 p,  float3 s)
{
	float3 d = fabs(p) - s;
	return length(max(d, make_float3(0.0)))
		+ min(max(d.x, max(d.y, d.z)), 0.0); // remove this line for an only partially signed sdf
}



HOST_DEVICE_FUNCTION
__forceinline__
float get_distance_scene(float3 r_orig, GPUVolumeObjectInstance instance, GPUBoundingBox * volumes)
{

	float min_dist_to_sdf = sphere_distance(r_orig, 2);

	return min_dist_to_sdf;
}


__device__
float get_distance_to_sdf(RenderingSettings render_settings, cudaTextureObject_t tex, GPUScene scene, GPUBoundingBox box, float3 from, HitResult hit_result, float3 step, int3 dim, GPUVolumeObjectInstance instance, float curr_t)
{
	float min_dist = K_INFINITY;
	float t_near = K_INFINITY, t_far;
	bool intersect_box = false;

	intersect_box = gpu_ray_box_intersect(box, from, hit_result.ray_dir, t_near, t_far);


	if (intersect_box)
	{

		float min_dist_to_sdf = get_distance(render_settings, tex, from, step, instance);
		min_dist = t_near + min_dist_to_sdf;
	}
	else
	{
		min_dist = K_INFINITY;
	}

	return min_dist;
}

__device__
float3 compute_sdf_normal(float3 p_hit, float t, RenderingSettings render_settings, cudaTextureObject_t tex, float3 step, GPUVolumeObjectInstance curr_obj)
{
	float delta = fmaxf(0.002, 10e-6 * t);
	float curr_dist = get_distance(render_settings, tex, p_hit, step, curr_obj);
	return normalize(make_float3(
		get_distance(render_settings, tex, p_hit + make_float3(delta, 0, 0), step, curr_obj) - curr_dist,
		get_distance(render_settings, tex, p_hit + make_float3(0, delta, 0), step, curr_obj) - curr_dist,
		get_distance(render_settings, tex, p_hit + make_float3(0, 0, delta), step, curr_obj) - curr_dist));
}