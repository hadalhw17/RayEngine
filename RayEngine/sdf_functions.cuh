#pragma once
#include "RayEngine.h"
#include "helper_math.h"
#include "cuda_helper_functions.h"
#include "cuda_memory_functions.cuh"
#include "ray_functions.cuh"
#include "sphere_tracing.cuh"

texture<float, cudaTextureType3D, cudaReadModeElementType> sdf_texture;

texture<float2, cudaTextureType3D, cudaReadModeElementType> scene_texture;

struct GPUScene
{
	GPUBoundingBox volume;
	float3 spacing;
	int3 dim;

	GPUScene()
	{

	}
};

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
float cubicTex3DSimple(float3 coord)
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
				result += bsplineXYZ * tex3D(sdf_texture, u, v, w);
			}
		}
	}
	return result;
}

__device__
float2 cubicTex3DSimple_scene(float3 coord)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	float3 index = floorf(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0f;
	float obj;
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
				float2 tex = tex3D(scene_texture, u, v, w);
				result += bsplineXYZ * tex.x;
				obj = tex.y;
			}
		}
	}
	return make_float2(result, obj);
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
float CUBICTEX3D(float3 coord)
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
	float tex000 = tex3D(sdf_texture, h0.x, h0.y, h0.z);
	float tex100 = tex3D(sdf_texture, h1.x, h0.y, h0.z);
	tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
	float tex010 = tex3D(sdf_texture, h0.x, h1.y, h0.z);
	float tex110 = tex3D(sdf_texture, h1.x, h1.y, h0.z);
	tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
	tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
	float tex001 = tex3D(sdf_texture, h0.x, h0.y, h1.z);
	float tex101 = tex3D(sdf_texture, h1.x, h0.y, h1.z);
	tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
	float tex011 = tex3D(sdf_texture, h0.x, h1.y, h1.z);
	float tex111 = tex3D(sdf_texture, h1.x, h1.y, h1.z);
	tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
	tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

	return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
}


__device__
__forceinline__
float get_distance(float3 r_orig, float3* step, int3* dim, GPUVolumeObjectInstance instance)
{
	int offset = (instance.index * (dim[instance.index].x - 1));
	float dist = cubicTex3DSimple(make_float3(offset + ((float)r_orig.x / (float)step[instance.index].x),
		(float)r_orig.y / (float)step[instance.index].y,
		(float)r_orig.z / (float)step[instance.index].z));
	//if(dist < 0.f)
	//	printf("%f \n", dist);
	return dist;
}

__device__
__forceinline__
float2 get_distance_scene(float2 * scene_tex, GPUScene scene, float3 r_orig, float3 step, int3 dim)
{
	// TODO: Fix this!
	//float2 dist = tex3D(scene_texture, r_orig.x / step.x, r_orig.y / step.y, r_orig.z / step.z);
	float2 dist = cubicTex3DSimple_scene(r_orig / step);
	//printf("%f\n", dist.y);
	//if(dist < 0.f)
	//	printf("%f \n", dist);
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
float aabb_distance(float3 p, float3 c, float3 s)
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
float2 get_scene_sdf(GPUScene scene, float3 from, HitResult hit_result, float curr_t, float& t_near, float& t_far, float2 * scene_tex)
{
	float2 res;
	float nearest_object = -1;

	float min_dist = K_INFINITY;
	bool intersect_box = false;

	intersect_box = gpu_ray_box_intersect(scene.volume, from, hit_result.ray_dir, t_near, t_far);


	if (intersect_box)
	{
		//printf("%f \n", scene.volume.Min.y);
		float2 min_dist_to_scene = get_distance_scene(scene_tex, scene, from, scene.spacing, scene.dim);
		nearest_object = min_dist_to_scene.y;
	}
	else
	{
		min_dist = K_INFINITY;
	}
	res.x = min_dist;
	res.y = nearest_object;

	return res;
}

__device__
float get_distance_to_sdf(GPUScene scene, GPUBoundingBox box, float3 from, HitResult hit_result, float3 * step, int3 * dim, GPUVolumeObjectInstance instance, float curr_t)
{
	float min_dist = K_INFINITY;
	float t_near = K_INFINITY, t_far;
	bool intersect_box = false;

	intersect_box = gpu_ray_box_intersect(box, from, hit_result.ray_dir, t_near, t_far);


	if (intersect_box)
	{

		float min_dist_to_sdf = get_distance(from, step, dim, instance);
		min_dist = t_near + min_dist_to_sdf;
	}
	else
	{
		min_dist = K_INFINITY;
	}

	return min_dist;
}

