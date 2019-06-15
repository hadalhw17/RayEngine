#pragma once
#include "../RayEngine/RayEngine.h"
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

const unsigned tableSize = 256;
const unsigned tableSizeMask = tableSize - 1;

__device__
float gradientDotV(
	size_t perm, // a value between 0 and 255 
	float x, float y, float z)
{
	switch (perm & 15) {
	case  0: return  x + y; // (1,1,0) 
	case  1: return -x + y; // (-1,1,0) 
	case  2: return  x - y; // (1,-1,0) 
	case  3: return -x - y; // (-1,-1,0) 
	case  4: return  x + z; // (1,0,1) 
	case  5: return -x + z; // (-1,0,1) 
	case  6: return  x - z; // (1,0,-1) 
	case  7: return -x - z; // (-1,0,-1) 
	case  8: return  y + z; // (0,1,1), 
	case  9: return -y + z; // (0,-1,1), 
	case 10: return  y - z; // (0,1,-1), 
	case 11: return -y - z; // (0,-1,-1) 
	case 12: return  y + x; // (1,1,0) 
	case 13: return -x + y; // (-1,1,0) 
	case 14: return -y + z; // (0,-1,1) 
	case 15: return -y - z; // (0,-1,-1) 
	}
}

inline __device__
float smoothstep(const float& t)
{
	return t * t * (3 - 2 * t);
}

inline __device__
float quintic(const float& t)
{
	return t * t * t * (t * (t * 6 - 15) + 10);
}

inline __device__
float smoothstepDeriv(const float& t)
{
	return t * (6 - 6 * t);
}

inline __device__
float quinticDeriv(const float& t)
{
	return 30 * t * t * (t * (t - 2) + 1);
}


/* inline */
__device__
uint8_t hash(const cudaTextureObject_t& permutation, const int& x, const int& y, const int& z)
{
	return tex2D<uint>(permutation, tex2D<uint>(permutation, x, y), z);
	//return tex2D<uint>(permutation, x, z);
}

__device__
float3 eval(const cudaTextureObject_t& permutation, const float3& p, float3& derivs)
{
	int xi0 = ((int)floor(p.x)) & tableSizeMask;
	int yi0 = ((int)floor(p.y)) & tableSizeMask;
	int zi0 = ((int)floor(p.z)) & tableSizeMask;

	int xi1 = (xi0 + 1) & tableSizeMask;
	int yi1 = (yi0 + 1) & tableSizeMask;
	int zi1 = (zi0 + 1) & tableSizeMask;

	float tx = p.x - ((int)floorf(p.x));
	float ty = p.y - ((int)floorf(p.y));
	float tz = p.z - ((int)floorf(p.z));

	float u = quintic(tx);
	float v = quintic(ty);
	float w = quintic(tz);
	
	// generate vectors going from the grid points to p
	float x0 = tx, x1 = tx - 1;
	float y0 = ty, y1 = ty - 1;
	float z0 = tz, z1 = tz - 1;

	float a = gradientDotV(hash(permutation, xi0, yi0, zi0), x0, y0, z0);
	float b = gradientDotV(hash(permutation, xi1, yi0, zi0), x1, y0, z0);
	float c = gradientDotV(hash(permutation, xi0, yi1, zi0), x0, y1, z0);
	float d = gradientDotV(hash(permutation, xi1, yi1, zi0), x1, y1, z0);	

	float e = gradientDotV(hash(permutation, xi0, yi0, zi1), x0, y0, z1);
	float f = gradientDotV(hash(permutation, xi1, yi0, zi1), x1, y0, z1);
	float g = gradientDotV(hash(permutation, xi0, yi1, zi1), x0, y1, z1);
	float h = gradientDotV(hash(permutation, xi1, yi1, zi1), x1, y1, z1);

	float du = quinticDeriv(tx);
	float dv = quinticDeriv(ty);
	float dw = quinticDeriv(tz);

	float k0 = a;
	float k1 = (b - a);
	float k2 = (c - a);
	float k3 = (e - a);
	float k4 = (a + d - b - c);
	float k5 = (a + f - b - e);
	float k6 = (a + g - c - e);
	float k7 = (b + c + e + h - a - d - f - g);

	derivs.x = du * (k1 + k4 * v + k5 * w + k7 * v * w);
	derivs.y = dv * (k2 + k4 * u + k6 * w + k7 * v * w);
	derivs.z = dw * (k3 + k5 * u + k6 * v + k7 * v * w);

	return make_float3(k0 + k1 * u + k2 * v + k3 * w + k4 * u * v + k5 * u * w + k6 * v * w + k7 * u * v * w);
}


__device__
BiomeTypes biomes(const float& lh, const float& elev)
{
	if (lh <= (0.2 * elev)) return BIOME_OCEAN;
	if (lh >= 0.8 * elev) return BiomeTypes::BIOME_SNOW;

}

__device__
float terrainH(const float2x2& m2, const float2& x, const cudaTextureObject_t& permutation, float3& p3, float3& derivs, const float& freq, const float& amp)
{
	float2  p = x * 0.003;
	p3 = make_float3(p.x, 0, p.y);
	float a = 0.0;
	float b = 1.0;
	float2  d = make_float2(0.0);
	for (int i = 0; i < freq + 6; i++)
	{
		float3 n = eval(permutation, p3, derivs);
		d.x += n.y;
		d.y += n.z;
		a += b * n.x / (1.0 + dot(d, d));
		b *= 0.5;
		p = mul(m2, p) * 2.0;
		p3 = make_float3(p.x, 0, p.y);
	}

	return a;
}
__device__
float terrainM(float2x2 m2, float2 x, const cudaTextureObject_t& permutation, float3& p3, float3& derivs, float freq, float amp)
{
	float2  p = x * 0.003;
	p3 = make_float3(p.x, 0, p.y);
	float a = 0.0;
	float b = 1.0;
	float2  d = make_float2(0.0);
	for (int i = 0; i < freq; i++)
	{
		float3 n = eval(permutation, p3, derivs);
		d.x += n.y;
		d.y += n.z;
		a += b * n.x / (1.0 + dot(d, d));
		b *= 0.5;
		p = mul(m2, p) * 2.0;
		p3 = make_float3(p.x, 0, p.y);
	}
	return a;
}
__device__
float terrainL(float2x2 m2, float2 x, const cudaTextureObject_t& permutation, float3& p3, float3& derivs)
{
	float2  p = x * 0.003;
	p3 = make_float3(p.x, 0, p.y);
	float a = 0.0;
	float b = 1.0;
	float2  d = make_float2(0.0);
	for (int i = 0; i < 3; i++)
	{
		float3 n = eval(permutation, p3, derivs);
		d.x += n.y;
		d.y += n.z;
		a += b * n.x / (1.0 + dot(d, d));
		b *= 0.5;
		p = mul(m2, p) * 2.0;
		p3 = make_float3(p.x, 0, p.y);
	}

	return a;
}

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
T interpolate(const cudaTextureObject_t& tex, const float3& location)
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
inline __device__
float2 sdf_fminf(float2 a, float2 b)
{
	return a.x < b.x ? a : b;
}

inline __device__
float2 sdf_fmaxf(float2 a, float2 b)
{
	return a.x > b.x ? a : b;
}
__forceinline__ __device__
float2 sdf_smin(float2 a, float2 b, float k = 0.1)
{
	float h = fmaxf(k - fabs(a.x - b.x), 0.0) / k;
	float2 res = sdf_fminf(a, b) - h * h * k * (1.0 / 4.0);
	return make_float2(res.x, sdf_fminf(a, b).y);
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
	else if (t < 2.0f) return a * a * a / 6.0f;
	else return 0.0f;
}



__device__
float2 cubicTex3DSimple(const cudaTextureObject_t& tex, const float3& coord)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float3 coord_grid = coord - 0.5f;
	float3 index = floorf(coord_grid);
	const float3 fraction = coord_grid - index;
	index = index + 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float2 fetch;
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
				fetch = tex3D<float2>(tex, u, v, w);
				result += bsplineXYZ * fetch.x;
			}
		}
	}
	return make_float2(result, floorf(fetch.y));
}



template<class T> inline
__device__
void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
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
float CUBICTEX3D(const cudaTextureObject_t& tex, const float3& coord)
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
float2 get_distance(const RenderingSettings& render_settings, const cudaTextureObject_t& tex, const float3& r_orig, const float3& step, bool normal)
{
	float2 dist;
	if (normal)
	{
		dist = cubicTex3DSimple(tex, r_orig / step);
		return dist;

	}
	switch (render_settings.quality)
	{
	case LOW:
		dist = tex3D<float2>(tex, r_orig.x / step.x, r_orig.y / step.y, r_orig.z / step.z);
		break;
	case MEDIUM:
		dist = interpolate<float2>(tex, r_orig / step);
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
float sphere_distance(const float3& p, const float& radius)
{
	return length(p) - radius;
}

HOST_DEVICE_FUNCTION
__forceinline__
float aabb_distance(const float3& p, const float3& s)
{
	float3 d = fabs(p) - s;
	return length(max(d, make_float3(0.0f)))
		+ min(max(d.x, max(d.y, d.z)), 0.0f); // remove this line for an only partially signed sdf
}



HOST_DEVICE_FUNCTION
__forceinline__
float get_distance_scene(const float3& r_orig, const GPUVolumeObjectInstance& instance, const GPUBoundingBox* volumes)
{

	float min_dist_to_sdf = sphere_distance(r_orig, 2);

	return min_dist_to_sdf;
}


__device__
float get_distance_to_sdf(const RenderingSettings& render_settings, const cudaTextureObject_t& tex, const GPUScene& scene, const GPUBoundingBox& box,
	const float3& from, const HitResult& hit_result, const float3& step, const int3& dim, const GPUVolumeObjectInstance& instance, const float& curr_t)
{
	float min_dist = K_INFINITY;
	float t_near = K_INFINITY, t_far;
	bool intersect_box = false;

	intersect_box = gpu_ray_box_intersect(box, from, hit_result.ray_dir, t_near, t_far);


	if (intersect_box)
	{

		float2 min_dist_to_sdf = get_distance(render_settings, tex, from, step, false);
		min_dist = t_near + min_dist_to_sdf.x;
	}
	else
	{
		min_dist = K_INFINITY;
	}

	return min_dist;
}

__device__
float get_terrain_distance(const float2x2& m2, const float3& poi, const float3& coord, float3& derivs, const SceneSettings &scene_settings, 
	const float3& transform, const GPUBoundingBox &volume, const cudaTextureObject_t& permutation, const bool& norm)
{
	float lh = 0.0f;
	float ly = 0.0f;

	float3 tr_coord = coord - transform;
	lh = norm ? terrainH(m2, make_float2(poi.x - transform.x, poi.z - transform.z), permutation, tr_coord, derivs, scene_settings.noise_freuency, scene_settings.noise_amplitude) 
		: terrainM(m2, make_float2(poi.x - transform.x, poi.z - transform.z), permutation, tr_coord, derivs, scene_settings.noise_freuency, scene_settings.noise_amplitude);


	lh = pow(lh + 1, scene_settings.noise_redistrebution);
	if(scene_settings.terracing != 1)
		lh = round(lh * scene_settings.terracing) / scene_settings.terracing;

	lh *= scene_settings.noise_amplitude;



	ly = poi.y;
	if (biomes(lh, volume.Max.y) == BIOME_OCEAN)
	{
		lh = (0.2 * volume.Max.y) - 1;
	}
	float sign = sgn(ly - lh);
	return sign * fabs(length(poi - make_float3(poi.x, lh, poi.z)));
}


__device__
float3 compute_sdf_normal(const float3& p_hit, const float& t, const RenderingSettings& render_settings, const cudaTextureObject_t& tex,
	const float3& step, const GPUVolumeObjectInstance& curr_obj)
{
	float delta = fminf(0.001 * t, 0.002);
	float curr_dist = get_distance(render_settings, tex, p_hit, step, true).x;
	return normalize(make_float3(
		get_distance(render_settings, tex, p_hit + make_float3(delta, 0, 0), step, true).x - curr_dist,
		get_distance(render_settings, tex, p_hit + make_float3(0, delta, 0), step, true).x - curr_dist,
		get_distance(render_settings, tex, p_hit + make_float3(0, 0, delta), step, true).x - curr_dist));
}

__device__
float3 compute_sdf_normal_tet(const float3& p_hit, const float& t, const RenderingSettings& render_settings,
	const cudaTextureObject_t& tex, const float3& step, const GPUVolumeObjectInstance& curr_obj)
{
	float delta = fminf(0.001 * t, 0.002);
	const int zero = fminf(0, t);

	float3 n = make_float3(0.0);
	for (int i = zero; i < 4; i++)
	{
		float3 e = 0.5773 * (2.0 * make_float3((((i + 3) >> 1) & 1), ((i >> 1) & 1), (i & 1)) - 1.0);
		n += e * get_distance(render_settings, tex, p_hit + e * delta, step, true).x;
	}
	return normalize(n);
}

__device__
float3 compute_terrain_normal(const float2x2& m2, const float3& p_hit, const float& t, const SceneSettings &scene_settings,
	const cudaTextureObject_t& permutation, GPUBoundingBox volume, const GPUVolumeObjectInstance& curr_obj, float3& deriv)
{
	float delta = 0.002 * t;
	float3 transform = curr_obj.location;
	return normalize(make_float3(
		get_terrain_distance(m2, p_hit - make_float3(delta, 0, 0), p_hit - make_float3(delta, 0, 0), deriv, scene_settings, -transform, volume, permutation, true)
		- get_terrain_distance(m2, p_hit + make_float3(delta, 0, 0), p_hit + make_float3(delta, 0, 0), deriv, scene_settings, -transform, volume, permutation, true), //X
		2.0 * delta, // Y
		get_terrain_distance(m2, p_hit - make_float3(0, 0, delta), p_hit - make_float3(0, 0, delta), deriv, scene_settings,-transform, volume, permutation, true)
		- get_terrain_distance(m2, p_hit + make_float3(0, 0, delta), p_hit + make_float3(0, 0, delta), deriv, scene_settings, -transform, volume, permutation, true))); // Z
}