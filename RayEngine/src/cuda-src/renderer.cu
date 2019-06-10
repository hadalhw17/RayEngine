////////////////////////////////////////////////////
// Main CUDA rendering file.
////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../Triangle.h"
#include "../Camera.h"
#include "../KDTree.h"
#include "../Light.h"
#include "../Object.h"
#include "../KDThreeGPU.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <curand_kernel.h>
#include "../GPUBoundingBox.h"
#include "../RayEngine/RayEngine.h"
#include "../Grid.h"
#include "Atmosphere.cuh"
#include "ray_functions.cuh"
#include "cuda_helper_functions.h"
#include "kd_tree_functions.cuh"
#include "sphere_tracing.cuh"
#include "filter_functions.cuh"
#include "cuda_memory_functions.cuh"
#include <sstream>
#include "cuda_occupancy.h"
#include <fstream>


__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

float2x2 m2;


__global__
void insert_sphere_to_texture(RenderingSettings render_settings, SceneSettings scene_settings, TerrainBrush brush, cudaTextureObject_t tex, HitResult hit_result, const GPUVolumeObjectInstance* instances,
	float3 step, float2* sdf_texute, GPUBoundingBox* volumes, uint3 tex_dim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= tex_dim.x || y >= tex_dim.y || z >= tex_dim.z)
		return;
	float scene_t_near, scene_t_far;
	bool intersect_scene = gpu_ray_box_intersect(volumes[0], hit_result.ray_o, hit_result.ray_dir, scene_t_near, scene_t_far);
	bool intersect_sdf = false;
	float prel;
	int material_index = -1;
	if (intersect_scene)
	{
		intersect_sdf = single_ray_sphere_trace(render_settings, scene_settings, tex, hit_result, instances,
			scene_t_near, scene_t_far, step, 0, prel, 10e-6, material_index);
	}
	if (intersect_sdf)
	{
		int index = x + tex_dim.x * (y + tex_dim.z * z);
		float3 poi = make_float3(x, y, z) * step;
		float3 sphere_pos = hit_result.ray_o + scene_t_near * hit_result.ray_dir;
		int mat_index = brush.material_index;
		switch (brush.brush_type)
		{
		case SPHERE_ADD:
			sdf_texute[index] = sdf_smin(sdf_texute[index], make_float2(sphere_distance(poi - sphere_pos, brush.brush_radius), mat_index));
			break;
		case SPHERE_SUBTRACT:
			sdf_texute[index] = sdf_fmaxf(sdf_texute[index], make_float2(-sphere_distance(poi - sphere_pos, brush.brush_radius), mat_index));
			break;
		case CUBE_ADD:
			GPUVolumeObjectInstance obj;
			//float3 normal = compute_sdf_normal(poi, 10e-6, render_settings, tex, step, obj);
			sdf_texute[index] = sdf_smin(sdf_texute[index], make_float2(aabb_distance(poi - sphere_pos + make_float3(0, 0.5, 0), make_float3(brush.brush_radius, 0.6, brush.brush_radius)), mat_index));
			//sdf_texute[index] = sdf_smin(sdf_texute[index], make_float2(aabb_distance(poi - (sphere_pos - normal * brush.brush_radius*2) - make_float3(0,4,0), make_float3(brush.brush_radius)), mat_index));
			break;
		case CUBE_SUBTRACT:
			sdf_texute[index] = sdf_fmaxf(sdf_texute[index], make_float2(-aabb_distance(poi - sphere_pos, make_float3(brush.brush_radius)), mat_index));
			break;
		default:
			break;
		}
	}

}
__device__
float3 fract(float3 a)
{
	return make_float3(a.x - (int)floorf(a.x), a.y - (int)floorf(a.y), a.z - (int)floorf(a.z));
}


__device__
float noised(float3 x, curandState* rand_state)
{
	int n = x.x + x.y * 57;
	n = (n << 13) ^ n;
	return (1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 1073741824.0);
}

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
uint8_t hash(const unsigned*& perm, const int& x, const int& y, const int& z)
{
	return perm[perm[perm[x] + y] + z];
}

__device__
float3 eval(const unsigned*& perm, const float3& p, float3& derivs)
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

	float u = smoothstep(tx);
	float v = smoothstep(ty);
	float w = smoothstep(tz);

	// generate vectors going from the grid points to p
	float x0 = tx, x1 = tx - 1;
	float y0 = ty, y1 = ty - 1;
	float z0 = tz, z1 = tz - 1;

	float a = gradientDotV(hash(perm, xi0, yi0, zi0), x0, y0, z0);
	float b = gradientDotV(hash(perm, xi1, yi0, zi0), x1, y0, z0);
	float c = gradientDotV(hash(perm, xi0, yi1, zi0), x0, y1, z0);
	float d = gradientDotV(hash(perm, xi1, yi1, zi0), x1, y1, z0);
	float e = gradientDotV(hash(perm, xi0, yi0, zi1), x0, y0, z1);
	float f = gradientDotV(hash(perm, xi1, yi0, zi1), x1, y0, z1);
	float g = gradientDotV(hash(perm, xi0, yi1, zi1), x0, y1, z1);
	float h = gradientDotV(hash(perm, xi1, yi1, zi1), x1, y1, z1);

	float du = smoothstepDeriv(tx);
	float dv = smoothstepDeriv(ty);
	float dw = smoothstepDeriv(tz);

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
float terrainH(const float2x2& m2, const float2& x, const unsigned*& perm, float3& p3, float3& derivs, const float& freq, const float& amp)
{
	float2  p = x * 0.003;
	p3 = make_float3(p.x, 0, p.y);
	float a = 0.0;
	float b = 1.0;
	float2  d = make_float2(0.0);
	for (int i = 0; i < freq + 6; i++)
	{
		float3 n = eval(perm, p3, derivs);
		d.x += n.y;
		d.y += n.z;
		a += b * n.x / (1.0 + dot(d, d));
		b *= 0.5;
		p = mul(m2, p) * 2.0;
		p3 = make_float3(p.x, 0, p.y);
	}

	return amp * a;
}
__device__
float terrainM(float2x2 m2, float2 x, const unsigned*& perm, float3& p3, float3& derivs, float freq, float amp)
{
	float2  p = x * 0.003;
	p3 = make_float3(p.x, 0, p.y);
	float a = 0.0;
	float b = 1.0;
	float2  d = make_float2(0.0);
	for (int i = 0; i < freq; i++)
	{
		float3 n = eval(perm, p3, derivs);
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
float terrainL(float2x2 m2, float2 x, const unsigned*& perm, float3& p3, float3& derivs)
{
	float2  p = x * 0.003;
	p3 = make_float3(p.x, 0, p.y);
	float a = 0.0;
	float b = 1.0;
	float2  d = make_float2(0.0);
	for (int i = 0; i < 3; i++)
	{
		float3 n = eval(perm, p3, derivs);
		d.x += n.y;
		d.y += n.z;
		a += b * n.x / (1.0 + dot(d, d));
		b *= 0.5;
		p = mul(m2, p) * 2.0;
		p3 = make_float3(p.x, 0, p.y);
	}

	return 120.0 * a;
}
__global__
void generate_noise(float2x2 m2, RenderingSettings render_settings, SceneSettings scene_settings, GPUVolumeObjectInstance* instances,
	float2* sdf_texute, float2* normal_texture, uint3 tex_dim, float3 spacing, GPUBoundingBox* volumes, curandState* rand_state, const unsigned* perm, float3* grads, float3 pos)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= tex_dim.x || y >= tex_dim.y || z >= tex_dim.z)
		return;
	volumes[0] = GPUBoundingBox(make_float3(0), make_float3((tex_dim.x - 1) * spacing.x, (tex_dim.y - 1) * spacing.y, (tex_dim.z - 1) * spacing.z));

	//instances->location = make_float3(pos.x - (volumes[0].dx() / 2), 0, pos.z - (volumes[0].dz() / 2));
	instances->location = pos;
	//volumes[0].Min += instances->location;
	//volumes[0].Max += instances->location;
	int index = x + (tex_dim.x) * (y + (tex_dim.z) * z);
	float3 poi = (make_float3(x, y, z)) * spacing;
	float3 coord = ((make_float3(x + 0.5f, 10, z + 0.5)) * spacing);

	float lh = 0.0f;
	float ly = 0.0f;
	float3 derivs;
	float3 nc = make_float3(coord.x / scene_settings.world_size.x, 0, coord.z / scene_settings.world_size.z);
	//lh = f(poi.x, poi.z, scene_settings.noise_freuency, scene_settings.noise_amplitude + 10);
	lh = terrainM(m2, make_float2(poi.x, poi.z), perm, coord, derivs, scene_settings.noise_freuency, scene_settings.noise_amplitude);
	//lh = eval(perm, nc  * scene_settings.noise_freuency, derivs).y;
	lh = pow(lh + 1, scene_settings.noise_redistrebution);
	lh = round(lh * scene_settings.terracing) / scene_settings.terracing;

	lh *= scene_settings.noise_amplitude;
	//lh += volumes[0].Max.y / 2;
	ly = poi.y;
	if (biomes(lh, volumes[0].Max.y) == BIOME_OCEAN)
	{
		lh = (0.2 * volumes[0].Max.y) - 1;
	}
	float sign = -1;
	if (ly > lh)
	{
		sign = 1;
	}

	sdf_texute[index] = make_float2(sign * fabs(length(poi - make_float3(poi.x, lh, poi.z))), 0);
}

__device__
bool draw_crosshair()
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if ((imageX == SCR_WIDTH / 2 && imageY == SCR_HEIGHT / 2) ||
		(imageX == (SCR_WIDTH / 2) + 1 && imageY == SCR_HEIGHT / 2) ||
		(imageX == (SCR_WIDTH / 2) - 1 && imageY == SCR_HEIGHT / 2))
		return true;
	return false;
}


__global__
void render_sphere_trace(const RenderingSettings render_settings, const RCamera render_camera, const SceneSettings scene,
	const float3* __restrict__ lights, const int num_lights,
	const  GPUVolumeObjectInstance* __restrict__ instances, const int num_instances,
	const GPUBoundingBox* __restrict__ volumes, const float3 step, const uint3 dim,
	uint* __restrict__ pixels, const int num_sdf, const cudaTextureObject_t tex, const cudaTextureObject_t normal_tex,
	const bool shade, const  uint width, const uint heigth, const  GPUMat* __restrict__ materials, const unsigned* __restrict__ d_permutationTable)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= width || imageY >= heigth)
		return;

	int index = imageY * width + imageX;

	float3 pixel_colour = make_float3(0);

	float u = (imageX / (float)width);
	float v = (imageY / (float)heigth);


	HitResult hit_result;
	generate_ray(hit_result.ray_o, hit_result.ray_dir, render_camera, width, heigth);
	// calculate eye ray in world space
	bool intersect_scene = false;
	float scene_t_near, scene_t_far, smallest_dist = K_INFINITY;
	int  nearest_shape = 0;
	int num_intersected = 0;

	// Interate over every object and test for intersection with their aabb.
#pragma unroll 1
	for (int i = 0; i < num_instances; ++i)
	{
		if (gpu_ray_box_intersect(volumes[instances[i].index], hit_result.ray_o, hit_result.ray_dir, scene_t_near, scene_t_far))
		{
			if (scene_t_near < smallest_dist)
			{
				intersect_scene = true;
				smallest_dist = scene_t_near;
				nearest_shape = i;
				++num_intersected;
			}
		}
	}
	float prel;
	if (intersect_scene)
	{

	}
	hit_result.ray_o -= instances[0].location;
	int material_index;
	if (!single_ray_sphere_trace(render_settings, scene, tex, hit_result, instances,
		smallest_dist, scene_t_far, step, nearest_shape, prel, 0.001, material_index))
	{
		pixel_colour = mix(pixel_colour, make_float3(1, 0.65, 0), prel);
		sky_mat(pixel_colour, hit_result.ray_dir);
		if (scene.enable_fog)
			apply_fog(pixel_colour, K_INFINITY, scene.fog_deisity, hit_result.ray_o, hit_result.ray_dir);
	}
	else
	{
		sphere_trace_shade(render_settings, tex, normal_tex, scene, lights, num_lights, hit_result, instances, smallest_dist, volumes,
			num_instances, step, dim, material_index, pixel_colour, 0, shade, materials, d_permutationTable);
		if (scene.enable_fog)
			apply_fog(pixel_colour, smallest_dist, scene.fog_deisity, hit_result.ray_o, hit_result.ray_dir);

	}

	// gamma	
	//pixel_colour = powf(clamp(pixel_colour, 0.0, 1.0), 0.45);

	// contrast, desat, tint and vignetting	
	//pixel_colour = pixel_colour * 0.3 + 0.7 * pixel_colour * pixel_colour * (make_float3(3.0) - 2.0 * pixel_colour);
	//pixel_colour = mix(pixel_colour, make_float3(pixel_colour.x + pixel_colour.y + pixel_colour.z) * 0.33, 0.2);
	//pixel_colour *= 1.25 * make_float3(1.02, 1.05, 1.0);
	if (render_settings.gamma) pixel_colour = powf(pixel_colour, 1.0 / 2.2);
	if (render_settings.vignetting) pixel_colour *= 0.5 + 0.5 * powf(16.0 * u * v * (1.0 - u) * (1.0 - v), render_settings.vignetting_k);
	pixel_colour = clip(pixel_colour);
	pixels[index] = rgbaFloatToInt(pixel_colour);
	return;
}

__global__
void Craze(float3* lights, float angle, Atmosphere* atmosphere)
{

	float x = cosf(angle) * 20.f;
	float z = sinf(angle) * 20.f;

	lights[0].x = x;
	lights[0].z = x;
	//lights[0].x = x;
	//lights[0].z = z;
	float ang = angle * M_PI * 0.6;
	atmosphere->sunDirection = make_float3(0, cosf(ang), sinf(ang));
}


////////////////////////////////////////////////////
// Initializes ray caster
////////////////////////////////////////////////////
__global__
void trace_scene(RKDTreeNodeGPU* tree, float4* pixels,
	const RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int root_index, int num_faces, int* indexList, uint width, uint height)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= width || imageY >= height)
		return;

	int index = imageY * width + imageX;

	float4 pixel_colour = make_float4(0);

	pixels = trace_pixel(tree, pixels, render_camera, scene_objs, num_objs,
		root_index, num_faces, indexList, width, height);

}


////////////////////////////////////////////////////
// Ray casting with brute force approach
////////////////////////////////////////////////////
__global__
void gpu_bruteforce_ray_cast(float4* image_buffer, const RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int num_faces, int stride, RKDTreeNodeGPU* tree, int* root_index, int* index_list, uint width, uint height)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= width || imageY >= height)
		return;

	int index = imageY * width + imageX;

	float3 pixel_colour = make_float3(0);

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, render_camera, width, height);

	float t_near, t_far;
	GPUBoundingBox bbox = tree[0].box;


	// Perform ray-box intersection test.
	bool intersects_aabb = gpu_ray_box_intersect(bbox, ray_o, ray_dir, t_near, t_far);
	if (intersects_aabb)
	{
		HitResult hit_result;

		for (int i = 0; i < num_faces; ++i) {

			float4 v0 = tex1Dfetch(triangle_texture, i * 3);
			float4 edge1 = tex1Dfetch(triangle_texture, i * 3 + 1);
			float4 edge2 = tex1Dfetch(triangle_texture, i * 3 + 2);

			// Perform ray-triangle intersection test.
			HitResult tmp_hit_result;
			bool intersects_tri = gpu_ray_tri_intersect(make_float3(v0.x, v0.y, v0.z), make_float3(edge1.x, edge1.y, edge1.z),
				make_float3(edge2.x, edge2.y, edge2.z), i, scene_objs[0], tmp_hit_result);

			if (intersects_tri)
			{
				if (tmp_hit_result.t < hit_result.t)
				{
					hit_result = tmp_hit_result;
					//narmals_mat(pixel_color, tmp_normal);
					//simple_shade(pixel_color, hit_result.normal, ray_dir);
					//phong_light(pixel_color, hit_point, ray_dir, tmp_normal, hit_point, tree, scene_objs, num_objs, root_index, index_list);
				}
			}
		}
	}

	ambient_light(pixel_colour);
	pixel_colour = clip(pixel_colour);

	//image_buffer[index] = pixel_colour;
	return;
}



////////////////////////////////////////////////////
// Perform ray-casting with kd-tree
////////////////////////////////////////////////////
__global__
void trace_primary_rays(RKDTreeNodeGPU* tree,
	const RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int* root_index, int* indexList, HitResult* hit_results, uint width, uint height)
{

	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= width || imageY >= height)
		return;

	int index = imageY * width + imageX;

	HitResult hit_result;
	generate_ray(hit_result.ray_o, hit_result.ray_dir, render_camera, width, height);

	trace_scene(tree, render_camera, scene_objs, num_objs, root_index, indexList, hit_result);

	hit_results[index] = hit_result;
	return;
}

__device__
float3 bilinear_filter(HitResult primary_hit_results, float3* texture)
{
	float u = (primary_hit_results.uv.x * 1000.f) - 0.5f;
	float v = (primary_hit_results.uv.y * 1000.f) - 0.5f;

	int x = floor(u);
	int y = floor(v);
	int xy = x + 1000 * y;
	int x1y = (x + 1) + 1000 * y;
	int xy1 = x + 1000 * (y + 1);
	int x1y1 = (x + 1) + 1000 * (y + 1);

	float u_ratio = u - (float)x;
	float v_ratio = v - (float)y;
	float u_opposite = 1.f - u_ratio;
	float v_opposite = 1.f - v_ratio;
	float3 result = (texture[xy] * u_opposite + texture[x1y] * u_ratio) * v_opposite +
		(texture[xy1] * u_opposite + texture[x1y1] * u_ratio) * v_ratio;
	return result;
}

__global__
void trace_secondary_rays(float3* lights, size_t num_lights, RKDTreeNodeGPU* tree,
	RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int* root_index, int* indexList, uint* pixels, HitResult* primary_hit_results, Atmosphere* atmosphere, float3* texture, size_t texture_size)
{
	float3 pixel_color = make_float3(0.f);
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= SCR_WIDTH || imageY >= SCR_HEIGHT)
		return;

	int index = imageY * SCR_WIDTH + imageX;

	HitResult shad_hit_result;
	if (primary_hit_results[index].hits)
	{
		HitResult hit_result;
		MaterialType hit_mat_type = scene_objs[primary_hit_results[index].obj_index].material.type;
		pixel_color = scene_objs[primary_hit_results[index].obj_index].material.color;
		if (hit_mat_type == TILE)
		{
			//int square = floor(primary_hit_results[index].hit_point.x) + floor(primary_hit_results[index].hit_point.z);
			//tile_pattern(pixel_color, square);


			pixel_color = bilinear_filter(primary_hit_results[index], texture);
		}
		else if (hit_mat_type == PHONG)
		{
			phong_light(lights, num_lights, pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
		}
		else if (hit_mat_type == REFLECT)
		{
			reflect(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], hit_result);
			primary_hit_results[index] = hit_result;
		}
		else if (hit_mat_type == REFRACT)
		{
			reflect_refract(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], hit_result);
			primary_hit_results[index] = hit_result;
		}
	}
	else
	{
		// if ray missed draw sky there.
		//sky_mat(pixel_color, primary_hit_results[index].ray_dir);
		float t_max = K_INFINITY;
		pixel_color = compute_incident_light(atmosphere, make_float3(0, atmosphere->earthRadius + 10000, 300000), primary_hit_results[index].ray_dir, 0, t_max);

		pixel_color.x = pixel_color.x < 1.413f ? pow(pixel_color.x * 0.38317f, 1.0f / 2.2f) : 1.0f - __expf(-pixel_color.x);
		pixel_color.y = pixel_color.y < 1.413f ? pow(pixel_color.y * 0.38317f, 1.0f / 2.2f) : 1.0f - __expf(-pixel_color.y);
		pixel_color.z = pixel_color.z < 1.413f ? pow(pixel_color.z * 0.38317f, 1.0f / 2.2f) : 1.0f - __expf(-pixel_color.z);
	}

	//pixel_color = clip(pixel_color);
	primary_hit_results[index].hit_color = pixel_color;
	pixels[index] += rgbaFloatToInt(pixel_color);
}


////////////////////////////////////////////////////
// Generate a shadow map and store it as
// an array of float4s
////////////////////////////////////////////////////
__global__
void generate_shadow_map(float3* lights, size_t num_lights, RKDTreeNodeGPU* tree,
	RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int* root_index, int* indexList, uint* pixels, HitResult* primary_hit_results)
{
	float3 pixel_color = make_float3(0);
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= SCR_WIDTH || imageY >= SCR_HEIGHT)
		return;

	int index = imageY * SCR_WIDTH + imageX;

	float4 pixel_colour = make_float4(0);
	if (!primary_hit_results[index].hits)
	{
		pixel_color = make_float3(0);
		//primary_hit_results[index].hit_color = pixel_color;
		pixels[index] = rgbaFloatToInt(pixel_color);
		return;
	}

	HitResult shad_hit_result;


	shade(lights, num_lights, pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);

	//primary_hit_results[index] = shad_hit_result;
	//primary_hit_results[index].hit_color = pixel_color;

	//pixel_color = clip(pixel_color);
	pixels[index] = rgbaFloatToInt(pixel_color);
}



__global__
void trace_secondary_shadow_rays(curandState* rand_state, float3* lights, size_t num_lights, RKDTreeNodeGPU* tree,
	const RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int* root_index, int* indexList, float4* pixels, HitResult* primary_hit_results)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= SCR_WIDTH || imageY >= SCR_HEIGHT)
		return;

	int index = imageY * SCR_WIDTH + imageX;

	float3 pixel_colour = make_float3(0);

	curandState local_random = rand_state[index];
	HitResult shad_hit_result;
	if (primary_hit_results[index].hits)
	{
		HitResult hit_result;
		MaterialType hit_mat_type = scene_objs[primary_hit_results[index].obj_index].material.type;
		// Compute indirect light for diffuse objects
		if (hit_mat_type == TILE || hit_mat_type == PHONG)
		{
			float3 direct_light = make_float3(0);
			shade(lights, num_lights, direct_light, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], hit_result);

			float3 indirectLigthing = make_float3(0);

			uint32_t N = 128;
			float3 Nt, Nb;
			createCoordinateSystem(primary_hit_results[index].normal, Nt, Nb);
			float pdf = 1 / (2 * M_PI);
			for (uint32_t n = 0; n < N; ++n) {
				HitResult local_hit_result;
				float r1 = curand_uniform(&local_random);
				float r2 = curand_uniform(&local_random);
				float3 sample = uniformSampleHemisphere(r1, r2);
				float3 sample_world = make_float3(
					sample.x * Nb.x + sample.y * primary_hit_results[index].normal.x + sample.z * Nt.x,
					sample.x * Nb.y + sample.y * primary_hit_results[index].normal.y + sample.z * Nt.y,
					sample.x * Nb.z + sample.y * primary_hit_results[index].normal.z + sample.z * Nt.z);
				// don't forget to divide by PDF and multiply by cos(theta)
				local_hit_result.ray_o = primary_hit_results[index].hit_point + sample_world * make_float3(K_EPSILON);
				local_hit_result.ray_dir = sample_world;
				trace_scene(tree, render_camera, scene_objs, num_objs, root_index, indexList, local_hit_result);
				indirectLigthing += r1 * local_hit_result.hit_color / pdf;
			}
			// divide by N
			indirectLigthing /= (float)N;
			pixel_colour = (direct_light / M_PI + 2 * indirectLigthing) * 0.18f;
			//shad_hit_result.hit_color = pixel_color;
		}
		else if (hit_mat_type == REFLECT)
		{
			reflect_light(pixel_colour, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
			//primary_hit_results[index] = hit_result;
		}
		else if (hit_mat_type == REFRACT)
		{
			refract_light(pixel_colour, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
			//primary_hit_results[index] = hit_result;
		}
		primary_hit_results[index] = shad_hit_result;

	}
	else
	{
		//pixel_color = make_float4(0.f);
	}


	//pixel_colour = clip(pixel_colour);
	gray_scale(pixel_colour);

	primary_hit_results[index].hit_color = pixel_colour;
	//pixels[index] += pixel_colour;
}


__global__
void register_random(curandState* rand_state, int3 dim)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= dim.x || y >= dim.y || z >= dim.z)
		return;
	int index = x + (dim.x) * (y + (dim.z) * z);

	curand_init(19622342346234384, index, 0, &rand_state[index]);
}


__global__
void mix_color_maps(float4* color_map, float4* shadow_map)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	// Mix the two color maps by using subtracive method.
	color_map[index] = color_map[index] - (color_map[index] - shadow_map[index]) / 2;

	// Postprocess light.
	//ambient_light(color_map[index]);
	//clip(color_map[index]);
}


__global__
void mix_direct_indirect_light(float4* direct_map, float4* indirect_map)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	// Mix the two color maps by using subtracive method.
	direct_map[index] = (direct_map[index] / M_PI + 2 * indirect_map[index]) * 0.18f;

	// Postprocess light.
	//ambient_light(direct_map[index]);
	//clip(direct_map[index]);
}


__global__
void copy_device(HitResult* dist, HitResult* source)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	dist[index] = source[index];
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}
__device__
float4* dev_tex_p;
int size = SCR_WIDTH * SCR_HEIGHT * sizeof(uint4);

// Number of threads in each thread block
//int  blockSize = SCR_WIDTH;

// Number of thread blocks in grid
dim3 primaryRaysBlockDim(32, 32);
dim3 primaryRaysGridDim(SCR_WIDTH >> 5, SCR_HEIGHT >> 5);
dim3 blockSize = (16, 16);
dim3 gridSize = dim3(iDivUp(SCR_WIDTH, blockSize.x), iDivUp(SCR_HEIGHT, blockSize.y));
////////////////////////////////////////////////////
// Main rendering pipeline
// Kernel is being executed from here
////////////////////////////////////////////////////
extern
void cuda_render_frame(uint* output, const uint& width, const uint& heigth)
{

#ifdef ray_tracing
	// Generate primary rays and cast them throught the scene.
	trace_primary_rays << < primaryRaysGridDim, primaryRaysBlockDim >> > (d_tree, sceneCam, d_scene_objects, d_object_number, d_root_index, d_index_list, d_hit_result, width, heigth);

	gpuErrchk(cudaDeviceSynchronize());
	// Generate primary shadow map
	//generate_shadow_map << < primaryRaysGridDim, primaryRaysBlockDim >> > (d_light, num_light, d_tree, sceneCam, d_scene_objects, d_object_number, d_root_index, d_index_list, output, d_hit_result);
	//for (int i = 0; i < 500; ++i)
	//{
	//	trace_secondary_shadow_rays << < blockSize, gridSize >> > (rand_state, d_light, num_light, d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list, d_indirect_map, d_shadow_hit_result, 0);
	//}
	//gpuErrchk(cudaDeviceSynchronize());
	// Generate secondary rays and evaluate scene colors from them.
	for (int i = 0; i < 5; ++i)
	{
		trace_secondary_rays << < primaryRaysGridDim, primaryRaysBlockDim >> > (d_light, num_light, d_tree, sceneCam, d_scene_objects, d_object_number, d_root_index, d_index_list,
			output, d_hit_result, d_atmosphere, d_textures, d_texture_size);
	}

#endif
#ifdef sphere_tracing
	gpuErrchk(cudaDeviceSynchronize());
	// Generate primary rays and cast them throught the scene.
	render_sphere_trace << < gridSize, blockSize >> > (cuda_render_settings, *h_camera, cuda_scene_settings, d_light, num_light, d_volume_instances, d_num_instances,
		d_sdf_volumes, sdf_spacing, sdf_dim, output, d_num_sdf, texObject, normalObject, should_shade, width, heigth, d_materials, d_permutationTable);

#endif
	gpuErrchk(cudaDeviceSynchronize());

	//Craze << <1, 1 >> > (d_light, angle, d_atmosphere);
	//angle += 0.001;  // or some other value.  Higher numbers = circles faster
	//if (angle > 1.1f) angle = 0.f;

}

extern
void toggle_shadow()
{
	switch (should_shade)
	{
	case true:
		should_shade = false;
		break;
	case false:
		should_shade = true;
		break;
	default:
		break;
	}
}

extern
void spawn_obj(RCamera cam, TerrainBrush brush, int x, int y)
{
	dim3 primaryRaysBlockDim(2, 2, 2);
	dim3 primaryRaysGridDim(sdf_dim.x / 2, sdf_dim.y / 2, sdf_dim.z / 2);
	float3 ray_o, ray_dir;
	int imageX = x;
	int imageY = y;

	int index = ((imageY * SCR_WIDTH) + imageX);
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	float sx = (float)imageX / (SCR_WIDTH - 1.0f);
	float sy = 1.0f - ((float)imageY / (SCR_HEIGHT - 1.0f));

	float3 rendercampos = cam.campos;

	// compute primary ray direction
	// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
	float3 rendercamview = cam.view; rendercamview = normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
	float3 rendercamup = cam.camdown; rendercamup = normalize(rendercamup);
	float3 horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis = normalize(horizontalAxis); // Important to normalize!
	float3 verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

	float3 middle = rendercampos + rendercamview;
	float3 horizontal = horizontalAxis * tanf(cam.fov.x * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5
	float3 vertical = verticalAxis * tanf(cam.fov.y * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5

	// compute pixel on screen
	float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
	float3 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cam.focial_distance); // Important for depth of field!		

	float3 aperturePoint = rendercampos;

	// calculate ray direction of next ray in path
	float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
	apertureToImagePlane = normalize(apertureToImagePlane); // ray direction needs to be normalised

	// ray direction
	float3 rayInWorldSpace = apertureToImagePlane;
	ray_dir = normalize(rayInWorldSpace);

	float3 pos = rendercampos + 10 * apertureToImagePlane;
	// ray origin
	ray_o = rendercampos;
	HitResult hit_result;
	hit_result.ray_o = ray_o;
	hit_result.ray_dir = apertureToImagePlane;
	insert_sphere_to_texture << < primaryRaysGridDim, primaryRaysBlockDim >> > (cuda_render_settings, cuda_scene_settings, brush, texObject, hit_result, d_volume_instances,
		sdf_spacing, d_sdf, d_sdf_volumes, sdf_dim);


	gpuErrchk(cudaDeviceSynchronize());
	bind_sdf_to_texture(&d_sdf, sdf_dim, 1);
}

extern
void generate_noise(const float3& pos)
{
	m2.m[0] = make_float2(0.8, -0.6);
	m2.m[1] = make_float2(0.6, 0.8);
	sdf_dim = cuda_scene_settings.volume_resolution;
	gpuErrchk(cudaFree(d_sdf));
	gpuErrchk(cudaMalloc((void**)& d_sdf, sdf_dim.x * sdf_dim.y * sdf_dim.z * sizeof(float2)));

	gpuErrchk(cudaFree(d_sdf_norm));
	gpuErrchk(cudaMalloc((void**)& d_sdf_norm, sdf_dim.x * sdf_dim.y * sdf_dim.z * sizeof(float2)));
	//sdf_spacing = cuda_scene_settings.volume_spacing;
	sdf_spacing = make_float3(cuda_scene_settings.world_size.x / sdf_dim.x, cuda_scene_settings.world_size.y / sdf_dim.y, cuda_scene_settings.world_size.z / sdf_dim.z);
	dim3 primaryRaysBlockDim(2, 2, 2);
	dim3 primaryRaysGridDim(sdf_dim.x / 2, sdf_dim.y / 2, sdf_dim.z / 2);
	//cudaMalloc(&rand_state, sdf_dim.x * sdf_dim.y * sdf_dim.z * sizeof(curandState));
	//register_random << <primaryRaysGridDim, primaryRaysBlockDim >> > (rand_state, sdf_dim);
	generate_noise << < primaryRaysGridDim, primaryRaysBlockDim >> > (m2, cuda_render_settings, cuda_scene_settings, d_volume_instances,
		d_sdf, d_sdf_norm, sdf_dim, sdf_spacing, d_sdf_volumes, rand_state, d_permutationTable, d_gradients, make_float3(0));

	//cudaFree(rand_state);
	gpuErrchk(cudaDeviceSynchronize());
	bind_sdf_to_texture(&d_sdf, sdf_dim, 1);
	bind_sdf_norm_to_texture(d_sdf_norm, sdf_dim, 1);
}

extern
void save_map()
{
	//delete h_grid;
	h_grid = new float2[sdf_dim.x * sdf_dim.y * sdf_dim.z];
	gpuErrchk(cudaMemcpy(h_grid, d_sdf, sdf_dim.x * sdf_dim.y * sdf_dim.z * sizeof(float2), cudaMemcpyDeviceToHost));

	std::ostringstream file_name;
	file_name << "Edited" << ".rsdf";
	std::ofstream volume_file_stream("SDFs/" + file_name.str(), std::ios::binary);
	volume_file_stream.write(reinterpret_cast<const char*> (&sdf_dim.x), sizeof(int));
	volume_file_stream.write(reinterpret_cast<const char*> (&sdf_dim.y), sizeof(int));
	volume_file_stream.write(reinterpret_cast<const char*> (&sdf_dim.z), sizeof(int));

	volume_file_stream.write(reinterpret_cast<const char*> (&sdf_spacing.x), sizeof(float));
	volume_file_stream.write(reinterpret_cast<const char*> (&sdf_spacing.y), sizeof(float));
	volume_file_stream.write(reinterpret_cast<const char*> (&sdf_spacing.z), sizeof(float));

	//Loop through data X changes first/fastest.
	for (unsigned int iz = 0; iz < sdf_dim.z; iz++)
		for (unsigned int iy = 0; iy < sdf_dim.y; iy++)
			for (unsigned int ix = 0; ix < sdf_dim.x; ix++) {

				volume_file_stream.write(reinterpret_cast<const char*> (&h_grid[ix + sdf_dim.y * (iy + sdf_dim.x * iz)]), sizeof(float2));

			}

	//Close the file when finished
	volume_file_stream.close();
}

extern
void load_map()
{
	Grid* distance_field = new Grid(std::string("SDFs/Edited.rsdf"));

	float size_sdf = distance_field->voxels.size() * sizeof(float2);

	sdf_spacing = distance_field->spacing;
	sdf_dim = distance_field->sdf_dim;

	h_grid = new float2[distance_field->voxels.size()];

	for (size_t iz = 0; iz < sdf_dim.z; ++iz)
	{
		for (size_t iy = 0; iy < sdf_dim.y; ++iy)
		{
			for (size_t ix = 0; ix < sdf_dim.x; ++ix)
			{
				h_grid[ix + sdf_dim.y * (iy + sdf_dim.x * iz)] = make_float2(distance_field->voxels.at((ix + sdf_dim.y * (iy + sdf_dim.x * iz))).distance, distance_field->voxels.at((ix + sdf_dim.y * (iy + sdf_dim.x * iz))).material);
			}

		}
	}
	gpuErrchk(cudaDeviceSynchronize());
	if (h_grid)
	{
		//delete[] h_grid, h_volumes;
		gpuErrchk(cudaFree(d_sdf_volumes));
		gpuErrchk(cudaFree(d_sdf));
	}
	gpuErrchk(cudaMalloc((void**)& d_sdf, size_sdf));
	gpuErrchk(cudaMalloc(&d_sdf_volumes, sizeof(GPUBoundingBox)));

	h_volumes = new GPUBoundingBox(make_float3(0.f), distance_field->box_max);

	gpuErrchk(cudaMemcpy(d_sdf, h_grid, size_sdf, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sdf_volumes, h_volumes, sizeof(GPUBoundingBox), cudaMemcpyHostToDevice));

	gpuErrchk(cudaDeviceSynchronize());
	bind_sdf_to_texture(&d_sdf, sdf_dim, 1);
	gpuErrchk(cudaDeviceSynchronize());
}



__global__
void sdf_collision_test(RCamera cam, RenderingSettings render_settings, cudaTextureObject_t tex, GPUBoundingBox* box, float3 step, volatile bool* overlaps, volatile bool* in_volume)
{
	float2 res = get_distance(render_settings, tex, cam.campos - box[0].Min, step);
	if (!point_in_aabb(box[0], cam.campos))
	{
		*in_volume = false;
		*overlaps = false;
		return;
	}
	*in_volume = true;
	if (res.x < 0)* overlaps = true;
}

extern
bool sdf_collision(RCamera cam)
{
	bool* collides;
	bool* in_volume;
	gpuErrchk(cudaMalloc(&collides, sizeof(bool)));
	gpuErrchk(cudaMalloc(&in_volume, sizeof(bool)));
	gpuErrchk(cudaMemset(collides, false, sizeof(bool)));
	gpuErrchk(cudaMemset(in_volume, false, sizeof(bool)));
	sdf_collision_test << <1, 1 >> > (cam, cuda_render_settings, texObject, d_sdf_volumes, sdf_spacing, collides, in_volume);
	bool h_overlaps, h_in_volume;
	gpuErrchk(cudaMemcpy(&h_overlaps, collides, sizeof(bool), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(&h_in_volume, in_volume, sizeof(bool), cudaMemcpyDeviceToHost));

	//if (!h_in_volume)
	//{
	//	m2.m[0] = make_float2(0.8, -0.6);
	//	m2.m[1] = make_float2(0.6, 0.8);
	//	sdf_dim = cuda_scene_settings.volume_resolution;
	//	gpuErrchk(cudaFree(d_sdf));
	//	gpuErrchk(cudaMalloc((void**)& d_sdf, sdf_dim.x * sdf_dim.y * sdf_dim.z * sizeof(float2)));

	//	gpuErrchk(cudaFree(d_sdf_norm));
	//	gpuErrchk(cudaMalloc((void**)& d_sdf_norm, sdf_dim.x * sdf_dim.y * sdf_dim.z * sizeof(float2)));
	//	//sdf_spacing = cuda_scene_settings.volume_spacing;
	//	sdf_spacing = make_float3(cuda_scene_settings.world_size.x / sdf_dim.x, cuda_scene_settings.world_size.y / sdf_dim.y, cuda_scene_settings.world_size.z / sdf_dim.z);
	//	dim3 primaryRaysBlockDim(2, 2, 2);
	//	dim3 primaryRaysGridDim(sdf_dim.x / 2, sdf_dim.y / 2, sdf_dim.z / 2);
	//	//cudaMalloc(&rand_state, sdf_dim.x * sdf_dim.y * sdf_dim.z * sizeof(curandState));
	//	//register_random << <primaryRaysGridDim, primaryRaysBlockDim >> > (rand_state, sdf_dim);
	//	generate_noise << < primaryRaysGridDim, primaryRaysBlockDim >> > (m2, cuda_render_settings, cuda_scene_settings, d_volume_instances,
	//		d_sdf, d_sdf_norm, sdf_dim, sdf_spacing, d_sdf_volumes, rand_state, d_permutationTable, d_gradients, cam.campos);

	//	//cudaFree(rand_state);
	//	gpuErrchk(cudaDeviceSynchronize());
	//	bind_sdf_to_texture(d_sdf, sdf_dim, 1);
	//	bind_sdf_norm_to_texture(d_sdf_norm, sdf_dim, 1);
	//}

	return h_overlaps;
}
