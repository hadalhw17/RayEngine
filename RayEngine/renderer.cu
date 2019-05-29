////////////////////////////////////////////////////
// Main CUDA rendering file.
////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "Triangle.h"
#include "Camera.h"
#include "KDTree.h"
#include "Light.h"
#include "Object.h"
#include "KDThreeGPU.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "MainWindow.h"
#include <curand_kernel.h>
#include "GPUBoundingBox.h"
#include "RayEngine.h"
#include "Grid.h"
#include "Atmosphere.cuh"
#include "ray_functions.cuh"
#include "cuda_helper_functions.h"
#include "kd_tree_functions.cuh"
#include "sphere_tracing.cuh"
#include "filter_functions.cuh"
#include "cuda_memory_functions.cuh"

typedef struct
{
	float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix


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



__global__
void insert_sphere_to_texture(TerrainBrushType brush_type, cudaTextureObject_t tex, HitResult hit_result, GPUVolumeObjectInstance* instances,
	float3* step, float *sdf_texute, GPUBoundingBox *volumes, int3 *tex_dim, float3 *tex_spacing)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= tex_dim[0].x || y >= tex_dim[0].y || z >= tex_dim[0].z)
		return;
	float scene_t_near, scene_t_far;
	bool intersect_scene = gpu_ray_box_intersect(volumes[0], hit_result.ray_o, hit_result.ray_dir, scene_t_near, scene_t_far);
	bool intersect_sdf = false;
	float prel;
	if (intersect_scene)
	{
		intersect_sdf = single_ray_sphere_trace(tex, hit_result, instances,
			scene_t_near, scene_t_far, step, 0, prel, 10e-6);
	}
	if (intersect_sdf)
	{
		int index = x + tex_dim[0].x * (y + tex_dim[0].z * z);
		float3 poi = make_float3(x, y, z) * tex_spacing[0];
		float3 sphere_pos = hit_result.ray_o + scene_t_near * hit_result.ray_dir;
		switch (brush_type)
		{
		case ADD:
			sdf_texute[index] = fminf(sdf_texute[index], aabb_distance(poi - sphere_pos, make_float3(1)));
			break;
		case SUBTRACT:
			sdf_texute[index] = fmaxf(sdf_texute[index], -sphere_distance(poi - sphere_pos, 1));
			break;
		case INTERSECT:
			break;
		default:
			break;
		}
	}
	
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
void render_sphere_trace(RCamera render_camera, const GPUScene scene, float3* lights, const int num_lights, GPUVolumeObjectInstance* instances, const int num_instances,
	GPUBoundingBox* volumes,  float3* step, int3* dim, uint *pixels, int num_sdf, cudaTextureObject_t	tex, bool shade, float4 *texture, uint width, uint heigth)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= width || imageY >= heigth)
		return;

	int index = imageY * width + imageX;

	float4 pixel_colour = make_float4(0);
	if (draw_crosshair())
	{
		pixels[index] = rgbaFloatToInt(pixel_colour);
		return;
	}

	float u = (imageX / (float)width) * 2.0f - 1.0f;
	float v = (imageY / (float)heigth) * 2.0f - 1.0f;


	HitResult hit_result;
	generate_ray(hit_result.ray_o, hit_result.ray_dir, render_camera, width, heigth);
	// calculate eye ray in world space

	bool intersect_scene = false;
	float scene_t_near, scene_t_far, smallest_dist = K_INFINITY;
	int  nearest_shape = 0;
	int num_intersected = 0;

	// Interate over every object and test for intersection with their aabb.
	for (int i = 0; i < num_instances; ++i)
	{
		if (gpu_ray_box_intersect(volumes[instances[i].index], hit_result.ray_o + instances[i].location, hit_result.ray_dir, scene_t_near, scene_t_far))
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
	bool intersect_sdf = false;
	float prel;
	if (intersect_scene)
	{
		intersect_sdf = single_ray_sphere_trace(tex, hit_result, instances,
			smallest_dist, scene_t_far, step, nearest_shape, prel, 10e-6);
	}

	if (!intersect_sdf)
	{
		sky_mat(pixel_colour, hit_result.ray_dir);
	}
	else
	{
		sphere_trace_shade(tex, scene, lights, num_lights, hit_result, instances, smallest_dist, volumes,
			num_instances, step, dim, nearest_shape, pixel_colour, 0, shade, texture);
	}
	pixel_colour = clip(pixel_colour);
	pixels[index] = rgbaFloatToInt(pixel_colour);
	return;
}

__global__
void Craze(float3 * lights, float angle, Atmosphere * atmosphere)
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
void trace_scene(RKDTreeNodeGPU * tree, float4 * pixels,
	const RCamera render_camera, GPUSceneObject * scene_objs, int num_objs,
	int root_index, int num_faces, int* indexList, uint width, uint height)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index > (width * height)) {
		return;
	}

	pixels = trace_pixel(tree, pixels, render_camera, scene_objs, num_objs,
		root_index, num_faces, indexList, width, height);

}


////////////////////////////////////////////////////
// Ray casting with brute force approach
////////////////////////////////////////////////////
__global__
void gpu_bruteforce_ray_cast(float4 * image_buffer, const RCamera render_camera, GPUSceneObject * scene_objs, int num_objs,
	int num_faces, int stride, RKDTreeNodeGPU * tree, int* root_index, int* index_list, uint width, uint height)
{
	int index = ((threadIdx.x * gridDim.x) + blockIdx.x) + stride;
	if (index > width * height)
		return;

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, render_camera, width, height);


	float4 pixel_color = make_float4(0);

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

	ambient_light(pixel_color);
	pixel_color = clip(pixel_color);

	image_buffer[index] = pixel_color;
	return;
}


__device__
bool point_in_aabb(const GPUBoundingBox & tBox, const float3 & vecPoint)
{
	return
		vecPoint.x > tBox.Min.x && vecPoint.x < tBox.Max.x&&
		vecPoint.y > tBox.Min.y && vecPoint.y < tBox.Max.y&&
		vecPoint.z > tBox.Min.z && vecPoint.z < tBox.Max.z;

}


////////////////////////////////////////////////////
// Perform ray-casting with kd-tree
////////////////////////////////////////////////////
__global__
void trace_primary_rays(RKDTreeNodeGPU* tree,
	const RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int* root_index, int* indexList, int stride, HitResult* hit_results, uint width, uint height)
{

	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * width + imageX;
	if (index > (width - 1) * (height - 1))
		return;

	HitResult hit_result;
	generate_ray(hit_result.ray_o, hit_result.ray_dir, render_camera, width, height);

	trace_scene(tree, render_camera, scene_objs, num_objs, root_index, indexList, stride, hit_result);

	hit_results[index + stride] = hit_result;
	return;
}

__device__
float4 bilinear_filter(HitResult primary_hit_results, float3* texture)
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
	return make_float4(result);
}

__global__
void trace_secondary_rays(float3 * lights, size_t num_lights, RKDTreeNodeGPU * tree,
	RCamera * render_camera, GPUSceneObject * scene_objs, int num_objs,
	int* root_index, int* indexList, uchar4 * pixels, HitResult * primary_hit_results, Atmosphere * atmosphere, float3 * texture, size_t texture_size, int stride)
{
	float4 pixel_color = make_float4(0.f);
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;
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
		pixel_color = make_float4(compute_incident_light(atmosphere, make_float3(0, atmosphere->earthRadius + 10000, 300000), primary_hit_results[index].ray_dir, 0, t_max), 0);

		pixel_color.x = pixel_color.x < 1.413f ? powf(pixel_color.x * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-pixel_color.x);
		pixel_color.y = pixel_color.y < 1.413f ? powf(pixel_color.y * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-pixel_color.y);
		pixel_color.z = pixel_color.z < 1.413f ? powf(pixel_color.z * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-pixel_color.z);
	}

	pixel_color = clip(pixel_color);
	primary_hit_results[index].hit_color = pixel_color;
	pixels[index + stride] = make_uchar4(0xFF * pixel_color.x, 0xFF * pixel_color.y, 0xFF * pixel_color.z, 0xFF * pixel_color.w);
}


////////////////////////////////////////////////////
// Generate a shadow map and store it as
// an array of float4s
////////////////////////////////////////////////////
__global__
void generate_shadow_map(float3 * lights, size_t num_lights, RKDTreeNodeGPU * tree,
	RCamera * render_camera, GPUSceneObject * scene_objs, int num_objs,
	int* root_index, int* indexList, float4 * pixels, HitResult * primary_hit_results, int stride)
{
	float4 pixel_color = make_float4(0);
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;
	if (!primary_hit_results[index].hits)
	{
		pixel_color = make_float4(0);
		//primary_hit_results[index].hit_color = pixel_color;
		pixels[index + stride] = pixel_color;
		return;
	}

	HitResult shad_hit_result;


	shade(lights, num_lights, pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);

	//primary_hit_results[index] = shad_hit_result;
	//primary_hit_results[index].hit_color = pixel_color;

	pixel_color = clip(pixel_color);
	pixels[index + stride] = pixel_color;
}



__global__
void trace_secondary_shadow_rays(curandState * rand_state, float3 * lights, size_t num_lights, RKDTreeNodeGPU * tree,
	const RCamera render_camera, GPUSceneObject * scene_objs, int num_objs,
	int* root_index, int* indexList, float4 * pixels, HitResult * primary_hit_results, int stride)
{
	float4 pixel_color = make_float4(0.f);
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	curandState local_random = rand_state[index];
	HitResult shad_hit_result;
	if (primary_hit_results[index].hits)
	{
		HitResult hit_result;
		MaterialType hit_mat_type = scene_objs[primary_hit_results[index].obj_index].material.type;
		// Compute indirect light for diffuse objects
		if (hit_mat_type == TILE || hit_mat_type == PHONG)
		{
			float4 direct_light = make_float4(0);
			shade(lights, num_lights, direct_light, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], hit_result);

			float4 indirectLigthing = make_float4(0);

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
				trace_scene(tree, render_camera, scene_objs, num_objs, root_index, indexList, stride, local_hit_result);
				indirectLigthing += r1 * local_hit_result.hit_color / pdf;
			}
			// divide by N
			indirectLigthing /= (float)N;
			pixel_color = (direct_light / M_PI + 2 * indirectLigthing) * 0.18f;
			//shad_hit_result.hit_color = pixel_color;
		}
		else if (hit_mat_type == REFLECT)
		{
			reflect_light(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
			//primary_hit_results[index] = hit_result;
		}
		else if (hit_mat_type == REFRACT)
		{
			refract_light(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
			//primary_hit_results[index] = hit_result;
		}
		primary_hit_results[index] = shad_hit_result;

	}
	else
	{
		//pixel_color = make_float4(0.f);
	}


	pixel_color = clip(pixel_color);
	gray_scale(pixel_color);

	primary_hit_results[index].hit_color = pixel_color;
	pixels[index + stride] += pixel_color;
}


__global__
void register_random(curandState * rand_state)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	curand_init(19622342346234384, index, 0, &rand_state[index]);
}


__global__
void mix_color_maps(float4 * color_map, float4 * shadow_map)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	// Mix the two color maps by using subtracive method.
	color_map[index] = color_map[index] - (color_map[index] - shadow_map[index]) / 2;

	// Postprocess light.
	ambient_light(color_map[index]);
	clip(color_map[index]);
}


__global__
void mix_direct_indirect_light(float4 * direct_map, float4 * indirect_map)
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
	clip(direct_map[index]);
}


__global__
void copy_device(HitResult * dist, HitResult * source)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	int index = imageY * SCR_WIDTH + imageX;
	if (index > (SCR_WIDTH - 1) * (SCR_HEIGHT - 1))
		return;

	dist[index] = source[index];
}



////////////////////////////////////////////////////
// Main rendering pipeline
// Kernel is being executed from here
////////////////////////////////////////////////////
extern
uchar4* render_frame(RCamera sceneCam, uint* output, uint width, uint heigth)
{
	int size = SCR_WIDTH * SCR_HEIGHT * sizeof(uint4);

	// Number of threads in each thread block
	int  blockSize = SCR_WIDTH;

	// Number of thread blocks in grid
	int  gridSize = SCR_HEIGHT;
	dim3 primaryRaysBlockDim(32, 32);
	dim3 primaryRaysGridDim(SCR_WIDTH >> 5, SCR_HEIGHT >> 5);

	//------------------------------------------------------------------------------------------------------

#ifdef ray_tracing
	// Generate primary rays and cast them throught the scene.
	trace_primary_rays << < blockSize, gridSize >> > (d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list, 0, d_hit_result);

	gpuErrchk(cudaDeviceSynchronize());
	// Generate primary shadow map
	generate_shadow_map << < blockSize, gridSize >> > (d_light, num_light, d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list, d_shadow_map, d_hit_result, 0);
	//for (int i = 0; i < 500; ++i)
	//{
	//	trace_secondary_shadow_rays << < blockSize, gridSize >> > (rand_state, d_light, num_light, d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list, d_indirect_map, d_shadow_hit_result, 0);
	//}
	gpuErrchk(cudaDeviceSynchronize());
	// Generate secondary rays and evaluate scene colors from them.
	for (int i = 0; i < 5; ++i)
	{
		trace_secondary_rays << < blockSize, gridSize >> > (d_light, num_light, d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list,
			d_pixels, d_hit_result, d_atmosphere, d_textures, d_texture_size, 0);
	}

	//mix_direct_indirect_light << <blockSize, gridSize >> > (d_shadow_map, d_indirect_map);
	//mix_color_maps << <blockSize, gridSize >> > (d_pixels, d_shadow_map);
#endif
#ifdef sphere_tracing
	// Generate primary rays and cast them throught the scene.
	render_sphere_trace << < primaryRaysGridDim, primaryRaysBlockDim >> > (sceneCam, scene, d_light, num_light, d_volume_instances, d_num_instances,
		d_sdf_volumes, d_sdf_steps, d_sdf_dim, output, d_num_sdf, texObject, should_shade, dev_tex_p, width, heigth);

#endif
	gpuErrchk(cudaDeviceSynchronize());

	Craze << <1, 1 >> > (d_light, angle, d_atmosphere);
	angle += 0.001;  // or some other value.  Higher numbers = circles faster
	if (angle > 1.1f) angle = 0.f;


	// Copy pixel array back to host.
	//gpuErrchk(cudaMemcpyAsync(output, d_pixels, size, cudaMemcpyDeviceToDevice));

	return h_pixels;
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
void spawn_obj(RCamera cam, TerrainBrushType brush_type)
{
	dim3 primaryRaysBlockDim(2, 2, 2);
	dim3 primaryRaysGridDim(125, 125, 125);
	float3 ray_o, ray_dir;
	int imageX = SCR_WIDTH / 2;
	int imageY = SCR_HEIGHT / 2;

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
	insert_sphere_to_texture << < primaryRaysGridDim, primaryRaysBlockDim >> > (brush_type, texObject, hit_result, d_volume_instances, d_sdf_steps, d_sdf, d_sdf_volumes, d_sdf_dim, d_sdf_steps);


	gpuErrchk(cudaDeviceSynchronize());
	bind_sdf_to_texture(d_sdf, make_int3(250), 1);
}

extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix)
{
	gpuErrchk(cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix));
}
