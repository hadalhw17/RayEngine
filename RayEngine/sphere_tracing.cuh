#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cuda_runtime_api.h"
#include "sdf_functions.cuh"
#include "material_functions.cuh"
#include "Grid.h"
#include "helper_math.h"
#include "sdf_functions.cuh"

texture<float4, cudaTextureType2D, cudaReadModeElementType> d_texture;

cudaArray* d_volumeArray = 0;
cudaTextureObject_t	texObject; // For 3D texture


__device__ __constant__
struct GPUBoundingBox* d_sdf_volumes;

GPUBoundingBox* h_volumes;

struct GPUVolumeObjectInstance* volume_objs;

__device__
float* d_sdf;

__device__ __constant__
float* dev_sdf_p;

__device__ __constant__
float3* d_sdf_steps;

__device__ __constant__
int3* d_sdf_dim;

__device__
struct GPUVolumeObjectInstance* d_volume_instances;

__device__
int d_num_instances;

__device__
int d_num_sdf;

__device__
float4* dev_tex_p;


__device__
float4 nearest_neightbour_filter(float x, float y,  float4 * texture)
{
	int tx = (int)(1000 * x), ty = (int)(1000 * y);
	float4* _texture = (1000 * ty + tx) + texture;

	return *_texture;
}

__device__
float4 bilinear_filter(float ix, float iy, float4* texture)
{
	float u = (ix * 1000.f) - 0.5f;
	float v = (iy * 1000.f) - 0.5f;

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
	float4 result = (texture[xy] * u_opposite + texture[x1y] * u_ratio) * v_opposite +
		(texture[xy1] * u_opposite + texture[x1y1] * u_ratio) * v_ratio;
	return result;
}

////////////////////////////////////////////////////
// Bind normals to texture memory
////////////////////////////////////////////////////
void bind_texture(float4* dev_texture, unsigned int tex_size)
{
	d_texture.normalized = false;                      // access with normalized texture coordinates
	d_texture.filterMode = cudaFilterModeLinear;        // Point mode, so no 
	d_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates
	d_texture.addressMode[1] = cudaAddressModeWrap;

	cudaArray *d_tex_array = 0;
	cudaChannelFormatDesc arr_channelDesc = cudaCreateChannelDesc<float4>();
	cudaMallocArray(&d_tex_array, &arr_channelDesc, 1000, 1000);

	cudaMemcpy2DToArray(d_tex_array, 0, 0, dev_texture, sizeof(float4) * 1000, sizeof(float4) * 1000, 1000, cudaMemcpyDeviceToDevice);


	size_t size = sizeof(float4) * tex_size;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	channelDesc.f = cudaChannelFormatKindFloat;

	//gpuErrchk(cudaBindTexture2D(0, (const textureReference*)&d_texture, (const void*)dev_texture, channelDesc, 1000, 1000, 1000 * sizeof(float4)));
	cudaBindTextureToArray(d_texture, d_tex_array);
}

__device__
bool single_ray_sphere_trace(cudaTextureObject_t tex, HitResult hit_result, GPUVolumeObjectInstance* instances,
	float& t_near, float t_far, float3* step, int nearest_shape, float &prel, float threshold)
{
	GPUVolumeObjectInstance curr_obj = instances[nearest_shape];
	float t = t_near;
	while (t < t_far)
	{
		float3 from = hit_result.ray_o + (t * hit_result.ray_dir) + curr_obj.location;
		float min_dist = K_INFINITY;
		if (curr_obj.index == -1)
		{

			min_dist = sphere_distance(from, 3);
		}
		else
		{
			//min_dist = get_distance(from, step, dim, curr_obj);
			min_dist = get_distance(tex, from, step, curr_obj);
		}
		if (min_dist <= threshold * t)
		{
			t_near = t;
			return true;
		}
		t += min_dist;
		prel = fminf(prel, 32 * min_dist / t);
	}
	return false;
}


__device__
void sphere_trace_shadow(cudaTextureObject_t tex, GPUVolumeObjectInstance* instances, float3 p_hit, float3 normal, GPUBoundingBox* volumes, int num_sdf, 
	float3* step, int3* dim, float3 light, float4& final_colour, int shape)
{
	float4 pixel_color = make_float4(0.f);
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= SCR_WIDTH || imageY >= SCR_HEIGHT)
		return;

	int index = imageX * SCR_HEIGHT + imageY;

	float light_dst = K_INFINITY;
	float3 lightpos = light, lightDir;
	float4 lightInt;
	illuminate(p_hit, lightpos, lightDir, lightInt, light_dst);

	HitResult hit_result;
	hit_result.ray_o = p_hit;
	hit_result.ray_dir = -lightDir;

	if (dot(light - p_hit, normal) <= 0)
	{
		return;
	}

	float res = 1.f;

	float t = 0;
	int step_count = 0;
	float t_near = 0;
	bool intersects = single_ray_sphere_trace(tex, hit_result, instances,
		t_near, light_dst, step, 0,res, K_EPSILON);
	final_colour += res * (lightInt * dot(hit_result.ray_dir, normal));
	return;
}

__device__
float4 triplanar_mapping(float3 norm, float3 p_hit, float4 *texture)
{
	// in wNorm is the world-space normal of the fragment
	float3 blending = fabs(norm);
	blending = normalize(fmaxf(blending, make_float3(0.00001))); // Force weights to sum to 1.0
	float b = (blending.x + blending.y + blending.z);
	blending /= make_float3(b);

	float4 xaxis = bilinear_filter( p_hit.y, p_hit.z, texture);
	float4 yaxis = bilinear_filter( p_hit.z, p_hit.x, texture);
	float4 zaxis = bilinear_filter( p_hit.x, p_hit.y, texture);

	//float4 xaxis = tex2D(d_texture, p_hit.y, p_hit.z);
	//float4 yaxis = tex2D(d_texture, p_hit.z, p_hit.x);
	//float4 zaxis = tex2D(d_texture, p_hit.x, p_hit.y);

	
	// blend the results of the 3 planar projections.
	float4 tex = blending.x * xaxis + blending.y * yaxis + blending.z * zaxis;
	return tex;
}

__device__
void sphere_trace_shade(cudaTextureObject_t tex, GPUScene scene, float3 * lights, int num_lights, HitResult hit_result, GPUVolumeObjectInstance * instance, float t, GPUBoundingBox * volumes,
 int num_sdf, float3 * step, int3 * dim, int curr_sdf, float4 & final_colour, int step_count, bool shade, float4 *texture)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= SCR_WIDTH || imageY >= SCR_HEIGHT)
		return;

	int index = imageX * SCR_HEIGHT + imageY;

	GPUVolumeObjectInstance curr_obj = instance[curr_sdf];
	float3 p_hit = hit_result.ray_o + (t)* hit_result.ray_dir;
	hit_result.ray_o = p_hit;
	float delta = 10e-5;
	float3 n = normalize(make_float3(
		get_distance(tex, p_hit + make_float3(delta, 0, 0), step, curr_obj) - get_distance(tex, p_hit + make_float3(-delta, 0, 0), step, curr_obj),
		get_distance(tex, p_hit + make_float3(0, delta, 0), step, curr_obj) - get_distance(tex, p_hit + make_float3(0, -delta, 0), step, curr_obj),
		get_distance(tex, p_hit + make_float3(0, 0, delta), step, curr_obj) - get_distance(tex, p_hit + make_float3(0, 0, -delta), step, curr_obj)));

	if (p_hit.y <= volumes[curr_obj.index].Max.y * 0.3) final_colour = make_float4(0.11, 0.16, 0.62, 0.f);
	else if (p_hit.y > volumes[curr_obj.index].Max.y * 0.3 && p_hit.y < volumes[curr_obj.index].Max.y * 0.6) final_colour = make_float4(0.59, 0.29, 0.f, 0.f);
	else if (p_hit.y >= volumes[curr_obj.index].Max.y * 0.6) final_colour = make_float4(0.26, 0.30, 0.27, 0.f);
	/*int square = floor(p_hit.x) + floor(p_hit.z);
	tile_pattern(final_colour, square);*/

	//final_colour = triplanar_mapping(n, normalize(fabs(p_hit / step[0]) / 250), texture);
	if (curr_obj.index != -1 && shade)
	{
		for (int i = 0; i < num_lights; ++i)
		{
			//p_hit = p_hit + K_EPSILON * n;
			sphere_trace_shadow(tex, instance, p_hit, n, volumes, num_sdf, step, dim, lights[i], final_colour, curr_sdf);
		}
	}
	else
	{
		simple_shade(final_colour, n, hit_result.ray_dir);
	}
}

////////////////////////////////////////////////////
// Bind triangles to texture memory
////////////////////////////////////////////////////
void bind_sdf_to_texture(float* dev_sdf_p, int3 dim, int num_sdf)
{
	if (texObject)
	{
		gpuErrchk(cudaDestroyTextureObject(texObject));
		gpuErrchk(cudaFreeArray(d_volumeArray));
	}
	cudaChannelFormatDesc arr_channelDesc = cudaCreateChannelDesc<float>();
	gpuErrchk(cudaMalloc3DArray(&d_volumeArray, &arr_channelDesc, make_cudaExtent(num_sdf * dim.x * sizeof(float), dim.y, dim.z), 0));

	cudaMemcpy3DParms copyParams = { 0 };

	//Array creation
	copyParams.srcPtr = make_cudaPitchedPtr(dev_sdf_p, num_sdf * dim.x * sizeof(float), dim.y, dim.z);
	copyParams.dstArray = d_volumeArray;
	copyParams.extent = make_cudaExtent(num_sdf * dim.x, dim.y, dim.z);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);
	//Array creation End

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = d_volumeArray;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));

	tex_desc.normalizedCoords = false;                      // access with normalized texture coordinates
	tex_desc.filterMode = cudaFilterModeLinear;        // Point mode, so no 
	tex_desc.addressMode[0] = cudaAddressModeBorder;    // wrap texture coordinates
	tex_desc.addressMode[1] = cudaAddressModeBorder;    // wrap texture coordinates
	tex_desc.addressMode[2] = cudaAddressModeBorder;    // wrap texture coordinates
	tex_desc.readMode = cudaReadModeElementType;

	gpuErrchk(cudaCreateTextureObject(&texObject, &res_desc, &tex_desc, NULL));
}


extern "C"
uchar4* initialize_volume_render(RCamera sceneCam, Grid* sdf, int num_sdf, std::vector<float4> textures)
{
	float size_sdf = num_sdf * sdf->voxels.size() * sizeof(float);
	size_t image_size = SCR_WIDTH * SCR_HEIGHT;
	int size = image_size * sizeof(uchar4);

	h_camera = &sceneCam;
	h_pixels = new uchar4[size];

	num_light = 1;
	float3* h_lights = new float3[num_light];

	h_lights[0] = make_float3(0, 25, 0);
	//h_lights[1] = make_float3(3, 15, 10);
	Atmosphere* h_atmosphere = new Atmosphere();
	size_t textures_size = textures.size() * sizeof(float4);
	float3* h_sdf_steps = new float3[num_sdf];
	int3* h_sdf_dim = new int3[num_sdf];
	float3* h_max_sdf = new float3[num_sdf];
	h_volumes = new GPUBoundingBox[num_sdf + 1];
	volume_objs = new GPUVolumeObjectInstance[200];
	volume_objs[0] = GPUVolumeObjectInstance(0, make_float3(0, 0, 0), make_float3(0));
	d_num_instances = 1;


	float* h_grid = new float[num_sdf * sdf[0].voxels.size()];

	for (size_t iz = 0; iz < sdf->sdf_dim.z; ++iz)
	{
		for (size_t iy = 0; iy < sdf->sdf_dim.y; ++iy)
		{
			size_t offset = 0;
			for (size_t g = 0; g < num_sdf; ++g)
			{
				for (size_t ix = 0; ix < sdf->sdf_dim.x; ++ix)
				{
					h_grid[offset + ix + sdf->sdf_dim.y * num_sdf * (iy + (sdf->sdf_dim.x) * iz)] = sdf[g].voxels.at((ix + sdf->sdf_dim.y * (iy + sdf->sdf_dim.x * iz))).distance;
				}
				offset += sdf[g].sdf_dim.x;
			}
		}
	}

	for (size_t g = 0; g < num_sdf; ++g)
	{
		h_sdf_steps[g] = sdf[g].spacing;
		h_sdf_dim[g] = sdf[g].sdf_dim;
		h_max_sdf[g] = sdf[g].box_max;
		h_volumes[g] = GPUBoundingBox(make_float3(0.f), sdf[g].box_max);

	}
	h_volumes[num_sdf] = GPUBoundingBox(make_float3(0.f), make_float3(6));
	scene.dim = make_int3(100);
	for (int i = 0; i < d_num_instances; ++i)
	{
		scene.volume.Max = max(scene.volume.Max, h_volumes[volume_objs[i].index].Max + volume_objs[i].location);
		scene.volume.Min = min(scene.volume.Min, h_volumes[volume_objs[i].index].Min + volume_objs[i].location);
	}

	scene.spacing.x = scene.volume.dx() / ((float)scene.dim.x - 1.f);
	scene.spacing.y = scene.volume.dy() / ((float)scene.dim.y - 1.f);
	scene.spacing.z = scene.volume.dz() / ((float)scene.dim.z - 1.f);

	// allocate memory for the triangle meshes on the GPU
	gpuErrchk(cudaMalloc((void**)& d_sdf, size_sdf));
	gpuErrchk(cudaMalloc(&d_pixels, size));
	gpuErrchk(cudaMalloc(&d_render_camera, sizeof(RCamera)));
	gpuErrchk(cudaMalloc(&d_light, num_light * sizeof(float3)));
	gpuErrchk(cudaMalloc(&d_sdf_steps, num_sdf * sizeof(float3)));
	gpuErrchk(cudaMalloc(&d_sdf_dim, num_sdf * sizeof(int3)));
	gpuErrchk(cudaMalloc(&d_atmosphere, sizeof(Atmosphere)));
	gpuErrchk(cudaMalloc(&d_sdf_volumes, num_sdf * sizeof(GPUBoundingBox)));
	gpuErrchk(cudaMalloc(&d_volume_instances, 200 * sizeof(GPUVolumeObjectInstance)));

	// copy triangle data to GPU
	gpuErrchk(cudaMemcpy(d_sdf, h_grid, size_sdf, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_light, h_lights, num_light * sizeof(float3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sdf_steps, h_sdf_steps, num_sdf * sizeof(float3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sdf_dim, h_sdf_dim, num_sdf * sizeof(int3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sdf_volumes, h_volumes, num_sdf * sizeof(GPUBoundingBox), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_atmosphere, h_atmosphere, sizeof(Atmosphere), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_volume_instances, volume_objs, 2 * sizeof(GPUVolumeObjectInstance), cudaMemcpyHostToDevice));
	d_num_sdf = num_sdf;

	// load triangle data into a CUDA texture
	bind_sdf_to_texture(d_sdf, sdf[0].sdf_dim, num_sdf);

	if (textures_size > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		gpuErrchk(cudaMalloc((void**)& dev_tex_p, textures_size));


		// copy triangle data to GPU
		gpuErrchk(cudaMemcpy(dev_tex_p, textures.data(), textures_size, cudaMemcpyHostToDevice));

		// load triangle data into a CUDA texture
		bind_texture(dev_tex_p, textures_size);
	}

	return d_pixels;
}

extern
void free_memory()
{
	gpuErrchk(cudaDestroyTextureObject(texObject));
	gpuErrchk(cudaFreeArray(d_volumeArray));
	gpuErrchk(cudaFree(d_atmosphere));
	gpuErrchk(cudaFree(d_sdf));
	gpuErrchk(cudaFree(d_sdf_volumes));
	gpuErrchk(cudaFree(d_render_camera));
	gpuErrchk(cudaFree(d_sdf_steps));
	gpuErrchk(cudaFree(d_sdf_dim));
	gpuErrchk(cudaFree(d_volume_instances));
	gpuErrchk(cudaFree(dev_tex_p));
}
