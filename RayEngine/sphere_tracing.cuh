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
#include "Material.h"

texture<float4, cudaTextureType2D, cudaReadModeElementType> d_texture;
texture<float4, cudaTextureType2D, cudaReadModeElementType> d_texture1;
texture<float4, cudaTextureType2D, cudaReadModeElementType> d_texture2;


cudaArray* d_volumeArray = 0;
cudaTextureObject_t	texObject; // For 3D texture


struct GPUMat
{
	texturess textttt;
};

__device__ __constant__
GPUMat *d_materials;


GPUMat *h_materials;


float2* h_grid;

__device__ __constant__
struct GPUBoundingBox* d_sdf_volumes;

GPUBoundingBox* h_volumes;

struct GPUVolumeObjectInstance* volume_objs;

__device__
float2* d_sdf;

__device__ __constant__
float* dev_sdf_p;

float3 sdf_spacing;

int3 sdf_dim;

__device__ __constant__	
struct GPUVolumeObjectInstance* d_volume_instances;

__device__
int d_num_instances;

__device__
int d_num_sdf;

RenderingSettings cuda_render_settings;
SceneSettings cuda_scene_settings;

float3 spacing;


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
void bind_texture(float4* dev_texture1, float4* dev_texture2, float4* dev_texture3, uint2 texture_resolution, size_t pitch, int curr_material)
{
	// Create 3 texture arrays( 1 for each plane(triplanar texture mapping)) 
	cudaArray* d_textureArray1;
	cudaArray* d_textureArray2;
	cudaArray* d_textureArray3;
	cudaChannelFormatDesc arr_channelDesc = cudaCreateChannelDesc<float4>();

	// Create array for texture 1
	gpuErrchk(cudaMallocArray(&d_textureArray1, &arr_channelDesc, texture_resolution.x, texture_resolution.y));
	gpuErrchk(cudaMemcpy2DToArray(d_textureArray1, 0, 0, dev_texture1, pitch, texture_resolution.x * sizeof(float4), texture_resolution.y, cudaMemcpyDeviceToDevice));
	//Array creation End

	// Create array for texture 2
	gpuErrchk(cudaMallocArray(&d_textureArray2, &arr_channelDesc, texture_resolution.x, texture_resolution.y));
	gpuErrchk(cudaMemcpy2DToArray(d_textureArray2, 0, 0, dev_texture2, pitch, texture_resolution.x * sizeof(float4), texture_resolution.y, cudaMemcpyDeviceToDevice));
	//Array creation End

	// Create array for texture 3
	gpuErrchk(cudaMallocArray(&d_textureArray3, &arr_channelDesc, texture_resolution.x, texture_resolution.y));
	gpuErrchk(cudaMemcpy2DToArray(d_textureArray3, 0, 0, dev_texture3, pitch, texture_resolution.x * sizeof(float4), texture_resolution.y, cudaMemcpyDeviceToDevice));
	//Array creation End

	cudaResourceDesc res_desc1;
	memset(&res_desc1, 0, sizeof(cudaResourceDesc));
	res_desc1.resType = cudaResourceTypeArray;
	res_desc1.res.array.array = d_textureArray1;

	cudaResourceDesc res_desc2;
	memset(&res_desc2, 0, sizeof(cudaResourceDesc));
	res_desc2.resType = cudaResourceTypeArray;
	res_desc2.res.array.array = d_textureArray2;

	cudaResourceDesc res_desc3;
	memset(&res_desc3, 0, sizeof(cudaResourceDesc));
	res_desc3.resType = cudaResourceTypeArray;
	res_desc3.res.array.array = d_textureArray3;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));

	tex_desc.normalizedCoords = true;                      // access with normalized texture coordinates
	tex_desc.filterMode = cudaFilterModeLinear;        // Point mode, so no 
	tex_desc.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates
	tex_desc.addressMode[1] = cudaAddressModeWrap;    // wrap texture coordinates
	tex_desc.readMode = cudaReadModeElementType;

	gpuErrchk(cudaCreateTextureObject(&h_materials[curr_material].textttt.texture1, &res_desc1, &tex_desc, NULL));
	gpuErrchk(cudaCreateTextureObject(&h_materials[curr_material].textttt.texture2, &res_desc2, &tex_desc, NULL));
	gpuErrchk(cudaCreateTextureObject(&h_materials[curr_material].textttt.texture3, &res_desc3, &tex_desc, NULL));
}


__device__
float f(float x, float z, float freq, float amp)
{
	float y = sinf(x)* cosf(z);
	return sinf(amp + (y * freq) * cosf(amp * 2 + y * freq/3));
}
__device__
bool single_ray_sphere_trace(const RenderingSettings render_settings, SceneSettings scene_settings, cudaTextureObject_t tex, HitResult hit_result,const  GPUVolumeObjectInstance* __restrict__ instances,
	float& t_near,const float t_far,const  float3 step,const int nearest_shape, float &prel,const float threshold, int &material_index)
{
	GPUVolumeObjectInstance curr_obj = instances[nearest_shape];
	float t = t_near;
	while (t < t_far)
	{
		float3 from = hit_result.ray_o + (t * hit_result.ray_dir) + curr_obj.location;
		float min_dist = K_INFINITY;
		float2 res;
		if (curr_obj.index == -1)
		{

			min_dist = sphere_distance(from, 3);
		}
		else
		{
			//min_dist = get_distance(from, step, dim, curr_obj);
			res = get_distance(render_settings, tex, from, step);
			min_dist = res.x;
			material_index = res.y;
		}
		if (min_dist <= threshold * t)
		{
			t_near = t;
			prel = 0;
			return true;
		}
		prel = fminf(prel, scene_settings.soft_shadow_k * min_dist / t);
		t += min_dist;
	}
	return false;
}

__device__
bool single_shadow_ray_sphere_trace(RenderingSettings render_settings, SceneSettings scene_settings, cudaTextureObject_t tex, HitResult hit_result,
	float& t_near, float t_far, float3 step, int nearest_shape, float& prel, float threshold, int& material_index)
{
	float t = t_near;
	while (t < t_far)
	{
		float3 from = hit_result.ray_o + (t * hit_result.ray_dir);
		float min_dist = K_INFINITY;
		float2 res;


		//min_dist = get_distance(from, step, dim, curr_obj);
		res = get_distance(render_settings, tex, from, step);
		min_dist = res.x;
		material_index = res.y;

		if (prel < K_EPSILON || min_dist <= threshold)
		{
			t_near = t;
			prel = 0;
			return true;
		}
		prel = fminf(prel, scene_settings.soft_shadow_k * min_dist / t);
		t += min_dist;
	}
	return false;
}


__device__
void sphere_trace_shadow(const RenderingSettings render_settings,const SceneSettings scene_settings,const cudaTextureObject_t tex,const GPUVolumeObjectInstance* __restrict__ instances,const float3 p_hit, const float3 normal,const GPUBoundingBox* __restrict__ volumes, const int num_sdf,
	const float3 step,const int3 dim,const float3 light, float3& final_colour,const int material_index)
{
	float3 pixel_color = make_float3(0.f);

	float light_dst = K_INFINITY;
	float x = cosf(scene_settings.light_pos.x) * 20.f;
	float z = sinf(scene_settings.light_pos.x) * 20.f;
	float3 lightpos = make_float3(x, scene_settings.light_pos.y, z), lightDir;
	float3 lightInt;
	illuminate(p_hit, lightpos, lightDir, lightInt, light_dst, scene_settings.light_intensity);

	HitResult hit_result;
	hit_result.ray_o = p_hit;
	hit_result.ray_dir = normalize(-lightDir);

	float sh = 1.f;

	float t = 0;
	int step_count = 0;
	float t_near = 0;
	int material_ind;
	bool intersects = single_shadow_ray_sphere_trace(render_settings, scene_settings, tex, hit_result,
		t_near, light_dst, step, 0, sh, 10e-6, material_ind);
	float amb = __saturatef(0.5 + 0.5 * normal.y);
	float soft_shadow = __saturatef(sh);
	float dif = fmaxf(.0f, dot(normal, hit_result.ray_dir));
	float bac = __saturatef(0.2 + 0.8 * dot(normalize(make_float3(-lightDir.x, 0.0, lightDir.z)), normal));

	pixel_color += soft_shadow * lightInt * dif;
	pixel_color += amb * make_float3(0.40, 0.60, 1.00) * 1.2;
	pixel_color += bac * make_float3(0.40, 0.50, 0.60);

	final_colour *= pixel_color;

	// gamma
	//final_colour = sqrtf(final_colour);
	//pixel_color = clip(pixel_color);
	//final_colour = sqrtf(final_colour);
	return;
}

inline __device__
float3 powf(float3 a, float b)
{
	return make_float3(powf(a.x, b), powf(a.y, b), powf(a.z, b));
}

inline __device__
float4 powf(float4 a, float b)
{
	return make_float4(powf(a.x, b), powf(a.y, b), powf(a.z, b), 0);
}
__device__
float3 triplanar_mapping(float3 norm, float3 p_hit, texturess textures)
{
	// in wNorm is the world-space normal of the fragment
	float3 blending = fabs(norm);
	blending = normalize(fmaxf(blending, make_float3(0.00001))); // Force weights to sum to 1.0
	float b = (blending.x + blending.y + blending.z);
	blending /= make_float3(b);

	//float4 x = bilinear_filter( p_hit.y, p_hit.z, texture);
	//float4 yaxis = bilinear_filter( p_hit.z, p_hit.x, texture);
	//float4 zaxis = bilinear_filter( p_hit.x, p_hit.y, texture);
	float4 xaxis = tex2D<float4>(textures.texture1, p_hit.y, p_hit.z);
	float4 yaxis = tex2D<float4>(textures.texture2, p_hit.z, p_hit.x);
	float4 zaxis = tex2D<float4>(textures.texture3, p_hit.x, p_hit.y);


	
	// blend the results of the 3 planar projections.
	float3 tex = blending.x * make_float3(xaxis.x, xaxis.y, xaxis.z) + blending.y * make_float3(yaxis.y, yaxis.y, yaxis.z) + blending.z * make_float3(zaxis.y, zaxis.y, zaxis.z);
	return tex;

	//float3 m = powf(fabs(norm), 2);
	//float4 x = tex2D(d_texture, p_hit.y, p_hit.z);
	//float4 y = tex2D(d_texture1, p_hit.z, p_hit.x);
	//float4 z = tex2D(d_texture2, p_hit.x, p_hit.y);
	//return (x * m.x + y * m.y + z * m.z) / (m.x + m.y + m.z);
}

__device__
void sphere_trace_shade(const RenderingSettings render_settings,const cudaTextureObject_t tex,const SceneSettings scene,const float3 * lights,const int num_lights, HitResult hit_result,const GPUVolumeObjectInstance * __restrict__ instance, const float t, const GPUBoundingBox * volumes,
 const int num_sdf,const float3 step,const int3 dim,const int material_index, float3 &final_colour,const int step_count,const bool shade, const  GPUMat* __restrict__ materials)
{
	final_colour = make_float3(0);

	GPUVolumeObjectInstance curr_obj;
	float3 p_hit = hit_result.ray_o + (t)* hit_result.ray_dir;
	hit_result.ray_o = p_hit;


	float3 n = compute_sdf_normal(p_hit, t, render_settings, tex, step, curr_obj);

	//if (p_hit.y <= volumes[curr_obj.index].Max.y * 0.3) final_colour = make_float4(0.11, 0.16, 0.62, 0.f);
	//else if (p_hit.y > volumes[curr_obj.index].Max.y * 0.3 && p_hit.y < volumes[curr_obj.index].Max.y * 0.6) final_colour = make_float4(0.59, 0.29, 0.f, 0.f);
	//else if (p_hit.y >= volumes[curr_obj.index].Max.y * 0.6) final_colour = make_float4(0.26, 0.30, 0.27, 0.f);
	/*int square = floor(p_hit.x) + floor(p_hit.z);
	tile_pattern(final_colour, square);*/

	if (material_index == -1)
	{
		int square = floor(p_hit.x) + floor(p_hit.z);
		tile_pattern(final_colour, square);
	}
	else
	{
		final_colour = mix(final_colour, triplanar_mapping(n, (p_hit/ step) / render_settings.texture_scale, materials[material_index].textttt), 0.3);
	}
	//final_colour = nearest_neightbour_filter((p_hit.z / step[0].z) / 250, (p_hit.x / step[0].x)/ 250, texture);
	
	if (shade)
	{
		for (int i = 0; i < num_lights; ++i)
		{
			//p_hit = p_hit + K_EPSILON * n;
			sphere_trace_shadow(render_settings, scene, tex, instance, p_hit, n, volumes, num_sdf, step, dim, lights[i], final_colour, material_index);
		}
	}
	else
	{
			//simple_shade(final_colour, n, hit_result.ray_dir);
	}
}

////////////////////////////////////////////////////
// Bind triangles to texture memory
////////////////////////////////////////////////////
void bind_sdf_to_texture(float2* dev_sdf_p, int3 dim, int num_sdf)
{
	if (texObject)
	{
		gpuErrchk(cudaDestroyTextureObject(texObject));
		gpuErrchk(cudaFreeArray(d_volumeArray));
	}
	cudaChannelFormatDesc arr_channelDesc = cudaCreateChannelDesc<float2>();
	gpuErrchk(cudaMalloc3DArray(&d_volumeArray, &arr_channelDesc, make_cudaExtent(num_sdf * dim.x * sizeof(float2), dim.y, dim.z), 0));

	cudaMemcpy3DParms copyParams = { 0 };

	//Array creation
	copyParams.srcPtr = make_cudaPitchedPtr(dev_sdf_p, num_sdf * dim.x * sizeof(float2), dim.y, dim.z);
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
uchar4* initialize_volume_render(RCamera sceneCam, Grid* sdf, int num_sdf, std::vector<VoxelMaterial> materials, RenderingSettings render_settings, SceneSettings scene_settings)
{
	float size_sdf = num_sdf * sdf->voxels.size() * sizeof(float2);
	size_t image_size = SCR_WIDTH * SCR_HEIGHT;
	int size = image_size * sizeof(uchar4);

	h_camera = &sceneCam;
	h_pixels = new uchar4[size];

	num_light = 1;
	float3* h_lights = new float3[num_light];

	h_lights[0] = make_float3(0, 25, 0);
	//h_lights[1] = make_float3(3, 15, 10);
	Atmosphere* h_atmosphere = new Atmosphere();
	float3* h_sdf_steps = new float3[num_sdf];
	int3* h_sdf_dim = new int3[num_sdf];
	float3* h_max_sdf = new float3[num_sdf];
	h_volumes = new GPUBoundingBox[num_sdf + 1];
	volume_objs = new GPUVolumeObjectInstance[200];
	volume_objs[0] = GPUVolumeObjectInstance(0, make_float3(0, 0, 0), make_float3(0));
	d_num_instances = 1;


	h_grid = new float2[num_sdf * sdf[0].voxels.size()];

	for (size_t iz = 0; iz < sdf->sdf_dim.z; ++iz)
	{
		for (size_t iy = 0; iy < sdf->sdf_dim.y; ++iy)
		{
			size_t offset = 0;
			for (size_t g = 0; g < num_sdf; ++g)
			{
				for (size_t ix = 0; ix < sdf->sdf_dim.x; ++ix)
				{
					h_grid[offset + ix + sdf->sdf_dim.y * num_sdf * (iy + (sdf->sdf_dim.x) * iz)] = make_float2(sdf[g].voxels.at((ix + sdf->sdf_dim.y * (iy + sdf->sdf_dim.x * iz))).distance, -1);
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
	sdf_spacing = sdf[0].spacing;
	sdf_dim = sdf[0].sdf_dim;
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
	gpuErrchk(cudaMalloc(&d_atmosphere, sizeof(Atmosphere)));
	gpuErrchk(cudaMalloc(&d_sdf_volumes, num_sdf * sizeof(GPUBoundingBox)));
	gpuErrchk(cudaMalloc(&d_volume_instances, 200 * sizeof(GPUVolumeObjectInstance)));

	// copy triangle data to GPU
	gpuErrchk(cudaMemcpy(d_sdf, h_grid, size_sdf, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_light, h_lights, num_light * sizeof(float3), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(d_sdf_volumes, h_volumes, num_sdf * sizeof(GPUBoundingBox), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_atmosphere, h_atmosphere, sizeof(Atmosphere), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_volume_instances, volume_objs, 2 * sizeof(GPUVolumeObjectInstance), cudaMemcpyHostToDevice));
	d_num_sdf = num_sdf;
	cuda_render_settings = render_settings;
	cuda_scene_settings = scene_settings;

	// load triangle data into a CUDA texture
	bind_sdf_to_texture(d_sdf, sdf[0].sdf_dim, num_sdf);

	if (materials.size() > 0)
	{
		h_materials = new GPUMat[materials.size()];
		gpuErrchk(cudaMalloc((void**)& d_materials, materials.size() * sizeof(GPUMat)));
		int curr_mat = 0;
		for (auto material : materials)
		{
			size_t pitch;
			float4* tex1;
			float4* tex2;
			float4* tex3;

			size_t tex_width = material.texture_resolution.x * sizeof(float4);
			size_t tex_heighth = material.texture_resolution.y;
			// allocate memory for the triangle meshes on the GPU
			//gpuErrchk(cudaMalloc((void**)& dev_tex_p, textures_size));
			gpuErrchk(cudaMallocPitch((void**)& tex1, &pitch, tex_width, tex_heighth));


			// copy triangle data to GPU
			//gpuErrchk(cudaMemcpy(dev_tex_p, textures.data(), textures_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy2D(tex1, pitch, material.texture_aray[0].data(), tex_width, tex_width, tex_heighth,
				cudaMemcpyHostToDevice));

			// allocate memory for the triangle meshes on the GPU
			//gpuErrchk(cudaMalloc((void**)& dev_tex_p, textures_size));
			gpuErrchk(cudaMallocPitch((void**)& tex2, &pitch, tex_width, tex_heighth));


			// copy triangle data to GPU
			//gpuErrchk(cudaMemcpy(dev_tex_p, textures.data(), textures_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy2D(tex2, pitch, material.texture_aray[1].data(), tex_width, tex_width, tex_heighth,
				cudaMemcpyHostToDevice));

			// allocate memory for the triangle meshes on the GPU
			//gpuErrchk(cudaMalloc((void**)& dev_tex_p, textures_size));
			gpuErrchk(cudaMallocPitch((void**)& tex3, &pitch, tex_width, tex_heighth));


			// copy triangle data to GPU
			//gpuErrchk(cudaMemcpy(dev_tex_p, textures.data(), textures_size, cudaMemcpyHostToDevice));
			gpuErrchk(cudaMemcpy2D(tex3, pitch, material.texture_aray[2].data(), tex_width, tex_width, tex_heighth,
				cudaMemcpyHostToDevice));

			// load triangle data into a CUDA texture
			bind_texture(tex1, tex2, tex3, material.texture_resolution, pitch, curr_mat);
			++curr_mat;
		}
		gpuErrchk(cudaMemcpy(d_materials, h_materials, materials.size() * sizeof(GPUMat), cudaMemcpyHostToDevice));
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
	gpuErrchk(cudaFree(d_volume_instances));
}

extern
void update_render_settings(RenderingSettings render_settings, SceneSettings scene_settings)
{
	cuda_render_settings = render_settings;
	cuda_scene_settings = scene_settings;
}