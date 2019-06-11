#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cuda_runtime_api.h"
#include "sdf_functions.cuh"
#include "../material_functions.cuh"
#include "../SDF/Chunk.h"
#include "helper_math.h"
#include "sdf_functions.cuh"
#include "../Material.h"
#include "../TextureObject.h"
#include "../PerlinNoise.h"


cudaArray* d_volumeArray = 0;
cudaArray* d_normalArray = 0;
cudaTextureObject_t	texObject; // For 3D texture
cudaTextureObject_t	normalObject; // For 3D texture


struct GPUMat
{
	texturess textttt;
};

const unsigned tableSize = 256;
const unsigned tableSizeMask = tableSize - 1;

__device__
float3* d_gradients;

__device__
unsigned* d_permutationTable;

__device__ __constant__
GPUMat* d_materials;


GPUMat* h_materials;


float2* h_grid;

__device__ __constant__
struct GPUBoundingBox* d_sdf_volumes;

GPUBoundingBox* h_volumes;

struct GPUVolumeObjectInstance* volume_objs;

__device__
float2* d_sdf;

__device__
float2* d_sdf_norm;

__device__ __constant__
float* dev_sdf_p;

float3 sdf_spacing;

uint3 sdf_dim;

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
float4 nearest_neightbour_filter(float x, float y, float4* texture)
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
void bind_texture(float4** dev_texture1, const uint2& texture_resolution, const size_t& pitch, const int& curr_material, const int& i)
{
	// Create 3 texture arrays( 1 for each plane(triplanar texture mapping)) 
	cudaArray* d_textureArray1;

	cudaChannelFormatDesc arr_channelDesc = cudaCreateChannelDesc<float4>();

	// Create array for texture 1
	gpuErrchk(cudaMallocArray(&d_textureArray1, &arr_channelDesc, texture_resolution.x, texture_resolution.y));
	gpuErrchk(cudaMemcpy2DToArray(d_textureArray1, 0, 0, *dev_texture1, pitch, texture_resolution.x * sizeof(float4), texture_resolution.y, cudaMemcpyDeviceToDevice));
	//Array creation End


	cudaResourceDesc res_desc1;
	memset(&res_desc1, 0, sizeof(cudaResourceDesc));
	res_desc1.resType = cudaResourceTypeArray;
	res_desc1.res.array.array = d_textureArray1;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));

	tex_desc.normalizedCoords = true;                      // access with normalized texture coordinates
	tex_desc.filterMode = cudaFilterModeLinear;        // Point mode, so no 
	tex_desc.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates
	tex_desc.addressMode[1] = cudaAddressModeWrap;    // wrap texture coordinates
	tex_desc.readMode = cudaReadModeElementType;
	h_materials[curr_material].textttt.resolution[i] = texture_resolution;
	gpuErrchk(cudaCreateTextureObject(&h_materials[curr_material].textttt.texture[i], &res_desc1, &tex_desc, NULL));
}


__device__
float f(float x, float z, float freq, float amp)
{
	float y = sinf(x) * cosf(z);
	return sinf(amp + (y * freq) * cosf(amp * 2 + y * freq / 3));
}

__device__
bool single_ray_sphere_trace(const RenderingSettings& render_settings, const SceneSettings& scene_settings, const cudaTextureObject_t& tex,
	const HitResult& hit_result, const  GPUVolumeObjectInstance*& __restrict__ instances,
	float& t_near, const const float& t_far, const  float3& step, const int& nearest_shape, float& prel, const float& threshold, int& material_index)
{
	GPUVolumeObjectInstance curr_obj = instances[nearest_shape];
	float t = t_near;
	while (t < t_far)
	{
		float3 from = hit_result.ray_o + (t * hit_result.ray_dir);
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
		t += min_dist * clamp(t * 0.6f, 0.45, 0.8f);
		if (min_dist <= threshold * t)
		{
			t_near = t;
			prel = 0;
			return true;
		}
		prel = fminf(prel, scene_settings.soft_shadow_k * min_dist / t);
	}
	return false;
}

__device__
bool single_shadow_ray_sphere_trace(const RenderingSettings& render_settings, const SceneSettings& scene_settings, const cudaTextureObject_t& tex, const HitResult& hit_result,
	float& t_near, const float& t_far, const float3& step, const int& nearest_shape, float& prel, const float& threshold, int& material_index)
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
void sphere_trace_shadow(const RenderingSettings& render_settings, const SceneSettings& scene_settings, const cudaTextureObject_t& tex,
	const GPUVolumeObjectInstance*& __restrict__ instances, const float3& p_hit, const float3& normal, const GPUBoundingBox*& __restrict__ volumes, const int& num_sdf,
	const float3& step, const uint3& dim, const float3& light, float3& final_colour, const int& material_index)
{
	float3 pixel_color = make_float3(0.f);

	float light_dst = K_INFINITY;
	float x = scene_settings.light_pos.x;
	float z = scene_settings.light_pos.x;
	float3 lightpos = make_float3(x, scene_settings.light_pos.y, x), lightDir;
	float3 lightInt;
	illuminate(p_hit, lightpos, lightDir, lightInt, light_dst, scene_settings.light_intensity);

	HitResult hit_result;
	hit_result.ray_o = p_hit;
	hit_result.ray_dir = normalize(-lightDir);
	float light_t_near, t_far;
	if (!gpu_ray_box_intersect(volumes[0], hit_result.ray_o, hit_result.ray_dir, light_t_near, t_far))
	{
		return;
	}
	float sh = 1.f;

	float t = 0;
	int step_count = 0;
	float t_near = 0;
	int material_ind;
	bool intersects = single_shadow_ray_sphere_trace(render_settings, scene_settings, tex, hit_result,
		t_near, t_far, step, 0, sh, 10e-6, material_ind);
	float amb = __saturatef(0.5 + 0.5 * normal.y);
	float soft_shadow = __saturatef(sh);
	float dif = fmaxf(.0f, dot(normal, hit_result.ray_dir));
	float bac = __saturatef(0.2 + 0.8 * dot(normalize(make_float3(-lightDir.x, 0.0, lightDir.z)), normal));

	pixel_color += soft_shadow * lightInt * dif;
	pixel_color += amb * make_float3(0.40, 0.60, 1.00) * 1.2;
	pixel_color += bac * make_float3(0.40, 0.50, 0.60);

	final_colour *= pixel_color;

	return;
}

inline __device__
float3 powf(float3 a, float b)
{
	return make_float3(pow(a.x, b), pow(a.y, b), pow(a.z, b));
}

inline __device__
float4 powf(float4 a, float b)
{
	return make_float4(pow(a.x, b), pow(a.y, b), pow(a.z, b), 0);
}
__device__
float3 triplanar_mapping(const float3& norm, const float3& p_hit, const texturess& textures)
{
	float3 blending = fabs(norm);
	blending = normalize(fmaxf(blending, make_float3(0.00001))); // Force weights to sum to 1.0
	const float b = (blending.x + blending.y + blending.z);
	blending /= make_float3(b);

	float4 xaxis = tex2D<float4>(textures.texture[0], fabs(p_hit.y), fabs(p_hit.z));
	float4 yaxis = tex2D<float4>(textures.texture[1], fabs(p_hit.z), fabs(p_hit.x));
	float4 zaxis = tex2D<float4>(textures.texture[2], fabs(p_hit.x), fabs(p_hit.y));

	// blend the results of the 3 planar projections.
	float3 tex = blending.x * make_float3(xaxis.x, xaxis.y, xaxis.z) + blending.y * make_float3(yaxis.x, yaxis.y, yaxis.z) + blending.z * make_float3(zaxis.x, zaxis.y, zaxis.z);
	return tex;
}

__device__
float fbm(float2& p, const unsigned* __restrict__ d_permutationTable)
{
	float2x2 m2;
	m2.m[0] = make_float2(0.8, -0.6);
	m2.m[1] = make_float2(0.6, 0.8);
	float f = 0.0;
	f += 0.5000 * d_permutationTable[int(clamp((p.x + 256 * p.y), 0.f, 255.f * 255.f))]; p = mul(m2, p) * 2.02;
	f += 0.2500 * d_permutationTable[int(clamp((p.x + 256 * p.y), 0.f, 255.f * 255.f))]; p = mul(m2, p) * 2.03;
	f += 0.1250 * d_permutationTable[int(clamp((p.x + 256 * p.y), 0.f, 255.f * 255.f))]; p = mul(m2, p) * 2.01;
	f += 0.0625 * d_permutationTable[int(clamp((p.x + 256 * p.y), 0.f, 255.f * 255.f))];
	return f / 0.9375;
}

__device__
BiomeTypes biomes(const float& lh, const float& elev)
{
	if (lh <= (0.2 * elev)) return BIOME_OCEAN;
	if (lh >= 0.8 * elev) return BiomeTypes::BIOME_SNOW;

}


inline __device__
float fract(float a)
{
	return a - (int)floorf(a);
}

__device__
void sphere_trace_shade(const RenderingSettings& render_settings, const cudaTextureObject_t& tex, const cudaTextureObject_t& normal_tex,
	const SceneSettings& scene, const float3* lights, const int& num_lights, HitResult hit_result,
	const GPUVolumeObjectInstance* __restrict__ instance, const float& t, const GPUBoundingBox* __restrict__ volumes,
	const int& num_sdf, const float3& step, const uint3& dim, const int& material_index,
	float3& final_colour, const int& step_count, const bool& shade, const GPUMat* __restrict__ materials, const unsigned* __restrict__ d_permutationTable)
{
	final_colour = make_float3(0);

	GPUVolumeObjectInstance curr_obj;
	float3 p_hit = hit_result.ray_o + (t)* hit_result.ray_dir;
	hit_result.ray_o = p_hit;

	//---------------------- Validate hit point and normal---------------------------
	if (!point_in_aabb(volumes[0], p_hit)) return;
	if (isinf(p_hit.x) || isinf(p_hit.y) || isinf(p_hit.z)) return;
	float3 n = compute_sdf_normal(p_hit, t, render_settings, tex, step, curr_obj);
	if (n.x != n.x || n.y != n.y || n.z != n.z) return;
	//-------------------------------------------------------------------------------


	if (material_index == -1)
	{
		int square = floor(p_hit.x) + floor(p_hit.z);
		tile_pattern(final_colour, square);
	}
	else
	{
		final_colour = mix(final_colour, triplanar_mapping(n, (p_hit) / render_settings.texture_scale, materials[material_index].textttt), 0.2);
	}

	switch (biomes(p_hit.y, volumes[0].Max.y))
	{
	case BiomeTypes::BIOME_OCEAN:
		final_colour = make_float3(0.1f, 0.1f, 0.8f);
		break;
	case BiomeTypes::BIOME_SNOW:
		// snow
		float h = smoothstep(55.0, 80.0, p_hit.y / (volumes[0].Max.y / 2) + 25.0 * fbm(0.01 * make_float2(p_hit.x, p_hit.z) / volumes[0].Max.y / 2, d_permutationTable));
		float e = smoothstep(1.0 - 0.5 * h, 1.0 - 0.1 * h, n.y);
		float o = 0.3 + 0.7 * smoothstep(0.0, 0.1, n.x + h * h);
		float s = h * e * o;
		final_colour = mix(final_colour, 0.29 * make_float3(0.62, 0.65, 0.7), smoothstep(0.1, 0.9, s));
		break;
	default:
		final_colour = make_float3(0.f);
		break;
	}

	//if (fract(p_hit.x / step.x) < 0.1f || fract(p_hit.y / step.y) < 0.1f || fract(p_hit.z / step.z) < 0.1f)
	//{
	//	final_colour = { 0 };
	//}
	if (shade)
	{
#pragma unroll 1
		for (int i = 0; i < num_lights; ++i)
		{
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
void bind_sdf_to_texture(float2** dev_sdf_p, const uint3& dim, const int& num_sdf)
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
	copyParams.srcPtr = make_cudaPitchedPtr(*dev_sdf_p, num_sdf * dim.x * sizeof(float2), dim.y, dim.z);
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

////////////////////////////////////////////////////
// Bind triangles to texture memory
////////////////////////////////////////////////////
void bind_sdf_norm_to_texture(float2* dev_sdf_p, const uint3& dim, const int& num_sdf)
{
	if (normalObject)
	{
		gpuErrchk(cudaDestroyTextureObject(normalObject));
		gpuErrchk(cudaFreeArray(d_normalArray));
	}
	cudaChannelFormatDesc arr_channelDesc = cudaCreateChannelDesc<float2>();
	gpuErrchk(cudaMalloc3DArray(&d_normalArray, &arr_channelDesc, make_cudaExtent(num_sdf * dim.x * sizeof(float2), dim.y, dim.z), 0));

	cudaMemcpy3DParms copyParams = { 0 };

	//Array creation
	copyParams.srcPtr = make_cudaPitchedPtr(dev_sdf_p, num_sdf * dim.x * sizeof(float2), dim.y, dim.z);
	copyParams.dstArray = d_normalArray;
	copyParams.extent = make_cudaExtent(num_sdf * dim.x, dim.y, dim.z);
	copyParams.kind = cudaMemcpyDeviceToDevice;
	cudaMemcpy3D(&copyParams);
	//Array creation End

	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(cudaResourceDesc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = d_normalArray;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));

	tex_desc.normalizedCoords = false;                      // access with normalized texture coordinates
	tex_desc.filterMode = cudaFilterModeLinear;        // Point mode, so no 
	tex_desc.addressMode[0] = cudaAddressModeBorder;    // wrap texture coordinates
	tex_desc.addressMode[1] = cudaAddressModeBorder;    // wrap texture coordinates
	tex_desc.addressMode[2] = cudaAddressModeBorder;    // wrap texture coordinates
	tex_desc.readMode = cudaReadModeElementType;

	gpuErrchk(cudaCreateTextureObject(&normalObject, &res_desc, &tex_desc, NULL));
}

extern "C"
void cuda_update_chunk(const RayEngine::RChunk& world_chunk, const RayEngine::RPerlinNoise& noise)
{
	printf("RENDER: updating chunk");

	gpuErrchk(cudaFree(d_sdf_volumes));
	gpuErrchk(cudaFree(d_sdf));
	const Grid& sdf = world_chunk.get_sdf();
	float size_sdf = sdf.voxels.size() * sizeof(float2);
	float3 h_sdf_steps;
	uint3 h_sdf_dim;
	float3 h_max_sdf;
	h_volumes = new GPUBoundingBox();

	volume_objs = new GPUVolumeObjectInstance();
	volume_objs[0] = GPUVolumeObjectInstance(0, world_chunk.get_location(), make_float3(0));
	d_num_instances = 1;

	sdf_spacing = (&sdf)[0].spacing;
	sdf_dim = (&sdf)[0].sdf_dim;

	h_grid = new float2[(&sdf)[0].voxels.size()];
	//h_grid = sdf[0].voxels.data();
	for (size_t iz = 0; iz < sdf_dim.z; ++iz)
	{
		for (size_t iy = 0; iy < sdf_dim.y; ++iy)
		{
			for (size_t ix = 0; ix < sdf_dim.x; ++ix)
			{
				h_grid[ix + sdf_dim.y * (iy + sdf_dim.x * iz)] = make_float2((&sdf)[0].voxels.at((ix + sdf_dim.y * (iy + sdf_dim.x * iz))).distance, (&sdf)[0].voxels.at((ix + sdf_dim.y * (iy + sdf_dim.x * iz))).material);
			}
		}
	}


	h_sdf_steps = (&sdf)[0].spacing;
	h_sdf_dim = (&sdf)[0].sdf_dim;
	h_max_sdf = (&sdf)[0].box_max;
	h_volumes = new GPUBoundingBox(make_float3(0.f) + 0.5f * h_sdf_steps, (&sdf)[0].box_max - 0.5f * h_sdf_steps);

	gpuErrchk(cudaMalloc((void**)& d_sdf, size_sdf));

	gpuErrchk(cudaMalloc(&d_sdf_volumes, sizeof(GPUBoundingBox)));
	gpuErrchk(cudaMalloc(&d_volume_instances, 200 * sizeof(GPUVolumeObjectInstance)));


	gpuErrchk(cudaMemcpy(d_sdf, h_grid, size_sdf, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_volume_instances, volume_objs, sizeof(GPUVolumeObjectInstance), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sdf_volumes, h_volumes, sizeof(GPUBoundingBox), cudaMemcpyHostToDevice));

	bind_sdf_to_texture(&d_sdf, (&sdf)[0].sdf_dim, 1);

}

extern "C"
void initialize_volume_render(RCamera& sceneCam, const RayEngine::RChunk& world_chunk, const int& num_sdf, const std::vector<VoxelMaterial>& materials, const RenderingSettings& render_settings,
	const SceneSettings& scene_settings, const RayEngine::RPerlinNoise& noise)
{
	const Grid& sdf = world_chunk.get_sdf();
	float size_sdf = num_sdf * sdf.voxels.size() * sizeof(float2);
	h_camera = &sceneCam;
	num_light = 1;
	float3* h_lights = new float3[num_light];
	h_lights[0] = make_float3(0, 25, 0);
	//h_lights[1] = make_float3(3, 15, 10);
	Atmosphere* h_atmosphere = new Atmosphere();
	float3* h_sdf_steps = new float3[num_sdf];
	uint3* h_sdf_dim = new uint3[num_sdf];
	float3* h_max_sdf = new float3[num_sdf];
	h_volumes = new GPUBoundingBox[num_sdf];

	volume_objs = new GPUVolumeObjectInstance[200];
	volume_objs[0] = GPUVolumeObjectInstance(0, world_chunk.get_location(), make_float3(0));
	d_num_instances = 1;

	sdf_spacing = (&sdf)[0].spacing;
	sdf_dim = (&sdf)[0].sdf_dim;

	h_grid = new float2[num_sdf * (&sdf)[0].voxels.size()];
	//h_grid = sdf[0].voxels.data();
	for (size_t iz = 0; iz < sdf_dim.z; ++iz)
	{
		for (size_t iy = 0; iy < sdf_dim.y; ++iy)
		{
			for (size_t g = 0; g < num_sdf; ++g)
			{
				for (size_t ix = 0; ix < sdf_dim.x; ++ix)
				{
					h_grid[ix + sdf_dim.y * (iy + sdf_dim.x * iz)] = make_float2((&sdf)[0].voxels.at((ix + sdf_dim.y * (iy + sdf_dim.x * iz))).distance, (&sdf)[0].voxels.at((ix + sdf_dim.y * (iy + sdf_dim.x * iz))).material);
				}
			}
		}
	}

	for (size_t g = 0; g < num_sdf; ++g)
	{
		h_sdf_steps[g] = (&sdf)[g].spacing;
		h_sdf_dim[g] = (&sdf)[g].sdf_dim;
		h_max_sdf[g] = (&sdf)[g].box_max;
		h_volumes[g] = GPUBoundingBox(make_float3(0.f) + 0.5f * h_sdf_steps[g], (&sdf)[g].box_max - 0.5f * h_sdf_steps[g]);

	}

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
	gpuErrchk(cudaMalloc((void**)& d_sdf_norm, size_sdf));
	//gpuErrchk(cudaMalloc(&d_pixels, size));
	gpuErrchk(cudaMalloc(&d_render_camera, sizeof(RCamera)));
	gpuErrchk(cudaMalloc(&d_light, num_light * sizeof(float3)));
	gpuErrchk(cudaMalloc(&d_gradients, tableSize * sizeof(float3)));
	gpuErrchk(cudaMalloc(&d_permutationTable, tableSize * 2 * sizeof(unsigned)));
	gpuErrchk(cudaMalloc(&d_atmosphere, sizeof(Atmosphere)));
	gpuErrchk(cudaMalloc(&d_sdf_volumes, num_sdf * sizeof(GPUBoundingBox)));
	gpuErrchk(cudaMalloc(&d_volume_instances, 200 * sizeof(GPUVolumeObjectInstance)));


	// copy triangle data to GPU
	gpuErrchk(cudaMemcpy(d_sdf, h_grid, size_sdf, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sdf_norm, h_grid, size_sdf, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice));
	//gpuErrchk(cudaMemcpy(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_light, h_lights, num_light * sizeof(float3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_gradients, noise.gradients, tableSize * sizeof(float3), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_permutationTable, noise.permutationTable, tableSize * 2 * sizeof(unsigned), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_sdf_volumes, h_volumes, num_sdf * sizeof(GPUBoundingBox), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_atmosphere, h_atmosphere, sizeof(Atmosphere), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_volume_instances, volume_objs, 2 * sizeof(GPUVolumeObjectInstance), cudaMemcpyHostToDevice));
	d_num_sdf = num_sdf;
	cuda_render_settings = render_settings;
	cuda_scene_settings = scene_settings;

	// load triangle data into a CUDA texture
	bind_sdf_to_texture(&d_sdf, (&sdf)[0].sdf_dim, num_sdf);
	//bind_sdf_norm_to_texture(d_sdf, sdf[0].sdf_dim, num_sdf);

	if (materials.size() > 0)
	{
		h_materials = new GPUMat[materials.size()];
		gpuErrchk(cudaMalloc((void**)& d_materials, materials.size() * sizeof(GPUMat)));
		int curr_mat = 0;
		for (auto material : materials)
		{
			for (int i = 0; i < 3; ++i)
			{
				size_t pitch;
				float4* tex1;
				size_t tex_width = material.texture_aray.at(i).resolution.x * sizeof(float4);
				size_t tex_heighth = material.texture_aray.at(i).resolution.x;

				// allocate memory for the triangle meshes on the GPU
				gpuErrchk(cudaMallocPitch((void**)& tex1, &pitch, tex_width, tex_heighth));


				// copy triangle data to GPU
				gpuErrchk(cudaMemcpy2D(tex1, pitch, material.texture_aray.at(i).texels.data(), tex_width, tex_width, tex_heighth,
					cudaMemcpyHostToDevice));

				bind_texture(&tex1, material.texture_aray.at(i).resolution, pitch, curr_mat, i);
				cudaFree(tex1);
			}
			++curr_mat;
		}
		gpuErrchk(cudaMemcpy(d_materials, h_materials, materials.size() * sizeof(GPUMat), cudaMemcpyHostToDevice));
		delete[] h_materials;
	}
	delete[] h_grid, h_atmosphere, h_sdf_dim, h_sdf_dim, h_lights, h_volumes, h_max_sdf, volume_objs;
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
void update_render_settings(const RenderingSettings& render_settings, const SceneSettings& scene_settings)
{
	cuda_render_settings = render_settings;
	cuda_scene_settings = scene_settings;
}

