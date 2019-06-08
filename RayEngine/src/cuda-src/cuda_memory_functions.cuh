#pragma once
#include "sphere_tracing.cuh"
#include "CUDARayTracing.cuh"
#include "../GPUBoundingBox.h"

////////////////////////////////////////////////////
// Bind triangles to texture memory
////////////////////////////////////////////////////
void bind_triangles_tro_texture(float4* dev_triangle_p, unsigned int number_of_triangles)
{
	triangle_texture.normalized = false;                      // access with normalized texture coordinates
	triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

	size_t size = sizeof(float4) * number_of_triangles * 3;
	const cudaChannelFormatDesc* channelDesc = &cudaCreateChannelDesc<float4>();
	cudaBindTexture(0, (const textureReference*)& triangle_texture, (const void*)dev_triangle_p, channelDesc, size);
}

////////////////////////////////////////////////////
// Bind normals to texture memory
////////////////////////////////////////////////////
void bind_normals_tro_texture(float4* dev_normals_p, unsigned int number_of_normals)
{
	normals_texture.normalized = false;                      // access with normalized texture coordinates
	normals_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	normals_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

	size_t size = sizeof(float4) * number_of_normals;
	const cudaChannelFormatDesc* channelDesc = &cudaCreateChannelDesc<float4>();
	cudaBindTexture(0, (const textureReference*)& normals_texture, (const void*)dev_normals_p, channelDesc, size);
}

////////////////////////////////////////////////////
// Bind uvs to texture memory
////////////////////////////////////////////////////
void bind_uvs_to_texture(float2* dev_uvs_p, unsigned int number_of_uvs)
{
	uvs_texture.normalized = false;                      // access with normalized texture coordinates
	uvs_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	uvs_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

	size_t size = sizeof(float2) * number_of_uvs;
	const cudaChannelFormatDesc* channelDesc = &cudaCreateChannelDesc<float2>();
	cudaBindTexture(0, (const textureReference*)& uvs_texture, (const void*)dev_uvs_p, channelDesc, size);
}


extern "C"
void update_objects(std::vector<GPUSceneObject> objs)
{
	cudaFree(d_scene_objects);
	cudaMalloc(&d_scene_objects, objs.size() * sizeof(GPUSceneObject));
	cudaMemcpy(d_scene_objects, &objs[0], objs.size() * sizeof(GPUSceneObject), cudaMemcpyHostToDevice);
}

extern "C"
void copy_memory(std::vector<RKDThreeGPU*>tree, RCamera sceneCam, std::vector<float4> triangles,
	std::vector<float4> normals, std::vector<float2> uvs, std::vector<GPUSceneObject> objs,
	std::vector<float3> textures, bool bruteforce = false)
{
	// --------------------------------Initialize host variables----------------------------------------------------
	size_t num_objs = objs.size();
	d_texture_size = textures.size();
	size_t textures_size = textures.size() * sizeof(float3);
	size_t image_size = SCR_WIDTH * SCR_HEIGHT;
	int size = image_size * sizeof(uchar4);
	angle = 0;
	int shadow_map_size = image_size * sizeof(float4);
	size_t size_hit_result = image_size * sizeof(HitResult);
	size_t size_kd_tree = 0;


	Atmosphere* h_atmosphere = new Atmosphere();

	for (auto t : tree)
	{
		size_kd_tree += t->GetNumNodes();
	}

	RKDTreeNodeGPU* h_tree = new RKDTreeNodeGPU[size_kd_tree];
	for (int k = 0, i = 0; i < num_objs; ++i)
	{
		for (int n = 0; n < tree[i]->GetNumNodes(); ++n, ++k)
		{
			h_tree[k] = tree[i]->GetNodes()[n];
		}
	}

	h_camera = &sceneCam;
	h_pixels = new uchar4[size];

	std::vector<int> h_root_index = {};
	int offset = 0;
	for (int i = 0; i < num_objs; i++)
	{
		h_root_index.push_back(tree.at(i)->get_root_index());
	}

	size_t light_size = 2;
	float3* h_lights = new float3[light_size];

	h_lights[0] = make_float3(0, 25, 0);
	h_lights[1] = make_float3(3, 15, 10);
	//------------------------------------------------------------------------------------------------------

	//--------------------------------Initialize device variables-------------------------------------------
	// initialise array of triangle indecies.
	std::vector<int> kd_tree_tri_indics = {};
	offset = 0;
	int count = 0;
	for (auto t : tree)
	{
		for (auto n : t->obj_index_list)
		{
			kd_tree_tri_indics.push_back(n + offset);
		}
		offset += objs[count].num_prims;

		count++;
	}

	size_t size_kd_tree_tri_indices = kd_tree_tri_indics.size() * sizeof(int);

	cudaMalloc(&rand_state, image_size * sizeof(curandState));
	cudaMalloc(&d_pixels, size);
	cudaMalloc(&d_atmosphere, sizeof(Atmosphere));
	cudaMalloc(&d_textures, textures_size);
	cudaMalloc(&d_shadow_map, shadow_map_size);
	cudaMalloc(&d_indirect_map, shadow_map_size);
	cudaMalloc(&d_light, light_size * sizeof(float3));
	cudaMalloc(&d_hit_result, size_hit_result);
	cudaMalloc(&d_shadow_hit_result, size_hit_result);
	cudaMalloc(&d_render_camera, sizeof(RCamera));
	cudaMalloc(&d_tree, size_kd_tree * sizeof(RKDTreeNodeGPU));
	cudaMalloc(&d_index_list, size_kd_tree_tri_indices);
	cudaMalloc(&d_scene_objects, num_objs * sizeof(GPUSceneObject));
	cudaMalloc(&d_root_index, num_objs * sizeof(int));


	// calculate total number of triangles in the scene
	size_t triangle_size = triangles.size() * sizeof(float4);
	int total_num_triangles = triangles.size() / 3;

	// calculate total number of normals in the scene
	size_t normals_size = normals.size() * sizeof(float4);
	size_t uvs_size = uvs.size() * sizeof(float2);

	if (uvs_size > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void**)& dev_uvs_p, uvs_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_uvs_p, uvs.data(), uvs_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_uvs_to_texture(dev_uvs_p, uvs_size);
	}

	if (num_objs > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void**)& dev_triangle_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_triangle_p, triangles.data(), triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_triangles_tro_texture(dev_triangle_p, total_num_triangles);

		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void**)& dev_normals_p, normals_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_normals_p, normals.data(), normals_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_normals_tro_texture(dev_normals_p, normals_size);
	}


	// Copy host vectors to device.
	cudaMemset(d_hit_result, 0, size_hit_result);
	cudaMemset(d_shadow_hit_result, 0, size_hit_result);
	cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atmosphere, h_atmosphere, sizeof(Atmosphere), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indirect_map, h_pixels, shadow_map_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_shadow_map, h_pixels, shadow_map_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_light, h_lights, light_size * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree, h_tree, size_kd_tree * sizeof(RKDTreeNodeGPU), cudaMemcpyHostToDevice);
	cudaMemcpy(d_index_list, kd_tree_tri_indics.data(), size_kd_tree_tri_indices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_textures, textures.data(), textures_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_scene_objects, &objs[0], num_objs * sizeof(GPUSceneObject), cudaMemcpyHostToDevice);
	cudaMemcpy(d_root_index, h_root_index.data(), num_objs * sizeof(int), cudaMemcpyHostToDevice);
	d_object_number = num_objs;
	num_light = light_size;

	kd_tree_tri_indics.clear(); //clear content
	kd_tree_tri_indics.resize(0); //resize it to 0
	kd_tree_tri_indics.shrink_to_fit(); //reallocate memory

	triangles.clear(); //clear content
	triangles.resize(0); //resize it to 0
	triangles.shrink_to_fit(); //reallocate memory

	//register_random << <SCR_WIDTH, SCR_HEIGHT >> > (rand_state);

}

