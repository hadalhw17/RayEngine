#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include "cuda_runtime_api.h"

////////////////////////////////////////////////////
// The scene triangles are stored in a 1D 
// CUDA texture of float4 for memory alignment
// Store two edges instead of vertices
// Each triangle is stored as three float4s: 
// (float4 first_vertex, float4 edge1, float4 edge2)
////////////////////////////////////////////////////
texture<float4, 1, cudaReadModeElementType> triangle_texture;

texture<float4, 1, cudaReadModeElementType> normals_texture;

texture<float2, 1, cudaReadModeElementType> uvs_texture;

uchar4 *h_pixels;

__device__
class Atmosphere *d_atmosphere;

__device__
curandState *rand_state;

__device__ __constant__
class RKDTreeNodeGPU *d_tree;

__device__
class float3 *d_light;

size_t num_light;

__device__
class RCamera *d_render_camera;

__device__ __constant__
int *d_index_list;

__device__
uchar4 *d_pixels;

__device__
float4 *d_shadow_map;

__device__
float4 *d_indirect_map;

__device__
struct HitResult *d_hit_result;

__device__
struct HitResult *d_shadow_hit_result;

__device__ __constant__
int *d_root_index;

__device__
RCamera *h_camera;

__device__ __constant__
struct GPUSceneObject *d_scene_objects;

__device__  __constant__
float3 *d_textures;

size_t d_texture_size;

__device__
int d_object_number;

__device__
float4 *dev_triangle_p;

__device__
float4 *dev_normals_p;

__device__
float2 *dev_uvs_p;

float angle;
