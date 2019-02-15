#pragma once

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
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

float4 *h_pixels;

__device__
class RKDTreeNodeGPU *d_tree;

__device__
class RCamera *d_render_camera;

__device__
int *d_index_list;

__device__
float4 *d_pixels;

__device__
int *d_root_index;

__device__
RCamera *h_camera;

__device__
struct GPUSceneObject *d_scene_objects;

__device__
int d_object_number;

float4 *Render(class RCamera sceneCam);
class CUDARenderer
{

public:
};