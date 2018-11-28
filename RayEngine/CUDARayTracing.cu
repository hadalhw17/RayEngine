////////////////////////////////////////////////////
// Main CUDA rendering file.
////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_math.h"

#include "Vector.h"
#include "Triangle.h"
#include "Camera.h"
#include "KDTree.h"
#include "Light.h"
#include "Color.h"
#include "Object.h"
#include "KDThreeGPU.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include "RayEngine.h"

////////////////////////////////////////////////////
// The scene triangles are stored in a 1D 
// CUDA texture of float4 for memory alignment
// Store two edges instead of vertices
// Each triangle is stored as three float4s: 
// (float4 first_vertex, float4 edge1, float4 edge2)
////////////////////////////////////////////////////
texture<float4, 1, cudaReadModeElementType> triangle_texture;

float4 *dev_triangle_p;


////////////////////////////////////////////////////
// Compute normal of a triangle by vertexes
////////////////////////////////////////////////////
__device__
float3 gpu_get_tri_normal(float3 &p1, float3 &p2, float3 &p3)
{
	float3 u = p2 - p1;
	float3 v = p3 - p1;

	float nx = u.y * v.z - u.z * v.y;
	float ny = u.z * v.x - u.x * v.z;
	float nz = u.x * v.y - u.y * v.x;

	
	return normalize(make_float3(nx, ny, nz));
}


////////////////////////////////////////////////////
// Compute normal of a triangle from texture array
////////////////////////////////////////////////////
__device__
float3 gpu_get_tri_normal(int tri_index)
{
	float4 u = tex1Dfetch(triangle_texture, tri_index * 3 + 1);
	float4 v = tex1Dfetch(triangle_texture, tri_index * 3 + 2);

	float nx = u.y * v.z - u.z * v.y;
	float ny = u.z * v.x - u.x * v.z;
	float nz = u.x * v.y - u.y * v.x;


	return normalize(make_float3(nx, ny, nz));
}


////////////////////////////////////////////////////
//Ray triangle intersect
////////////////////////////////////////////////////
__device__
bool gpu_ray_tri_intersect(float3 ray_o, float3 ray_dir, float3 v0, float3 v0v1, float3 v0v2, float &t, float3 &normal)
{
	float kEpsilon1 = 0.00001f;

	float u, v;

	float3 pvec = cross(ray_dir, v0v2);
	float det = dot(v0v1, pvec);

	// Ray and triangle are parallel if det is close to 0.
	if (fabs(det) < -kEpsilon1) return false;

	float invDet = 1 / det;

	float3 tvec = ray_o - v0;
	u = dot(tvec, pvec) * invDet;
	if (u < 0 || u > 1) return false;

	float3 qvec = cross(tvec, v0v1);
	v = dot(ray_dir, qvec) * invDet;
	if (v < 0 || u + v > 1) return false;

	t = dot(v0v2, qvec) * invDet;

	
	return (t > kEpsilon1) ? true : false;
}



////////////////////////////////////////////////////
// Ray-box intersection
////////////////////////////////////////////////////
__device__
bool gpu_ray_box_intersect(GPUBoundingBox bbox, float3 ray_o, float3 ray_dir, float &t_near, float &t_far)
{
	float3 dirfrac = make_float3(1.0f / ray_dir.x, 1.0f / ray_dir.y, 1.0f / ray_dir.z);

	float t1 = (bbox.Min.x - ray_o.x) * dirfrac.x;
	float t2 = (bbox.Max.x - ray_o.x) * dirfrac.x;
	float t3 = (bbox.Min.y - ray_o.y) * dirfrac.y;
	float t4 = (bbox.Max.y - ray_o.y) * dirfrac.y;
	float t5 = (bbox.Min.z - ray_o.z) * dirfrac.z;
	float t6 = (bbox.Max.z - ray_o.z) * dirfrac.z;

	float tmin = max(max(min(t1, t2), min(t3, t4)), min(t5, t6));
	float tmax = min(min(max(t1, t2), max(t3, t4)), max(t5, t6));

	// If tmax < 0, ray intersects AABB, but entire AABB is behind ray, so reject.
	if (tmax < 0.000001f) {
		return false;
	}

	// If tmin > tmax, ray does not intersect AABB.
	if (tmin > tmax) {
		return false;
	}

	t_near = tmin;
	t_far = tmax;

	return true;
}


////////////////////////////////////////////////////
// Checks if point is to the left of split plane
////////////////////////////////////////////////////
__device__
bool is_point_to_the_left_of_split(RKDTreeNodeGPU node, const float3 &p)
{
	if (node.axis == X_Axis) {
		return (p.x < node.split_val);
	}
	else if (node.axis == Y_Axis) {
		return (p.y < node.split_val);
	}
	else if (node.axis == Z_Axis) {
		return (p.z < node.split_val);
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else {
		return false;
	}
}


////////////////////////////////////////////////////
// Returns a node index roped to a corresponding face
////////////////////////////////////////////////////
__device__
int get_neighboring_node_index(RKDTreeNodeGPU node, float3 p)
{
	const float GPU_KD_TREE_EPSILON = 0.00001f;

	// Check left face.
	if (fabs(p.x - node.box.Min.x) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[LEFT];
	}
	// Check front face.
	else if (fabs(p.z - node.box.Max.z) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[FRONT];
	}
	// Check right face.
	else if (fabs(p.x - node.box.Max.x) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[RIGHT];
	}
	// Check back face.
	else if (fabs(p.z - node.box.Min.z) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[BACK];
	}
	// Check top face.
	else if (fabs(p.y - node.box.Max.y) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[TOP];
	}
	// Check bottom face.
	else if (fabs(p.y - node.box.Min.y) < GPU_KD_TREE_EPSILON) {
		return node.neighbor_node_indices[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else {
		return -1;
	}
}


////////////////////////////////////////////////////
// Stackless kd-tree traversal algorithm
////////////////////////////////////////////////////
__device__
bool stackless_kdtree_traversal(RKDTreeNodeGPU *node, float3 ray_o, float3 ray_dir, 
	float &t, int root_index, int *indexList, int &hitObject)
{
	const float CUDAkInfinity = 1e20;
	RKDTreeNodeGPU currentNode = node[root_index];
	float t_near, t_far;
	float3 normal;

	bool intersectsBox = gpu_ray_box_intersect(currentNode.box, ray_o, ray_dir, t_near, t_far);
	if (!intersectsBox)
	{
		return false;
	}
	
	while (t_near < t_far)
	{

		float3 entry_point = ray_o + (ray_dir * t_near);
		while (!currentNode.is_leaf)
		{
			currentNode = is_point_to_the_left_of_split(currentNode, entry_point) ? node[currentNode.left_index] : node[currentNode.right_index];
		}

		// We are in a leaf node.
		float tmp_t = CUDAkInfinity, u, v;
		for (size_t i = currentNode.index_of_first_object; i < currentNode.index_of_first_object + currentNode.num_objs; ++i)
		{

			float4 v0 =		tex1Dfetch(triangle_texture, i * 3);
			float4 edge1 =	tex1Dfetch(triangle_texture, i * 3 + 1);
			float4 edge2 =	tex1Dfetch(triangle_texture, i * 3 + 2);
			if (gpu_ray_tri_intersect(ray_o, ray_dir, make_float3(v0.x, v0.y, v0.z), make_float3(edge1.x, edge1.y, edge1.z), 
				make_float3(edge2.x, edge2.y, edge2.z), tmp_t, normal) && tmp_t <  t_far)
			{
				hitObject = (int) i;
				t = tmp_t;
				t_far = tmp_t;
			}

		}

		
		// Compute distance to exit current node.
		float tmp_t_near, tmp_t_far;
		bool intersects_curr_node_bounding_box = gpu_ray_box_intersect(currentNode.box, ray_o, ray_dir, tmp_t_near, tmp_t_far);
		if (intersects_curr_node_bounding_box) {
			// Set t_entry to be the entrance point of the next (neighboring) node.
			t_near = tmp_t_far;
		}
		else {
			// This should never happen.
			// If it does, then that means we're checking triangles in a node that the ray never intersects.
			break;
		}

		// Get neighboring node using ropes attached to current node.
		float3 p_exit = ray_o + (t_near * ray_dir);
		int new_node_index = get_neighboring_node_index(currentNode, p_exit);

		// Break if neighboring node not found, meaning we've exited the kd-tree.
		if (new_node_index == -1) {
			break;
		}

		currentNode = node[new_node_index];

	}
	if (hitObject != 1) {
		t = t_far;
		return true;
	}
	
	return false;
}


////////////////////////////////////////////////////
// Call tree traversal and paind pixels
////////////////////////////////////////////////////
__device__ 
float4 *device_trace_ray(RKDTreeNodeGPU *tree, float3 ray_o, float3 ray_dir, 
	float4 *Pixel, int root_index, int num_faces, int *indexList)
{
	int flat_index = (blockIdx.x * blockDim.x) + threadIdx.x;

	int hit_object = -1;
	float3 normal;
	float t = 9999999, u,v;
	float4 pixel_color;
	if (stackless_kdtree_traversal(tree, ray_o, ray_dir, t, root_index, indexList, hit_object))
	{
		float3 tmp_normal = gpu_get_tri_normal(hit_object);
		pixel_color.x = (tmp_normal.x < 0.0f) ? (tmp_normal.x * -1.0f) : tmp_normal.x;;
		pixel_color.y = (tmp_normal.y < 0.0f) ? (tmp_normal.y * -1.0f) : tmp_normal.y;
		pixel_color.z = (tmp_normal.z < 0.0f) ? (tmp_normal.z * -1.0f) : tmp_normal.z;

	}
	return Pixel;
}


////////////////////////////////////////////////////
// Generates ray origin and ray direction thread index
////////////////////////////////////////////////////
__device__ 
void generate_ray(float3 &ray_o, float3 &ray_dir, int width, int heigth,
	float3 camera_position, float3 camera_direction, float3 camera_down, float3 camera_right, int stride)
{
	int index = ((blockIdx.x * blockDim.x) + threadIdx.x) + stride;

	int x = index % width;
	int y = index / width;

	if (index > (width * heigth)) {
		return;
	}

	float sx = (float)x / (width - 1.0f);
	float sy = 1.0f - ((float)y / (heigth - 1.0f));

	// Cast rays.
	ray_o = camera_position;
	float3 cam_ray_direction = camera_direction + (camera_down + (2.f * sx - 1.f) +
		(camera_right * (2.f * sy - 1.f)));

	ray_dir = normalize(cam_ray_direction - ray_o);
}


////////////////////////////////////////////////////
// Cast ray from a pixel
////////////////////////////////////////////////////
__device__ 
float4 *trace_pixel(RKDTreeNodeGPU *tree, float4 *pixels, int width, int heigth,
	float3 camera_position, float3 camera_direction, float3 camera_down, float3 camera_right,
	int root_index, int num_faces, int *indexList, int stride)
{
	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, heigth, camera_position, camera_direction, camera_down, camera_right, stride);

	pixels = device_trace_ray(tree, ray_o, ray_dir, pixels, root_index, num_faces, indexList);

	return pixels;
}

////////////////////////////////////////////////////
// Initializes ray caster
////////////////////////////////////////////////////
__global__ 
void trace_scene(RKDTreeNodeGPU *tree, int width, int height, float4 *pixels, 
	float3 camera_position, float3 camera_direction, float3 camera_down, float3 camera_right, 
	int root_index, int num_faces, int *indexList, int stride)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index > (width * height)) {
		return;
	}

	pixels = trace_pixel(tree, pixels, width, height, camera_position, camera_direction, 
			camera_down, camera_right, root_index, num_faces, indexList, stride);

}

////////////////////////////////////////////////////
// Perform ray-casting with kd-tree
////////////////////////////////////////////////////
__global__
void stackless_trace_scene(RKDTreeNodeGPU *tree, int width, int height, float4 *pixels,
	float3 camera_position, float3 camera_direction, float3 camera_down, float3 camera_right,
	int root_index, int num_faces, int *indexList, int stride)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index > width * height)
		return;

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, height, camera_position, camera_direction, camera_down, camera_right, stride);

	int hit_object = -1;
	float t = 9999999, u, v;


	float4 pixel_color = make_float4(0);

	// Perform ray-box intersection test.
	float t_near, t_far;
	bool intersects_aabb = stackless_kdtree_traversal(tree, ray_o, ray_dir, t, root_index, indexList, hit_object);
	if (intersects_aabb)
	{

		float3 tmp_normal = gpu_get_tri_normal(hit_object);
		pixel_color.x = (tmp_normal.x < 0.0f) ? (tmp_normal.x * -1.0f) : tmp_normal.x;;
		pixel_color.y = (tmp_normal.y < 0.0f) ? (tmp_normal.y * -1.0f) : tmp_normal.y;
		pixel_color.z = (tmp_normal.z < 0.0f) ? (tmp_normal.z * -1.0f) : tmp_normal.z;
	}

	pixels[index + stride] = pixel_color;
	return;
}

////////////////////////////////////////////////////
// std::swap()
////////////////////////////////////////////////////
template <typename T>
__device__
void inline swap(T& a, T& b)
{
	T c(a); a = b; b = c;
}


////////////////////////////////////////////////////
// Tile material
////////////////////////////////////////////////////
__device__
void tile_pattern(float4 &color, int square)
{
	if ((square % 2) == 0) {
		// black tile
		color.x = 0;
		color.y = 1;
		color.z = 0;
	}
	else {
		// white tile
		color.x = 1;
		color.y = 1;
		color.z = 1;
	}
}


////////////////////////////////////////////////////
// Compute fresnel equation
////////////////////////////////////////////////////
__device__
void fresnel(float3 &I, float3 &N, float &ior, float &kr)
{
	const float dot_p = dot(I, N);
	float cosi = clamp(-1.f, 1.f, dot_p);
	float etai = 1, etat = ior;
	if (cosi > 0) { swap(etai, etat); }
	// Compute sini using Snell's law.
	const float sint = etai / etat * sqrtf(max(0.f, 1 - cosi * cosi));
	// Total internal reflection.
	if (sint >= 1) {
		kr = 1;
	}
	else {
		const float cost = sqrtf(max(0.f, 1 - sint * sint));
		cosi = fabsf(cosi);
		const float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
		const float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
		kr = (Rs * Rs + Rp * Rp) / 2;
	}
}


////////////////////////////////////////////////////
// Refract
////////////////////////////////////////////////////
__device__
float3 refract(float3 &I, float3 &N, float &ior)
{
	float dot_p = dot(I,N);
	float cosi = clamp(-1.f, 1.f, dot_p);
	float etai = 1, etat = ior;
	float3 n = N;
	if (cosi < 0) { cosi = -cosi; }
	else { swap(etai, etat); n = -N; }
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? make_float3(0) : I * (eta + n * eta * cosi - sqrtf(k));
}


////////////////////////////////////////////////////
// Trace for shadows
////////////////////////////////////////////////////
__device__
bool trace_shadow(RKDTreeNodeGPU *tree, float3 ray_o, float3 ray_dir,
	float &t_near, int root_index, int *index_list, int hitObject)
{
	hitObject = -1;

	bool inter = stackless_kdtree_traversal(tree, ray_o, ray_dir, t_near, root_index, index_list, hitObject);
	return inter;
}


////////////////////////////////////////////////////
// Compute light intensity
////////////////////////////////////////////////////
__device__
void illuminate(float3 & P, float3 light_pos, float3 & lightDir, float4 & lightIntensity, float & distance)
{
	// Return not to devide by zero.
	if (distance == 0)
		return;

	lightDir = P- light_pos;
	float r2 = light_pos.x * light_pos.x + light_pos.y * light_pos.y + light_pos.z * light_pos.z;;
	distance = sqrt(r2);
	lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
	lightIntensity = make_float4(1, 1, 1, 1) *500 / (4 * 3.14 * r2);
}


////////////////////////////////////////////////////
// Clip color
////////////////////////////////////////////////////
__device__
float4 clip(float4 color)
{
	float Red = color.x, Green = color.y, Blue = color.z, special = color.w;
	float alllight = color.x + color.y + color.z;
	float excesslight = alllight - 3;
	if (excesslight > 0) {
		Red = Red + excesslight * (Red / alllight);
		Green = Green + excesslight * (Green / alllight);
		Blue = Blue + excesslight * (Blue / alllight);
	}
	if (Red > 1) { Red = 1; }
	if (Green > 1) { Green = 1; }
	if (Blue > 1) { Blue = 1; }
	if (Red < 0) { Red = 0; }
	if (Green < 0) { Green = 0; }
	if (Blue < 0) { Blue = 0; }

	return make_float4(Red, Green, Blue, special);
}


////////////////////////////////////////////////////
// Ambient light
////////////////////////////////////////////////////
__device__
void ambient_light(float4 &color)
{
	color = color * 0.2;
}


////////////////////////////////////////////////////
// Ray casting with brute force approach
////////////////////////////////////////////////////
__global__
void gpu_bruteforce_ray_cast(float4 *image_buffer,
	int width, int height, float3 camera_position, float3 camera_direction, float3 camera_down, float3 camera_right,
	int num_faces, GPUBoundingBox bbox, int stride, RKDTreeNodeGPU *tree, int root_index, int *index_list)
{
	int index = (((blockIdx.x) * blockDim.x) + threadIdx.x) + stride;
	if (index > width * height)
		return;

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, height, camera_position, camera_direction, camera_down, camera_right, stride);

	
	float4 pixel_color = make_float4(0);

	float t_near, t_far;

	// Perform ray-box intersection test.
	bool intersects_aabb = gpu_ray_box_intersect(bbox, ray_o, ray_dir, t_near, t_far);
	if (intersects_aabb)
	{
		float t = 999999999;

		for (int i = 0; i < num_faces; ++i) {

			float4 v0 =	   tex1Dfetch(triangle_texture, i * 3);
			float4 edge1 = tex1Dfetch(triangle_texture, i * 3 + 1);
			float4 edge2 = tex1Dfetch(triangle_texture, i * 3 + 2);

			// Perform ray-triangle intersection test.
			float tmp_t = 999999999;
			float3 tmp_normal = make_float3(0.5f, 0.5f, 0.5f);
			bool intersects_tri = gpu_ray_tri_intersect(ray_o, ray_dir, make_float3(v0.x, v0.y, v0.z), make_float3(edge1.x, edge1.y, edge1.z),
				make_float3(edge2.x, edge2.y, edge2.z), tmp_t, tmp_normal);
			tmp_normal = gpu_get_tri_normal(i);


			if (intersects_tri) 
			{
				if (tmp_t < t)
				{
					t = tmp_t;
					pixel_color.x = (tmp_normal.x < 0.0f) ? (tmp_normal.x * -1.0f) : tmp_normal.x;;
					pixel_color.y = (tmp_normal.y < 0.0f) ? (tmp_normal.y * -1.0f) : tmp_normal.y;
					pixel_color.z = (tmp_normal.z < 0.0f) ? (tmp_normal.z * -1.0f) : tmp_normal.z;

					float3 intersectionPosition = ray_o + ray_dir * t;
					float3 intersectingRayDirection = ray_dir;
					float3 normal = gpu_get_tri_normal(i);
					float3 bias = normal * make_float3(1e-4);
				}
			}
		}
	}
	ambient_light(pixel_color);

	image_buffer[index] = pixel_color;
	return;
}


////////////////////////////////////////////////////
// Bind triangles to texture memory
////////////////////////////////////////////////////
void bind_triangles_tro_texture(float4 *dev_triangle_p, unsigned int number_of_triangles)
{
	triangle_texture.normalized = false;                      // access with normalized texture coordinates
	triangle_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	triangle_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

	size_t size = sizeof(float4)*number_of_triangles * 3;
	const cudaChannelFormatDesc *channelDesc = &cudaCreateChannelDesc<float4>();
	cudaBindTexture(0,(const textureReference *) &triangle_texture, (const void *)dev_triangle_p, channelDesc, size);
}


////////////////////////////////////////////////////
// Main render function
// All variables are initialized here
// Kernel is being executed
////////////////////////////////////////////////////
extern "C"
float4 *Render(RKDThreeGPU *tree, RCamera sceneCam, std::vector<float4> triangles, bool bruteforce = false)
{


	// --------------------------------Initialize host variables----------------------------------------------------
	// Size of vectors
	int n = SCR_WIDTH * SCR_HEIGHT;

	float aspectratio = SCR_WIDTH / (float)SCR_HEIGHT;

	// Make camera from CUDA types for optimisation
	float3 camera_position =	sceneCam.getCameraPosition();
	float3 camera_direction =	sceneCam.getCameraDirection();
	float3 camera_down =		sceneCam.getCameraDown();
	float3 camera_right =		sceneCam.getCameraRight();

	int size = SCR_WIDTH * SCR_HEIGHT * sizeof(float4);
	float4 *h_pixels = new float4[size];

	RKDTreeNodeGPU *h_tree = tree->GetNodes();
	float size_kd_tree = tree->GetNumNodes() * sizeof(RKDTreeNodeGPU);

	//------------------------------------------------------------------------------------------------------

	//--------------------------------Initialize device variables-------------------------------------------
	float4 *d_pixels;
	int d_widh, d_height;

	RKDTreeNodeGPU *d_tree;
	GPUBoundingBox bbox;
	int d_root_index;
	int *d_index_list;

	bbox = tree->GetNodes()[0].box;

	// initialise array of triangle indecies.
	std::vector<int> kd_tree_tri_indics = tree->GetIndexList();
	float size_kd_tree_tri_indices = kd_tree_tri_indics.size() * sizeof(int);
	int *tri_index_array = new int[kd_tree_tri_indics.size()];
	for (int i = 0; i < kd_tree_tri_indics.size(); ++i) {
		tri_index_array[i] = kd_tree_tri_indics[i];
	}

	float3 *verts = tree->get_verts();
	float size_verts = sizeof(float3) * tree->get_num_verts();

	float3 *faces = tree->get_faces();
	float size_faces = sizeof(float3) * tree->get_num_faces();

	cudaMalloc(&d_pixels, size);
	cudaMalloc(&d_tree, size_kd_tree);
	cudaMalloc(&d_index_list, size_kd_tree_tri_indices);

	// calculate total number of triangles in the scene
	size_t triangle_size = triangles.size() * sizeof(float4);
	int total_num_triangles = triangles.size() / 3;


	if (tree->get_num_faces() > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void **)&dev_triangle_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_triangle_p, &triangles[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_triangles_tro_texture(dev_triangle_p, total_num_triangles);
	}


	// Copy host vectors to device.
	cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree, h_tree, size_kd_tree, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index_list, tri_index_array, size_kd_tree_tri_indices, cudaMemcpyHostToDevice);
	d_root_index = tree->GoutRootIndes();

	// Number of threads in each thread block
	int  blockSize = 1280;

	// Number of thread blocks in grid
	int  gridSize = 720;

	int num_verts = tree->get_num_verts();
	int nun_faces = tree->get_num_faces();

	int stride = 0;

	//------------------------------------------------------------------------------------------------------

	printf("Starting rendering on the GPU with blockSize = %d and gridSize = %d\n", blockSize, gridSize);

	cudaDeviceSynchronize();

	// Start timer that is used for benchmarking.
	auto start = std::chrono::steady_clock::now();

	// Perform bruteforce approach or use kd-tree acceleration.
	if (bruteforce)
	{
		gpu_bruteforce_ray_cast << < 1024, 1024 >> > (d_pixels, SCR_WIDTH, SCR_HEIGHT, 
			camera_position, camera_direction, camera_down, camera_right, nun_faces, bbox, 0, d_tree, d_root_index, d_index_list);
	}
	else
	{
		stackless_trace_scene << < blockSize, gridSize >> > (d_tree, SCR_WIDTH, SCR_HEIGHT, d_pixels,
			camera_position, camera_direction, camera_down, camera_right,
			d_root_index, nun_faces, d_index_list, 0);
	}

	cudaDeviceSynchronize();


	// Finish timer used for benchmarking.
	auto finish = std::chrono::steady_clock::now();

	float elapsed_seconds = std::chrono::duration_cast<
	std::chrono::duration<float>>(finish - start).count();
	std::cout << "Rendering of a on the GPU frame has finished in " << elapsed_seconds << " seconds." << std::endl;

	// Copy pixel array back to host.
	cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);

	// Release device memory.
	cudaFree(d_pixels);
	cudaFree(d_tree);
	cudaFree(d_index_list);

	// Release host memory.
	delete[] tri_index_array;

	triangles.clear(); //clear content
	triangles.resize(0); //resize it to 0
	triangles.shrink_to_fit(); //reallocate memory

	kd_tree_tri_indics.clear(); //clear content
	kd_tree_tri_indics.resize(0); //resize it to 0
	kd_tree_tri_indics.shrink_to_fit(); //reallocate memory

	// Check for CUDA runtime API calls errors.
	cudaError_t cudaError;
	cudaError = cudaGetLastError();

	if (cudaError != cudaSuccess)
	{
		printf("cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}

	return h_pixels;
}