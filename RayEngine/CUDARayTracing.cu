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

#include "RayEngine.h"

////////////////////////////////////////////////////
// The scene triangles are stored in a 1D 
// CUDA texture of float4 for memory alignment
// Store two edges instead of vertices
// Each triangle is stored as three float4s: 
// (float4 first_vertex, float4 edge1, float4 edge2)
////////////////////////////////////////////////////
texture<float4, 1, cudaReadModeElementType> triangle_texture;

texture<float4, 1, cudaReadModeElementType> normals_texture;


float4 *dev_triangle_p;
float4 *dev_normals_p;

#define PI_OVER_TWO 1.5707963267948966192313216916397514420985
#define M_PI 3.14156265


////////////////////////////////////////////////////
// Compute normal of a triangle by vertexes
////////////////////////////////////////////////////
__device__
float3 gpu_get_tri_normal(float3 &p1, float3 &u, float3 &v)
{
	float nx = u.y * v.z - u.z * v.y;
	float ny = u.z * v.x - u.x * v.z;
	float nz = u.x * v.y - u.y * v.x;

	
	return normalize(make_float3(nx, ny, nz));
}


////////////////////////////////////////////////////
// Compute normal of a triangle from texture array
////////////////////////////////////////////////////
__device__
float3 gpu_get_tri_normal(int tri_index, float u, float v)
{
	float4 n0 = tex1Dfetch(normals_texture, tri_index * 3);
	float4 n1 = tex1Dfetch(normals_texture, tri_index * 3 + 1);
	float4 n2 = tex1Dfetch(normals_texture, tri_index * 3 + 2);

	return (1 - u - v) * make_float3(n0.x, n0.y, n0.z) + u * make_float3(n1.x, n1.y, n1.z) + v * make_float3(n2.x, n2.y, n2.z);
}


////////////////////////////////////////////////////
// Möller–Trumbore intersection algorithm
// Between ray and triangle
////////////////////////////////////////////////////
__device__
bool gpu_ray_tri_intersect(float3 ray_o, float3 ray_dir, float3 v0, float3 e1, float3 e2, float &t, float3 &normal, float3 &hit_point, int tri_index)
{
	float3 h, s, q;
	float a, f, u, v;

	h = cross(ray_dir, e2);
	a = dot(e1, h);

	if (a > -0.00001f && a < 0.00001f) {
		return false;
	}

	f = 1.0f / a;
	s = ray_o - v0;
	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	q = cross(s, e1);
	v = f * dot(ray_dir, q);

	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}

	// at this stage we can compute t to find out where the intersection point is on the line
	t = f * dot(e2, q);

	if (t > 0.00001f) { // ray intersection
		normal = gpu_get_tri_normal(v0, e1, e2);
		return true;
	}
	else { // this means that there is a line intersection but not a ray intersection
		return false;
	}
}



////////////////////////////////////////////////////
// Ray-box intersection
////////////////////////////////////////////////////
__device__
__forceinline__
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
	if (tmax < .0f) {
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
	float &t, int root_index, int *indexList, float3 &normal, float3 &hit_point)
{

	const float GPU_INFINITY = 999999999.0f;

	RKDTreeNodeGPU curr_node = node[root_index];

	// Perform ray/AABB intersection test.
	float t_entry, t_exit;
	bool intersects_root_node_bounding_box = gpu_ray_box_intersect(curr_node.box, ray_o, ray_dir, t_entry, t_exit);
	float3 tmp_hit_point;

	if (!intersects_root_node_bounding_box) {
		return false;
	}

	bool intersection_detected = false;
	int limit = 0;
	float t_entry_prev = -GPU_INFINITY;
	while (t_entry < t_exit && limit < 500) {
		++limit;
		t_entry_prev = t_entry;

		// Down traversal - Working our way down to a leaf node.
		float3 p_entry = ray_o + (t_entry * ray_dir);
		while (!curr_node.is_leaf) {
			//curr_node = gpu_ray_box_intersect(node[curr_node.left_index].box, ray_o, ray_dir, t_entry, t_exit) ? node[curr_node.left_index] : node[curr_node.right_index];
			curr_node = is_point_to_the_left_of_split(curr_node, p_entry) ? node[curr_node.left_index] : node[curr_node.right_index];
		}

		// We've reached a leaf node.
		// Check intersection with triangles contained in current leaf node.
		for (size_t i = curr_node.index_of_first_object; i < curr_node.index_of_first_object + curr_node.num_objs; ++i)
		{
			int tri = indexList[i];
			float4 v0 =		tex1Dfetch(triangle_texture, tri * 3);
			float4 edge1 =	tex1Dfetch(triangle_texture, tri * 3 + 1);
			float4 edge2 =	tex1Dfetch(triangle_texture, tri * 3 + 2);

			// Perform ray/triangle intersection test.
			float tmp_t = GPU_INFINITY;
			float3 tmp_normal = make_float3(0.0f, 0.0f, 0.0f);
			bool intersects_tri = gpu_ray_tri_intersect(ray_o, ray_dir, make_float3(v0.x, v0.y, v0.z), make_float3(edge1.x, edge1.y, edge1.z),
				make_float3(edge2.x, edge2.y, edge2.z), tmp_t, tmp_normal, tmp_hit_point, tri);

			if (intersects_tri) {
				if (tmp_t < t_exit) {
					intersection_detected = true;
					t_exit = tmp_t;
					normal = tmp_normal;
					tmp_hit_point = ray_o + (t_exit * ray_dir);
				}
			}
		}

		if (intersection_detected)
			break;
		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;
		bool intersects_curr_node_bounding_box = gpu_ray_box_intersect(curr_node.box, ray_o, ray_dir, tmp_t_near, tmp_t_far);
		if (intersects_curr_node_bounding_box) {
			// Set t_entry to be the entrance point of the next (neighboring) node.
			t_entry = tmp_t_far + 0.00001f;
		}
		else {
			// This should never happen.
			// If it does, then that means we're checking triangles in a node that the ray never intersects.
			break;
		}


		//// Get neighboring node using ropes attached to current node.
		//float3 p_exit = ray_o + (t_entry * ray_dir);
		//int new_node_index = get_neighboring_node_index(curr_node, p_exit);

		//// Break if neighboring node not found, meaning we've exited the kd-tree.
		//if (new_node_index == -1) {
		//	//break;
		//}

		curr_node = node[root_index];

	}

	if (intersection_detected) {
		t = t_exit;
		hit_point = tmp_hit_point;
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
	float3 hit_point;
	float3 normal;
	float t = 9999999;
	float4 pixel_color;
	if (stackless_kdtree_traversal(tree, ray_o, ray_dir, t, root_index, indexList, normal, hit_point))
	{
		pixel_color.x = (normal.x < 0.0f) ? (normal.x * -1.0f) : normal.x;;
		pixel_color.y = (normal.y < 0.0f) ? (normal.y * -1.0f) : normal.y;
		pixel_color.z = (normal.z < 0.0f) ? (normal.z * -1.0f) : normal.z;

	}
	Pixel = &pixel_color;
	return Pixel;
}


////////////////////////////////////////////////////
// Generates ray origin and ray direction thread index
////////////////////////////////////////////////////
__device__ 
void generate_ray(float3 &ray_o, float3 &ray_dir, int width, int heigth,
	const RCamera *render_camera, int stride)
{
	int index = ((threadIdx.x * gridDim.x) + blockIdx.x) + stride;

	int x = index % width;
	int y = index / width;

	if (index > (width * heigth)) {
		return;
	}

	float sx = (float)x / (width - 1.0f);
	float sy = 1.0f - ((float)y / (heigth - 1.0f));

	float3 rendercampos = render_camera->campos;

	// compute primary ray direction
	// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
	float3 rendercamview = render_camera->view; rendercamview = normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
	float3 rendercamup = render_camera->camdown; rendercamup = normalize(rendercamup);
	float3 horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis = normalize(horizontalAxis); // Important to normalize!
	float3 verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

	float3 middle = rendercampos + rendercamview;
	float3 horizontal = horizontalAxis * tanf(render_camera->fov.x * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5
	float3 vertical = verticalAxis * tanf(render_camera->fov.y * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5

	// compute pixel on screen
	float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
	float3 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * render_camera->focial_distance); // Important for depth of field!		

	float3 aperturePoint = rendercampos;

	// calculate ray direction of next ray in path
	float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
	apertureToImagePlane = normalize(apertureToImagePlane); // ray direction needs to be normalised

	// ray direction
	float3 rayInWorldSpace = apertureToImagePlane;
	ray_dir = normalize(rayInWorldSpace);

	// ray origin
	ray_o = rendercampos;
}


////////////////////////////////////////////////////
// Cast ray from a pixel
////////////////////////////////////////////////////
__device__ 
float4 *trace_pixel(RKDTreeNodeGPU *tree, float4 *pixels, int width, int heigth,
	const RCamera *render_camera,
	int root_index, int num_faces, int *indexList, int stride)
{
	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, heigth, render_camera, stride);

	pixels = device_trace_ray(tree, ray_o, ray_dir, pixels, root_index, num_faces, indexList);

	return pixels;
}

////////////////////////////////////////////////////
// Initializes ray caster
////////////////////////////////////////////////////
__global__ 
void trace_scene(RKDTreeNodeGPU *tree, int width, int height, float4 *pixels, 
	const RCamera *render_camera,
	int root_index, int num_faces, int *indexList, int stride)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index > (width * height)) {
		return;
	}

	pixels = trace_pixel(tree, pixels, width, height, render_camera,
		root_index, num_faces, indexList, stride);

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
		color.y = 0;
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
// Normal visualisation material
////////////////////////////////////////////////////
__device__
void narmals_mat(float4 &color, float3 normal)
{
	color.x = (normal.x < 0.0f) ? (normal.x * -1.0f) : normal.x;
	color.y = (normal.y < 0.0f) ? (normal.y * -1.0f) : normal.y;
	color.z = (normal.z < 0.0f) ? (normal.z * -1.0f) : normal.z;
}

////////////////////////////////////////////////////
// Normal visualisation material
////////////////////////////////////////////////////
__device__
void simple_shade(float4 &color, float3 normal, float3 ray_dir)
{
	color = make_float4(max(0.f, dot(normal, -ray_dir))); // facing ratio 
}


////////////////////////////////////////////////////
// Sky material represent ray directions
////////////////////////////////////////////////////
__device__
void sky_mat(float4 &color, float3 ray_dir)
{
	// Visualise ray directions on the sky.
	color = make_float4(ray_dir, 0);
	color.x = (color.x < 0.0f) ? (color.x * -1.0f) : color.x;
	color.y = (color.y < 0.0f) ? (color.y * -1.0f) : color.y;
	color.z = (color.z < 0.0f) ? (color.z * -1.0f) : color.z;
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
	float &t_near, int root_index, int *index_list, float3 &normal, float3 &hip_point)
{
	bool inter = stackless_kdtree_traversal(tree, ray_o, ray_dir, t_near, root_index, index_list, normal, hip_point);
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

	lightDir = P - light_pos;
	
	float r2 = light_pos.x * light_pos.x + light_pos.y * light_pos.y + light_pos.z * light_pos.z;
	distance = sqrt(r2);
	lightDir.x /= distance, lightDir.y /= distance, lightDir.z /= distance;
	lightIntensity = make_float4(1, 0.804, 0.8, 1) * 25000 / (4 * M_PI * r2);
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
// Phong light
////////////////////////////////////////////////////
__device__
void phong_light(float4 &finalColor, float3 P, float3 dir, float3 &normal, float3 &hit_point, RKDTreeNodeGPU *tree, 
	int root_index, int *index_list)
{
	float3 bias = normal * make_float3(1e-4);
	float4 diffuse = make_float4(0), specular = make_float4(0);
	float3 lightpos = make_float3(10, 10, 10), lightDir;
	float4 lightInt;
	float tShadowed;
	illuminate(P, lightpos, lightDir, lightInt, tShadowed);

	const float3 ray_o = P + bias;
	float3 ray_dir = -lightDir;

	bool vis = trace_shadow(tree, ray_o, ray_dir, tShadowed, root_index, index_list , normal, hit_point);

	diffuse += (lightInt * make_float4(vis * 0.18) * (max(0.0, dot(normal, (-lightDir)))));

	float3 R = reflect(lightDir, normal);
	specular += (lightInt * make_float4(vis) * (powf(max(0.0, dot(R,(-dir))), 10)));

	finalColor = finalColor + (diffuse * (0.8)) + (specular * (0.2));
}

////////////////////////////////////////////////////
// Shade
////////////////////////////////////////////////////
__device__
void shade(float4 &finalColor, float3 &normal, float3 &hit_point, RKDTreeNodeGPU *tree,
	int root_index, int *index_list)
{
	float3 bias = normal ;
	float3 lightpos = make_float3(1, 30, 5), lightDir;
	float4 lightInt;
	float4 lightColor = make_float4(1);

	float tShad;
	illuminate(hit_point, lightpos, lightDir, lightInt, tShad);

	const float3 ray_o = hit_point + bias;
	float3 ray_dir = -lightDir;

	bool vis = !trace_shadow(tree, ray_o, ray_dir, tShad, root_index, index_list, normal, hit_point);

	finalColor += vis * lightInt * max(0.0, dot(normal,(ray_dir)));

}


////////////////////////////////////////////////////
// Ray casting with brute force approach
////////////////////////////////////////////////////
__global__
void gpu_bruteforce_ray_cast(float4 *image_buffer,
	int width, int height, const RCamera *render_camera,
	int num_faces, GPUBoundingBox bbox, int stride, RKDTreeNodeGPU *tree, int root_index, int *index_list)
{
	int index = ((threadIdx.x * gridDim.x) + blockIdx.x) + stride;
	if (index > width * height)
		return;

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, height, render_camera, stride);

	
	float4 pixel_color = make_float4(0);

	float t_near, t_far;

	// Perform ray-box intersection test.
	bool intersects_aabb = gpu_ray_box_intersect(bbox, ray_o, ray_dir, t_near, t_far);
	if (true)
	{
		float t = 999999999;

		for (int i = 0; i < num_faces; ++i) {

			float4 v0 =	   tex1Dfetch(triangle_texture, i * 3);
			float4 edge1 = tex1Dfetch(triangle_texture, i * 3 + 1);
			float4 edge2 = tex1Dfetch(triangle_texture, i * 3 + 2);

			// Perform ray-triangle intersection test.
			float tmp_t = 999999999;
			float3 tmp_normal;
			float3 hit_point;
			bool intersects_tri = gpu_ray_tri_intersect(ray_o, ray_dir, make_float3(v0.x, v0.y, v0.z), make_float3(edge1.x, edge1.y, edge1.z),
				make_float3(edge2.x, edge2.y, edge2.z), tmp_t, tmp_normal, hit_point, i);

			if (intersects_tri) 
			{
				if (tmp_t < t)
				{
					t = tmp_t;
					hit_point = ray_o + (t * ray_dir);
					//narmals_mat(pixel_color, tmp_normal);
					simple_shade(pixel_color, tmp_normal, ray_dir);
					phong_light(pixel_color, hit_point, ray_dir, tmp_normal, hit_point, tree, root_index, index_list);
				}
			}
		}
	}

	//ambient_light(pixel_color);
	pixel_color = clip(pixel_color);

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
// Bind normals to texture memory
////////////////////////////////////////////////////
void bind_normals_tro_texture(float4 *dev_normals_p, unsigned int number_of_normals)
{
	normals_texture.normalized = false;                      // access with normalized texture coordinates
	normals_texture.filterMode = cudaFilterModePoint;        // Point mode, so no 
	normals_texture.addressMode[0] = cudaAddressModeWrap;    // wrap texture coordinates

	size_t size = sizeof(float4)*number_of_normals;
	const cudaChannelFormatDesc *channelDesc = &cudaCreateChannelDesc<float4>();
	cudaBindTexture(0, (const textureReference *)&normals_texture, (const void *)dev_normals_p, channelDesc, size);
}

////////////////////////////////////////////////////
// Perform ray-casting with kd-tree
////////////////////////////////////////////////////
__global__
void stackless_trace_scene(RKDTreeNodeGPU *tree, int width, int height, float4 *pixels,
	RCamera *render_camera,
	int root_index, int num_faces, int *indexList, int stride)
{

	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > width * height)
		return;

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, height, render_camera, stride);

	float3 normal;
	float3 hit_point;
	float t = 9999999;


	float4 pixel_color = make_float4(0);

	// Perform ray-box intersection test.
	bool intersects_aabb = stackless_kdtree_traversal(tree, ray_o, ray_dir, t, root_index, indexList, normal, hit_point);
	if (intersects_aabb)
	{
		int square = (int)floor(hit_point.x) + (int)floor(hit_point.z);
		//narmals_mat(pixel_color, normal);
		

		//tile_pattern(pixel_color, square);
		simple_shade(pixel_color, normal, ray_dir);
		phong_light(pixel_color, hit_point, ray_dir, normal,hit_point, tree, root_index, indexList);
		//shade(pixel_color, normal, hit_point, tree, root_index, indexList);
	}
	else
	{
		// If ray missed draw sky there.
		sky_mat(pixel_color, ray_dir);
	}

	//ambient_light(pixel_color);
	pixel_color = clip(pixel_color);
	pixels[index + stride] = pixel_color;
	return;
}


////////////////////////////////////////////////////
// Main render function
// All variables are initialized here
// Kernel is being executed
////////////////////////////////////////////////////
extern "C"
float4 *Render(RKDThreeGPU *tree, RCamera sceneCam, std::vector<float4> triangles, std::vector<float4> normals, bool bruteforce = false)
{
	// --------------------------------Initialize host variables----------------------------------------------------

	int size = SCR_WIDTH * SCR_HEIGHT * sizeof(float4);
	float4 *h_pixels = new float4[size];

	RKDTreeNodeGPU *h_tree = tree->GetNodes();
	float size_kd_tree = tree->GetNumNodes() * sizeof(RKDTreeNodeGPU);

	RCamera * h_camera = &sceneCam;

	//------------------------------------------------------------------------------------------------------

	//--------------------------------Initialize device variables-------------------------------------------
	float4 *d_pixels;

	RKDTreeNodeGPU *d_tree;
	GPUBoundingBox bbox;
	RCamera *d_render_camera;

	int d_root_index;
	int *d_index_list;

	bbox = tree->GetNodes()[0].box;

	// initialise array of triangle indecies.
	std::vector<int> kd_tree_tri_indics = tree->GetIndexList();

	float size_kd_tree_tri_indices = kd_tree_tri_indics.size() * sizeof(int);

	std::unique_ptr<int[]> tri_index_array  (new int[kd_tree_tri_indics.size()]);

	for (size_t i = 0; i < kd_tree_tri_indics.size(); ++i) {
		tri_index_array[i] = kd_tree_tri_indics[i];
	}

	float3 *verts = tree->get_verts();
	float size_verts = sizeof(float3) * tree->get_num_verts();

	float3 *faces = tree->get_faces();
	float size_faces = sizeof(float3) * tree->get_num_faces();

	cudaMalloc(&d_pixels, size);
	cudaMalloc(&d_render_camera, sizeof(RCamera));
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

		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void **)&dev_normals_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_normals_p, &normals[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_normals_tro_texture(dev_normals_p, total_num_triangles);
	}


	// Copy host vectors to device.
	cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree, h_tree, size_kd_tree, cudaMemcpyHostToDevice);
	cudaMemcpy(d_index_list, tri_index_array.get(), size_kd_tree_tri_indices, cudaMemcpyHostToDevice);
	d_root_index = tree->get_root_index();

	// Number of threads in each thread block
	int  blockSize = 1280;

	// Number of thread blocks in grid
	int  gridSize = 800;

	int num_verts = tree->get_num_verts();
	int nun_faces = tree->get_num_faces();


	//------------------------------------------------------------------------------------------------------

	//printf("Starting rendering on the GPU with blockSize = %d and gridSize = %d\n", blockSize, gridSize);

	//cudaDeviceSynchronize();

	// Start timer that is used for benchmarking.
	//auto start = std::chrono::steady_clock::now();

	// Perform bruteforce approach or use kd-tree acceleration.
	//if (!bruteforce)
	//{
		//gpu_bruteforce_ray_cast << < blockSize, gridSize >> > (d_pixels, SCR_WIDTH, SCR_HEIGHT,
		//	d_render_camera, nun_faces, bbox, 0, d_tree, d_root_index, d_index_list);
	//}
	//else
	//{
	//}
	stackless_trace_scene << < blockSize, gridSize >> > (d_tree, SCR_WIDTH, SCR_HEIGHT, d_pixels,
		d_render_camera, d_root_index, nun_faces, d_index_list, 0);

	cudaDeviceSynchronize();


	// Finish timer used for benchmarking.
	//auto finish = std::chrono::steady_clock::now();

	//float elapsed_seconds = std::chrono::duration_cast<
	//std::chrono::duration<float>>(finish - start).count();
	//std::cout << "Rendering of a on the GPU frame has finished in " << elapsed_seconds << " seconds." << std::endl;

	// Copy pixel array back to host.
	cudaMemcpy(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);

	// Release device memory.
	cudaFree(d_pixels);
	cudaFree(d_tree);
	cudaFree(d_index_list);
	cudaFree(dev_triangle_p);
	cudaFree(dev_normals_p);
	cudaFree(d_render_camera);

	// Release host memory.
//	free(tri_index_array);

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