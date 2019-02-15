////////////////////////////////////////////////////
// Main CUDA rendering file.
////////////////////////////////////////////////////

#include "CUDARayTracing.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


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
bool gpu_ray_tri_intersect(float3 ray_o, float3 ray_dir, float3 v0, float3 e1, float3 e2, int tri_index, HitResult &hit_result)
{
	float3 h, s, q;
	float a, f, u, v;

	h = cross(ray_dir, e2);
	a = dot(e1, h);

	if (a > -1e-4 && a < 1e-4) {
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
	hit_result.t = f * dot(e2, q);

	if (hit_result.t > 1e-4) { // ray intersection
		hit_result.normal = gpu_get_tri_normal(v0, e1, e2);
		hit_result.hit_point = ray_o + (hit_result.t * ray_dir);
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

	float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
	float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

	// If tmax < 0, ray intersects AABB, but entire AABB is behind ray, so reject.
	if (tmax < 0) {
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
	const float GPU_KD_TREE_EPSILON = 1e-4;

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
	GPUSceneObject *scene_objs, int num_objs, int curr_obj_count,
	int *root_index, int *indexList, HitResult &hit_result)
{
	// Iterate ofer all of the individual trees in the scene.
	RKDTreeNodeGPU curr_node = node[root_index[curr_obj_count]];

	// Perform ray/AABB intersection test.
	float t_entry, t_exit;
	GPUBoundingBox transformed_box = curr_node.box;
	transformed_box.Min += scene_objs[curr_obj_count].location;
	transformed_box.Max += scene_objs[curr_obj_count].location;
	bool intersects_root_node_bounding_box = gpu_ray_box_intersect(curr_node.box, ray_o, ray_dir, t_entry, t_exit);
	

	if (!intersects_root_node_bounding_box) {
		return false;
	}


	HitResult tmp_hit_result;
	bool intersection_detected = false;
	int limit = 0;
	float t_entry_prev = -kInfinity;

	while (t_entry < t_exit && limit < 500) {
		++limit;
		t_entry_prev = t_entry;

		// Down traversal - Working our way down to a leaf node.
		float3 p_entry = ray_o + (t_entry * ray_dir);
		while (!curr_node.is_leaf) {
			curr_node = is_point_to_the_left_of_split(curr_node, p_entry) ? node[curr_node.left_index] : node[curr_node.right_index];
		}

		// We've reached a leaf node.
		// Check intersection with triangles contained in current leaf node.
		

		for (size_t i = curr_node.index_of_first_object; i < curr_node.index_of_first_object + curr_node.num_objs; ++i)
		{
			int tri = indexList[i + scene_objs[curr_obj_count].offset];
			float4 v00 =	tex1Dfetch(triangle_texture, tri * 3 + 0);
			float4 edge1 =	tex1Dfetch(triangle_texture, tri * 3 + 1);
			float4 edge2 =	tex1Dfetch(triangle_texture, tri * 3 + 2);

			float3 v0 = make_float3(v00.x, v00.y, v00.z);

			float3 e1 = make_float3(edge1.x, edge1.y, edge1.z);
			float3 e2 = make_float3(edge2.x, edge2.y, edge2.z);

			float3 v1 = e1 + v0;
			float3 v2 = e2 + v0;
			v0  += scene_objs[curr_obj_count].location;
			v1  += scene_objs[curr_obj_count].location;
			v2  += scene_objs[curr_obj_count].location;

			// Perform ray/triangle intersection test.
			HitResult local_hit_result;

			bool intersects_tri = gpu_ray_tri_intersect(ray_o, ray_dir, 
				v0, v1 - v0, v2 - v0, tri, local_hit_result);

			if (intersects_tri) {
				if (local_hit_result.t < t_exit) {
					intersection_detected = true;

					tmp_hit_result = local_hit_result;
					t_exit = tmp_hit_result.t;
				}
			}
		}
		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;
		transformed_box = curr_node.box;
		transformed_box.Min += scene_objs[curr_obj_count].location;
		transformed_box.Max += scene_objs[curr_obj_count].location;
		bool intersects_curr_node_bounding_box = gpu_ray_box_intersect(transformed_box, ray_o, ray_dir, tmp_t_near, tmp_t_far);
		if (intersects_curr_node_bounding_box) {
			// Set t_entry to be the entrance point of the next (neighboring) node.
			t_entry = tmp_t_far + 1e-4;
		}
		else {
			// This should never happen.
			// If it does, then that means we're checking triangles in a node that the ray never intersects.
			//break;
		}


		//// Get neighboring node using ropes attached to current node.
		//float3 p_exit = ray_o + (t_entry * ray_dir);
		//int new_node_index = get_neighboring_node_index(curr_node, p_exit);

		//// Break if neighboring node not found, meaning we've exited the kd-tree.
		//if (new_node_index == -1) {
		//	break;
		//}
		
		curr_node = node[root_index[curr_obj_count]];

	}

	if (intersection_detected) {
		hit_result = tmp_hit_result;
		return true;
	}

	return false;

}


////////////////////////////////////////////////////
// Call tree traversal and paind pixels
////////////////////////////////////////////////////
__device__ 
float4 *device_trace_ray(RKDTreeNodeGPU *tree, float3 ray_o, float3 ray_dir, 
	GPUSceneObject *scene_objs, int num_objs,
	float4 *Pixel, int *root_index, int num_faces, int *indexList)
{
	float3 hit_point;
	float3 normal;
	float t = 9999999; 
	HitResult hit_result;
	float4 pixel_color;
	for (int i = 0; i < num_objs; ++i)
	{
		if (stackless_kdtree_traversal(tree, ray_o, ray_dir,scene_objs, num_objs, i, root_index, indexList, hit_result))
		{
			pixel_color.x = (normal.x < 0.0f) ? (normal.x * -1.0f) : normal.x;;
			pixel_color.y = (normal.y < 0.0f) ? (normal.y * -1.0f) : normal.y;
			pixel_color.z = (normal.z < 0.0f) ? (normal.z * -1.0f) : normal.z;

		}
		Pixel = &pixel_color;
	}
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
	const RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int root_index, int num_faces, int *indexList, int stride)
{
	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, heigth, render_camera, stride);

	pixels = device_trace_ray(tree, ray_o, ray_dir, scene_objs, num_objs, pixels, &root_index, num_faces, indexList);

	return pixels;
}

////////////////////////////////////////////////////
// Initializes ray caster
////////////////////////////////////////////////////
__global__ 
void trace_scene(RKDTreeNodeGPU *tree, int width, int height, float4 *pixels, 
	const RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int root_index, int num_faces, int *indexList, int stride)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index > (width * height)) {
		return;
	}

	pixels = trace_pixel(tree, pixels, width, height, render_camera, scene_objs, num_objs,
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
	color = make_float4(fmaxf(0.f, dot(normal, -ray_dir))); // facing ratio 
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
	const float sint = etai / etat * sqrtf(fmaxf(0.f, 1 - cosi * cosi));
	// Total internal reflection.
	if (sint >= 1) {
		kr = 1;
	}
	else {
		const float cost = sqrtf(fmaxf(0.f, 1 - sint * sint));
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
bool trace_shadow(RKDTreeNodeGPU *tree, float3 ray_o, float3 ray_dir, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list, HitResult &hit_result)
{
	bool inter = false;
	for (int i = 0; i < num_objs; i++)
	{
		HitResult tmp_hit_result;
		bool hits = stackless_kdtree_traversal(tree, ray_o, ray_dir, scene_objs, num_objs,i, root_index, index_list, tmp_hit_result);
		if (hits)
		{
			if (hit_result.t< tmp_hit_result.t)
			{
				inter = hits;
				hit_result = tmp_hit_result;
			}
		}
	}
	return inter;
}


////////////////////////////////////////////////////
// Compute light intensity
////////////////////////////////////////////////////
__device__
void illuminate(float3 & P, float3 light_pos, float3 & lightDir, float4 & lightIntensity, float &distance)
{
	// Return not to devide by zero.
	if (distance == 0)
		return;

	lightDir = P - light_pos;
	
	float r2 = light_pos.x * light_pos.x + light_pos.y * light_pos.y + light_pos.z * light_pos.z;
	distance = sqrtf(r2);
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
	color += color * 0.2;
}


////////////////////////////////////////////////////
// Phong light
////////////////////////////////////////////////////
__device__
void phong_light(float4 &finalColor, float3 P, float3 dir, RKDTreeNodeGPU *tree, 
	GPUSceneObject *scene_objs, int num_objs, int *root_index, int *index_list, HitResult &hit_result)
{
	float3 bias = hit_result.normal * make_float3(1e-4);
	float4 diffuse = make_float4(0), specular = make_float4(0);
	float3 lightpos = make_float3(10, 10, 10), lightDir;
	float4 lightInt;
	illuminate(P, lightpos, lightDir, lightInt, hit_result.t);

	const float3 ray_o = P + bias;
	float3 ray_dir = -lightDir;

	bool vis = trace_shadow(tree, ray_o, ray_dir, scene_objs, num_objs, root_index, index_list, hit_result);

	diffuse += (lightInt * make_float4(vis * 0.18) * (fmaxf(0.0, dot(hit_result.normal, (-lightDir)))));

	float3 R = reflect(lightDir, hit_result.normal);
	specular += (lightInt * make_float4(vis) * (powf(fmaxf(0.0, dot(R,(-dir))), 10)));

	finalColor = finalColor + (diffuse * (0.8)) + (specular * (0.2));
}

////////////////////////////////////////////////////
// Shade
////////////////////////////////////////////////////
__device__
void shade(float4 &finalColor, float3 &normal, float3 &hit_point, RKDTreeNodeGPU *tree, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list)
{
	float3 bias = normal + make_float3(1e-4);
	float3 lightpos = make_float3(1, 50, 5), lightDir;
	float4 lightInt;
	float4 lightColor = make_float4(1);
	HitResult hit_result;
	float tShad;
	illuminate(hit_point, lightpos, lightDir, lightInt, tShad);

	const float3 ray_o = hit_point + bias;
	float3 ray_dir = -lightDir;

	bool vis = !trace_shadow(tree, ray_o, ray_dir, scene_objs, num_objs, root_index, index_list, hit_result);

	finalColor += vis * lightInt * fmaxf(0.0, dot(normal,(ray_dir)));

}


////////////////////////////////////////////////////
// Ray casting with brute force approach
////////////////////////////////////////////////////
__global__
void gpu_bruteforce_ray_cast(float4 *image_buffer,
	int width, int height, const RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int num_faces, int stride, RKDTreeNodeGPU *tree, int *root_index, int *index_list)
{
	int index = ((threadIdx.x * gridDim.x) + blockIdx.x) + stride;
	if (index > width * height)
		return;

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, height, render_camera, stride);

	
	float4 pixel_color = make_float4(0);

	float t_near, t_far;
	GPUBoundingBox bbox = tree[0].box;


	// Perform ray-box intersection test.
	bool intersects_aabb = gpu_ray_box_intersect(bbox, ray_o, ray_dir, t_near, t_far);
	if (intersects_aabb)
	{
		HitResult hit_result;

		for (int i = 0; i < num_faces; ++i) {

			float4 v0 =	   tex1Dfetch(triangle_texture, i * 3);
			float4 edge1 = tex1Dfetch(triangle_texture, i * 3 + 1);
			float4 edge2 = tex1Dfetch(triangle_texture, i * 3 + 2);

			// Perform ray-triangle intersection test.
			HitResult tmp_hit_result;
			bool intersects_tri = gpu_ray_tri_intersect(ray_o, ray_dir, make_float3(v0.x, v0.y, v0.z), make_float3(edge1.x, edge1.y, edge1.z),
				make_float3(edge2.x, edge2.y, edge2.z), i, tmp_hit_result);

			if (intersects_tri) 
			{
				if (tmp_hit_result.t < hit_result.t)
				{
					hit_result = tmp_hit_result;
					//narmals_mat(pixel_color, tmp_normal);
					simple_shade(pixel_color, hit_result.normal, ray_dir);
					//phong_light(pixel_color, hit_point, ray_dir, tmp_normal, hit_point, tree, scene_objs, num_objs, root_index, index_list);
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
	RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *indexList, int stride)
{

	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > width * height)
		return;

	float3 ray_o, ray_dir;
	generate_ray(ray_o, ray_dir, width, height, render_camera, stride);


	HitResult hit_result;


	float4 pixel_color = make_float4(0);

	// Perform ray-box intersection test.
	bool intersects_aabb = false;
	int tmp_tree_counter = 0;

	for (int i = 0; i < num_objs; i++)
	{
		HitResult tmp_hit_result;
		bool hits = stackless_kdtree_traversal(tree , ray_o, ray_dir, scene_objs, num_objs, i, root_index, indexList, tmp_hit_result);
		if (hits) {
			if (tmp_hit_result.t< hit_result.t) {
				intersects_aabb = hits;
				hit_result = tmp_hit_result;
			}
		}
	}
	if (intersects_aabb)
	{
		//int square = (int)floor(hit_point.x) + (int)floor(hit_point.z);
		//narmals_mat(pixel_color, normal);
		

		//tile_pattern(pixel_color, square);
		simple_shade(pixel_color, hit_result.normal, ray_dir);
		//phong_light(pixel_color, hit_point, ray_dir, normal,hit_point, tree, scene_objs, num_objs, root_index, indexList);
		//shade(pixel_color, normal, hit_point, tree, scene_objs, num_objs, root_index, indexList);
	}
	else
	{
		// If ray missed draw sky there.
		sky_mat(pixel_color, ray_dir);
	}

	ambient_light(pixel_color);
	pixel_color = clip(pixel_color);
	pixels[index + stride] = pixel_color;
	return;
}

extern "C"
void free_memory()
{
	// Release device memory.
	cudaFree(d_pixels);
	cudaFree(d_tree);
	cudaFree(d_index_list);
	cudaFree(dev_triangle_p);
	cudaFree(dev_normals_p);
	cudaFree(d_render_camera);
}

extern "C"
void update_objects(GPUSceneObject *objs, int num_objs)
{
	cudaFree(d_scene_objects);
	cudaMalloc(&d_scene_objects, num_objs * sizeof(GPUSceneObject));
	cudaMemcpy(d_scene_objects, objs, num_objs * sizeof(GPUSceneObject), cudaMemcpyHostToDevice);
}

extern "C"
void copy_memory(std::vector<RKDThreeGPU *>tree, RCamera sceneCam, std::vector<float4> triangles, std::vector<float4> normals, GPUSceneObject *objs, int num_objs, bool bruteforce = false)
{
	// --------------------------------Initialize host variables----------------------------------------------------

	int size = SCR_WIDTH * SCR_HEIGHT * sizeof(float4);
	size_t size_kd_tree = 0;

	for (auto t : tree)
	{
		size_kd_tree += t->GetNumNodes();
	}

	RKDTreeNodeGPU *h_tree = new RKDTreeNodeGPU[size_kd_tree];
	for (int k = 0, int i = 0; i < num_objs; i++)
	{
		for (int n = 0; n < tree[i]->GetNumNodes(); n++, k++)
		{
			h_tree[k] = tree[i]->GetNodes()[n];
		}
	}

	h_camera = &sceneCam;
	h_pixels = new float4[size];

	std::vector<int> h_root_index = {};
	int offset = 0;
	for (int i = 0; i < num_objs; i++)
	{
		h_root_index.push_back(tree.at(i)->get_root_index());
	}


	//------------------------------------------------------------------------------------------------------

	//--------------------------------Initialize device variables-------------------------------------------


	// initialise array of triangle indecies.
	std::vector<int> kd_tree_tri_indics = {};
	offset = 0;
	int count = 0;
	for (auto t:tree)
	{
		for (auto n: t->obj_index_list)
		{
			kd_tree_tri_indics.push_back(n + offset);
		}
		offset += objs[count].num_prims;

		count++;
	}

	size_t size_kd_tree_tri_indices = kd_tree_tri_indics.size() * sizeof(int);

	cudaMalloc(&d_pixels, size);
	cudaMalloc(&d_render_camera, sizeof(RCamera));
	cudaMalloc(&d_tree, size_kd_tree * sizeof(RKDTreeNodeGPU));
	cudaMalloc(&d_index_list, size_kd_tree_tri_indices);
	cudaMalloc(&d_scene_objects, num_objs * sizeof(GPUSceneObject));
	cudaMalloc(&d_root_index, num_objs * sizeof(int));

	// calculate total number of triangles in the scene
	size_t triangle_size = triangles.size() * sizeof(float4);
	int total_num_triangles = triangles.size() / 3;


	if (num_objs > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void **)&dev_triangle_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_triangle_p, &triangles[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_triangles_tro_texture(dev_triangle_p, total_num_triangles);

		// allocate memory for the triangle meshes on the GPU
		//cudaMalloc((void **)&dev_normals_p, triangle_size);

		// copy triangle data to GPU
		//cudaMemcpy(dev_normals_p, &normals[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		//bind_normals_tro_texture(dev_normals_p, total_num_triangles);
	}


	// Copy host vectors to device.
	cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree, h_tree, size_kd_tree * sizeof(RKDTreeNodeGPU), cudaMemcpyHostToDevice);
	cudaMemcpy(d_index_list, kd_tree_tri_indics.data(), size_kd_tree_tri_indices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_scene_objects, objs, num_objs * sizeof(GPUSceneObject), cudaMemcpyHostToDevice);
	cudaMemcpy(d_root_index, h_root_index.data(), num_objs * sizeof(int), cudaMemcpyHostToDevice);
	d_object_number = num_objs;

	kd_tree_tri_indics.clear(); //clear content
	kd_tree_tri_indics.resize(0); //resize it to 0
	kd_tree_tri_indics.shrink_to_fit(); //reallocate memory

	triangles.clear(); //clear content
	triangles.resize(0); //resize it to 0
	triangles.shrink_to_fit(); //reallocate memory

}

////////////////////////////////////////////////////
// Main render function
// All variables are initialized here
// Kernel is being executed
////////////////////////////////////////////////////
extern
float4 *Render(RCamera sceneCam)
{
	h_camera = &sceneCam;
	cudaMemcpyAsync(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice);
	
	int size = SCR_WIDTH * SCR_HEIGHT * sizeof(float4);

	// Number of threads in each thread block
	int  blockSize = SCR_WIDTH;

	// Number of thread blocks in grid
	int  gridSize = SCR_HEIGHT;


	//------------------------------------------------------------------------------------------------------

	//printf("Starting rendering on the GPU with blockSize = %d and gridSize = %d\n", blockSize, gridSize);

	//cudaDeviceSynchronize();

	// Start timer that is used for benchmarking.
	//auto start = std::chrono::steady_clock::now();

	// Perform bruteforce approach or use kd-tree acceleration.
	//if (!bruteforce)
	//{
		//gpu_bruteforce_ray_cast << < blockSize, gridSize >> > (d_pixels, SCR_WIDTH, SCR_HEIGHT,
		//	d_render_camera, d_scene_objects, d_object_number, num_faces, 0, d_tree, d_root_index, d_index_list);

	//}
	//else
	//{
	//}
	stackless_trace_scene << < blockSize, gridSize >> > (d_tree, SCR_WIDTH, SCR_HEIGHT, d_pixels,
		d_render_camera, d_scene_objects, d_object_number, d_root_index,  d_index_list, 0);

	cudaDeviceSynchronize();


	// Finish timer used for benchmarking.
	//auto finish = std::chrono::steady_clock::now();

	//float elapsed_seconds = std::chrono::duration_cast<
	//std::chrono::duration<float>>(finish - start).count();
	//std::cout << "Rendering of a on the GPU frame has finished in " << elapsed_seconds << " seconds." << std::endl;

	// Copy pixel array back to host.
	cudaMemcpyAsync(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);

	// Check for CUDA runtime API calls errors.
	cudaError_t cudaError;
	cudaError = cudaGetLastError();

	//cudaFree(d_render_camera);

	if (cudaError != cudaSuccess)
	{
		printf("cudaGetLastError() returned %d: %s\n", cudaError, cudaGetErrorString(cudaError));
	}

	return h_pixels;
}