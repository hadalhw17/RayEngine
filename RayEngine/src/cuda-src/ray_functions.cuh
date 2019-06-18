#pragma once
#include "cuda-src/gpu_structs.h"
#include "cuda_helper_functions.h"
#include "../Primitives/Camera.h"
#include "../Primitives/GPUBoundingBox.h"
#include "CUDARayTracing.cuh"
#include "kd_tree_functions.cuh"


////////////////////////////////////////////////////
// Generates ray origin and ray direction thread index
////////////////////////////////////////////////////
__host__ __device__
void generate_ray(RayData& ray,
	const RCamera render_camera, uint width, uint heigth, const uint2& xy)
{
	uint imageX =xy.x;
	uint imageY = xy.y;

	if (imageX >= width || imageY >= heigth)
		return;

	float sx = (float)imageX / (width - 1.0f);
	float sy = 1.0f - ((float)imageY / (heigth - 1.0f));

	float3 rendercampos = render_camera.campos;

	// compute primary ray direction
	// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
	float3 rendercamview = render_camera.view; rendercamview = normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
	float3 rendercamup = render_camera.camdown; rendercamup = normalize(rendercamup);
	float3 horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis = normalize(horizontalAxis); // Important to normalize!
	float3 verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

	float3 middle = rendercampos + rendercamview;
	float3 horizontal = horizontalAxis * tanf(render_camera.fov.x * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5
	float3 vertical = verticalAxis * tanf(render_camera.fov.y * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5

	// compute pixel on screen
	float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
	float3 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * render_camera.focial_distance); // Important for depth of field!		

	float3 aperturePoint = rendercampos;

	// calculate ray direction of next ray in path
	float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
	apertureToImagePlane = normalize(apertureToImagePlane); // ray direction needs to be normalised

	// ray direction
	float3 rayInWorldSpace = apertureToImagePlane;
	ray.direction = normalize(rayInWorldSpace);

	// ray origin
	ray.origin = rendercampos;
	ray.max_distance = K_INFINITY;
}


////////////////////////////////////////////////////
// Ray-box intersection
////////////////////////////////////////////////////
__device__
bool gpu_ray_box_intersect(const GPUBoundingBox& bbox, RayData& ray)
{
	float3 dirfrac = make_float3(1.0f / ray.direction.x, 1.0f / ray.direction.y, 1.0f / ray.direction.z);

	float t1 = (bbox.Min.x - ray.origin.x) * dirfrac.x;
	float t2 = (bbox.Max.x - ray.origin.x) * dirfrac.x;
	float t3 = (bbox.Min.y - ray.origin.y) * dirfrac.y;
	float t4 = (bbox.Max.y - ray.origin.y) * dirfrac.y;
	float t5 = (bbox.Min.z - ray.origin.z) * dirfrac.z;
	float t6 = (bbox.Max.z - ray.origin.z) * dirfrac.z;

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

	ray.min_distance = tmin;
	ray.max_distance = tmax;

	ray.min_distance = fsel(ray.min_distance, ray.min_distance, 0.f);

	return true;
}

__device__
bool solve_quadratic(const float& a, const float& b, const float& c, float& x0, float& x1)
{
	float discr = b * b - 4.f * a * c;
	if (discr < .0f) return false;
	else if (discr == 0) {
		x0 = x1 = -0.5f * b / a;
	}
	else {
		float q = (b > .0f) ?
			-0.5f * (b + sqrt(discr)) :
			-0.5f * (b - sqrt(discr));
		x0 = q / a;
		x1 = c / q;
	}

	return true;
}

__device__
bool gpu_ray_sphere_intersect(const float3 ray_o, const float3 ray_dir, const float radius, float& t0, float& t1)
{
	// analytic solution
	float a = dot(ray_dir, ray_dir);
	float b = 2 * dot(ray_dir, ray_o);
	float radius2 = pow(radius, 2);
	float c = dot(ray_o, ray_o) - radius2;
	if (!solve_quadratic(a, b, c, t0, t1)) return false;

	if (t0 > t1) swap(t0, t1);

	return true;
}

////////////////////////////////////////////////////
// Compute normal of a triangle by vertexes
////////////////////////////////////////////////////
__device__
float3 gpu_get_tri_normal(float3& p1, float3& u, float3& v)
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
float3 gpu_get_tri_normal(const int tri_index, const float u, const float v)
{
	const float w = 1.f - (u + v);
	const float4 n0 = tex1Dfetch(normals_texture, tri_index * 3);
	const float4 n1 = tex1Dfetch(normals_texture, tri_index * 3 + 1);
	const float4 n2 = tex1Dfetch(normals_texture, tri_index * 3 + 2);

	return normalize(w * make_float3(n0.x, n0.y, n0.z) + u * make_float3(n1.x, n1.y, n1.z) + v * make_float3(n2.x, n2.y, n2.z));
}

__device__
float2 get_uvs(const int tri_index, const float u, const float v)
{
	const float w = 1.f - u - v;
	const float2 uv0 = tex1Dfetch(uvs_texture, tri_index * 3);
	const float2 uv1 = tex1Dfetch(uvs_texture, tri_index * 3 + 1);
	const float2 uv2 = tex1Dfetch(uvs_texture, tri_index * 3 + 2);

	return w * uv0 + u * uv1 + v * uv2;
}


////////////////////////////////////////////////////
// Möller–Trumbore intersection algorithm
// Between ray and triangle
////////////////////////////////////////////////////
__device__
bool gpu_ray_tri_intersect(float3 v0, float3 e1, float3 e2, int tri_index, GPUSceneObject curr_obj, HitResult& hit_result)
{
	float3 h, s, q;
	float a, f, u, v;

	h = cross(hit_result.ray.direction, e2);
	a = dot(e1, h);

	if (a < K_EPSILON) {
		return false;
	}

	f = 1.0f / a;
	s = hit_result.ray.origin - v0;
	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	q = cross(s, e1);
	v = f * dot(hit_result.ray.direction, q);

	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}

	// at this stage we can compute t to find out where the intersection point is on the line
	hit_result.t = f * dot(e2, q);

	if (hit_result.t > K_EPSILON) { // ray intersection
		hit_result.normal = gpu_get_tri_normal(tri_index, u, v);
		//hit_result.normal = gpu_get_tri_normal(v0, e1, e2);

		if (curr_obj.material.uvs)
		{
			hit_result.uv = get_uvs(tri_index, u, v);
		}
		else
		{
			hit_result.uv = make_float2(u, v);
		}
		hit_result.hits = true;
		return true;
	}
	else { // this means that there is a line intersection but not a ray intersection
		return false;
	}
}



////////////////////////////////////////////////////
// Stackless kd-tree traversal algorithm
////////////////////////////////////////////////////
__device__
bool stackless_kdtree_traversal(RKDTreeNodeGPU* node,
	GPUSceneObject* scene_objs, int num_objs, int curr_obj_count,
	int* root_index, int* indexList, HitResult& hit_result, bool is_secondary)
{
	// Iterate ofer all of the individual trees in the scene.
	RKDTreeNodeGPU curr_node = node[root_index[curr_obj_count]];

	float siny = sinf(scene_objs[curr_obj_count].rotation.y);
	float cosy = cosf(scene_objs[curr_obj_count].rotation.y);

	float3 new_dir, new_o = make_float3(0);
	new_dir.x = hit_result.ray.direction.x * cosy + hit_result.ray.direction.z * siny;
	new_dir.y = hit_result.ray.direction.y;
	new_dir.z = -hit_result.ray.direction.x * siny + hit_result.ray.direction.z * cosy;

	if (!scene_objs[curr_obj_count].is_character)
	{
		new_o.x = hit_result.ray.origin.x * cosy + hit_result.ray.origin.z * siny;
		new_o.y = hit_result.ray.origin.y;
		new_o.z = -hit_result.ray.origin.x * siny + hit_result.ray.origin.z * cosy;
		new_o += scene_objs[curr_obj_count].location;
	}
	if (scene_objs[curr_obj_count].is_character && is_secondary)
	{

		siny = sinf(-scene_objs[curr_obj_count].rotation.y);
		cosy = cosf(-scene_objs[curr_obj_count].rotation.y);
		new_dir.x = hit_result.ray.direction.x * cosy + hit_result.ray.direction.z * siny;
		new_dir.y = hit_result.ray.direction.y;
		new_dir.z = -hit_result.ray.direction.x * siny + hit_result.ray.direction.z * cosy;

		new_o.x = scene_objs[curr_obj_count].location.x * cosy + scene_objs[curr_obj_count].location.z * siny;
		new_o.y = -scene_objs[curr_obj_count].location.y;
		new_o.z = -scene_objs[curr_obj_count].location.x * siny + scene_objs[curr_obj_count].location.z * cosy;
	}

	new_dir = normalize(new_dir);

	// Perform ray/AABB intersection test.
	RayData transformed_ray;
	transformed_ray.direction = new_dir;
	transformed_ray.origin = new_o;
	bool intersects_root_node_bounding_box = gpu_ray_box_intersect(curr_node.box, transformed_ray);


	if (!intersects_root_node_bounding_box) {
		return false;
	}


	HitResult tmp_hit_result;
	bool intersection_detected = false;
	int limit = 0;

	float t_entry = transformed_ray.min_distance, t_exit = transformed_ray.max_distance;
	while (t_entry < t_exit && limit < 50) {
		++limit;

		// Down traversal - Working our way down to a leaf node.
		float3 p_entry = new_o + (t_entry * new_dir);
		while (!curr_node.is_leaf) {
			curr_node = is_point_to_the_left_of_split(curr_node, p_entry) ? node[curr_node.left_index] : node[curr_node.right_index];
		}

		// We've reached a leaf node.
		// Check intersection with triangles contained in current leaf node.
		for (size_t i = curr_node.index_of_first_object; i < curr_node.index_of_first_object + curr_node.num_objs; ++i)
		{
			int tri = indexList[i + scene_objs[curr_obj_count].offset];
			float4 v00 = tex1Dfetch(triangle_texture, tri * 3 + 0);
			float4 edge1 = tex1Dfetch(triangle_texture, tri * 3 + 1);
			float4 edge2 = tex1Dfetch(triangle_texture, tri * 3 + 2);

			float3 v0 = make_float3(v00.x, v00.y, v00.z);
			float3 e1 = make_float3(edge1.x, edge1.y, edge1.z);
			float3 e2 = make_float3(edge2.x, edge2.y, edge2.z);

			// Perform ray/triangle intersection test.
			HitResult local_hit_result;
			local_hit_result.ray = transformed_ray;

			gpu_ray_tri_intersect(v0, e1, e2, tri, scene_objs[curr_obj_count], local_hit_result);

			if (local_hit_result.hits) {
				if (local_hit_result.t < t_exit) {
					intersection_detected = true;
					local_hit_result.obj_index = curr_obj_count;
					local_hit_result.ray = hit_result.ray;
					tmp_hit_result = local_hit_result;

					t_exit = tmp_hit_result.t;
				}
			}
		}
		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;

		bool intersects_curr_node_bounding_box = gpu_ray_box_intersect(curr_node.box, transformed_ray);
		if (intersects_curr_node_bounding_box) {
			// Set t_entry to be the entrance point of the next (neighboring) node.
			t_entry = tmp_t_far;
		}
		else {
			// This should never happen.
			// If it does, then that means we're checking triangles in a node that the ray never intersects.
			break;
		}


		// Get neighboring node using ropes attached to current node.
		float3 p_exit = new_o + (t_entry * new_dir);
		int new_node_index = get_neighboring_node_index(curr_node, p_exit);

		// Break if neighboring node not found, meaning we've exited the kd-tree.
		if (new_node_index == -1) {
			break;
		}

		curr_node = node[new_node_index];

	}

	if (intersection_detected) {
		hit_result = tmp_hit_result;
		return true;
	}

	return false;

}

////////////////////////////////////////////////////
// Trace for shadows
////////////////////////////////////////////////////
__device__
void trace_shadow(RKDTreeNodeGPU* tree, GPUSceneObject* scene_objs, int num_objs,
	int* root_index, int* index_list, HitResult& hit_result)
{
	for (int i = 0; i < num_objs; ++i)
	{
		stackless_kdtree_traversal(tree, scene_objs, num_objs, i, root_index, index_list, hit_result, true);
	}

}

__device__
void trace_scene(RKDTreeNodeGPU* tree,
	const RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int* root_index, int* indexList, HitResult& hit_result)
{
	// Perform ray-box intersection test.
	for (int i = 0; i < num_objs; i++)
	{
		HitResult tmp_hit_result;
		tmp_hit_result.ray.direction = hit_result.ray.direction;
		tmp_hit_result.ray.origin = hit_result.ray.origin;

		bool hits = stackless_kdtree_traversal(tree, scene_objs, num_objs, i, root_index, indexList, tmp_hit_result, false);
		if (hits) {
			if (tmp_hit_result.t < hit_result.t) {
				hit_result = tmp_hit_result;
				hit_result.hits = hits;
			}
		}
	}
}

////////////////////////////////////////////////////
// Call tree traversal and paind pixels
////////////////////////////////////////////////////
__device__
float4* device_trace_ray(RKDTreeNodeGPU* tree, RayData ray,
	GPUSceneObject* scene_objs, int num_objs,
	float4* Pixel, int* root_index, int num_faces, int* indexList)
{
	HitResult hit_result;
	hit_result.ray = ray;
	float4 pixel_color;
	for (int i = 0; i < num_objs; ++i)
	{
		if (stackless_kdtree_traversal(tree, scene_objs, num_objs, i, root_index, indexList, hit_result, false))
		{
			pixel_color.y = (hit_result.normal.y < 0.0f) ? (hit_result.normal.y * -1.0f) : hit_result.normal.y;
			pixel_color.z = (hit_result.normal.z < 0.0f) ? (hit_result.normal.z * -1.0f) : hit_result.normal.z;
			pixel_color.x = (hit_result.normal.x < 0.0f) ? (hit_result.normal.x * -1.0f) : hit_result.normal.x;

		}
		Pixel = &pixel_color;
	}
	return Pixel;
}



////////////////////////////////////////////////////
// Cast ray from a pixel
////////////////////////////////////////////////////
__device__
float4* trace_pixel(RKDTreeNodeGPU* tree, float4* pixels,
	const RCamera render_camera, GPUSceneObject* scene_objs, int num_objs,
	int root_index, int num_faces, int* indexList, uint width, uint heigth)
{
	int imageX = blockIdx.x * blockDim.x + threadIdx.x;
	int imageY = blockIdx.y * blockDim.y + threadIdx.y;

	if (imageX >= width || imageY >= heigth)
		return;

	RayData ray;
	generate_ray(ray, render_camera, width, heigth, { imageX, imageY });

	pixels = device_trace_ray(tree, ray, scene_objs, num_objs, pixels, &root_index, num_faces, indexList);

	return pixels;
}
