////////////////////////////////////////////////////
// Main CUDA rendering file.
////////////////////////////////////////////////////

#include "CUDARayTracing.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
#include "Atmosphere.cuh"

#include "RayEngine.h"



float4 *dev_triangle_p;
float4 *dev_normals_p;

#define PI_OVER_TWO 1.5707963267948966192313216916397514420985
#define M_PI 3.14156265

__device__
void gray_scale(float4 &color)
{
	color = make_float4((0.3 * color.x) + (0.59 * color.y) + (0.11 * color.z));
	
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


__device__
bool solve_quadratic(const float & a, const float & b, const float & c, float & x0, float & x1)
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
bool gpu_ray_sphere_intersect(const float3 ray_o, const float3 ray_dir, const float radius, float &t0, float &t1)
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
// Algorithm for computing color of sky sphere
// Code is taken from:
// https://www.scratchapixel.com/code.php?id=52&origin=/lessons/procedural-generation-virtual-worlds/simulating-sky
////////////////////////////////////////////////////
__device__
float3 compute_incident_light(Atmosphere *armosphere, const float3 orig, const float3 dir, float tmin, float tmax)
{
	float t0 , t1;
	if (!gpu_ray_sphere_intersect(orig, dir, armosphere->atmosphereRadius, t0, t1) || t1 < 0) return make_float3(1);
	if (t0 > tmin && t0 > 0) tmin = t0;
	if (t1 < tmax) tmax = t1;
	size_t numSamples = 32;
	size_t numSamplesLight = 8;
	float segmentLength = (tmax - tmin) / numSamples;
	float tCurrent = tmin;
	float3 sumR = make_float3(0), sumM = make_float3(0); // mie and rayleigh contribution 
	float opticalDepthR = 0, opticalDepthM = 0;
	float mu = dot(dir, armosphere->sunDirection); // mu in the paper which is the cosine of the angle between the sun direction and the ray direction 
	float phaseR = 3.f / (16.f * M_PI) * (1 + mu * mu);
	float g = 0.76f;
	float phaseM = 3.f / (8.f * M_PI) * ((1.f - g * g) * (1.f + mu * mu)) / ((2.f + g * g) * pow(1.f + g * g - 2.f * g * mu, 1.5f));
	for (size_t i = 0; i < numSamples; ++i) {
		float3 samplePosition = orig + (tCurrent + segmentLength * 0.5f) * dir;
		float height = length(samplePosition) - armosphere->earthRadius;
		// compute optical depth for light
		float hr = exp(-height / armosphere->Hr) * segmentLength;
		float hm = exp(-height / armosphere->Hm) * segmentLength;
		opticalDepthR += hr;
		opticalDepthM += hm;
		// light optical depth
		float t0Light, t1Light;
		gpu_ray_sphere_intersect(samplePosition, armosphere->sunDirection, armosphere->atmosphereRadius, t0Light, t1Light);
		float segmentLengthLight = t1Light / numSamplesLight, tCurrentLight = 0;
		float opticalDepthLightR = 0, opticalDepthLightM = 0;
		size_t j;
		for (j = 0; j < numSamplesLight; ++j) {
			float3 samplePositionLight = samplePosition + (tCurrentLight + segmentLengthLight * 0.5f) * armosphere->sunDirection;
			float heightLight = length(samplePositionLight) - armosphere->earthRadius;
			if (heightLight < 0) break;
			opticalDepthLightR += exp(-heightLight / armosphere->Hr) * segmentLengthLight;
			opticalDepthLightM += exp(-heightLight / armosphere->Hm) * segmentLengthLight;
			tCurrentLight += segmentLengthLight;
		}
		if (j == numSamplesLight) {
			float3 tau = armosphere->betaR * (opticalDepthR + opticalDepthLightR) + armosphere->betaM * 1.1f * (opticalDepthM + opticalDepthLightM);
			float3 attenuation = make_float3(exp(-tau.x), exp(-tau.y), exp(-tau.z));
			sumR += attenuation * hr;
			sumM += attenuation * hm;
		}
		tCurrent += segmentLength;
	}

	return (sumR * armosphere->betaR * phaseR + sumM + 100 * armosphere->betaM * phaseM) * 20;
}


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

	return normalize((1 - u - v) * make_float3(n0.x, n0.y, n0.z) + u * make_float3(n1.x, n1.y, n1.z) + v * make_float3(n2.x, n2.y, n2.z));
}


////////////////////////////////////////////////////
// Möller–Trumbore intersection algorithm
// Between ray and triangle
////////////////////////////////////////////////////
__device__
bool gpu_ray_tri_intersect(float3 v0, float3 e1, float3 e2, int tri_index, HitResult &hit_result)
{
	float3 h, s, q;
	float a, f, u, v;

	h = cross(hit_result.ray_dir, e2);
	a = dot(e1, h);

	if (a <  1e-4) {
		return false;
	}

	f = 1.0f / a;
	s = hit_result.ray_o - v0;
	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	q = cross(s, e1);
	v = f * dot(hit_result.ray_dir, q);

	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}

	// at this stage we can compute t to find out where the intersection point is on the line
	hit_result.t = f * dot(e2, q);

	if (hit_result.t > 1e-4) { // ray intersection
		hit_result.normal = gpu_get_tri_normal(tri_index, u, v);
		hit_result.hit_point = hit_result.ray_o + (hit_result.t * hit_result.ray_dir);
		hit_result.hits = true;
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

	if (t_near < 0) {
		t_near = 1e-4;
	}

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
bool stackless_kdtree_traversal(RKDTreeNodeGPU *node,
	GPUSceneObject *scene_objs, int num_objs, int curr_obj_count,
	int *root_index, int *indexList, HitResult &hit_result)
{
	// Iterate ofer all of the individual trees in the scene.
	RKDTreeNodeGPU curr_node = node[root_index[curr_obj_count]];

	// Perform ray/AABB intersection test.
	float t_entry, t_exit;

	bool intersects_root_node_bounding_box = gpu_ray_box_intersect(curr_node.box, hit_result.ray_o + scene_objs[curr_obj_count].location, hit_result.ray_dir, t_entry, t_exit);
	

	if (!intersects_root_node_bounding_box) {
		return false;
	}


	HitResult tmp_hit_result;
	bool intersection_detected = false;
	int limit = 0;
	float t_entry_prev = -kInfinity;

	while (t_entry < t_exit && limit < 50) {
		++limit;
		t_entry_prev = t_entry;

		// Down traversal - Working our way down to a leaf node.
		float3 p_entry = hit_result.ray_o + scene_objs[curr_obj_count].location + (t_entry * hit_result.ray_dir);
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

			// Perform ray/triangle intersection test.
			HitResult local_hit_result;
			local_hit_result.ray_o = hit_result.ray_o + scene_objs[curr_obj_count].location;
			local_hit_result.ray_dir = hit_result.ray_dir;

			gpu_ray_tri_intersect(v0, e1, e2, tri, local_hit_result);

			if (local_hit_result.hits) {
				if (local_hit_result.t < t_exit) {
					intersection_detected = true;
					local_hit_result.obj_index = curr_obj_count;
					local_hit_result.ray_o -= scene_objs[curr_obj_count].location;
					tmp_hit_result = local_hit_result;

					t_exit = tmp_hit_result.t;
				}
			}
		}
		// Compute distance along ray to exit current node.
		float tmp_t_near, tmp_t_far;

		bool intersects_curr_node_bounding_box = gpu_ray_box_intersect(curr_node.box, hit_result.ray_o + scene_objs[curr_obj_count].location, hit_result.ray_dir, tmp_t_near, tmp_t_far);
		if (intersects_curr_node_bounding_box) {
			// Set t_entry to be the entrance point of the next (neighboring) node.
			t_entry = tmp_t_far;
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
	HitResult hit_result;
	float4 pixel_color;
	for (int i = 0; i < num_objs; ++i)
	{
		if (stackless_kdtree_traversal(tree,scene_objs, num_objs, i, root_index, indexList, hit_result))
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
// Generates ray origin and ray direction thread index
////////////////////////////////////////////////////
__device__ 
void generate_ray(float3 &ray_o, float3 &ray_dir,
	const RCamera *render_camera, int stride)
{
	int index = ((threadIdx.x * gridDim.x) + blockIdx.x) + stride;

	int x = index % SCR_WIDTH;
	int y = index / SCR_WIDTH;

	if (index > (SCR_WIDTH * SCR_HEIGHT)) {
		return;
	}

	float sx = (float)x / (SCR_WIDTH - 1.0f);
	float sy = 1.0f - ((float)y / (SCR_HEIGHT - 1.0f));

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
	generate_ray(ray_o, ray_dir, render_camera, stride);

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
	color += make_float4(fmaxf(0.f, dot(normal, -ray_dir)/2)); // facing ratio 
}


////////////////////////////////////////////////////
// Sky material represent ray directions
////////////////////////////////////////////////////
__device__
void sky_mat(float4 &color, float3 ray_dir)
{
	//// Visualise ray directions on the sky.
	//color = make_float4(ray_dir, 0);
	//color.x = (color.x < 0.0f) ? (color.x * -1.0f) : color.x;
	//color.y = (color.y < 0.0f) ? (color.y * -1.0f) : color.y;
	//color.z = (color.z < 0.0f) ? (color.z * -1.0f) : color.z;

	float t = 0.5f * (ray_dir.y + 1.f);
	color = make_float4(1.f) - t * make_float4(1.f) = t * make_float4(0.5f, 0.7f, 1.f, 0.f);
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
	return k < 0 ? make_float3(0) : eta * I * (eta * cosi - sqrtf(k)) * n;
}


////////////////////////////////////////////////////
// Trace for shadows
////////////////////////////////////////////////////
__device__
void trace_shadow(RKDTreeNodeGPU *tree, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list, HitResult &hit_result)
{
	bool inter = false;
	for (int i = 0; i < num_objs; ++i)
	{
		stackless_kdtree_traversal(tree, scene_objs, num_objs, i, root_index, index_list, hit_result);
	}

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
	lightIntensity = make_float4(1, 1,1 , 1) * 2500 / (4 * M_PI * r2);
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


__device__
float3 uniformSampleHemisphere(const float &r1, const float &r2)
{
	// cos(theta) = u1 = y
	// cos^2(theta) + sin^2(theta) = 1 -> sin(theta) = srtf(1 - cos^2(theta))
	float sinTheta = sqrtf(1 - r1 * r1);
	float phi = 2 * M_PI * r2;
	float x = sinTheta * cosf(phi);
	float z = sinTheta * sinf(phi);
	return make_float3(x, r1, z);
}

__device__
void createCoordinateSystem(const float3 &N, float3 &Nt, float3 &Nb)
{
	if (fabs(N.x) > fabs(N.y))
		Nt = make_float3(N.z, 0, -N.x) / sqrtf(N.x * N.x + N.z * N.z);
	else
		Nt = make_float3(0, -N.z, N.y) / sqrtf(N.y * N.y + N.z * N.z);
	Nb = cross(N,Nt);
}

////////////////////////////////////////////////////
// Phong light
////////////////////////////////////////////////////
__device__
void phong_light(float3 *lights,size_t num_lights, float4 &finalColor, RKDTreeNodeGPU *tree,
	GPUSceneObject *scene_objs, int num_objs, int *root_index, int *index_list, HitResult &hit_result, HitResult &shadow_hit_result)
{
	float3 bias = hit_result.normal * make_float3(-1e-4);
	for (int i = 0; i < num_lights; ++i)
	{
		float4 diffuse = make_float4(0), specular = make_float4(0);
		float3 lightpos = lights[i], lightDir;
		float4 lightInt;
		float t = kInfinity;
		illuminate(hit_result.hit_point, lightpos, lightDir, lightInt, t);

		shadow_hit_result.ray_o = hit_result.hit_point + bias;
		shadow_hit_result.ray_dir = -lightDir;

		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, shadow_hit_result);

		diffuse += lightInt * make_float4(shadow_hit_result.hits * 0.18) * fmaxf(0.0, dot(hit_result.normal, -lightDir));

		float3 R = reflect(lightDir, hit_result.normal);
		specular += lightInt * make_float4(shadow_hit_result.hits * powf(fmaxf(0.0, dot(R, -hit_result.ray_dir)), 10));

		finalColor += diffuse * 0.8 + specular * 0.2;
		shadow_hit_result = HitResult();
	}
}

////////////////////////////////////////////////////
// Shade
////////////////////////////////////////////////////
__device__
void shade(float3 *lights, size_t num_lights, float4 &finalColor, RKDTreeNodeGPU *tree, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list, HitResult &hit_result, HitResult &shading_hit_result)
{
	float3 bias = hit_result.normal* make_float3(1e-4);
	for (int i = 0; i < num_lights; ++i)
	{
		float3 lightpos = lights[i], lightDir;
		float4 lightInt;
		float4 lightColor = make_float4(1, 0, 0, 0);
		float t = kInfinity;
		illuminate(hit_result.hit_point, lightpos, lightDir, lightInt, t);

		shading_hit_result.ray_o = hit_result.hit_point + bias;
		shading_hit_result.ray_dir = -lightDir;

		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, shading_hit_result);
		finalColor += !shading_hit_result.hits * lightInt * fmaxf(.0f, dot(hit_result.normal, (-lightDir)));
	}
	shading_hit_result.hit_color = finalColor;
}


__device__
void reflect_refract(float4 &finalColor, RKDTreeNodeGPU *tree, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list, HitResult &hit_result, HitResult &shading_hit_result)
{
	float4 refractionRColor = make_float4(0), reflectionRColor = make_float4(0);
	float kr = 0, ior = 1.3;
	float3 direct = hit_result.ray_dir;
	bool outside = dot(direct, hit_result.normal) < 0;
	fresnel(direct, hit_result.normal, ior, kr);

	if (kr < 1)
	{
		float3 refractionDirection = refract(direct, hit_result.normal, ior);
		float3 refractionRayOrig = outside ? hit_result.hit_point - hit_result.normal * make_float3(1e-4) : hit_result.hit_point + hit_result.normal * make_float3(1e-4);
		HitResult refraction_result;
		refraction_result.ray_dir = refractionDirection;
		refraction_result.ray_o = refractionRayOrig;
		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, refraction_result);

		if (refraction_result.hits)
		{
			refractionRColor += scene_objs[refraction_result.obj_index].material.color;
		}

		shading_hit_result = refraction_result;
		
	}
	HitResult reflection_result;
	float3 reflectionDirection = reflect(direct, hit_result.normal);
	float3 reflectionRayOrig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(1e-4) : hit_result.hit_point - hit_result.normal * make_float3(1e-4);
	reflection_result.ray_dir = reflectionDirection;
	reflection_result.ray_o = reflectionRayOrig;
	trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_result);

	if (reflection_result.hits)
	{
		reflectionRColor += scene_objs[reflection_result.obj_index].material.color;
	}
	// mix the two
	finalColor += reflectionRColor * (kr) + refractionRColor * (1 - kr);
	finalColor = clip(finalColor);
}

__device__
void refract_light(float4 &finalColor, RKDTreeNodeGPU *tree, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list, HitResult &hit_result, HitResult &shading_hit_result)
{
	float4 refractionRColor = make_float4(0), reflectionRColor = make_float4(0);
	float kr = 0, ior = 1.3;
	float3 direct = hit_result.ray_dir;
	bool outside = dot(direct, hit_result.normal) < 0;
	fresnel(direct, hit_result.normal, ior, kr);

	if (kr < 1)
	{
		float3 refractionDirection = refract(direct, hit_result.normal, ior);
		float3 refractionRayOrig = outside ? hit_result.hit_point - hit_result.normal * make_float3(1e-4) : hit_result.hit_point + hit_result.normal * make_float3(1e-4);
		HitResult refraction_result;
		refraction_result.ray_dir = refractionDirection;
		refraction_result.ray_o = refractionRayOrig;
		trace_shadow(tree, scene_objs, num_objs, root_index, index_list, refraction_result);

		if (refraction_result.hits)
		{
			refractionRColor += refraction_result.hits * 2 * fmaxf(.0f, dot(hit_result.normal, (-refractionDirection)));
		}

		shading_hit_result = refraction_result;

	}
	HitResult reflection_result;
	float3 reflectionDirection = reflect(direct, hit_result.normal);
	float3 reflectionRayOrig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(1e-4) : hit_result.hit_point - hit_result.normal * make_float3(1e-4);
	reflection_result.ray_dir = reflectionDirection;
	reflection_result.ray_o = reflectionRayOrig;
	trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_result);

	if (reflection_result.hits)
	{
		reflectionRColor += reflection_result.hits * 2 * fmaxf(.0f, dot(hit_result.normal, (-reflectionDirection)));
	}
	// mix the two
	finalColor = reflectionRColor * (kr)+refractionRColor * (1 - kr);
	finalColor = clip(finalColor);
}
		
__device__
void reflect(float4 &finalColor, RKDTreeNodeGPU *tree, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list, HitResult &hit_result, HitResult &shading_hit_result)
{
	float3 direct = hit_result.ray_dir;
	bool outside = dot(direct, hit_result.normal) < 0;
	float3 dir = reflect(direct, hit_result.normal);
	float3 orig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(1e-4) : hit_result.hit_point - hit_result.normal * make_float3(1e-4);
	HitResult reflection_hit_result;
	reflection_hit_result.ray_dir = dir;
	reflection_hit_result.ray_o = orig;
	trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_hit_result);
	shading_hit_result = reflection_hit_result;
	if (reflection_hit_result.hits)
	{
		finalColor += 0.8 * scene_objs[reflection_hit_result.obj_index].material.color;
	}
}

__device__
void reflect_light(float4 &finalColor, RKDTreeNodeGPU *tree, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *index_list, HitResult &hit_result, HitResult &shading_hit_result)
{
	float3 direct = hit_result.ray_dir;
	bool outside = dot(direct, hit_result.normal) < 0;
	float3 dir = reflect(direct, hit_result.normal);
	float3 orig = outside < 0 ? hit_result.hit_point + hit_result.normal * make_float3(1e-4) : hit_result.hit_point - hit_result.normal * make_float3(1e-4);
	HitResult reflection_hit_result;
	reflection_hit_result.ray_dir = dir;
	reflection_hit_result.ray_o = orig;
	trace_shadow(tree, scene_objs, num_objs, root_index, index_list, reflection_hit_result);
	shading_hit_result = reflection_hit_result;
	if (reflection_hit_result.hits)
	{
		finalColor += reflection_hit_result.hits * 10 * fmaxf(.0f, dot(hit_result.normal, (-dir)));
		//finalColor = make_float4((finalColor.x + finalColor.y + finalColor.z) / 3);
	}
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
	generate_ray(ray_o, ray_dir, render_camera, stride);

	
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
			bool intersects_tri = gpu_ray_tri_intersect(make_float3(v0.x, v0.y, v0.z), make_float3(edge1.x, edge1.y, edge1.z),
				make_float3(edge2.x, edge2.y, edge2.z), i, tmp_hit_result);

			if (intersects_tri) 
			{
				if (tmp_hit_result.t < hit_result.t)
				{
					hit_result = tmp_hit_result;
					//narmals_mat(pixel_color, tmp_normal);
					//simple_shade(pixel_color, hit_result.normal, ray_dir);
					//phong_light(pixel_color, hit_point, ray_dir, tmp_normal, hit_point, tree, scene_objs, num_objs, root_index, index_list);
				}
			}
		}
	}

	ambient_light(pixel_color);
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

__device__ 
void trace_scene(RKDTreeNodeGPU *tree,
	RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *indexList, int stride, HitResult &hit_result)
{
	// Perform ray-box intersection test.
	for (int i = 0; i < num_objs; i++)
	{
		HitResult tmp_hit_result;
		tmp_hit_result.ray_dir = hit_result.ray_dir;
		tmp_hit_result.ray_o = hit_result.ray_o;

		bool hits = stackless_kdtree_traversal(tree, scene_objs, num_objs, i, root_index, indexList, tmp_hit_result);
		if (hits) {
			if (tmp_hit_result.t < hit_result.t) {
				hit_result = tmp_hit_result;
				hit_result.hits = hits;
			}
		}
	}
}

////////////////////////////////////////////////////
// Perform ray-casting with kd-tree
////////////////////////////////////////////////////
__global__
void trace_primary_rays(RKDTreeNodeGPU *tree,
	RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *indexList, int stride, HitResult *hit_results)
{

	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > SCR_WIDTH * SCR_HEIGHT)
		return;

	HitResult hit_result;
	generate_ray(hit_result.ray_o, hit_result.ray_dir, render_camera, stride);

	trace_scene(tree, render_camera, scene_objs, num_objs, root_index, indexList, stride, hit_result);

	hit_results[index + stride] = hit_result;
	return;
}


__global__
void trace_secondary_rays(float3 *lights, size_t num_lights, RKDTreeNodeGPU *tree,
	RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *indexList, float4 *pixels, HitResult *primary_hit_results, Atmosphere *atmosphere, int stride)
{
	float4 pixel_color = make_float4(0.f);
	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > SCR_WIDTH * SCR_HEIGHT)
		return;

	int x = index % SCR_WIDTH;
	int y = index / SCR_WIDTH;

	float sx = (float)x / (SCR_WIDTH - 1.0f);
	float sy = 1.0f - ((float)y / (SCR_HEIGHT - 1.0f));

	HitResult shad_hit_result;
	if (primary_hit_results[index].hits)
	{
		HitResult hit_result;
		MaterialType hit_mat_type = scene_objs[primary_hit_results[index].obj_index].material.type;
		pixel_color = scene_objs[primary_hit_results[index].obj_index].material.color;
		if (hit_mat_type == TILE)
		{
			int square = (int)floor(primary_hit_results[index].hit_point.x) + (int)floor(primary_hit_results[index].hit_point.z);
			tile_pattern(pixel_color, square);
		}
		else if (hit_mat_type == PHONG)
		{
			phong_light(lights, num_lights, pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
		}
		else if (hit_mat_type == REFLECT)
		{
			reflect(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], hit_result);
			primary_hit_results[index] = hit_result;
		}
		else if (hit_mat_type == REFRACT)
		{
			reflect_refract(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], hit_result);
			primary_hit_results[index] = hit_result;
		}
	}
	else
	{
		// if ray missed draw sky there.
		//sky_mat(pixel_color, primary_hit_results[index].ray_dir);
		float t_min = 0, t1 = kInfinity, t_max = kInfinity;
		pixel_color = make_float4(compute_incident_light(atmosphere, make_float3(0, atmosphere->earthRadius + 10000, 300000), primary_hit_results[index].ray_dir, 0, t_max),0);

		pixel_color.x = pixel_color.x < 1.413f ? powf(pixel_color.x * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-pixel_color.x);
		pixel_color.y = pixel_color.y < 1.413f ? powf(pixel_color.y * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-pixel_color.y);
		pixel_color.z = pixel_color.z < 1.413f ? powf(pixel_color.z * 0.38317f, 1.0f / 2.2f) : 1.0f - exp(-pixel_color.z);
	}

	pixel_color = clip(pixel_color);
	primary_hit_results[index].hit_color = pixel_color;
	pixels[index + stride] = pixel_color;
}


////////////////////////////////////////////////////
// Generate a shadow map and store it as
// an array of float4s
////////////////////////////////////////////////////
__global__
void generate_shadow_map(float3 *lights, size_t num_lights, RKDTreeNodeGPU *tree,
	RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *indexList, float4 *pixels, HitResult *primary_hit_results, int stride)
{
	float4 pixel_color = make_float4(0);
	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > SCR_WIDTH * SCR_HEIGHT)
		return;
	if (!primary_hit_results[index].hits)
	{
		pixel_color = make_float4(0);
		primary_hit_results[index].hit_color = pixel_color;
		pixels[index + stride] = pixel_color;
		return;
	}

	HitResult shad_hit_result;


	shade(lights, num_lights, pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);

	primary_hit_results[index] = shad_hit_result;
	primary_hit_results[index].hit_color = pixel_color;

	pixel_color = clip(pixel_color);
	pixels[index + stride] = pixel_color;
}



__global__
void trace_secondary_shadow_rays(curandState *rand_state, float3 *lights, size_t num_lights, RKDTreeNodeGPU *tree,
	RCamera *render_camera, GPUSceneObject *scene_objs, int num_objs,
	int *root_index, int *indexList, float4 *pixels, HitResult *primary_hit_results, int stride)
{
	float4 pixel_color = make_float4(0.f);
	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > SCR_WIDTH * SCR_HEIGHT)
		return;

	curandState local_random = rand_state[index];
	HitResult shad_hit_result;
	if (primary_hit_results[index].hits)
	{
		HitResult hit_result;
		MaterialType hit_mat_type = scene_objs[primary_hit_results[index].obj_index].material.type;
		// Compute indirect light for diffuse objects
		if (hit_mat_type == TILE || hit_mat_type == PHONG)
		{
			float4 direct_light = make_float4(0);
			shade(lights, num_lights, direct_light, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], hit_result);

			float4 indirectLigthing = make_float4(0);

			uint32_t N = 128;
			float3 Nt, Nb;
			createCoordinateSystem(primary_hit_results[index].normal, Nt, Nb);
			float pdf = 1 / (2 * M_PI);
			for (uint32_t n = 0; n < N; ++n) {
				HitResult local_hit_result;
				float r1 = curand_uniform(&local_random);
				float r2 = curand_uniform(&local_random);
				float3 sample = uniformSampleHemisphere(r1, r2);
				float3 sample_world = make_float3(
					sample.x * Nb.x + sample.y * primary_hit_results[index].normal.x + sample.z * Nt.x,
					sample.x * Nb.y + sample.y * primary_hit_results[index].normal.y + sample.z * Nt.y,
					sample.x * Nb.z + sample.y * primary_hit_results[index].normal.z + sample.z * Nt.z);
				// don't forget to divide by PDF and multiply by cos(theta)
				local_hit_result.ray_o = primary_hit_results[index].hit_point + sample_world * make_float3(1e-4);
				local_hit_result.ray_dir = sample_world;
				trace_scene(tree, render_camera, scene_objs, num_objs, root_index, indexList, stride, local_hit_result);
				indirectLigthing += r1 *local_hit_result.hit_color / pdf;
			}
			// divide by N
			indirectLigthing /= (float)N;
			pixel_color = (direct_light / M_PI + 2 * indirectLigthing) * 0.18f;
			//shad_hit_result.hit_color = pixel_color;
		}
		else if(hit_mat_type == REFLECT)
		{
			reflect_light(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
			//primary_hit_results[index] = hit_result;
		}
		else if (hit_mat_type == REFRACT)
		{
			refract_light(pixel_color, tree, scene_objs, num_objs, root_index, indexList, primary_hit_results[index], shad_hit_result);
			//primary_hit_results[index] = hit_result;
		}
		primary_hit_results[index] = shad_hit_result;

	}
	else
	{
		//pixel_color = make_float4(0.f);
	}

	
	pixel_color = clip(pixel_color);
	gray_scale(pixel_color);

	primary_hit_results[index].hit_color = pixel_color;
	pixels[index + stride] += pixel_color;
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

	cudaDeviceReset();
}

extern "C"
void update_objects(std::vector<GPUSceneObject> objs)
{
	cudaFree(d_scene_objects);
	cudaMalloc(&d_scene_objects, objs.size() * sizeof(GPUSceneObject));
	cudaMemcpy(d_scene_objects, &objs[0], objs.size() * sizeof(GPUSceneObject), cudaMemcpyHostToDevice);
}

__global__
void register_random(curandState *rand_state)
{
	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > SCR_WIDTH * SCR_HEIGHT)
		return;

	curand_init(19622342346234384, index, 0, &rand_state[index]);
}

extern "C"
void copy_memory(std::vector<RKDThreeGPU *>tree, RCamera sceneCam, std::vector<float4> triangles, std::vector<float4> normals, std::vector<GPUSceneObject> objs, bool bruteforce = false)
{
	// --------------------------------Initialize host variables----------------------------------------------------
	size_t num_objs = objs.size();
	size_t image_size = SCR_WIDTH * SCR_HEIGHT;
	angle = 0;
	int size = image_size * sizeof(float4);
	int shadow_map_size = image_size * sizeof(float4);
	size_t size_hit_result = image_size * sizeof(HitResult);
	size_t size_kd_tree = 0;

	Atmosphere *h_atmosphere = new Atmosphere();

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

	size_t light_size = 2;
	float3 *h_lights = new float3[light_size];

	h_lights[0] = make_float3(0, 25, 0);
	h_lights[1] = make_float3(3, 15, 10);
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

	cudaMalloc(&rand_state, image_size * sizeof(curandState));
	cudaMalloc(&d_pixels, size);
	cudaMalloc(&d_atmosphere, sizeof(Atmosphere));
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

	if (num_objs > 0)
	{
		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void **)&dev_triangle_p, triangle_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_triangle_p, &triangles[0], triangle_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_triangles_tro_texture(dev_triangle_p, total_num_triangles);

		// allocate memory for the triangle meshes on the GPU
		cudaMalloc((void **)&dev_normals_p, normals_size);

		// copy triangle data to GPU
		cudaMemcpy(dev_normals_p, &normals[0], normals_size, cudaMemcpyHostToDevice);

		// load triangle data into a CUDA texture
		bind_normals_tro_texture(dev_normals_p, normals_size);
	}


	// Copy host vectors to device.
	cudaMemset(d_hit_result, 0, size_hit_result);
	cudaMemset(d_shadow_hit_result, 0,size_hit_result);
	cudaMemcpy(d_pixels, h_pixels, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atmosphere, h_atmosphere, sizeof(Atmosphere), cudaMemcpyHostToDevice);
	cudaMemcpy(d_indirect_map, h_pixels, shadow_map_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_shadow_map, h_pixels, shadow_map_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_light, h_lights, light_size * sizeof(float3), cudaMemcpyHostToDevice);
	cudaMemcpy(d_render_camera, h_camera, sizeof(RCamera), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tree, h_tree, size_kd_tree * sizeof(RKDTreeNodeGPU), cudaMemcpyHostToDevice);
	cudaMemcpy(d_index_list, kd_tree_tri_indics.data(), size_kd_tree_tri_indices, cudaMemcpyHostToDevice);
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

	register_random<<<SCR_WIDTH, SCR_HEIGHT>>>(rand_state);

}


__global__
void mix_color_maps(float4 *color_map, float4 *shadow_map)
{
	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > SCR_WIDTH * SCR_HEIGHT)
		return;

	// Mix the two color maps by using subtracive method.
	color_map[index] = color_map[index] - (color_map[index] - shadow_map[index]) / 2;

	// Postprocess light.
	ambient_light(color_map[index]);
	clip(color_map[index]);
}

__global__
void mix_direct_indirect_light(float4 *direct_map, float4 *indirect_map)
{
	int index = (threadIdx.x * gridDim.x) + blockIdx.x;
	if (index > SCR_WIDTH * SCR_HEIGHT)
		return;

	// Mix the two color maps by using subtracive method.
	direct_map[index] = (direct_map[index]/M_PI + 2 * indirect_map[index]) * 0.18f;

	// Postprocess light.
	//ambient_light(direct_map[index]);
	clip(direct_map[index]);
}

__global__
void Craze(float3 *lights, float angle, Atmosphere *atmosphere)
{

	float x = cosf(angle) * 20.f;
	float z = sinf(angle) * 20.f;

	lights[0] = make_float3(x, 30, x);
	lights[1] = make_float3(x, 15, z);
	float ang = angle * M_PI * 0.6;
	atmosphere->sunDirection = make_float3(0, cosf(ang), sinf(ang));
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

	// Generate primary rays and cast them throught the scene.
	trace_primary_rays << < blockSize, gridSize >> > ( d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index,  d_index_list, 0, d_hit_result);
	cudaDeviceSynchronize();
	cudaMemcpy(d_shadow_hit_result, d_hit_result, SCR_WIDTH * SCR_HEIGHT * sizeof(HitResult), cudaMemcpyDeviceToDevice);
	// Generate primary shadow map
	generate_shadow_map << < blockSize, gridSize >> > (d_light, num_light, d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list, d_shadow_map, d_shadow_hit_result, 0);
	//for (int i = 0; i < 500; ++i)
	//{
	//	trace_secondary_shadow_rays << < blockSize, gridSize >> > (rand_state, d_light, num_light, d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list, d_indirect_map, d_shadow_hit_result, 0);
	//}
	
	// Generate secondary rays and evaluate scene colors from them.
	for (int i = 0; i < 2; ++i)
	{
		trace_secondary_rays << < blockSize, gridSize >> > (d_light, num_light, d_tree, d_render_camera, d_scene_objects, d_object_number, d_root_index, d_index_list, d_pixels, d_hit_result,d_atmosphere, 0);
	}

	//mix_direct_indirect_light << <blockSize, gridSize >> > (d_shadow_map, d_indirect_map);
	mix_color_maps << <blockSize, gridSize >> > (d_pixels, d_shadow_map);

	cudaDeviceSynchronize();

	Craze<<<1, 1>>>(d_light, angle, d_atmosphere);
	angle += 0.001;  // or some other value.  Higher numbers = circles faster


	// Finish timer used for benchmarking.
	//auto finish = std::chrono::steady_clock::now();

	//float elapsed_seconds = std::chrono::duration_cast<
	//std::chrono::duration<float>>(finish - start).count();
	//std::cout << "Rendering of a on the GPU frame has finished in " << elapsed_seconds << " seconds." << std::endl;

	// Copy pixel array back to host.
	cudaMemcpyAsync(h_pixels, d_pixels, size, cudaMemcpyDeviceToHost);

	//cudaMemset(d_indirect_map, 0, size);

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