#include "RayTracer.h"
#include "Camera.h"
#include "Sphere.h"


#include "Light.h"
#include "Plane.h"
#include "RStaticMesh.h"
#include "Source.h"
#include "Triangle.h"
#include "KDTree.h"
#include "Object.h"

#include <limits>
#include <random>
#include <cmath>
#include <iostream>
#include <memory>

#include "cuda_runtime_api.h"
#include "helper_math.h"
#include "Grid.h"

float4 _clip(float4 color)
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

	RRayTracer::RRayTracer()
	{
		float4 whiteLight = make_float4(1.0, 1.0, 1.0, 0);
		//Position of light
		float3 lightPosition = make_float3(0, 3, 0);

		//Light
		RLight sceneLight(lightPosition, whiteLight);

		//Set of light sources(can be several)
		lightSources.push_back(std::shared_ptr<RSource>(&sceneLight));
	}

	float4* RRayTracer::sphere_trace(Grid* sdf, RCamera* scene_camera)
	{
		float aspectratio = SCR_WIDTH / (float)SCR_HEIGHT;

		size_t size = SCR_WIDTH * SCR_HEIGHT * sizeof(float4);

		float4* pixels = new float4[size];

		auto start = std::chrono::system_clock::now();

		std::cout << "Starting rendering on CPU" << std::endl;

#pragma omp distribute parallel for schedule(dynamic)
		{
			for (int i = 0; i < SCR_WIDTH; i++)
			{
				for (int j = 0; j < SCR_HEIGHT; j++)
				{
					int flatIndex = j * SCR_WIDTH + i;

					float sx = (float)i / (SCR_WIDTH - 1.0f);
					float sy = 1.0f - ((float)j / (SCR_HEIGHT - 1.0f));

					float3 rendercampos = scene_camera->campos;

					// compute primary ray direction
					// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
					float3 rendercamview = scene_camera->view; rendercamview = normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
					float3 rendercamup = scene_camera->camdown; rendercamup = normalize(rendercamup);
					float3 horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis = normalize(horizontalAxis); // Important to normalize!
					float3 verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

					float3 middle = rendercampos + rendercamview;
					float3 horizontal = horizontalAxis * tanf(scene_camera->fov.x * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5
					float3 vertical = verticalAxis * tanf(scene_camera->fov.y * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5

					// compute pixel on screen
					float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
					float3 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * scene_camera->focial_distance); // Important for depth of field!		

					float3 aperturePoint = rendercampos;

					// calculate ray direction of next ray in path
					float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
					apertureToImagePlane = normalize(apertureToImagePlane); // ray direction needs to be normalised

					// ray direction
					float3 rayInWorldSpace = apertureToImagePlane;
					float3 ray_dir = normalize(rayInWorldSpace);

					// ray origin
					float3 ray_o = rendercampos;
					RRay cam_ray = RRay(ray_o, ray_dir);
					float t = 0;
					int step_count = 0;
					while (t < 100)
					{
						float min_distance = K_INFINITY;
						float3 from = ray_o + t * ray_dir;
						//min_distance = sdf->get_distance(from);

						if (min_distance <= 10e-6 * t)
						{
							float4 finalRColor = make_float4(t / step_count);
							pixels[flatIndex].x = finalRColor.x;
							pixels[flatIndex].y = finalRColor.y;
							pixels[flatIndex].z = finalRColor.z;

							_clip(pixels[flatIndex]);
							break;
						}
						t += min_distance;
						step_count++;
					}


				}
			}
		}

		auto end = std::chrono::system_clock::now();

		float elapsed = std::chrono::duration_cast<
			std::chrono::duration<float>>(end - start).count();
		std::cout << elapsed << " seconds took rendering of a frame on a CPU" << std::endl;
		return pixels;
	}




	float4* RRayTracer::trace(RKDTreeCPU* tree, RCamera* scene_camera)
	{

		float aspectratio = SCR_WIDTH / (float)SCR_HEIGHT;

		size_t size = SCR_WIDTH * SCR_HEIGHT * sizeof(float4);

		float4* pixels = new float4[size];

		auto start = std::chrono::system_clock::now();

		std::cout << "Starting rendering on CPU" << std::endl;

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < SCR_WIDTH; i++)
		{
			for (int j = 0; j < SCR_HEIGHT; j++)
			{
				int flatIndex = j * SCR_WIDTH + i;

				float sx = (float)i / (SCR_WIDTH - 1.0f);
				float sy = 1.0f - ((float)j / (SCR_HEIGHT - 1.0f));

				float3 rendercampos = scene_camera->campos;

				// compute primary ray direction
				// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
				float3 rendercamview = scene_camera->view; rendercamview = normalize(rendercamview); // view is already supposed to be normalized, but normalize it explicitly just in case.
				float3 rendercamup = scene_camera->camdown; rendercamup = normalize(rendercamup);
				float3 horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis = normalize(horizontalAxis); // Important to normalize!
				float3 verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis = normalize(verticalAxis); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

				float3 middle = rendercampos + rendercamview;
				float3 horizontal = horizontalAxis * tanf(scene_camera->fov.x * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5
				float3 vertical = verticalAxis * tanf(scene_camera->fov.y * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5

				// compute pixel on screen
				float3 pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
				float3 pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * scene_camera->focial_distance); // Important for depth of field!		

				float3 aperturePoint = rendercampos;

				// calculate ray direction of next ray in path
				float3 apertureToImagePlane = pointOnImagePlane - aperturePoint;
				apertureToImagePlane = normalize(apertureToImagePlane); // ray direction needs to be normalised

				// ray direction
				float3 rayInWorldSpace = apertureToImagePlane;
				float3 ray_dir = normalize(rayInWorldSpace);

				// ray origin
				float3 ray_o = rendercampos;
				RRay cam_ray = RRay(ray_o, ray_dir);
				float4 finalRColor = castRay(cam_ray, 0, tree);
				pixels[flatIndex].x = finalRColor.x;
				pixels[flatIndex].y = finalRColor.y;
				pixels[flatIndex].z = finalRColor.z;
			}
		}

		auto end = std::chrono::system_clock::now();

		float elapsed = std::chrono::duration_cast<
			std::chrono::duration<float>>(end - start).count();
		std::cout << elapsed << " seconds took rendering of a frame on a CPU" << std::endl;
		return pixels;
	}


	inline float modulo(const float& f)
	{
		return f - std::floor(f);
	}

	float4 RRayTracer::castRay(RRay ray, int depth, RKDTreeCPU* node)
	{
		//float4 finalRColor = make_float4(0, 0, 0, 0);
		//if (depth > 2) return make_float4(0, 0, 0, 0);
		//float tNear = K_INFINITY;
		//float3 tmp_normal;

		////distance to the intersection and bariocentric coordinates
		//if (traceShadow(ray, tNear, tmp_normal, node))
		//{
		//	//qDebug() << node->intersectionAmount;
		//	//simple_shade(finalRColor, tmp_normal, ray->getRayDirection());
		//	finalRColor.x = (tmp_normal.x < 0.0f) ? (tmp_normal.x * -1.0f) : tmp_normal.x;
		//	finalRColor.y = (tmp_normal.y < 0.0f) ? (tmp_normal.y * -1.0f) : tmp_normal.y;
		//	finalRColor.z = (tmp_normal.z < 0.0f) ? (tmp_normal.z * -1.0f) : tmp_normal.z;
		//	float3 intersectionPosition = ray.getRayOrigin() + (ray.getRayDirection() * tNear);
		//	float3 intersectingRayDirection = ray.getRayDirection();
		//	float3 normal = hitObjectGetNormalAt(intersectionPosition);
		//	float3 bias = normal * (K_EPSILON);

		//	//if object has tile pattern
		//	if (hitObject->GetColor().w == 2)
		//	{
		//		int square = (int)floor(intersectionPosition.x) + (int)floor(intersectionPosition.z);
		//		tilePattern(finalRColor, square);
		//	}
		//	ambientLight(finalRColor);
		//	//Phong color model
		//	if (hitObject->GetColor().w > 0 && hitObject->GetColor().w <= 1)
		//	{
		//		float4 diffuse, specular;
		//		for (int i = 0; i < lightSources.size(); i++)
		//		{
		//			float3 lightDir;
		//			float4 lightInt;
		//			float tShadowed;
		//			lightSources[i]->Illuminate(intersectionPosition, lightDir, lightInt, tShadowed);
		//			RObject *intObj;

		//			RRay difRay(intersectionPosition + (normal), -lightDir);

		//			bool vis = traceShadow(difRay, tShadowed, &intObj, node);

		//			diffuse = diffuse + (lightInt * (vis * 0.18) * (std::max(0.0, (double)dot(normal,(-lightDir)))));

		//			float3 R = reflect(lightDir, normal);
		//			float3 dir = ray.getRayDirection();
		//			specular = specular + (lightInt * (vis) * (std::pow(std::max(0.0, (double)dot(R, -dir)), 10)));
		//		}
		//		 finalRColor = finalRColor + (diffuse * (0.8)) + (specular * (0.2));
		//	}
		//	//reflections
		//	if (hitObject->GetColor().w > 0 && hitObject->GetColor().w <= 1)
		//	{
		//		float4 refractionRColor, reflectionRColor;
		//		float kr = 0, ior = 1.5;
		//		float3 direct = ray.getRayDirection();
		//		fresnel(direct, normal, ior, kr);
		//		if (kr < 1)
		//		{
		//			float3 refractionDirection = normalize(refract(direct, normal, ior));
		//			float3 refractionRayOrig = dot(refractionDirection, normal) < 0 ? intersectionPosition  - (bias) : intersectionPosition + (bias);
		//			RRay refractionRay(refractionRayOrig, refractionDirection);
		//			refractionRColor = castRay(refractionRay, depth + 1, node);
		//		}

		//		float3 reflectionDirection = normalize(reflect(direct, normal));
		//		float3 reflectionRayOrig = dot(reflectionDirection, normal) < 0 ? intersectionPosition + (bias) : intersectionPosition -(bias);
		//		RRay reflectionRay(reflectionRayOrig, reflectionDirection);
		//		reflectionRColor = castRay(reflectionRay, depth + 1, node);

		//		// mix the two
		//		finalRColor = finalRColor + (reflectionRColor * (kr)) + (refractionRColor * (1 - kr));
		//	}

		//	//shadows
		//	for (size_t i = 0; i < lightSources.size(); ++i)
		//	{
		//		float3 lightDir;
		//		float4 lightIntensity;
		//		float tShad;
		//		RObject *shadObject;
		//		lightSources[i]->Illuminate(intersectionPosition, lightDir, lightIntensity, tShad);
		//		RRay shadowRay(intersectionPosition + (bias), -lightDir);
		//		bool vis = !traceShadow(shadowRay, tShad, &shadObject, node);
		//		if (vis)
		//			finalRColor = finalRColor + (lightIntensity * (std::max(0.0, (double)dot(normal,-lightDir))) * (0.099));

		//	}
		//}
		//return finalRColor;
		return make_float4(0);
	}

	void RRayTracer::tilePattern(float4& color, int square)
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

	void RRayTracer::ambientLight(float4& color)
	{
		color = color * (0.2f);
	}

	void RRayTracer::fresnel(float3& I, float3& N, float& ior, float& kr)
	{
		const float _dot = dot(I, N);
		float cosi = clamp(-1, 1, _dot);
		float etai = 1, etat = ior;
		if (cosi > 0) { std::swap(etai, etat); }
		// Compute sini using Snell's law
		const float sint = etai / etat * sqrtf(std::max(0.f, 1 - cosi * cosi));
		// Total internal reflection
		if (sint >= 1) {
			kr = 1;
		}
		else {
			const float cost = sqrtf(std::max(0.f, 1 - sint * sint));
			cosi = fabsf(cosi);
			const float Rs = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
			const float Rp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
			kr = (Rs * Rs + Rp * Rp) / 2;
		}
	}

	float3 RRayTracer::refract(float3& I, float3& N, float& ior)
	{
		float _dot = dot(I, N);
		float cosi = clamp(-1, 1, _dot);
		float etai = 1, etat = ior;
		float3 n = N;
		if (cosi < 0) { cosi = -cosi; }
		else { std::swap(etai, etat); n = -N; }
		float eta = etai / etat;
		float k = 1 - eta * eta * (1 - cosi * cosi);
		if (k < 0)
			return make_float3(0);
		else
			return I * (eta)+(n * (eta * cosi - sqrtf(k)));
	}

	float3 RRayTracer::reflect(float3& I, float3& N)
	{
		float num = 2 * dot(I, N);
		return I - (N * (num));
	}

	bool RRayTracer::traceShadow(RRay ray,
		float& tNear, float3& normal, RKDTreeCPU* tree)
	{
		bool inter = tree->singleRayStacklessIntersect(ray, tNear, normal);
		return inter;
	}

	float RRayTracer::clamp(float lo, float hi, float v)
	{
		return std::max(lo, std::min(hi, v));
	}



	void RRayTracer::simple_shade(float4& color, float3 normal, float3 ray_dir)
	{
		color = make_float4(fmaxf(0.f, dot(normal, -ray_dir))); // facing ratio 
	}

	//void trace(std::vector<RObject> scene_objects, double xCamPos, double zCamPos, double xLookAt, double yLookAt, double zLookAt)
	//{
	//	size_t width = 500;
	//	size_t height = 500;
	//	float3 Y = make_float3(0, 1, 0);
	//
	//	double aspectratio = width / (double)height;
	//
	//	float3 cam_pos = make_float3(xCamPos, 0.3, zCamPos);
	//
	//	// Where camera looks at.
	//	float3 look_at = make_float3(xLookAt, yLookAt, zLookAt);
	//
	//	// Vector between camera and the point where to look at.
	//	float3 diff_btw = cam_pos - look_at;
	//
	//	// Compute variables that define camera.
	//	float3 camDir = normalize(-diff_btw);
	//	float3 camDown = normalize(cross(Y, camDir));
	//	float3 camRight = cross(camDown, camDir);
	//
	//	// Create camera.
	//	RCamera sceneCam(cam_pos, camDir, camDown, camRight);
	//
	//	// Rendered image.
	//	RGBType *pixels = new RGBType[width * height * sizeof(RGBType)];
	//
	//	// Loop over all if the pixels of the image.
	//	for (size_t i = 0; i < width; i++)
	//	{
	//		for (size_t j = 0; j < height; j++)
	//		{
	//			int flatIndex = j * width + i;
	//
	//			// Find position of pixels for perspective projection.
	//			double x = (i + 0.5) / width;
	//			double y = ((height - j) + 0.5) / height;
	//
	//			// Initialise variables that define ray.
	//			float3 cam_ray_origin = sceneCam.getCameraPosition();
	//			float3 cam_ray_direction = normalize(camDir + camDown * x + camRight * y);
	//			RRay cam_ray(cam_ray_origin, cam_ray_direction, t_hit);
	//
	//			float t_hit;
	//			float t_min = K_INFINITY;
	//			
	//			// Loops over all of the scene objects and test for the intersection with each of them.
	//			if (find_intersection(cam_ray, scene_objects)
	//			{
	//				if (t_hit < t_min)
	//				{
	//					// If intersection is found, then paint the pixel.
	//					t_min = t_hit;
	//					float4 finalColor = castRay(cam_ray, 0, scene_objects);
	//					pixels[flatIndex].r = finalColor.x;
	//					pixels[flatIndex].g = finalColor.y;
	//					pixels[flatIndex].b = finalColor.z;
	//				}
	//			}
	//		}
	//	}
	//}
