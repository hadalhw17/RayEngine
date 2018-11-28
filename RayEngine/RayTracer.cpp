#include "RayTracer.h"
#include "Camera.h"
#include "Sphere.h"


#include "Light.h"
#include "Plane.h"
#include "RStaticMesh.h"
#include "Color.h"
#include "Source.h"
#include "Triangle.h"
#include "KDTree.h"
#include "Object.h"

#include <limits>
#include <random>
#include <cmath>
#include <iostream>

#include "cuda_runtime.h"
#include "cutil_math.h"



RVectorF Y(0, 1, 0);
RRayTracer::RRayTracer()
{

}


RGBType *RRayTracer::trace(RKDTreeCPU *tree, float xCamPos, float zCamPos, float xLookAt, float yLookAt, float zLookAt)
{

	RVectorF A(0, 3, 0);
	RVectorF B(3, 3, 0);
	RVectorF C(3, 0, 0);
	RVectorF D(0, 0, 0);

	float aspectratio = SCR_WIDTH / (float)SCR_HEIGHT;

	float3 camPos = make_float3(xCamPos, 0.3, zCamPos);

	float3 diffBtw = make_float3(camPos.x - xLookAt,
		camPos.y - yLookAt,
		camPos.z - zLookAt);
	float3 Y = make_float3(0.f, 1.f, 0.f);
	//Variables defining camera
	float3 camdir = normalize(-diffBtw);
	float3 camdown = normalize(cross(Y, camdir));
	float3 camright = cross(camdown, camdir);
	//RCamera
	RCamera sceneCam(camPos, camdir, camdown, camright);

	RColor whiteLight(1.0, 1.0, 1.0, 0);
	//Position of light
	RVectorF lightPosition(0, 3, 0);

	//Light
	RLight sceneLight(lightPosition, whiteLight);

	//Set of light sources(can be several)
	lightSources.push_back(std::shared_ptr<RSource>(&sceneLight));


	RGBType *pixels = new RGBType[SCR_WIDTH * SCR_HEIGHT * sizeof(pixels)];

	auto start = std::chrono::system_clock::now();

	std::cout << "Starting rendering on CPU" << std::endl;

#pragma omp parallel for schedule(dynamic)
		for (int i = 0; i < SCR_WIDTH; i++)
		{
			for (int j = 0; j < SCR_HEIGHT; j++)
			{
				int flatIndex = j * SCR_WIDTH + i;

				float x = (i + 0.5) / SCR_WIDTH;
				float y = ((SCR_HEIGHT - j) + 0.5) / SCR_HEIGHT;


				//cast rays
				float3 cam_ray_origin = sceneCam.getCameraPosition();
				float3 cam_ray_direction = normalize(camdir +  camdown * (x) + (camright * (y)));

				RRay *cam_ray = new RRay(cam_ray_origin, cam_ray_direction);
				RColor finalRColor = castRay(cam_ray, 0, tree);
				pixels[flatIndex].r = finalRColor.GetColorRed();
				pixels[flatIndex].g = finalRColor.GetColorGreen();
				pixels[flatIndex].b = finalRColor.GetColorBlue();
			}
		}

	auto end = std::chrono::system_clock::now();

	float elapsed = std::chrono::duration_cast<
		std::chrono::duration<float>>(end - start).count();
	std::cout << elapsed << " seconds took rendering of a frame on a CPU" << std::endl;
	return pixels;
}


inline float modulo(const float &f)
{
	return f - std::floor(f);
}

RColor RRayTracer::castRay(RRay *ray, int depth, RKDTreeCPU *node)
{
	//RColor finalRColor(0, 0, 0, 0);
	//if (depth > 2) return RColor(0, 0, 0, 0);
	//RObject *hitObject = nullptr;
	//float tNear = kInfinity;

	////distance to the intersection and bariocentric coordinates
	//if (traceShadow(ray, tNear, &hitObject, node))
	//{
	//	//qDebug() << node->intersectionAmount;
	//	finalRColor = hitObject->GetColor();
	//	RVectorF intersectionPosition = ray->getRayOrigin().VectorAdd(ray->getRayDirection().VectorMult(tNear));
	//	RVectorF intersectingRayDirection = ray->getRayDirection();
	//	RVectorF normal = hitObject->GetNormalAt(intersectionPosition);
	//	RVectorF bias = normal.VectorMult(1e-4);

	//	//if object has tile pattern
	//	if (hitObject->GetColor().GetColorSpecial() == 2)
	//	{
	//		int square = (int)floor(intersectionPosition.getVecX()) + (int)floor(intersectionPosition.getVecZ());
	//		tilePattern(finalRColor, square);
	//	}
	//	ambientLight(finalRColor);
	//	//Phong color model
	//	if (hitObject->GetColor().GetColorSpecial() > 0 && hitObject->GetColor().GetColorSpecial() <= 1)
	//	{
	//		RColor diffuse, specular;
	//		for (int i = 0; i < lightSources.size(); i++)
	//		{
	//			float3 lightDir;
	//			RColor lightInt;
	//			float tShadowed;
	//			lightSources[i]->Illuminate(intersectionPosition, lightDir, lightInt, tShadowed);
	//			RObject *intObj;

	//			RRay *difRay = new RRay(intersectionPosition.VectorAdd(normal), -lightDir);

	//			bool vis = traceShadow(difRay, tShadowed, &intObj, node);

	//			diffuse = diffuse.ColorAdd(lightInt.ColorScalar(vis * 0.18).ColorScalar(std::max(0.0, (double)normal.DotProduct(lightDir.Negative()))));

	//			RVectorF R = reflect(lightDir, normal);
	//			specular = specular.ColorAdd(lightInt.ColorScalar(vis).ColorScalar(std::pow(std::max(0.0, (double)R.DotProduct(-ray->getRayDirection())), 10)));
	//		}
	//		finalRColor = finalRColor.ColorMultiply(diffuse.ColorScalar(0.8)).ColorAdd(specular.ColorScalar(0.2));
	//	}
	//	//reflections
	//	if (hitObject->GetColor().GetColorSpecial() > 0 && hitObject->GetColor().GetColorSpecial() <= 1)
	//	{
	//		RColor refractionRColor, reflectionRColor;
	//		float kr = 0, ior = 1.5;
	//		RVectorF direct = ray->getRayDirection();
	//		fresnel(direct, normal, ior, kr);
	//		if (kr < 1)
	//		{
	//			RVectorF refractionDirection = refract(direct, normal, ior).Normalize();
	//			RVectorF refractionRayOrig = refractionDirection.DotProduct(normal) < 0 ? intersectionPosition.VectorAdd(bias.Negative()) : intersectionPosition.VectorAdd(bias);
	//			RRay *refractionRay = new RRay(refractionRayOrig, refractionDirection);
	//			refractionRColor = castRay(refractionRay, depth + 1, node);
	//		}

	//		RVectorF reflectionDirection = reflect(direct, normal).Normalize();
	//		RVectorF reflectionRayOrig = reflectionDirection.DotProduct(normal) < 0 ? intersectionPosition.VectorAdd(bias) : intersectionPosition.VectorAdd(bias.Negative());
	//		RRay *reflectionRay = new RRay(reflectionRayOrig, reflectionDirection);
	//		reflectionRColor = castRay(reflectionRay, depth + 1, node);

	//		// mix the two
	//		finalRColor = finalRColor.ColorAdd(reflectionRColor.ColorScalar(kr)).ColorAdd(refractionRColor.ColorScalar(1 - kr));
	//	}

	//	//shadows
	//	for (size_t i = 0; i < lightSources.size(); ++i)
	//	{
	//		RVectorF lightDir;
	//		RColor lightIntensity;
	//		float tShad;
	//		RObject *shadObject;
	//		lightSources[i]->Illuminate(intersectionPosition, lightDir, lightIntensity, tShad);
	//		RRay *shadowRay = new RRay(intersectionPosition.VectorAdd(bias), lightDir.Negative());
	//		bool vis = !traceShadow(shadowRay, tShad, &shadObject, node);
	//		if (vis)
	//			finalRColor = finalRColor.ColorAdd(lightIntensity.ColorMultiply(lightSources[i]->getLightColor())
	//				.ColorScalar(std::max(0.0, (double)normal.DotProduct(lightDir.Negative()))).ColorScalar(0.099));

	//	}
	//}
	//return finalRColor.Clip();
}

void RRayTracer::tilePattern(RColor &color, int square)
{
	if ((square % 2) == 0) {
		// black tile
		color.SetColorRed(0);
		color.SetColorGreen(0);
		color.SetColorBlue(0);
	}
	else {
		// white tile
		color.SetColorRed(1);
		color.SetColorGreen(1);
		color.SetColorRed(1);
	}
}

void RRayTracer::ambientLight(RColor &color)
{
	color = color.ColorScalar(0.2);
}

void RRayTracer::fresnel(RVectorF &I, RVectorF &N, float &ior, float &kr)
{
	const float dot = I.DotProduct(N);
	float cosi = clamp(-1, 1, dot);
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

RVectorF RRayTracer::refract(RVectorF &I, RVectorF &N, float &ior)
{
	float dot = I.DotProduct(N);
	float cosi = clamp(-1, 1, dot);
	float etai = 1, etat = ior;
	RVectorF n = N;
	if (cosi < 0) { cosi = -cosi; }
	else { std::swap(etai, etat); n = N.Negative(); }
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? 0 : I.VectorMult(eta).VectorAdd(n.VectorMult(eta * cosi - sqrtf(k)));
}

RVectorF RRayTracer::reflect(RVectorF &I, RVectorF &N)
{
	float num = 2 * I.DotProduct(N);
	return I.VectorAdd(N.VectorMult(num)).Negative();
}

bool RRayTracer::traceShadow(RRay *ray,
	float &tNear,
	RObject **hitObject,
	RKDTreeCPU *tree)
{
	*hitObject = nullptr;

	bool inter = tree->intersect(tree->root, ray, tNear, hitObject);
	return inter;
}

float RRayTracer::clamp(float lo, float hi, float v)
{
	return std::max(lo, std::min(hi, v));
}


