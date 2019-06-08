#pragma once

#include <vector>
#include <memory>



class RObject;
class RKDTreeCPU;
struct float4;
struct float3;
class RSource;
class RTriangle;
class RRay;
class Grid;

class RRayTracer
{
public:

	std::vector<std::shared_ptr<RSource>> lightSources;
	RRayTracer();

	float4* sphere_trace(Grid* sdf, class RCamera* scene_cam);

	float4 *trace(RKDTreeCPU *tree, class RCamera *scene_cam);


	bool traceShadow(RRay , float &, float3 &normal, RKDTreeCPU *tree);
	float4 castRay(RRay ray, int depth, RKDTreeCPU *node);
	void generateScene(std::vector<std::shared_ptr<RObject>> &);

	//Color options
	void tilePattern(float4 &, int square);
	void ambientLight(float4 &);
	void specularReflection(RRay, std::vector<std::shared_ptr<RObject>> &, float4 &, float4);
	void fresnel(float3 &, float3 &, float &, float &);
	float3 refract(float3 &, float3 &, float &);

	float3 reflect(float3 &, float3 &);

	float clamp(float lo, float hi, float v);

	void simple_shade(float4 &color, float3 normal, float3 ray_dir);

private:
	int counterFor = 0;
};

