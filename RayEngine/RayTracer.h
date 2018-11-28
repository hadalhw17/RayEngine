#pragma once

#include "Vector.h"

#include <vector>
#include <memory>



class RObject;
class RKDTreeCPU;
class RColor;
class RSource;
class RTriangle;
class RRay;

using RVectorF = RVector<float>;

class RRayTracer
{
public:

	std::vector<std::shared_ptr<RSource>> lightSources;
	RRayTracer();


	RGBType *trace(RKDTreeCPU *tree, float xCamPos, float zCamPos, float xLookAt, float yLookAt, float zLookAt);


	bool traceShadow(RRay *, float &, RObject **, RKDTreeCPU *tree);
	RColor castRay(RRay *ray, int depth, RKDTreeCPU *node);
	void generateScene(std::vector<std::shared_ptr<RObject>> &);

	//Color options
	void tilePattern(RColor &, int square);
	void ambientLight(RColor &);
	void specularReflection(RRay, std::vector<std::shared_ptr<RObject>> &, RColor &, RVectorF);
	void fresnel(RVectorF &, RVectorF &, float &, float &);
	RVectorF refract(RVectorF &, RVectorF &, float &);

	RVectorF reflect(RVectorF &, RVectorF &);

	float clamp(float lo, float hi, float v);



private:
	int counterFor = 0;
};

