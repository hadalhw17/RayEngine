#pragma once
#include <vector>
#include <memory>

#include "RayEngine.h"

class RKDTree;
class RKDTreeCPU;
class RObject;
class RSphere;
class RTriangle;
class RTriMesh;
class RPlane;

class RScene
{
public:
	HOST_DEVICE_FUNCTION RScene();

	HOST_DEVICE_FUNCTION ~RScene();

	HOST_DEVICE_FUNCTION 
	void BuildScene();

	HOST_DEVICE_FUNCTION 
	RKDTreeCPU *GetSceneTree();
	HOST_DEVICE_FUNCTION 
	inline std::vector<std::shared_ptr<RTriangle>> getObjects() { return sceneObjects; }

	RKDTreeCPU *tree;
	std::vector<std::shared_ptr<RTriangle>> sceneObjects;
	float3 *normals;
	int num_normals;
private:
	HOST_DEVICE_FUNCTION 
	void AddObjects(std::vector<std::shared_ptr<RTriangle>> *sceneObject, std::vector<std::shared_ptr<RTriangle>> *complexObject);
};

