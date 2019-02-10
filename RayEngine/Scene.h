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
class RStaticMesh;

class RScene
{
public:
	static float moveCounter;
	HOST_DEVICE_FUNCTION RScene();

	HOST_DEVICE_FUNCTION ~RScene();

	HOST_DEVICE_FUNCTION 
	RKDTreeCPU *GetSceneTree();
	HOST_DEVICE_FUNCTION 
	inline std::vector<std::shared_ptr<RTriangle>> getObjects() { return sceneObjects; }
	
	void rebuild_scene();

	RKDTreeCPU *tree;
	std::vector<std::shared_ptr<RTriangle>> sceneObjects;
	float3 *normals;
	int num_normals;

private:
	void initialise_scene();
	void load_meshes_from_file(RStaticMesh *complexObject, RStaticMesh *complexObject2);
	std::pair<size_t, size_t> merge_meshes();

	void build_tree();

	RStaticMesh *complexObject;
	RStaticMesh *complexObject2;

	float3 *arrv;
	float3 *arrf;

	size_t numFaces;
	size_t numVerts;
};

