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

	
	void rebuild_scene();

	RKDTreeCPU *tree;


	void Tick(float delta_time);

private:
	void initialise_scene();
	void load_meshes_from_file(std::vector<char *>);
	void clear_memory();
	std::pair<size_t, size_t> merge_meshes();

	void build_tree();

	RStaticMesh *complexObject;
	RStaticMesh *complexObject2;
	std::vector<RStaticMesh *> sceneObjects;

	float3 *arrv;
	float3 *arrf;
	float3 *normals;
	
	size_t num_normals;
	size_t numFaces;
	size_t numVerts;
};

