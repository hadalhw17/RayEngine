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
class RSceneObject;

class RScene
{
public:
	static float moveCounter;
	HOST_DEVICE_FUNCTION RScene();

	HOST_DEVICE_FUNCTION ~RScene();

	HOST_DEVICE_FUNCTION 
	std::vector<RKDTreeCPU *> GetSceneTree();
	HOST_DEVICE_FUNCTION 

	
	void rebuild_scene();

	std::vector<RKDTreeCPU *> tree;
	std::vector<RSceneObject *> sceneObjects;


	size_t numFaces;


	void Tick(float delta_time);

private:
	void initialise_scene();
	void load_meshes_from_file(std::vector<char *>);
	void clear_memory();
	void build_gpu_structs();
	std::pair<size_t, size_t> merge_meshes();

	void build_tree();

	RStaticMesh *complexObject;
	RStaticMesh *complexObject2;



	float3 *arrv;
	float3 *arrf;
	float3 *normals;
	
	size_t num_normals;
	size_t numVerts;
};

