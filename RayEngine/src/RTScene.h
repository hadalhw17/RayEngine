#pragma once
#include "Scene.h"
#include "RayEngine/RayEngine.h"


class RAY_ENGINE_API RTScene
	: public RScene
{
public:
	RTScene();
	virtual ~RTScene();
	std::vector<RKDTreeCPU*> GetSceneTree();
private:
	void initialise_scene();
	void load_meshes_from_file(std::vector<char*>);
	void build_gpu_structs();
	void init_triangles();
	std::pair<size_t, size_t> merge_meshes();

	void build_tree();

protected:
	void clear_memory() override;
	virtual void build_scene() override;
private:
	float3* arrv;
	float3* arrf;
	float3* normals;
	float2* uvs;
	std::vector<float4> m_triangles;
	std::vector<float4> m_normals;
	std::vector<float2> m_uvs;

	size_t num_normals;
	size_t num_uvs;
	size_t numVerts;
	std::vector<class RKDTreeCPU*>Tree;
	std::vector<class RKDThreeGPU*> CUDATree;
	std::vector<class RKDTreeCPU*> tree;
	size_t numFaces;
};