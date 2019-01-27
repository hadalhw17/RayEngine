#include "Scene.h"
#include "Plane.h"
#include "Object.h"
#include "Triangle.h"
#include "Sphere.h"
#include "KDTree.h"
#include "RStaticMesh.h"
#include "ObjFileReader.h"

#include <iostream>
#include <vector>


RScene::RScene()
{
}


RScene::~RScene()
{
}

void RScene::BuildScene()
{
	size_t numFaces;
	size_t numVerts;
	char *FileName = (char *)"Meshes/cow.obj";
	WavefrontOBJ *reader = new WavefrontOBJ(FileName);
	RStaticMesh *complexObject = reader->loadObjFromFile(FileName);
	FileName = (char *)"Meshes/cow.obj";
	RStaticMesh *complexObject2 = reader->loadObjFromFile(FileName);
	std::cout << "Constructing tree" << std::endl;
	auto start = std::chrono::steady_clock::now();
	numFaces = complexObject->get_num_faces() + complexObject2->get_num_faces();
	numVerts = complexObject->get_num_verts() + complexObject2->get_num_verts();
	std::vector<float3> tmp_faces(complexObject->get_faces(), complexObject->get_faces() + complexObject->get_num_faces());
	std::vector<float3> tmp_verts(complexObject->get_verts(), complexObject->get_verts() + complexObject->get_num_verts());
	
	for (int i = 0; i < complexObject2->get_num_faces(); i++)
	{
		tmp_faces.push_back(complexObject2->get_faces()[i]);
	}
	for (int i = 0; i < complexObject2->get_num_verts(); i++)
	{
		tmp_verts.push_back(complexObject2->get_verts()[i]);
	}
	numFaces = tmp_faces.size();
	numVerts = tmp_verts.size();
	normals = new float3[complexObject->num_norms];
	for (int i = 0; i < complexObject->num_norms; ++i)
	{
		normals[i] = complexObject->norms[i];
	}
	num_normals = complexObject->num_norms;
	float3 *arrv = new float3[numVerts];
	std::copy(tmp_verts.begin(), tmp_verts.end(), arrv);

	float3 *arrf = new float3[numFaces];
	std::copy(tmp_faces.begin(), tmp_faces.end(), arrf);

	tree = new RKDTreeCPU(arrv, arrf, normals, numVerts, numFaces);
	auto finish = std::chrono::steady_clock::now();
	float elapsed_seconds = std::chrono::duration_cast<
	std::chrono::duration<float>>(finish - start).count();
	std::cout << "Tree is constructed in " << elapsed_seconds << " seconds." << std::endl;
}

RKDTreeCPU * RScene::GetSceneTree()
{
	return tree;
}

void RScene::AddObjects(std::vector<std::shared_ptr<RTriangle>>* sceneObject, std::vector<std::shared_ptr<RTriangle>>* complexObject)
{
	for (size_t i = 0; i < complexObject->size(); i++)
	{
		sceneObject->push_back(std::shared_ptr<RTriangle>(complexObject->at(i)));
	}
}
