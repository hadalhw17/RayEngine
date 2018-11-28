#include "Scene.h"
#include "Vector.h"
#include "Color.h"
#include "Plane.h"
#include "Object.h"
#include "Triangle.h"
#include "Sphere.h"
#include "KDTree.h"
#include "RStaticMesh.h"

#include <iostream>


RScene::RScene()
{
}


RScene::~RScene()
{
}

void RScene::BuildScene()
{
	int numObjs;
	char *FileName = (char *)"Meshes/dragon_vrip.ply";
	RStaticMesh *complexObject = new RStaticMesh(FileName);
	std::cout << "Constructing tree" << std::endl;
	auto start = std::chrono::steady_clock::now();
	numObjs = sceneObjects.size();
	RTriangle *tmp_objs = new RTriangle[numObjs];
	
	for (int i = 0; i < numObjs; i++)
	{
		tmp_objs[i] = *sceneObjects[i];
	}
	tree = new RKDTreeCPU(complexObject->get_verts(), complexObject->get_faces(), complexObject->get_num_verts(), complexObject->get_num_faces());
	auto finish = std::chrono::steady_clock::now();
	float elapsed_seconds = std::chrono::duration_cast<
	std::chrono::duration<float>>(finish - start).count();
	std::cout << "Tree is constructed in " << elapsed_seconds << " seconds." << std::endl;

	delete[] tmp_objs;
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
