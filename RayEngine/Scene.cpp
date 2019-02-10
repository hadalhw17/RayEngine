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
	load_meshes_from_file(complexObject, complexObject2);

	initialise_scene();
}


RScene::~RScene()
{
}
float RScene::moveCounter = 0.f;


void RScene::initialise_scene()
{
	std::cout << "Initialising scene" << std::endl;

	auto mesh_sizes = merge_meshes();

	numFaces = mesh_sizes.first;
	numVerts = mesh_sizes.second;

	build_tree();
}

RKDTreeCPU * RScene::GetSceneTree()
{
	return tree;
}

void RScene::rebuild_scene()
{
	moveCounter += 0.001f;
	auto mesh_sizes = merge_meshes();

	numFaces = mesh_sizes.first;
	numVerts = mesh_sizes.second;

	build_tree();
}


void RScene::load_meshes_from_file(RStaticMesh * _complexObject, RStaticMesh * _complexObject2)
{
	char *FileName = (char *)"Meshes/character.obj";
	
	WavefrontOBJ *reader = new WavefrontOBJ(FileName);
	complexObject = reader->loadObjFromFile(FileName);
	FileName = (char *)"Meshes/cat.obj";
	complexObject2 = reader->loadObjFromFile(FileName);
}

std::pair<size_t, size_t> RScene::merge_meshes()
{
	size_t numFaces;
	size_t numVerts;

	numFaces = complexObject->get_num_faces() + complexObject2->get_num_faces();
	numVerts = complexObject->get_num_verts() + complexObject2->get_num_verts();
	
	for (int i = 0; i < complexObject->get_num_verts(); i++)
	{
		complexObject->verts[i].y += moveCounter;
	}


	for (int i = 0; i < complexObject2->get_num_verts(); i++)
	{
		complexObject2->verts[i].z += moveCounter;
	}

	std::vector<float3> tmp_faces(complexObject->get_faces(), complexObject->get_faces() + complexObject->get_num_faces());
	std::vector<float3> tmp_verts(complexObject->get_verts(), complexObject->get_verts() + complexObject->get_num_verts());


	for (int i = 0; i < complexObject2->get_num_faces(); i++)
	{
		tmp_faces.push_back(complexObject2->get_faces()[i] + complexObject->get_num_verts());
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

	arrv = new float3[numVerts];
	arrf = new float3[numFaces];

	std::copy(tmp_verts.begin(), tmp_verts.end(), arrv);

	std::copy(tmp_faces.begin(), tmp_faces.end(), arrf);

	return std::pair<size_t, size_t>(numFaces, numVerts);
}

void RScene::build_tree()
{
	std::cout << "Constructing tree" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	tree = new RKDTreeCPU(arrv, arrf, normals, numVerts, numFaces);
	auto finish = std::chrono::high_resolution_clock::now();
	float elapsed_seconds = std::chrono::duration_cast<
		std::chrono::duration<float>>(finish - start).count();
	std::cout << "Tree is constructed in " << elapsed_seconds << " seconds." << std::endl;
}
