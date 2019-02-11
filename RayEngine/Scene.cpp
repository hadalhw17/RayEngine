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
	load_meshes_from_file({ (char *)"Meshes/floor.obj" ,
							(char *)"Meshes/cat.obj"});

	initialise_scene();
}


RScene::~RScene()
{
	clear_memory();
}
float RScene::moveCounter = 0.f;


void RScene::Tick(float delta_time)
{
	rebuild_scene();
}

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
	clear_memory();
	moveCounter += 0.001f;
	auto mesh_sizes = merge_meshes();

	numFaces = mesh_sizes.first;
	numVerts = mesh_sizes.second;

	build_tree();
}


void RScene::load_meshes_from_file(std::vector<char *> files)
{
	for (auto file : files)
	{
		WavefrontOBJ *reader = new WavefrontOBJ((char *)file);
		sceneObjects.push_back(reader->loadObjFromFile((char *)file));

		delete reader;
	}
}

void RScene::clear_memory()
{
	delete[] arrf;
	delete[] arrv;
	delete[] normals;
}

std::pair<size_t, size_t> RScene::merge_meshes()
{
	size_t numFaces;
	size_t numVerts;

	numFaces = sceneObjects[0]->get_num_faces() + sceneObjects[1]->get_num_faces();
	numVerts = sceneObjects[0]->get_num_verts() + sceneObjects[1]->get_num_verts();
	
	for (size_t i = 0; i < sceneObjects[0]->get_num_verts(); i++)
	{
		sceneObjects[0]->verts[i].y -= 0.01f;
	}


	for (int i = 0; i < sceneObjects[1]->get_num_verts(); i++)
	{
		sceneObjects[1]->verts[i].z += 0.01;
	}

	std::vector<float3> tmp_faces = {};
	std::vector<float3> tmp_verts = {};
	std::vector<float3> tmp_norms = {};
	size_t stride = 0;
	for (int counter = 0; counter < sceneObjects.size(); counter++)
	{
		for (size_t i = 0; i < sceneObjects.at(counter)->get_num_faces(); i++)
		{
			tmp_faces.push_back(sceneObjects.at(counter)->get_faces()[i] + stride);
		}

		for (size_t i = 0; i < sceneObjects.at(counter)->get_num_verts(); i++)
		{
			tmp_verts.push_back(sceneObjects.at(counter)->get_verts()[i]);
		}

		for (size_t i = 0; i < sceneObjects.at(counter)->get_num_verts(); i++)
		{
			tmp_norms.push_back(sceneObjects.at(counter)->norms[i]);
		}

		stride += sceneObjects.at(counter)->get_num_verts();
	}

	numFaces = tmp_faces.size();
	numVerts = tmp_verts.size();
	num_normals = tmp_norms.size();

	arrv = new float3[numVerts];
	arrf = new float3[numFaces];
	normals = new float3[num_normals];

	std::copy(tmp_verts.begin(), tmp_verts.end(), arrv);

	std::copy(tmp_faces.begin(), tmp_faces.end(), arrf);
	
	std::copy(tmp_norms.begin(), tmp_norms.end(), normals);

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
