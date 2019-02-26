#include "Scene.h"
#include "Plane.h"
#include "Object.h"
#include "Triangle.h"
#include "Sphere.h"
#include "KDTree.h"
#include "RStaticMesh.h"
#include "Cow.h"
#include "Floor.h"
#include "Glass.h"

#include <iostream>
#include <vector>




RScene::RScene()
{
	
	ACow *cow = new ACow;
	sceneObjects.push_back(cow);
	AFloor *floor = new AFloor;
	sceneObjects.push_back(floor);

	AGlass *glass = new AGlass;
	sceneObjects.push_back(glass);

	initialise_scene();
}
extern "C"
void update_objects(std::vector<GPUSceneObject> objs);

RScene::~RScene()
{
	clear_memory();
}
float RScene::moveCounter = 0.f;


void RScene::Tick(float delta_time)
{
	rebuild_scene();


	std::vector<GPUSceneObject> tmp_objs;
	for (auto obj : sceneObjects)
	{
		obj->tick(delta_time);
		tmp_objs.push_back(obj->object_properties);
	}
	update_objects(tmp_objs);
}

void RScene::initialise_scene()
{
	//std::cout << "Initialising scene" << std::endl;

	//auto mesh_sizes = merge_meshes();
	build_gpu_structs();

	//numFaces = mesh_sizes.first;
	//numVerts = mesh_sizes.second;

	build_tree();
}

std::vector<RKDTreeCPU *> RScene::GetSceneTree()
{
	return tree;
}

void RScene::rebuild_scene()
{
	//delete[] objs;
	
	//build_gpu_structs();

	//build_tree();
}


void RScene::load_meshes_from_file(std::vector<char *> files)
{
	for (auto file : files)
	{
		sceneObjects.push_back(new RSceneObject((char *)file));

	}

}

void RScene::clear_memory()
{
	delete[] arrf;
	delete[] arrv;
	delete[] normals;
}

void RScene::build_gpu_structs()
{
	//objs = new GPUSceneObject[sceneObjects.size()];
	int first_index = 0;
	int i = 0;
	for (auto obj : sceneObjects)
	{
		obj->object_properties.num_prims = obj->root_component->get_num_faces();
		obj->object_properties.index_of_first_prim = first_index;
		first_index += obj->root_component->get_num_faces();

		++i;
	}

	//num_objs = sceneObjects.size();
}

std::pair<size_t, size_t> RScene::merge_meshes()
{
	size_t numFaces = 0;
	size_t numVerts = 0;
	
	//for (size_t i = 0; i < sceneObjects[0]->get_num_verts(); i++)
	//{
	//	sceneObjects[0]->verts[i].y -= 0.01f;
	//}


	//for (int i = 0; i < sceneObjects[1]->get_num_verts(); i++)
	//{
	//	sceneObjects[1]->verts[i].z += 0.01;
	//}

	std::vector<float3> tmp_faces = {};
	std::vector<float3> tmp_verts = {};
	std::vector<float3> tmp_norms = {};
	size_t stride = 0;
	for (int counter = 0; counter < sceneObjects.size(); counter++)
	{
		for (size_t i = 0; i < sceneObjects.at(counter)->root_component->get_num_faces(); i++)
		{
			tmp_faces.push_back(sceneObjects.at(counter)->root_component->get_faces()[i] + stride);
			sceneObjects.at(counter)->root_component->get_faces()[i] = tmp_faces[i];
		}

		for (size_t i = 0; i < sceneObjects.at(counter)->root_component->get_num_verts(); i++)
		{
			tmp_verts.push_back(sceneObjects.at(counter)->root_component->get_verts()[i]);
			sceneObjects.at(counter)->root_component->get_verts()[i] = tmp_verts[i];
		}

		for (size_t i = 0; i < sceneObjects.at(counter)->root_component->get_num_verts(); i++)
		{
			tmp_norms.push_back(sceneObjects.at(counter)->root_component->norms[i]);
			sceneObjects.at(counter)->root_component->norms[i] = tmp_norms[i];
		}

		stride += sceneObjects.at(counter)->root_component->get_num_verts();
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
	tree = {};
	for (auto obj : sceneObjects)
	{
		std::cout << "Constructing tree" << "\" .. " << std::endl;
		auto start = std::chrono::high_resolution_clock::now();
		RKDTreeCPU *new_tree = new RKDTreeCPU(obj->root_component->get_verts(), obj->root_component->get_faces(), 
			obj->root_component->get_norms(), obj->root_component->num_verts, obj->root_component->num_faces, obj->root_component->num_norms);

		obj->object_properties.num_nodes = new_tree->numNodes;
		obj->collision_box = GPUBoundingBox(&new_tree->root->box);
		tree.push_back(new_tree);
		auto finish = std::chrono::high_resolution_clock::now();
		float elapsed_seconds = std::chrono::duration_cast<
			std::chrono::duration<float>>(finish - start).count();
		std::cout << "Tree is constructed in " << elapsed_seconds << " seconds." << std::endl;

	}
	//std::cout << "Constructing tree" << std::endl;
	//auto start = std::chrono::high_resolution_clock::now();
	//tree = new RKDTreeCPU(arrv, arrf, normals, numVerts, numFaces);
	//auto finish = std::chrono::high_resolution_clock::now();
	//float elapsed_seconds = std::chrono::duration_cast<
	//	std::chrono::duration<float>>(finish - start).count();
	////std::cout << "Tree is constructed in " << elapsed_seconds << " seconds." << std::endl;
}
