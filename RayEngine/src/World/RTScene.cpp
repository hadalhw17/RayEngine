#include "repch.h"


#include "RTScene.h"

#include "Primitives/KDThreeGPU.h"
#include "Objects/MeshObject.h"

extern "C" void copy_memory(std::vector<RKDThreeGPU*> tree, RCamera _sceneCam, std::vector<float4> h_triangles,
	std::vector<float4> h_normals, std::vector<float2> h_uvs, std::vector<GPUSceneObject> objs, std::vector<float4> textures, class Grid* grid);

void RTScene::initialise_scene()
{
	init_triangles();

	std::vector<GPUSceneObject> tmp_objs;
	for (auto objs : scene_objects)
	{
		tmp_objs.push_back(objs->object_properties);
	}
	//copy_memory(CUDATree, scene_camera, m_triangles, m_normals, m_uvs, tmp_objs, textures, distance_field);
	//std::cout << "Initialising scene" << std::endl;

	//auto mesh_sizes = merge_meshes();
	build_gpu_structs();

	//numFaces = mesh_sizes.first;
	//numVerts = mesh_sizes.second;

	build_tree();
}

void RTScene::clear_memory()
{
	delete[] arrf;
	delete[] arrv;
	delete[] normals;
}

void RTScene::build_scene()
{
	setup_camera();
	update_camera();
	RE_LOG("Building scene" << "\" .. ");
	int i = 0;
	for (auto t : GetSceneTree())
	{
		RKDThreeGPU* gpu_tree = new RKDThreeGPU(t);
		scene_objects.at(i)->object_properties.index_list_size = gpu_tree->GetIndexList().size();

		CUDATree.push_back(gpu_tree);
		++i;
	}
	for (i = 0; i < scene_objects.size(); i++)
	{
		for (int k = 0; k < i; k++)
		{
			scene_objects.at(i)->object_properties.offset += scene_objects.at(k)->object_properties.index_list_size - scene_objects.at(k)->object_properties.num_nodes;
		}
	}
	RE_LOG("Done building scene");
}

RTScene::RTScene()
	:RScene()
{
	//AFloor *floor = new AFloor;
	//sceneObjects.push_back(floor);

	//ACow *cow = new ACow;
	//sceneObjects.push_back(cow);


	//AGlass *glass = new AGlass;
	//sceneObjects.push_back(glass);

	initialise_scene();
}

RTScene::~RTScene()
{
}

std::vector<RKDTreeCPU*> RTScene::GetSceneTree()
{
	return tree;
}

void RTScene::load_meshes_from_file(std::vector<char*> files)
{
	for (auto file : files)
	{
		scene_objects.push_back(new RMeshObject((char*)file));
	}

}

void RTScene::build_gpu_structs()
{
	size_t first_index = 0;
	int i = 0;
	for (auto obj : scene_objects)
	{
		if (RMeshObject * mesh_obj = dynamic_cast<RMeshObject*>(obj))
		{
			obj->object_properties.num_prims = mesh_obj->root_component->get_num_faces();
			obj->object_properties.index_of_first_prim = first_index;
			first_index += mesh_obj->root_component->get_num_faces();
		}


		++i;
	}
}

std::pair<size_t, size_t> RTScene::merge_meshes()
{
	size_t numFaces = 0;
	size_t numVerts = 0;

	std::vector<uint3> tmp_faces = {};
	std::vector<float3> tmp_verts = {};
	std::vector<float3> tmp_norms = {};
	std::vector<float2> tmp_uvs = {};
	size_t stride = 0;
	for (int counter = 0; counter < scene_objects.size(); counter++)
	{
		if (RMeshObject * mesh_obj = dynamic_cast<RMeshObject*>(scene_objects.at(counter)))
		{
			for (size_t i = 0; i < mesh_obj->root_component->get_num_faces(); i++)
			{
				tmp_faces.push_back(mesh_obj->root_component->get_faces()[i] + stride);
				mesh_obj->root_component->get_faces()[i] = tmp_faces[i];
			}

			for (size_t i = 0; i < mesh_obj->root_component->get_num_verts(); i++)
			{
				tmp_verts.push_back(mesh_obj->root_component->get_verts()[i]);
				mesh_obj->root_component->get_verts()[i] = tmp_verts[i];
			}

			for (size_t i = 0; i < mesh_obj->root_component->get_num_norms(); i++)
			{
				tmp_norms.push_back(mesh_obj->root_component->get_norms()[i]);
				mesh_obj->root_component->get_norms()[i] = tmp_norms[i];
			}

			for (size_t i = 0; i < mesh_obj->root_component->get_num_uvs(); i++)
			{
				tmp_uvs.push_back(mesh_obj->root_component->get_uvs()[i]);
				mesh_obj->root_component->get_uvs()[i] = tmp_uvs[i];
			}

			stride += mesh_obj->root_component->get_num_verts();
		}
	}

	numFaces = tmp_faces.size();
	numVerts = tmp_verts.size();
	num_normals = tmp_norms.size();
	num_uvs = tmp_uvs.size();

	arrv = new float3[numVerts];
	arrf = new uint3[numFaces];
	normals = new float3[num_normals];
	uvs = new float2[num_uvs];

	std::copy(tmp_verts.begin(), tmp_verts.end(), arrv);

	std::copy(tmp_faces.begin(), tmp_faces.end(), arrf);

	std::copy(tmp_norms.begin(), tmp_norms.end(), normals);

	std::copy(tmp_uvs.begin(), tmp_uvs.end(), uvs);

	return std::pair<size_t, size_t>(numFaces, numVerts);
}

void RTScene::build_tree()
{
	tree = {};
	for (auto obj : scene_objects)
	{
		if (RMeshObject * mesh_obj = dynamic_cast<RMeshObject*>(obj))
		{
			std::cout << "Constructing tree" << "\" .. " << std::endl;
			auto start = std::chrono::high_resolution_clock::now();
			RKDTreeCPU* new_tree = new RKDTreeCPU(mesh_obj->root_component->get_verts(), mesh_obj->root_component->get_faces(),
				mesh_obj->root_component->get_norms(), mesh_obj->root_component->get_uvs(), mesh_obj->root_component->num_verts,
				mesh_obj->root_component->num_faces, mesh_obj->root_component->num_norms, mesh_obj->root_component->num_uvs);

			mesh_obj->object_properties.num_nodes = new_tree->numNodes;
			mesh_obj->collision_box = GPUBoundingBox(&new_tree->root->box);
			tree.push_back(new_tree);
			auto finish = std::chrono::high_resolution_clock::now();
			float elapsed_seconds = std::chrono::duration_cast<
				std::chrono::duration<float>>(finish - start).count();
			std::cout << "Tree is constructed in " << elapsed_seconds << " seconds." << std::endl;
		}
	}
	//std::cout << "Constructing tree" << std::endl;
	//auto start = std::chrono::high_resolution_clock::now();
	//tree = new RKDTreeCPU(arrv, arrf, normals, numVerts, numFaces);
	//auto finish = std::chrono::high_resolution_clock::now();
	//float elapsed_seconds = std::chrono::duration_cast<
	//	std::chrono::duration<float>>(finish - start).count();
	////std::cout << "Tree is constructed in " << elapsed_seconds << " seconds." << std::endl;
}

void RTScene::init_triangles()
{
	RE_LOG("Initialising triangle buffer" << "\" .. ");
	m_triangles = {};
	m_normals = {};
	m_uvs = {};
	int count = 0;
	size_t offset = 0;
	size_t root_offset = 0;

	for (auto t : CUDATree)
	{

		float3* verts = t->get_verts();
		uint3* faces = t->get_faces();
		float3* norms = t->get_normals();
		float2* uv = t->uvs;

		for (size_t i = 0; i < t->get_num_faces(); ++i)
		{
			// make a local copy of the triangle vertices
			uint3 tri = faces[i];
			float3 v0 = verts[(size_t)tri.x];
			float3 v1 = verts[(size_t)tri.y];
			float3 v2 = verts[(size_t)tri.z];

			// store triangle data as float4
			// store two edges per triangle instead of vertices, to save some calculations in the
			// ray triangle intersection test
			m_triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
			m_triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));
			m_triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));

			float3 n0 = norms[(size_t)tri.x];
			float3 n1 = norms[(size_t)tri.y];
			float3 n2 = norms[(size_t)tri.z];
			m_normals.push_back(make_float4(n0.x, n0.y, n0.z, 0));
			m_normals.push_back(make_float4(n1.x, n1.y, n1.z, 0));
			m_normals.push_back(make_float4(n2.x, n2.y, n2.z, 0));

			if (t->num_uvs > 0)
			{
				float2 uv0 = uv[(size_t)tri.x];
				float2 uv1 = uv[(size_t)tri.y];
				float2 uv2 = uv[(size_t)tri.z];
				m_uvs.push_back(make_float2(uv0.x, uv0.y));
				m_uvs.push_back(make_float2(uv0.x, uv0.y));
				m_uvs.push_back(make_float2(uv0.x, uv0.y));
			}
			//t->GetIndexList()[i] += offset;
		}
		RE_LOG("Old root index: ");

		for (int k = 0; k < t->GetNumNodes(); ++k)
		{
			t->GetNodes()[k].left_index += root_offset;
			t->GetNodes()[k].right_index += root_offset;
			t->GetNodes()[k].index_of_first_object += root_offset;

			for (int i = 0; i < 6; ++i)
			{
				if (t->GetNodes()[k].neighbor_node_indices[i] != -1)
					t->GetNodes()[k].neighbor_node_indices[i] += root_offset;
			}
		}
		t->root_index += root_offset;

		offset += t->obj_index_list.size();

		root_offset += t->GetNumNodes();
		RE_LOG("New root index: ");

		++count;
		delete[] verts, faces, norms;
	}
	RE_LOG("Done initialising triangle buffer");
}