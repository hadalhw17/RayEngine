#include "repch.h"


#include "MeshAdjacencyTable.h"

#include "RStaticMesh.h"



	RMeshAdjacencyTable::RMeshAdjacencyTable()
	{
	}


	RMeshAdjacencyTable::~RMeshAdjacencyTable()
	{
	}

	void RMeshAdjacencyTable::build_table(RStaticMesh* mesh)
	{
		triangle_adjacency = std::vector<std::vector<int>>(mesh->num_verts);
		for (size_t i = 0; i < mesh->num_faces; ++i)
		{
			// Get triangle.
			uint3 tri = mesh->faces[i];
			int v0 = tri.x;
			int v1 = tri.y;
			int v2 = tri.z;

			edge e0 = edge(v0, v1);
			edge e1 = edge(v1, v2);
			edge e2 = edge(v2, v0);

			adjacency_list[e0].push_back(i);
			adjacency_list[e1].push_back(i);
			adjacency_list[e2].push_back(i);

			triangle_adjacency[v0].push_back(i);
			triangle_adjacency[v1].push_back(i);
			triangle_adjacency[v2].push_back(i);
		}
	}

	std::vector<int> RMeshAdjacencyTable::get_adjacent_edges(edge edge)
	{
		return adjacency_list[edge];
	}
