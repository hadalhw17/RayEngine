#pragma once

#include <vector>
#include <unordered_map>

struct edge_hash
{
	template <class T1, class T2>
	std::size_t operator() (const std::pair<T1, T2>& pair) const
	{
		return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
	}
};

// Data structure for adjacency edges.
// The whole idea is to store an edge as a pair of integers along with a vector of integers that represent end vertecies of adjacent edges. 
class RMeshAdjacencyTable
{
	using edge = std::pair<int, int>;

public:
	RMeshAdjacencyTable();
	~RMeshAdjacencyTable();

	std::unordered_map<edge, std::vector<int>, edge_hash> adjacency_list;
	std::vector<std::vector<int>> triangle_adjacency;

	void build_table(class RStaticMesh *mesh);

	std::vector<int> get_adjacent_edges(edge edge);

};

