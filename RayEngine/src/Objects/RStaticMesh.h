#pragma once

#include "rply.h"

#include "Objects/ObjectComponent.h"
#include "RayEngine/RayEngine.h"
#include <memory>
#include <vector>

class RTriangle;
class RRay;
class RStaticMesh : public RObjectComponent
{
	std::vector<std::shared_ptr<RTriangle>> StaticMesh;



	static float count;
	static std::vector<float> xCoordinates;
	static std::vector<float> yCoordinates;
	static std::vector<float> zCoordinates;
	static std::vector<int> vertexIndecies;

public:
	HOST_DEVICE_FUNCTION RStaticMesh();
	~RStaticMesh();

	HOST_DEVICE_FUNCTION
		RStaticMesh(char* fileName) { LoadFromFile(fileName); }

	// method function
	HOST_DEVICE_FUNCTION
		static int vertex_cb(p_ply_argument argument);

	HOST_DEVICE_FUNCTION
		static int face_cb(p_ply_argument argument);

	HOST_DEVICE_FUNCTION
		void LoadFromFile(char* fileName);

	HOST_DEVICE_FUNCTION
		void LoadPolyMeshFromFile(const char* file);

	HOST_DEVICE_FUNCTION
		std::vector<float> Intersect(RRay* r);

	HOST_DEVICE_FUNCTION
		std::vector<std::shared_ptr<RTriangle>> GetTriangles() { return StaticMesh; }

	HOST_DEVICE_FUNCTION
		size_t GetVertexN() { return RStaticMesh::xCoordinates.size(); }

	void generate_face_normals();

	float3* get_verts() { return verts; }
	uint3* get_faces() { return faces; }
	float3* get_norms() { return norms; }
	float2* get_uvs() { return uvs; }
	size_t get_num_verts() { return num_verts; }
	size_t get_num_faces() { return num_faces; }
	size_t get_num_norms() { return num_norms; }
	size_t get_num_uvs() { return num_uvs; }
	float3* verts;
	uint3* faces;
	float3* norms;
	float2* uvs;

	std::vector<float3> face_normals;

	class RMeshAdjacencyTable* adjacency_table;

	size_t num_verts, num_faces, num_norms, num_uvs;
private:


};

