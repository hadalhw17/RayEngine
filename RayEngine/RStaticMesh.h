#pragma once
#include <memory>
#include <vector>

#include "rply.h"
#include "RayEngine.h"
#include "ObjectComponent.h"

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
	RStaticMesh(char *fileName) { LoadFromFile(fileName); }

	// method function
	HOST_DEVICE_FUNCTION
	static int vertex_cb(p_ply_argument argument);

	HOST_DEVICE_FUNCTION
	static int face_cb(p_ply_argument argument);

	HOST_DEVICE_FUNCTION
	void LoadFromFile(char *fileName);

	HOST_DEVICE_FUNCTION
	void LoadPolyMeshFromFile(const char *file);

	HOST_DEVICE_FUNCTION
	std::vector<float> Intersect(RRay *r);

	HOST_DEVICE_FUNCTION
	std::vector<std::shared_ptr<RTriangle>> GetTriangles() { return StaticMesh; }

	HOST_DEVICE_FUNCTION
	size_t GetVertexN() { return RStaticMesh::xCoordinates.size(); }

	float3 *get_verts() { return verts; }
	float3 *get_faces() { return faces; }
	float3 *get_norms() { return norms; }
	size_t get_num_verts() { return num_verts; }
	size_t get_num_faces() { return num_faces; }
	float3 *verts;
	float3 *faces;
	float3 *norms;
	size_t num_verts, num_faces, num_norms;
private:


};

