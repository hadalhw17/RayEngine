#pragma once
#include <memory>
#include <vector>

#include "rply.h"
#include "Vector.h"

class RTriangle;
class RRay;

class RStaticMesh
{
	std::vector<std::shared_ptr<RTriangle>> StaticMesh;



	static float count;
	static std::vector<float> xCoordinates;
	static std::vector<float> yCoordinates;
	static std::vector<float> zCoordinates;
	static std::vector<int> vertexIndecies;

public:
	HOST_DEVICE_FUNCTION RStaticMesh();

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
	int GetVertexN() { return RStaticMesh::xCoordinates.size(); }

	float3 *get_verts() { return verts; }
	float3 *get_faces() { return faces; }
	int get_num_verts() { return num_verts; }
	int get_num_faces() { return num_faces; }

private:
	float3 *verts;
	float3 *faces;
	int num_verts, num_faces;
};

