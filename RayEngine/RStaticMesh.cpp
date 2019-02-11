#include "RStaticMesh.h"
#include "Triangle.h"
#include "Ray.h"

#include "vector_types.h"

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "helper_math.h"


std::vector<float> RStaticMesh::xCoordinates = {};
std::vector<float> RStaticMesh::yCoordinates = {};
std::vector<float> RStaticMesh::zCoordinates = {};
std::vector<int> RStaticMesh::vertexIndecies = {};

std::vector<float3> vecArray = {};

RStaticMesh::RStaticMesh()
{
}

RStaticMesh::~RStaticMesh()
{
	delete[] verts;
	delete[] faces;
	delete[] norms;
}

int RStaticMesh::vertex_cb(p_ply_argument argument)
{
	long eol;
	ply_get_argument_user_data(argument, NULL, &eol);
	RStaticMesh::xCoordinates.push_back(ply_get_argument_value(argument));
	RStaticMesh::yCoordinates.push_back(ply_get_argument_value(argument));
	RStaticMesh::zCoordinates.push_back(ply_get_argument_value(argument));
	return 1;
}

int RStaticMesh::face_cb(p_ply_argument argument)
{
	long length, value_index;
	ply_get_argument_property(argument, NULL, &length, &value_index);
	switch (value_index) {
	case 0:

		//RRayTracer::vertexIndecies.push_back(ply_get_argument_value(argument));
	case 1:
		//printf("%g ", ply_get_argument_value(argument));
		RStaticMesh::vertexIndecies.push_back(ply_get_argument_value(argument));
		//qDebug() << ply_get_argument_value(argument);
		break;
	case 2:
		//printf("%g\n", ply_get_argument_value(argument));
		RStaticMesh::vertexIndecies.push_back(ply_get_argument_value(argument));
		break;
	default:
		break;
	}
	return 1;
}

void RStaticMesh::LoadFromFile(char * fileName)
{
	p_ply ply = ply_open(fileName, NULL, 0, NULL);

	if (!ply)
	{
		std::cout << fileName << " file not opened" << std::endl;
		exit(1);
	}
	if (!ply_read_header(ply))
	{
		std::cout << "Header corrupted!" << std::endl;
		exit(1);
	}
	else std::cout << fileName << " opened successfully" << std::endl;

	num_verts = RPLY_H::ply_set_read_cb(ply, "vertex", "x", vertex_cb, NULL, 0);
	RPLY_H::ply_set_read_cb(ply, "vertex", "y", vertex_cb, NULL, 1);
	RPLY_H::ply_set_read_cb(ply, "vertex", "z", vertex_cb, NULL, 0);
	num_faces = RPLY_H::ply_set_read_cb(ply, "face", "vertex_indices", face_cb, NULL, 0);

	if (!ply_read(ply))
	{
		 std::cout << "File is corrupted! We are having some troubles, Houston!" << std::endl;
	}

	int i = 0;
	for (i = 0; i < num_verts * 3; i += 3)
	{
		float3 verticies = make_float3(xCoordinates.at(i), yCoordinates.at(i + 1), zCoordinates.at(i + 2));
		vecArray.push_back(verticies);
	}

	verts = new float3[num_verts];
	for (i = 0; i < num_verts; ++i)
	{
		verts[i] = make_float3(xCoordinates.at(i*3), yCoordinates.at(i*3 + 1), zCoordinates.at(i*3 + 2));
	}

	faces = new float3[num_faces];
	for (i = 0; i < num_faces; ++i)
	{
		float x = vertexIndecies.at(i*3);
		float y = vertexIndecies.at(i*3 + 1);
		float z = vertexIndecies.at(i*3 + 2);
		faces [i] = make_float3(x, y, z);
	}

	RTriangle *triangle2;
	i = 1;
	for (i = 0; i < num_faces * 3; i += 3)
	{
		float x = vertexIndecies.at(i);
		float y = vertexIndecies.at(i + 1);
		float z = vertexIndecies.at(i + 2);
		triangle2 = new RTriangle(vecArray.at(x), vecArray.at(y), vecArray.at(z),
			make_float4(0.5, 0.5, 0.5, 3));
	
		StaticMesh.push_back(std::shared_ptr<RTriangle>(triangle2));
	}
}

void RStaticMesh::LoadPolyMeshFromFile(const char * file)
{
	std::ifstream ifs;
	try {
		ifs.open(file);
		if (ifs.fail()) throw;
		std::stringstream ss;
		ss << ifs.rdbuf();
		uint32_t numFaces;
		ss >> numFaces;
		std::unique_ptr<uint32_t[]> faceIndex(new uint32_t[numFaces]);
		uint32_t vertsIndexArraySize = 0;
		// reading face index array
		for (uint32_t i = 0; i < numFaces; ++i) {
			ss >> faceIndex[i];
			vertsIndexArraySize += faceIndex[i];
		}
		std::unique_ptr<uint32_t[]> vertsIndex(new uint32_t[vertsIndexArraySize]);
		uint32_t vertsArraySize = 0;
		// reading vertex index array
		for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
			ss >> vertsIndex[i];
			if (vertsIndex[i] > vertsArraySize) vertsArraySize = vertsIndex[i];
		}
		vertsArraySize += 1;
		// reading vertices
		std::unique_ptr<float3[]> vert(new float3[vertsArraySize]);
		for (uint32_t i = 0; i < vertsArraySize; ++i) {
			ss >> vert[i].x >> vert[i].y >> vert[i].z;
		}
		// reading normals
		std::unique_ptr<float3[]> normals(new float3[vertsIndexArraySize]);
		for (uint32_t i = 0; i < vertsIndexArraySize; ++i) {
			ss >> normals[i].x >> normals[i].y >> normals[i].z;
		}

		for (uint32_t i = 0, k = 0; i < numFaces; ++i) {

		}

		std::unique_ptr<uint32_t[]> trisIndex = std::unique_ptr<uint32_t[]>(new uint32_t[numFaces * 3]);
		float3 triVerts[3];
		uint32_t l = 0;
		for (uint32_t i = 0, k = 0; i < numFaces; ++i) { // for each  face
			int a = 0;
			for (uint32_t j = 0; j < faceIndex[i] - 2; ++j) { // for each triangle in the face
				trisIndex[l] = vertsIndex[k];
				trisIndex[l + 1] = vertsIndex[k + j + 1];
				trisIndex[l + 2] = vertsIndex[k + j + 2];
				float3 A = vert[trisIndex[l]];
				float3 B = vert[trisIndex[l + 1]];
				float3 C = vert[trisIndex[l + 2]];
				//qDebug() << trisIndex[l];
				triVerts[a] = make_float3(trisIndex[l], trisIndex[l + 1], trisIndex[l + 2]);
				l += 3;
				a++;
				RTriangle *tr = new RTriangle(A, B, C, make_float4(1, 0, 0, 0));
				StaticMesh.push_back(std::shared_ptr<RTriangle>(tr));
			}
			k += faceIndex[i];

		}
		//return new TriangleMesh(numFaces, faceIndex, vertsIndex, verts, normals, st);
	}

	catch (...) {
		ifs.close();
	}
	ifs.close();

	//return nullptr;
}

std::vector<float> RStaticMesh::Intersect(RRay * r)
{
	std::vector<float> intersections;
	float t, u, v;
	for (auto&& s : GetTriangles())
	{
		if (s->FindIntersection(r, t, u, v))
		{
			intersections.push_back(t);
		}
	}
	return intersections;
}
