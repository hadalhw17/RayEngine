#pragma once

/*
	This file is part of Nori, a simple educational ray tracer
	Copyright (c) 2015 by Wenzel Jakob
	Nori is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License Version 3
	as published by the Free Software Foundation.
	Nori is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.
	You should have received a copy of the GNU General Public License
	along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "RStaticMesh.h"
#include <filesystem/resolver.h>
#include <unordered_map>
#include <fstream>
#include <functional>
#include <iostream>
#include "MeshAdjacencyTable.h"

namespace RayEngine
{
	/// Tokenize a string into a list by splitting at 'delim'
	std::vector<std::string> tokenize(const std::string& string, const std::string& delim, bool includeEmpty)
	{
		std::string::size_type lastPos = 0, pos = string.find_first_of(delim, lastPos);
		std::vector<std::string> tokens;

		while (lastPos != std::string::npos) {
			if (pos != lastPos || includeEmpty)
				tokens.push_back(string.substr(lastPos, pos - lastPos));
			lastPos = pos;
			if (lastPos != std::string::npos) {
				lastPos += 1;
				pos = string.find_first_of(delim, lastPos);
			}
		}

		return tokens;
	}

	/// Convert a string into an unsigned integer value
	unsigned int toUInt(const std::string& str) {
		char* end_ptr = nullptr;
		unsigned int result = (int)strtoul(str.c_str(), &end_ptr, 10);
		if (*end_ptr != '\0')
			std::cout << ("Could not parse integer value \"%s\"", str) << std::endl;
		return result;
	}

	/**
	 * \brief Loader for Wavefront OBJ triangle meshes
	 */
	class WavefrontOBJ {
	public:


		RStaticMesh* loadObjFromFile(const char* fileName)
		{
			RStaticMesh* mesh = new RStaticMesh();
			typedef std::unordered_map<OBJVertex, uint32_t, OBJVertexHash> VertexMap;

			std::ifstream is(fileName);


			std::cout << "Loading \"" << fileName << "\" .. " << std::endl;
			std::cout.flush();

			std::vector<float3>   positions;
			std::vector<float2>   texcoords;
			std::vector<float3>   normals;
			std::vector<uint32_t>   indices;
			std::vector<OBJVertex>  vertices;
			VertexMap vertexMap;

			std::string line_str;
			while (std::getline(is, line_str)) {
				std::istringstream line(line_str);

				std::string prefix;
				line >> prefix;

				if (prefix == "v") {
					float3 p;
					line >> p.x >> p.y >> p.z;
					positions.push_back(p);
				}
				else if (prefix == "vt") {
					float2 tc;
					line >> tc.x >> tc.y;
					texcoords.push_back(tc);
				}
				else if (prefix == "vn") {
					float3 n;
					line >> n.x >> n.y >> n.z;
					normals.push_back((n));
				}
				else if (prefix == "f") {
					std::string v1, v2, v3, v4;
					line >> v1 >> v2 >> v3 >> v4;
					OBJVertex verts[6];
					int nVertices = 3;

					verts[0] = OBJVertex(v1);
					verts[1] = OBJVertex(v2);
					verts[2] = OBJVertex(v3);

					if (!v4.empty()) {
						/* This is a quad, split into two triangles */
						verts[3] = OBJVertex(v4);
						verts[4] = verts[0];
						verts[5] = verts[2];
						nVertices = 6;
					}
					/* Convert to an indexed vertex list */
					for (int i = 0; i < nVertices; ++i) {
						const OBJVertex& v = verts[i];
						VertexMap::const_iterator it = vertexMap.find(v);
						if (it == vertexMap.end()) {
							vertexMap[v] = (uint32_t)vertices.size();
							indices.push_back((uint32_t)vertices.size());
							vertices.push_back(v);
						}
						else {
							indices.push_back(it->second);
						}
					}
				}
			}

			mesh->faces = new float3[indices.size() / 3];
			memcpy(mesh->faces, indices.data(), sizeof(uint32_t) * indices.size());
			for (uint32_t i = 0; i < indices.size() / 3; ++i)
			{
				mesh->faces[i].x = indices[i * 3];
				mesh->faces[i].y = indices[i * 3 + 1];
				mesh->faces[i].z = indices[i * 3 + 2];
			}


			mesh->verts = new float3[vertices.size()];
			for (uint32_t i = 0; i < vertices.size(); ++i)
				mesh->verts[i] = positions.at(vertices[i].p - 1);

			mesh->norms = new float3[vertices.size()];
			for (uint32_t i = 0; i < vertices.size(); ++i)
				mesh->norms[i] = normals.at(vertices[i].n - 1);

			if (texcoords.size() > 0)
			{
				mesh->uvs = new float2[vertices.size()];
				for (uint32_t i = 0; i < vertices.size(); ++i)
					mesh->uvs[i] = texcoords.at(vertices[i].uv - 1);
				mesh->num_uvs = vertices.size();
			}
			else
			{
				mesh->uvs = new float2[vertices.size()];
				for (uint32_t i = 0; i < vertices.size(); ++i)
					mesh->uvs[i] = make_float2(0.f, 0.f);
				mesh->num_uvs = vertices.size();
			}


			mesh->num_faces = indices.size() / 3;
			mesh->num_verts = vertices.size();
			mesh->num_norms = vertices.size();

			mesh->adjacency_table = new RMeshAdjacencyTable();
			mesh->adjacency_table->build_table(mesh);
			return mesh;
		}

	protected:

		/// Vertex indices used by the OBJ format
		struct OBJVertex {
			uint32_t p = (uint32_t)-1;
			uint32_t n = (uint32_t)-1;
			uint32_t uv = (uint32_t)-1;

			inline OBJVertex() { }

			inline OBJVertex(const std::string& string) {
				std::vector<std::string> tokens = tokenize(string, "/", true);

				if (tokens.size() < 1 || tokens.size() > 3)
					std::cout << ("Invalid vertex data: \"%s\"", string) << std::endl;

				p = toUInt(tokens[0]);

				if (tokens.size() >= 2 && !tokens[1].empty())
					uv = toUInt(tokens[1]);

				if (tokens.size() >= 3 && !tokens[2].empty())
					n = toUInt(tokens[2]);
			}

			inline bool operator==(const OBJVertex& v) const {
				return v.p == p && v.n == n && v.uv == uv;
			}
		};

		/// Hash function for OBJVertex
		struct OBJVertexHash : std::unary_function<OBJVertex, size_t> {
			std::size_t operator()(const OBJVertex& v) const {
				size_t hash = std::hash<uint32_t>()(v.p);
				hash = hash * 37 + std::hash<uint32_t>()(v.uv);
				hash = hash * 37 + std::hash<uint32_t>()(v.n);
				return hash;
			}
		};
	};
}