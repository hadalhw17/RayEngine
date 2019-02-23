#pragma once
#include "BoundingVolume.h"
#include <vector>
#include "RayEngine.h"

class RTriangle;
class RObject;
class RRay;
class RPlane;






class KDNodeCPU
{
public:
	KDNodeCPU();
	~KDNodeCPU();


	RBoundingVolume box;

	KDNodeCPU *LeftNode;
	KDNodeCPU *RightNode;
	int vid;
	bool isLeaf;
	int numVerts;
	int numFaces;
	int *objIndeces;

	Axis axis;
	float split_val;

	KDNodeCPU *ropes[6];

	void PrintDebugString();

	bool isPointToLeftOfSplittingPlane(const float3 &p) const;
	KDNodeCPU* getNeighboringNode(float3 p);

};


class RKDTreeCPU
{
public:
	KDNodeCPU *root;

	int numNodes;
	int numLevels;
	int numLeaves;
	int depth = 0;
	float3 lowerBound;
	float3 upperBound;
	int intersectionAmount;

	RBoundingVolume box;
	float3 *verts, *faces, *norms;
	size_t num_verts, num_faces, num_norms;
	bool isLeaf = false;

	HOST_DEVICE_FUNCTION RKDTreeCPU();

	HOST_DEVICE_FUNCTION 
	RKDTreeCPU(RKDTreeCPU* node, RBoundingVolume b, float3 lb, float3 ub, int depth);

	HOST_DEVICE_FUNCTION 
	RKDTreeCPU(float3 *_verts, float3 *_faces, float3 *_norms, int numVerts, int numFaces, int num_norms);

	~RKDTreeCPU();

	Axis getLongestBoundingBoxSide(float3 min, float3 max);

	float getMinTriValue(int tri_index, Axis axis);

	float getMaxTriValue(int tri_index, Axis axis);


	HOST_DEVICE_FUNCTION 
	KDNodeCPU *build(int dep, int objCount, int *tri_indecies, RBoundingVolume bbox);

	RBoundingVolume generateBox(float3 *verts, int num_verts);
	RBoundingVolume generateBox(int num_tris, int *tri_indices);

	void buildRopeStructure(KDNodeCPU *curr_node, KDNodeCPU *ropes[], bool is_single_ray_case);

	void buildRopeStructure();

	void optimizeRopes(KDNodeCPU *ropes[], RBoundingVolume bbox);

	HOST_DEVICE_FUNCTION
	bool intersect(KDNodeCPU *node, RRay r, float &t, float3 &normal);
	bool singleRayStacklessIntersect(KDNodeCPU *node, float3 ray_o, float3 ray_dir, float &t, float3 &normal);
	bool singleRayStacklessIntersect(RRay ray, float &t, float3 &normal);

	HOST_DEVICE_FUNCTION
	bool intersect(RRay r, float &t, float3 &normal);
};