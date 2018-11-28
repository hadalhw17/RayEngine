#pragma once
#include "Vector.h"
#include "BoundingVolume.h"
#include <vector>
#include "RayEngine.h"

class RTriangle;
class RObject;
class RRay;
class RPlane;

using RVectorF = RVector<float>;




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
	int numObjs;
	int numVerts;
	int numFaces;
	int *objIndeces;

	Axis axis;
	float split_val;

	KDNodeCPU *ropes[6];

	void PrintDebugString();
};


class RKDTreeCPU
{
public:
	KDNodeCPU *root;

	int numNodes;
	int numLevels;
	int numLeaves;
	int depth = 0;
	RVectorF lowerBound;
	RVectorF upperBound;
	int intersectionAmount;

	RBoundingVolume box;
	float3 *verts, *faces;
	int num_verts, num_faces;
	bool isLeaf = false;

	HOST_DEVICE_FUNCTION RKDTreeCPU();

	HOST_DEVICE_FUNCTION 
	RKDTreeCPU(RKDTreeCPU* node, RBoundingVolume b, RVectorF lb, RVectorF ub, int depth);

	HOST_DEVICE_FUNCTION 
	RKDTreeCPU(float3 *_verts, float3 *_faces, int numVerts, int numFaces);

	Axis getLongestBoundingBoxSide(float3 min, float3 max);

	float getMinTriValue(int tri_index, Axis axis);

	float getMaxTriValue(int tri_index, Axis axis);

	HOST_DEVICE_FUNCTION 
	KDNodeCPU *build(int dep, int objCount, int *tri_indecies, RBoundingVolume bbox);

	RBoundingVolume generateBox(float3 *verts, int num_verts);

	void buildRopeStructure(KDNodeCPU *curr_node, KDNodeCPU *ropes[], bool is_single_ray_case);

	void optimizeRopes(KDNodeCPU *ropes[], RBoundingVolume bbox);

	HOST_DEVICE_FUNCTION
	bool intersect(KDNodeCPU *node, RRay *r, float &t, RObject **hitObject);

	HOST_DEVICE_FUNCTION
	bool intersect(RRay *r, float &t, RObject **hitObject);
};