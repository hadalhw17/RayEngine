#include "KDTree.h"
#include "Plane.h"
#include "Triangle.h"

#include "cuda_runtime.h"
#include "cutil_math.h"


#include <memory>
#include <iostream>

const int MAX_DEPTH = 48;
const  int OBJECTS_IN_LEAF = 2000;
const int  MAX_SPLITS_OF_VOXEL = 5;
const int SPLIT_COST = 5;

KDNodeCPU::KDNodeCPU()
{
	LeftNode = nullptr;
	RightNode = nullptr;
	isLeaf = false;
	for (int i = 0; i < 6; ++i) {
		ropes[i] = nullptr;
	}
	vid = -99;
}

KDNodeCPU::~KDNodeCPU()
{
	if (numFaces > 0) {
		delete[] objIndeces;
	}

	if (LeftNode) {
		delete LeftNode;
	}
	if (RightNode) {
		delete RightNode;
	}
}

void KDNodeCPU::PrintDebugString()
{
	std::cout << "bounding box min: ( " << box.bounds[0].x << ", " << box.bounds[0].y << ", " << box.bounds[0].z << " )" << std::endl;
	std::cout << "bounding box max: ( " << box.bounds[1].x << ", " << box.bounds[1].y << ", " << box.bounds[1].z << " )" << std::endl;
	std::cout << "num_tris: " << numObjs << std::endl;

	// Print split plane axis.
	if (axis == X_Axis) {
		std::cout << "split plane axis: X_AXIS" << std::endl;
	}
	else if (axis == Y_Axis) {
		std::cout << "split plane axis: Y_AXIS" << std::endl;
	}
	else if (axis == Z_Axis) {
		std::cout << "split plane axis: Z_AXIS" << std::endl;
	}
	else {
		std::cout << "split plane axis: invalid" << std::endl;
	}

	// Print whether or not node is a leaf node.
	if (isLeaf) {
		std::cout << "is leaf node: YES" << std::endl;
	}
	else {
		std::cout << "is leaf node: NO" << std::endl;
	}

	// Print pointers to children.
	if (LeftNode) {
		std::cout << "left child: " << LeftNode << std::endl;
	}
	else {
		std::cout << "left child: NULL" << std::endl;
	}
	if (RightNode) {
		std::cout << "right child: " << RightNode << std::endl;
	}
	else {
		std::cout << "right child: NULL" << std::endl;
	}

	// Print empty line.
	std::cout << std::endl;
}


RKDTreeCPU::RKDTreeCPU()
{
	numNodes = 0;
	numLevels = 0;
	numLeaves = 0;
	depth = 0;
	lowerBound = RVectorF(kInfinity);
	upperBound = RVectorF(-kInfinity);
}

RKDTreeCPU::RKDTreeCPU(RKDTreeCPU * node, RBoundingVolume b, RVectorF lb, RVectorF ub, int d)
{
	this->box = b;
	this->lowerBound = lb;
	this->upperBound = ub;
	numNodes = 0;
	numLevels = 0;
	numLeaves = 0;
	depth = d;
}



RKDTreeCPU::RKDTreeCPU(float3 *_verts, float3 *_faces, int num_verts, int num_faces)
{
	this->verts = _verts;
	this->faces = _faces;
	this->num_verts = num_verts;
	this->num_faces = num_faces;

	this->numNodes = 0;
	this->numLevels = 0;
	this->numLeaves = 0;
	this->depth = 1;

	// Create list of triangle indices for first level of kd-tree.
	int *tri_indices = new int[num_faces];
	for (int i = 0; i < num_faces; ++i) {
		tri_indices[i] = i;
	}
	RBoundingVolume bbox;
	bbox = generateBox(_verts, num_verts);

	this->root = build(depth, num_faces, tri_indices, bbox);

	// build rope structure
	KDNodeCPU* ropes[6] = { NULL };
	buildRopeStructure( root, ropes, true );

}

Axis RKDTreeCPU::getLongestBoundingBoxSide(float3 min, float3 max)
{
	// max > min is guaranteed.
	float xlength = max.x - min.x;
	float ylength = max.y - min.y;
	float zlength = max.z - min.z;
	return (xlength > ylength && xlength > zlength) ? X_Axis : (ylength > zlength ? Y_Axis : Z_Axis);
}

float RKDTreeCPU::getMinTriValue(int tri_index, Axis axis)
{
	float3 tri = faces[tri_index];
	float3 v0 = verts[(int)tri.x];
	float3 v1 = verts[(int)tri.y];
	float3 v2 = verts[(int)tri.z];

	if (axis == X_Axis) {
		return (v0.x < v1.x && v0.x < v2.x) ? v0.x : (v1.x < v2.x ? v1.x : v2.x);
	}
	else if (axis == Y_Axis) {
		return (v0.y < v1.y && v0.y < v2.y) ? v0.y : (v1.y < v2.y ? v1.y : v2.y);
	}
	else {
		return (v0.z < v1.z && v0.z < v2.z) ? v0.z : (v1.z < v2.z ? v1.z : v2.z);
	}
}

float RKDTreeCPU::getMaxTriValue(int tri_index, Axis axis)
{
	float3 tri = faces[tri_index];
	float3 v0 = verts[(int)tri.x];
	float3 v1 = verts[(int)tri.y];
	float3 v2 = verts[(int)tri.z];

	if (axis == X_Axis) {
		return (v0.x > v1.x && v0.x > v2.x) ? v0.x : (v1.x > v2.x ? v1.x : v2.x);
	}
	else if (axis == Y_Axis) {
		return (v0.y > v1.y && v0.y > v2.y) ? v0.y : (v1.y > v2.y ? v1.y : v2.y);
	}
	else {
		return (v0.z > v1.z && v0.z > v2.z) ? v0.z : (v1.z > v2.z ? v1.z : v2.z);
	}
}


KDNodeCPU * RKDTreeCPU::build(int dep, int objCount, int *tri_indecies, RBoundingVolume bbox)
{
	KDNodeCPU *node = new KDNodeCPU();
	node->box = bbox;
	node->numObjs = objCount;

	node->objIndeces = tri_indecies;


	// Reached minimum abount of primitives in the node => make leaf and return it.
	if (objCount <= OBJECTS_IN_LEAF || dep > MAX_DEPTH)
	{

		node->isLeaf = true;
		node->vid = numNodes;

		if (depth > numLevels)
		{
			this->numLevels = depth;
		}
		
		++numNodes;
		++numLevels;

		return node;
	}

	// Allocate and initialize memory for temporary buffers to hold triangle indices for left and right subtrees.
	int *temp_left_tri_indices = new int[objCount];
	int *temp_right_tri_indices = new int[objCount];

	int leftObjCount = 0, rightObjCount = 0;

	// Bounding boxes for left and right children.
	RBoundingVolume leftBBox = bbox;
	RBoundingVolume rightBBox = bbox;


	// Average midpoint of the bounding box.
	float midp;
	Axis split_axis = getLongestBoundingBoxSide(make_float3(bbox.bounds[0].x, bbox.bounds[0].y, bbox.bounds[0].z),
		make_float3(bbox.bounds[1].x, bbox.bounds[1].y, bbox.bounds[1].z));

	// Update boxes boundaries according to split axis and mid point.
	switch (split_axis) {
	case X_Axis:		//X_Axis
		midp = bbox.bounds[0].x + ((bbox.bounds[1].x - (bbox.bounds[0].x) / (2.f)));
		node->axis = X_Axis;
		leftBBox.bounds[1].x = midp;
		rightBBox.bounds[0].x = midp;
		break;
	case Y_Axis:		//Y_Axis
		midp = bbox.bounds[0].y + ((bbox.bounds[1].y - (bbox.bounds[0].y) / (2.f)));
		node->axis = Y_Axis;
		leftBBox.bounds[1].y = midp;
		rightBBox.bounds[0].y = midp;
		break;
	case Z_Axis:		//Z_Axis
		midp = bbox.bounds[0].z + ((bbox.bounds[1].z - (bbox.bounds[0].z) / (2.f)));
		node->axis = Z_Axis;
		leftBBox.bounds[1].z = midp;
		rightBBox.bounds[0].z = midp;
		break;
	}

	node->split_val = midp;


	// Decide whether primitive goes to the right or to the left node.
	float minVal, maxVal;
	for (size_t i = 0; i < objCount; i++)
	{
		// Split objects based on their midpoints size of average in axises.
		switch (split_axis) {
		case X_Axis:
			minVal = getMinTriValue(tri_indecies[i], X_Axis);
			maxVal = getMaxTriValue(tri_indecies[i], X_Axis);
			break;
		case Y_Axis:
			minVal = getMinTriValue(tri_indecies[i], Y_Axis);
			maxVal = getMaxTriValue(tri_indecies[i], Y_Axis);

			break;
		case Z_Axis:
			minVal = getMinTriValue(tri_indecies[i], Z_Axis);
			maxVal = getMaxTriValue(tri_indecies[i], Z_Axis);
			break;
		}

		// Goes to the left node, index is added to left list, -1 is added to right list
		if (midp > minVal)
		{
			temp_left_tri_indices[i] = tri_indecies[i];
			++leftObjCount;
		}
		else
		{
			temp_left_tri_indices[i] = -1;
		}

		// Goes to the right node, index is added to right list, -1 is added to left list
		if (midp <= maxVal)
		{
			temp_right_tri_indices[i] = tri_indecies[i];
			++rightObjCount;
		}
		else
		{
			temp_right_tri_indices[i] = -1;
		}
	}

	// List of object indecies for left and right node
	int *left_tri_indices = new int[leftObjCount];
	int *right_tri_indices = new int[rightObjCount];

	// Filter indecies between right and left by removing the one that have -1.
	int left_index = 0, right_index = 0;
	for (int i = 0; i < objCount; ++i) {
		if (temp_left_tri_indices[i] != -1) {
			left_tri_indices[left_index] = temp_left_tri_indices[i];
			++left_index;
		}
		if (temp_right_tri_indices[i] != -1) {
			right_tri_indices[right_index] = temp_right_tri_indices[i];
			++right_index;
		}
	}

	// Free temporary triangle indices buffers.
	delete[] temp_left_tri_indices;
	delete[] temp_right_tri_indices;

	node->LeftNode = build(dep + 1, leftObjCount, left_tri_indices, leftBBox);
	node->RightNode = build(dep + 1, rightObjCount, right_tri_indices, rightBBox);

	// Assign final parametres.
	node->vid = numNodes;
	++numNodes;


	delete[] left_tri_indices;
	delete[] right_tri_indices;

	return node;
}

void RKDTreeCPU::buildRopeStructure(KDNodeCPU *curr_node, KDNodeCPU *ropes[], bool is_single_ray_case)
{
	// Base case.
	if (curr_node->isLeaf) {
		//std::cout<<curr_node->id<<": "<<std::endl;
		for (int i = 0; i < 6; ++i) {
			curr_node->ropes[i] = ropes[i];
		}
	}
	else {
		// Only optimize ropes on single ray case.
		// It is not optimal to optimize on packet traversal case.
		if (is_single_ray_case) {
			optimizeRopes(ropes, curr_node->box);
		}

		BoxFace SL, SR;
		if (curr_node->axis == X_Axis) {
			SL = LEFT;
			SR = RIGHT;
		}
		else if (curr_node->axis == Y_Axis) {
			SL = BOTTOM;
			SR = TOP;
		}
		// Split plane is Z_AXIS.
		else {
			SL = BACK;
			SR = FRONT;
		}

		KDNodeCPU* RS_left[6];
		KDNodeCPU* RS_right[6];
		for (int i = 0; i < 6; ++i) {
			RS_left[i] = ropes[i];
			RS_right[i] = ropes[i];
		}

		// Recurse.
		RS_left[SR] = curr_node->RightNode;
		buildRopeStructure(curr_node->LeftNode, RS_left, is_single_ray_case);

		// Recurse.
		RS_right[SL] = curr_node->LeftNode;
		buildRopeStructure(curr_node->RightNode, RS_right, is_single_ray_case);
	}
}

void RKDTreeCPU::optimizeRopes(KDNodeCPU *ropes[], RBoundingVolume bbox)
{
	// Loop through ropes of all faces of node bounding box.
	for (int i = 0; i < 6; ++i) {
		KDNodeCPU *rope_node = ropes[i];

		if (rope_node == NULL) {
			continue;
		}

		// Process until leaf node is reached.
		// The optimization - We try to push the ropes down into the tree as far as possible
		// instead of just having the ropes point to the roots of neighboring subtrees.
		while (!rope_node->isLeaf) {

			// Case I.

			if (i == LEFT || i == RIGHT) {

				// Case I-A.

				// Handle parallel split plane case.
				if (rope_node->axis == X_Axis) {
					rope_node = (i == LEFT) ? rope_node->RightNode : rope_node->LeftNode;
				}

				// Case I-B.

				else if (rope_node->axis == Y_Axis) {
					if (rope_node->split_val < (bbox.bounds[0].y - 0.00001)) {
						rope_node = rope_node->RightNode;
					}
					else if (rope_node->split_val > (bbox.bounds[1].y + 0.00001)) {
						rope_node = rope_node->LeftNode;
					}
					else {
						break;
					}
				}

				// Case I-C.

				// Split plane is Z_AXIS.
				else {
					if (rope_node->split_val < (bbox.bounds[0].z - 0.00001)) {
						rope_node = rope_node->RightNode;
					}
					else if (rope_node->split_val > (bbox.bounds[1].z + 0.00001)) {
						rope_node = rope_node->LeftNode;
					}
					else {
						break;
					}
				}
			}

			// Case II.

			else if (i == FRONT || i == BACK) {

				// Case II-A.

				// Handle parallel split plane case.
				if (rope_node->split_val == Z_Axis) {
					rope_node = (i == BACK) ? rope_node->RightNode : rope_node->LeftNode;
				}

				// Case II-B.

				else if (rope_node->split_val == X_Axis) {
					if (rope_node->split_val < (bbox.bounds[0].x - 0.00001)) {
						rope_node = rope_node->RightNode;
					}
					else if (rope_node->split_val > (bbox.bounds[1].x + 0.00001)) {
						rope_node = rope_node->LeftNode;
					}
					else {
						break;
					}
				}

				// Case II-C.

				// Split plane is Y_AXIS.
				else {
					if (rope_node->split_val < (bbox.bounds[0].y - 0.00001)) {
						rope_node = rope_node->RightNode;
					}
					else if (rope_node->split_val > (bbox.bounds[1].y + 0.00001)) {
						rope_node = rope_node->LeftNode;
					}
					else {
						break;
					}
				}
			}

			// Case III.

			// TOP and BOTTOM.
			else {

				// Case III-A.

				// Handle parallel split plane case.
				if (rope_node->split_val == Y_Axis) {
					rope_node = (i == BOTTOM) ? rope_node->RightNode : rope_node->LeftNode;
				}

				// Case III-B.

				else if (rope_node->split_val == Z_Axis) {
					if (rope_node->split_val < (bbox.bounds[0].z - 0.00001)) {
						rope_node = rope_node->RightNode;
					}
					else if (rope_node->split_val > (bbox.bounds[1].z + 0.00001)) {
						rope_node = rope_node->LeftNode;
					}
					else {
						break;
					}
				}

				// Case III-C.

				// Split plane is X_AXIS.
				else {
					if (rope_node->split_val < (bbox.bounds[0].x - 0.00001)) {
						rope_node = rope_node->RightNode;
					}
					else if (rope_node->split_val > (bbox.bounds[1].x + 0.00001)) {
						rope_node = rope_node->LeftNode;
					}
					else {
						break;
					}
				}
			}
		}
	}
}


RBoundingVolume RKDTreeCPU::generateBox(float3 *verts, int num_verts)
{
	// Compute bounding box for input mesh.
	RVectorF max = RVectorF(-INFINITY, -INFINITY, -INFINITY);
	RVectorF min = RVectorF(INFINITY, INFINITY, INFINITY);

	for (int i = 0; i < num_verts; ++i) {
		if (verts[i].x < min.x) {
			min.x = verts[i].x;
		}
		if (verts[i].y < min.y) {
			min.y = verts[i].y;
		}
		if (verts[i].z < min.z) {
			min.z = verts[i].z;
		}
		if (verts[i].x > max.x) {
			max.x = verts[i].x;
		}
		if (verts[i].y > max.y) {
			max.y = verts[i].y;
		}
		if (verts[i].z > max.z) {
			max.z = verts[i].z;
		}
	}

	RBoundingVolume bbox;
	bbox.bounds[0] = min;
	bbox.bounds[1] = max;

	return bbox;
}

bool triIntersect(float3 ray_o, float3 ray_dir, float3 v0, float3 v1, float3 v2, float &t)
{
	float3 e1, e2, h, s, q;
	float a, f, u, v;

	e1 = v1 - v0;
	e2 = v2 - v0;

	h = cross(ray_dir, e2);
	a = dot(e1, h);

	if (a > -0.00001f && a < 0.00001f) {
		return false;
	}

	f = 1.0f / a;
	s = ray_o - v0;
	u = f * dot(s, h);

	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	q = cross(s, e1);
	v = f * dot(ray_dir, q);

	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}

	// at this stage we can compute t to find out where the intersection point is on the line
	t = f * dot(e2, q);

	if (t > 0.00001f) { // ray intersection
		return true;
	}
	else { // this means that there is a line intersection but not a ray intersection
		return false;
	}
}

bool RKDTreeCPU::intersect(KDNodeCPU * node, RRay * r, float & t, RObject ** hitObject)
{	float tNear, tFar;
	//if (node->box->intersect(r, tNear, tFar))
	//{
	//	if ((node->LeftNode && !node->LeftNode->isLeaf) || (node->RightNode && !node->RightNode->isLeaf))
	//	{
	//		bool leftHit = node->LeftNode ? intersect(node->LeftNode, r, t, hitObject) : false;
	//		bool rightHit = node->RightNode ? intersect(node->RightNode, r, t, hitObject) : false;

	//		return leftHit || rightHit;
	//	}
	//	else
	//	{
	//		float tNearObject = kInfinity, u, v;

	//		for (int i = 0; i < node->numObjs; ++i)
	//		{
	//			float3 tri = faces[node->objIndeces[i]];
	//			float3 v0 = verts[(int)tri.x];
	//			float3 v1 = verts[(int)tri.y];
	//			float3 v2 = verts[(int)tri.z];
	//			if (triIntersect() && t > tNearObject)
	//			{

	//				*hitObject = &objects[node->objIndeces[i]];
	//				t = tNearObject;
	//				intersectionAmount++;
	//			}
	//		}
	//		return(*hitObject != nullptr);
	//	}
	//}
	//return false;
}


bool RKDTreeCPU::intersect(RRay * r, float &t, RObject ** hitObject)
{
	t = kInfinity;
	return intersect(root, r, t, hitObject);
}
