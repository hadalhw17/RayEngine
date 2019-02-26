#include "KDTree.h"
#include "Plane.h"
#include "Triangle.h"

#include "cuda_runtime.h"
#include "cutil_math.h"


#include <memory>
#include <iostream>

const int MAX_DEPTH = 41;
const  int OBJECTS_IN_LEAF = 20;
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
	if (numFaces > 0 && objIndeces) {
		delete[] objIndeces;
	}

	if (LeftNode) {
		delete LeftNode;
	}
	if (RightNode) {
		delete RightNode;
	}
	for (auto rope : ropes)
	{
		delete rope;
	}
}

void KDNodeCPU::PrintDebugString()
{
	std::cout << "bounding box min: ( " << box.bounds[0].x << ", " << box.bounds[0].y << ", " << box.bounds[0].z << " )" << std::endl;
	std::cout << "bounding box max: ( " << box.bounds[1].x << ", " << box.bounds[1].y << ", " << box.bounds[1].z << " )" << std::endl;
	std::cout << "num_tris: " << numFaces << std::endl;
	std::cout << "id: " << vid << std::endl;

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
	lowerBound = make_float3(kInfinity);
	upperBound = make_float3(-kInfinity);
}

RKDTreeCPU::RKDTreeCPU(RKDTreeCPU * node, RBoundingVolume b, float3 lb, float3 ub, int d)
{
	this->box = b;
	this->lowerBound = lb;
	this->upperBound = ub;
	numNodes = 0;
	numLevels = 0;
	numLeaves = 0;
	depth = d;
}



RKDTreeCPU::RKDTreeCPU(float3 *_verts, float3 *_faces, float3 *_norms,  int num_verts, int num_faces, int num_norms)
{
	this->num_verts = num_verts;
	this->num_faces = num_faces;
	this->num_norms = num_norms;

	this->numNodes = 0;
	this->numLevels = 0;
	this->numLeaves = 0;
	this->depth = 1;

	this->verts = new float3[num_verts];
	this->norms = new float3[num_norms];
	this->faces = new float3[num_faces];

	for (int i = 0; i < num_verts; ++i)
	{
		this->verts[i] = _verts[i];
	}

	for (int i = 0; i < num_faces; ++i)
	{
		this->faces[i] = _faces[i];
	}

	for (int i = 0; i < num_norms; ++i)
	{
		this->norms[i] = _norms[i];
	}

	// Create list of triangle indices for first level of kd-tree.
	int *tri_indices = new int[num_faces];
	for (int i = 0; i < num_faces; ++i) {
		tri_indices[i] = i;
	}
	RBoundingVolume bbox;
	bbox = generateBox(num_faces, tri_indices);

	this->root = build(1, num_faces, tri_indices, bbox);

	// build rope structure
	KDNodeCPU* ropes[6] = { nullptr };
	buildRopeStructure();


}

RKDTreeCPU::~RKDTreeCPU()
{
	if (num_verts)
	{
		delete[] verts;
		delete[] norms;
	}
	if (num_faces)
	{
		delete[] faces;
	}
	if (root)
	{
		delete root;
	}
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

bool KDNodeCPU::isPointToLeftOfSplittingPlane(const float3 &p) const
{
	if (axis == X_Axis) {
		return (p.x < split_val);
	}
	else if (axis == Y_Axis) {
		return (p.y < split_val);
	}
	else if (axis == Z_Axis) {
		return (p.z < split_val);
	}
	// Something went wrong because split_plane_axis is not set to one of the three allowed values.
	else {
		std::cout << "ERROR: split_plane_axis not set to valid value." << std::endl;
		return false;
	}
}

KDNodeCPU* KDNodeCPU::getNeighboringNode(float3 p)
{

	// Check left face.
	if (fabs(p.x - box.bounds[0].x) < kEpsilon) {
		return ropes[LEFT];
	}
	// Check front face.
	else if (fabs(p.z - box.bounds[1].z) < kEpsilon) {
		return ropes[FRONT];
	}
	// Check right face.
	else if (fabs(p.x - box.bounds[1].x) < kEpsilon) {
		return ropes[RIGHT];
	}
	// Check back face.
	else if (fabs(p.z - box.bounds[0].z) < kEpsilon) {
		return ropes[BACK];
	}
	// Check top face.
	else if (fabs(p.y - box.bounds[1].y) < kEpsilon) {
		return ropes[TOP];
	}
	// Check bottom face.
	else if (fabs(p.y - box.bounds[0].y) < kEpsilon) {
		return ropes[BOTTOM];
	}
	// p should be a point on one of the faces of this node's bounding box, but in this case, it isn't.
	else {
		std::cout << "ERROR: Node neighbor could not be returned." << std::endl;
		return nullptr;
	}
}

KDNodeCPU * RKDTreeCPU::build(int dep, int objCount, int *tri_indecies, RBoundingVolume bbox)
{
	KDNodeCPU *node = new KDNodeCPU();
	node->box = bbox;
	node->numFaces = objCount;

	node->objIndeces = tri_indecies;

	//node->box = generateBox(objCount, tri_indecies);


	// Reached minimum abount of primitives in the node => make leaf and return it.
	if (objCount <= OBJECTS_IN_LEAF || dep >= MAX_DEPTH)
	{
		node->isLeaf = true;

		if (dep > numLevels)
		{
			numLevels = dep;
		}

		node->vid = numNodes;
		++numNodes;

		++numLeaves;

		return node;
	}

	// Bounding boxes for left and right children.
	RBoundingVolume leftBBox = bbox;
	RBoundingVolume rightBBox = bbox;


	// Average midpoint of the bounding box.
	float midp = 0;
	Axis split_axis = getLongestBoundingBoxSide(make_float3(bbox.bounds[0].x, bbox.bounds[0].y, bbox.bounds[0].z),
		make_float3(bbox.bounds[1].x, bbox.bounds[1].y, bbox.bounds[1].z));
	node->axis = split_axis;

	// Update boxes boundaries according to split axis and mid point.
	switch (split_axis) {
	case X_Axis:		//X_Axis
		midp = bbox.bounds[0].x + ((bbox.bounds[1].x - bbox.bounds[0].x) / (2.f));
		leftBBox.bounds[1].x = midp + kEpsilon;
		rightBBox.bounds[0].x = midp - kEpsilon;
		break;
	case Y_Axis:		//Y_Axis
		midp = bbox.bounds[0].y + ((bbox.bounds[1].y - bbox.bounds[0].y) / (2.f));
		leftBBox.bounds[1].y = midp + kEpsilon;
		rightBBox.bounds[0].y = midp - kEpsilon;
		break;
	case Z_Axis:		//Z_Axis
		midp = bbox.bounds[0].z + ((bbox.bounds[1].z - bbox.bounds[0].z) / (2.f));
		leftBBox.bounds[1].z = midp + kEpsilon;
		rightBBox.bounds[0].z = midp - kEpsilon;
		break;
	}

	node->split_val = midp;


	// Allocate and initialize memory for temporary buffers to hold triangle indices for left and right subtrees.
	int *temp_left_tri_indices = new int[objCount];
	int *temp_right_tri_indices = new int[objCount];

	int leftObjCount = 0, rightObjCount = 0;

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

	for (size_t i = 0; i < objCount; ++i) {
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

	return node;
}

void RKDTreeCPU::buildRopeStructure(KDNodeCPU *curr_node, KDNodeCPU *ropes[], bool is_single_ray_case)
{
	// Base case.
	if (curr_node->isLeaf) 
	{
		//std::cout<<curr_node->id<<": "<<std::endl;
		for (int i = 0; i < 6; ++i) {
			curr_node->ropes[i] = ropes[i];
		}
	}
	else 
	{
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

void RKDTreeCPU::buildRopeStructure()
{
	KDNodeCPU* ropes[6] = { NULL };
	buildRopeStructure(root, ropes, true);
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
	float3 max = make_float3(-kInfinity, -kInfinity, -kInfinity);
	float3 min = make_float3(kInfinity, kInfinity, kInfinity);

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

RBoundingVolume RKDTreeCPU::generateBox(int num_tris, int *tri_indices)
{
	int num_vert = num_tris * 3;
	float3 *vert = new float3[num_vert];

	int verts_index;
	for (int i = 0; i < num_tris; ++i) {
		float3 tri = faces[i];
		verts_index = i * 3;
		vert[verts_index + 0] = verts[(int)tri.x];
		vert[verts_index + 1] = verts[(int)tri.y];
		vert[verts_index + 2] = verts[(int)tri.z];
	}

	RBoundingVolume bbox = generateBox(vert, num_vert);
	delete[] vert;
	return bbox;
}

float3 get_tri_normal(float3 &p1, float3 &p2, float3 &p3)
{
	float3 u = p2 - p1;
	float3 v = p3 - p1;

	float nx = u.y * v.z - u.z * v.y;
	float ny = u.z * v.x - u.x * v.z;
	float nz = u.x * v.y - u.y * v.x;


	return normalize(make_float3(nx, ny, nz));
}

float3 get_tri_normal(float3 &n0, float3 &n1, float3 &n2, float &u, float &v)
{

	return normalize((1 - u - v) * make_float3(n0.x, n0.y, n0.z) + u * make_float3(n1.x, n1.y, n1.z) + v * make_float3(n2.x, n2.y, n2.z));
}

bool triIntersect(float3 ray_o, float3 ray_dir, float3 v0, float3 v1, float3 v2, float &t, float3 &normal, float3 &n0, float3 &n1, float3 &n2)
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
		normal = get_tri_normal(n0, n1, n2, u, v);
		return true;
	}
	else { // this means that there is a line intersection but not a ray intersection
		return false;
	}
}

bool RKDTreeCPU::intersect(KDNodeCPU * node, RRay r, float & t, float3 &normal)
{	
	float tNear, tFar;
	bool has_intersected = false;
	if (node->box.intersect(r, tNear, tFar))
	{
		if ((node->LeftNode && !node->LeftNode->isLeaf) || (node->RightNode && !node->RightNode->isLeaf))
		{
			bool leftHit = node->LeftNode ? intersect(node->LeftNode, r, t, normal) : false;
			bool rightHit = node->RightNode ? intersect(node->RightNode, r, t, normal) : false;

			return leftHit || rightHit;
		}
		else
		{
			float tNearObject = kInfinity;

			for (int i = 0; i < node->numFaces; ++i)
			{
				float3 tri = faces[node->objIndeces[i]];
				float3 v0 = verts[(int)tri.x];
				float3 v1 = verts[(int)tri.y];
				float3 v2 = verts[(int)tri.z];

				float3 n0 = norms[(int)tri.x];
				float3 n1 = norms[(int)tri.y];
				float3 n2 = norms[(int)tri.z];
				if (triIntersect(r.getRayOrigin(), r.getRayDirection(), v0, v1, v2, t, normal, n0, n1, n2))
				{
					has_intersected = true;
					//*hitObject = &objects[node->objIndeces[i]];
					t = tNearObject;
					intersectionAmount++;
				}
			}
			return has_intersected;
		}
	}
	return false;
}

bool RKDTreeCPU::singleRayStacklessIntersect(KDNodeCPU *node, float3 ray_o, float3 ray_dir, float &t, float3 & normal)
{
	// Perform ray/AABB intersection test.
	float t_near, t_far;
	RRay ray (ray_o, ray_dir);

	node->box.intersect(ray,t_near, t_far);
	bool intersects_aabb = node->box.intersect(ray, t_near, t_far);

	if (intersects_aabb) {
		// If current node is a leaf node.
		if (!node->LeftNode && !node->RightNode) {
			// Check triangles for intersections.
			bool intersection_detected = false;
			for (int i = 0; i < node->numFaces; ++i) {
				float3 tri = faces[node->objIndeces[i]];
				float3 v0 = verts[(int)tri.x];
				float3 v1 = verts[(int)tri.y];
				float3 v2 = verts[(int)tri.z];

				float3 n0 = norms[(int)tri.x];
				float3 n1 = norms[(int)tri.y];
				float3 n2 = norms[(int)tri.z];

				// Perform ray/triangle intersection test.
				float tmp_t = INFINITY;
				float3 tmp_normal = make_float3(0.0f, 0.0f, 0.0f);
				bool intersects_tri = triIntersect(ray.getRayOrigin(), ray.getRayDirection(), v0, v1, v2, tmp_t, tmp_normal, n0, n1, n2);

				if (intersects_tri) {
					intersection_detected = true;
					if (tmp_t < t) {
						t = tmp_t;
						normal = tmp_normal;
					}
				}
			}

			return intersection_detected;
		}
		// Else, recurse.
		else {
			bool hit_left = false, hit_right = false;
			if (node->LeftNode) {
				hit_left = singleRayStacklessIntersect(node->LeftNode, ray_o, ray_dir, t, normal);
			}
			if (node->RightNode) {
				hit_right = singleRayStacklessIntersect(node->RightNode, ray_o, ray_dir, t, normal);
			}
			return hit_left || hit_right;
		}
	}

	return false;
}
bool RKDTreeCPU::singleRayStacklessIntersect(RRay ray, float & t, float3 & normal)
{
	// Perform ray/AABB intersection test.
	t = kInfinity;
	float t_near, t_far;
	float3 ray_o = ray.getRayOrigin();
	float3 ray_dir = ray.getRayDirection();
	bool hit = singleRayStacklessIntersect(root, ray_o, ray_dir, t, normal);
	if (hit) {
		//t = t_far;
	}
	return hit;
	
}

bool RKDTreeCPU::intersect(RRay r, float &t, float3 &normal)
{
	t = kInfinity;
	return intersect(root, r, t, normal);
}
