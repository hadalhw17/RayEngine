#pragma once

#include <vector>


class RRayTracer;
class RScene;
class RKDTreeCPU;
class RKDThreeGPU;
struct RCamera;
struct float4;
struct GLFWwindow;
class RMovableCamera;

class MainWindow
{

public:
	MainWindow();
	~MainWindow();

	void RenderFrame(RKDThreeGPU *CUDATree);
	void processInput(GLFWwindow *window);


	RRayTracer *RayTracer;
	RScene *Scene;
	RKDTreeCPU *Tree;
	RCamera *SceneCam;

	RKDThreeGPU *CUDATree;
	std::vector<float4> triangles;
	float4 *pixels;

	double x_pos = 0;
	double y_pos;
	double z_pos = 0;
	double x_look_at = 0;
	double y_look_at = 0;
	double z_look_at = 0.8;
	double xDelta = 0;
	double yDelta = 0;
	double oldMouseX = 0;
	double oldMouseY = 0;




private:
	void init_triangles();
	void setup_camera();
	void build_scene();


};
