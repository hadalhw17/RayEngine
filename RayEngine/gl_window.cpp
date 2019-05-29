/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /*
	 Volume rendering sample

	 This sample loads a 3D volume from disk and displays it using
	 ray marching and 3D textures.

	 Note - this is intended to be an example of using 3D textures
	 in CUDA, not an optimized volume renderer.

	 Changes
	 sgg 22/3/2010
	 - updated to use texture for display instead of glDrawPixels.
	 - changed to render from front-to-back rather than back-to-front.
 */

 // OpenGL Graphics includes
#include <helper_gl.h>
#define NOMINMAX
#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif

// CUDA Runtime, Interop, and includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cuda_profiler_api.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <driver_functions.h>
#include <vector>
#include <iostream>
#include <string>

#include "RayTracer.h"
#include "RayEngine.h"
#include "Scene.h"
#include "KDTree.h"
#include "Camera.h"
#include "KDThreeGPU.h"
#include "SceneObject.h"
#include "Grid.h"
#include "MovableCamera.h"
#include "KDTree.h"
#include "KDThreeGPU.h"

// CUDA utilities
#include <helper_cuda.h>

// Helper functions
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>


typedef unsigned int uint;
typedef unsigned char uchar;

const char* sSDKsample = "CUDA 3D Volume Render";


uint width = SCR_WIDTH, height = SCR_HEIGHT;



float density = 0.05f;
float brightness = 1.0f;
float transferOffset = 0.0f;
float transferScale = 1.0f;
bool linearFiltering = true;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

StopWatchInterface* timer = 0;
StopWatchInterface* click_timer = 0;

// map PBO to get CUDA device pointer
uint* d_output;

bool should_spawn = false;
TerrainBrushType brush_type;
bool wraped = false;

// Auto-Verification Code
const int frameCheckNumber = 2;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int g_Index = 0;
unsigned int frameCount = 0;

int* pArgc;
char** pArgv;

#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

extern uchar4* render_frame(RCamera sceneCamm, uint *output, uint width, uint heigth);
extern "C" void initialize_volume_render(RCamera sceneCam, Grid* sdf, int num_sdf, std::vector<float4> textures);
extern void spawn_obj(RCamera pos, TerrainBrushType brush_type);
extern void toggle_shadow();
extern "C" void copy_memory(std::vector<RKDThreeGPU*> tree, RCamera _sceneCam, std::vector<float4> h_triangles,
	std::vector<float4> h_normals, std::vector<float2> h_uvs, std::vector<GPUSceneObject> objs, std::vector<float3> textures, Grid* grid);
extern "C"
void copyInvViewMatrix(float* invViewMatrix, size_t sizeofMatrix);
extern void free_memory();

RCamera* SceneCam;
RScene* Scene;
RMovableCamera* movable_camera;
std::vector<RKDTreeCPU*>Tree;
std::vector<RKDThreeGPU*> CUDATree;

void initPixelBuffer();

void computeFPS()
{
	frameCount++;
	fpsCount++;

	if (fpsCount == fpsLimit)
	{
		char fps[256];
		float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
		sprintf(fps, "Volume Render: %3.1f fps", ifps);

		glutSetWindowTitle(fps);
		fpsCount = 0;

		fpsLimit = (int)MAX(1.f, ifps);
		sdkResetTimer(&timer);
	}
}

void setup_camera()
{
	std::cout << "Camera initial setup." << std::endl;
	SceneCam = new RCamera();
	movable_camera = new RMovableCamera();
	movable_camera->build_camera(SceneCam);
}

void build_scene()
{
	std::cout << "Building scene" << "\" .. " << std::endl;
	Scene = new RScene;
	Scene->update_camera(movable_camera);
	int i = 0;
	Tree = Scene->GetSceneTree();
	for (auto t : Tree)
	{
		RKDThreeGPU* gpu_tree = new RKDThreeGPU(t);
		Scene->sceneObjects.at(i)->object_properties.index_list_size = gpu_tree->GetIndexList().size();

		CUDATree.push_back(gpu_tree);
		++i;
	}
	for (i = 0; i < Scene->sceneObjects.size(); i++)
	{
		for (int k = 0; k < i; k++)
		{
			Scene->sceneObjects.at(i)->object_properties.offset += Scene->sceneObjects.at(k)->object_properties.index_list_size - Scene->sceneObjects.at(k)->object_properties.num_nodes;
		}
	}
	std::cout << "Done building scene" << std::endl;
}

// render image using CUDA
void render()
{
	// map PBO to get CUDA device pointer
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)& d_output, &num_bytes,
		cuda_pbo_resource));
	//printf("CUDA mapped PBO: May access %ld bytes\n", num_bytes);

	// clear image
	checkCudaErrors(cudaMemset(d_output, 0, width * height * 4));

	movable_camera->build_camera(SceneCam);
	// call CUDA kernel, writing results to PBO
	render_frame(*SceneCam, d_output, width, height);

	getLastCudaError("kernel failed");

	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));
}

// display results using OpenGL (called by GLUT)
void display()
{
	sdkStartTimer(&timer);
	if (should_spawn && sdkGetTimerValue(&click_timer) > 100.f)
	{
		sdkResetTimer(&click_timer);
		spawn_obj(*SceneCam, brush_type);
	}

	render();

	// display results
	glClear(GL_COLOR_BUFFER_BIT);

	// draw image from PBO
	glDisable(GL_DEPTH_TEST);

	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	// copy from pbo to texture
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// draw textured quad
	glEnable(GL_TEXTURE_2D);
	glBegin(GL_QUADS);
	glTexCoord2f(0, 0);
	glVertex2f(0, 0);
	glTexCoord2f(1, 0);
	glVertex2f(1, 0);
	glTexCoord2f(1, 1);
	glVertex2f(1, 1);
	glTexCoord2f(0, 1);
	glVertex2f(0, 1);
	glEnd();

	glDisable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);

	glutSwapBuffers();
	glutReportErrors();

	sdkStopTimer(&timer);

	computeFPS();
}

void idle()
{
	glutPostRedisplay();
}

void keyboard(unsigned char key, int x, int y)
{
	RMovableCamera* tmp_cam = new RMovableCamera();
	memcpy(tmp_cam, movable_camera, sizeof(RMovableCamera));
	float scale = .5f;
	switch (key)
	{
	case 27:
#if defined (__APPLE__) || defined(MACOSX)
		exit(EXIT_SUCCESS);
#else
		glutDestroyWindow(glutGetWindow());
		return;
#endif
		break;

	case 'w':
		tmp_cam->move_forward(scale);
		break;

	case 's':
		tmp_cam->move_forward(-scale);
		break;

	case 'a':
		tmp_cam->strafe(scale);
		break;

	case 'd':
		tmp_cam->strafe(-scale);
		break;

	case 'q':
		//click_timer = 0.f;
		toggle_shadow();
		break;

	case ';':
		transferOffset += 0.01f;
		break;

	case '\'':
		transferOffset -= 0.01f;
		break;

	case '.':
		transferScale += 0.01f;
		break;

	case ',':
		transferScale -= 0.01f;
		break;

	default:
		break;
	}

	memcpy(movable_camera, tmp_cam, sizeof(RMovableCamera));
	delete tmp_cam;
	glutPostRedisplay();
	


	//
	//	bool overlaps = false;
	//	for (int i = 0; i < main_window->CUDATree.size(); ++i)
	//	{
	//		if (point_aabb_collision(main_window->Scene->sceneObjects[i]->collision_box,  tmp_cam->position))
	//			overlaps = true;
	//	}
	//
	//	click_timer += delta_time;

}

int ox, oy;
int buttonState = 0;

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		
		switch (button)
		{
		case 0:
			brush_type = TerrainBrushType::ADD;
			break;
		case 2:
			brush_type = TerrainBrushType::SUBTRACT;
			break;
		default:
			break;
		}
		should_spawn = true;
	}
	else if (state == GLUT_UP)
	{
		should_spawn = false;
	}
	ox = x;
	oy = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = (float)(glutGet(GLUT_WINDOW_WIDTH) / 2 - x);
	dy = (float)(glutGet(GLUT_WINDOW_HEIGHT) / 2 - y);



	if (dx != 0 || dy != 0)
	{
		
	}
	movable_camera->change_yaw(-dx * 0.005);
	movable_camera->change_pitch(-dy * 0.005);
	ox = x;
	oy = y;
	glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH) / 2, glutGet(GLUT_WINDOW_HEIGHT) / 2);
	glutPostRedisplay();
}

int iDivUp(int a, int b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

void reshape(int w, int h)
{
	width = w;
	height = h;
	initPixelBuffer();

	glViewport(0, 0, w, h);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	free_memory();

	if (pbo)
	{
		cudaGraphicsUnregisterResource(cuda_pbo_resource);
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}
	// Calling cudaProfilerStop causes all profile data to be
	// flushed before the application exits
	cudaProfilerStop();
}

void initGL(int* argc, char** argv)
{
	// initialize GLUT callback functions
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutCreateWindow("CUDA volume rendering");
	glutSetCursor(GLUT_CURSOR_NONE);
	glutGameModeString("1920x1080:24");
	glutEnterGameMode();

	if (!isGLVersionSupported(2, 0) ||
		!areGLExtensionsSupported("GL_ARB_pixel_buffer_object"))
	{
		printf("Required OpenGL extensions are missing.");
		exit(EXIT_SUCCESS);
	}
}

void initPixelBuffer()
{
	if (pbo)
	{
		checkCudaErrors(cudaDeviceSynchronize());
		// unregister this buffer object from CUDA C
		checkCudaErrors(cudaGraphicsUnregisterResource(cuda_pbo_resource));

		// delete old buffer
		glDeleteBuffers(1, &pbo);
		glDeleteTextures(1, &tex);
	}

	// create pixel buffer object for display
	glGenBuffers(1, &pbo);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLubyte) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

	// create texture for display
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);
}


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	pArgc = &argc;
	pArgv = argv;


#if defined(__linux__)
	setenv("DISPLAY", ":0", 0);
#endif
	//start logs
	printf("%s Starting...\n\n", sSDKsample);


	fpsLimit = frameCheckNumber;

	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL(&argc, argv);

	findCudaDevice(argc, (const char**)argv);

	sdkCreateTimer(&timer);

	sdkCreateTimer(&click_timer);
	sdkStartTimer(&click_timer);

	setup_camera();
	build_scene();
	Grid *distance_field = new Grid[1];
	distance_field[0] = Grid(std::string(PATH_TO_VOLUMES) + std::string("terrain250.rsdf"));
	//distance_field[1] = Grid(std::string(PATH_TO_VOLUMES) + std::string("cat250.rsdf"));

	initialize_volume_render(*SceneCam, distance_field, 1, Scene->textures);
	// This is the normal rendering path for VolumeRender
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutPassiveMotionFunc(motion);
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	glutIdleFunc(idle);

	initPixelBuffer();

#if defined (__APPLE__) || defined(MACOSX)
	atexit(cleanup);
#else
	glutCloseFunc(cleanup);
#endif

	glutMainLoop();
}
