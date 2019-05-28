#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "RayTracer.h"
#include "RayEngine.h"
#include "Scene.h"
#include "KDTree.h"
#include "Camera.h"
#include "KDThreeGPU.h"
#include "SceneObject.h"

#include "helper_math.h"

#include <iostream>
#include <vector>
#include "MainWindow.h"
#include "MovableCamera.h"
#include <thread>
#include "Grid.h"
#include <cuda_gl_interop.h>
#include <helper_gl.h>

 
extern uchar4* render_frame(RCamera sceneCam);
extern "C" void initialize_volume_render(RCamera sceneCam, Grid* sdf, int num_sdf);
extern void spawn_obj(RCamera pos);
extern "C" void copy_memory(std::vector<RKDThreeGPU*> tree, RCamera _sceneCam, std::vector<float4> h_triangles,
	std::vector<float4> h_normals, std::vector<float2> h_uvs, std::vector<GPUSceneObject> objs, std::vector<float3> textures, Grid* grid);
extern void free_memory();

double currentFrame;
double lastFrame;
double deltaTime;
float click_timer;

MainWindow *main_window;
RMovableCamera *movable_camera;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint tex = 0;     // OpenGL texture object
struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(float delta_dime, GLFWwindow *window);

static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos);
void cursorEnterCallback(GLFWwindow *widnow, int entered);
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
bool point_aabb_collision(const GPUBoundingBox& tBox, const float3& vecPoint);




GLubyte* image;
int frame_counter = 0;

MainWindow::MainWindow()
{
	image = new GLubyte[4 * SCR_WIDTH * SCR_HEIGHT];
	setup_camera();
	click_timer = 0.f;

	build_scene();
#ifdef ray_tracing
	init_triangles();

	std::vector<GPUSceneObject> tmp_objs;
	for (auto objs : Scene->sceneObjects)
	{
		tmp_objs.push_back(objs->object_properties);
	}
	copy_memory(CUDATree, *SceneCam, triangles, normals, uvs, tmp_objs, Scene->textures, distance_field);
#endif

#ifdef sphere_tracing
	distance_field = new Grid[1];
	distance_field[0] = Grid(std::string(PATH_TO_VOLUMES) + std::string("terrain250.rsdf"));
	//distance_field[1] = Grid(std::string(PATH_TO_VOLUMES) + std::string("cat250.rsdf"));

	initialize_volume_render(*SceneCam, distance_field, 1);
#endif
	currentFrame = glfwGetTime();
	lastFrame = currentFrame;
}

MainWindow::~MainWindow()
{
	delete Scene, SceneCam;
	//if(pixels)
	delete[] pixels, image, distance_field;
}


void MainWindow::RenderFrame()
{
	//Text->RenderText("Hello World!!", 5.0f, 5.0f, 1.0f);
	RRayTracer *tracer = new RRayTracer();
	movable_camera->build_camera(SceneCam);
	pixels = render_frame(*SceneCam);

	//pixels = tracer->sphere_trace(distance_field,SceneCam);
	// create some image data
	
	for (int j = 0; j < SCR_HEIGHT; ++j) {
		size_t indexY = j * SCR_WIDTH;
		for (int i = 0; i < SCR_WIDTH; ++i) {
			size_t index = indexY + i;
			image[4 * index + 0] = pixels[index].x; // R
			image[4 * index + 1] = pixels[index].y; // G
			image[4 * index + 2] = pixels[index].z; // B
			//image[4 * index + 3] = 0xFF;
		}
	}
	// set texture content
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);
	//delete[] pixels;
	//delete[] image;
}


bool should_spawn = false;
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void MainWindow::processInput(float delta_time, GLFWwindow *window)
{
	RMovableCamera *tmp_cam = new RMovableCamera();
	memcpy(tmp_cam, movable_camera, sizeof(RMovableCamera));



	float scale = .5f;
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		tmp_cam->move_forward(scale);

	}
	else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		tmp_cam->move_forward(-scale);

	}
	else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		tmp_cam->strafe(scale);
	}
	else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		tmp_cam->strafe(-scale);
	}
	else if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
	{
		x_look_at -= 10.f;
	}
	else if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
	{
		x_look_at += 10.f;

	}
	else if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
	{
		yDelta++;


	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
	{
		yDelta--;

	}
	if (!should_spawn && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_1) == GLFW_PRESS && click_timer > 10.f)
	{
		click_timer = 0.f;
		should_spawn = true;
	}

	bool overlaps = false;
	for (int i = 0; i < main_window->CUDATree.size(); ++i)
	{
		if (point_aabb_collision(main_window->Scene->sceneObjects[i]->collision_box,  tmp_cam->position))
			overlaps = true;
	}

	click_timer += delta_time;

	memcpy(movable_camera, tmp_cam, sizeof(RMovableCamera));
	delete tmp_cam;
}


void MainWindow::init_triangles()
{
	std::cout << "Initialising triangle buffer" << "\" .. " << std::endl;
	triangles = {};
	normals = {};
	uvs = {};
	int count = 0;
	size_t offset = 0;
	size_t root_offset = 0;

	for (auto t : CUDATree)
	{

		float3 *verts = t->get_verts();
		float3 *faces = t->get_faces();
		float3 *norms = t->get_normals();
		float2 *uv = t->uvs;

		for (size_t i = 0; i < t->get_num_faces(); ++i)
		{
			// make a local copy of the triangle vertices
			float3 tri = faces[i];
			float3 v0 = verts[(size_t)tri.x];
			float3 v1 = verts[(size_t)tri.y];
			float3 v2 = verts[(size_t)tri.z];

			// store triangle data as float4
			// store two edges per triangle instead of vertices, to save some calculations in the
			// ray triangle intersection test
			triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
			triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));
			triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));

			float3 n0 = norms[(size_t)tri.x];
			float3 n1 = norms[(size_t)tri.y];
			float3 n2 = norms[(size_t)tri.z];
			normals.push_back(make_float4(n0.x, n0.y, n0.z, 0));
			normals.push_back(make_float4(n1.x, n1.y, n1.z, 0));
			normals.push_back(make_float4(n2.x, n2.y, n2.z, 0));

			if (t->num_uvs > 0)
			{
				float2 uv0 = uv[(size_t)tri.x];
				float2 uv1 = uv[(size_t)tri.y];
				float2 uv2 = uv[(size_t)tri.z];
				uvs.push_back(make_float2(uv0.x, uv0.y));
				uvs.push_back(make_float2(uv0.x, uv0.y));
				uvs.push_back(make_float2(uv0.x, uv0.y));
			}
			//t->GetIndexList()[i] += offset;
		}
		std::cout << "Old root index: " << t->root_index << std::endl;

		for (int k = 0; k < t->GetNumNodes(); ++k)
		{
			t->GetNodes()[k].left_index += root_offset;
			t->GetNodes()[k].right_index += root_offset;
			t->GetNodes()[k].index_of_first_object += root_offset;

			for (int i = 0; i < 6; ++i)
			{
				if(t->GetNodes()[k].neighbor_node_indices[i] != -1)
					t->GetNodes()[k].neighbor_node_indices[i] += root_offset;
			}
		}
		t->root_index += root_offset;
		
		offset += t->obj_index_list.size();
		
		root_offset += t->GetNumNodes();
		std::cout << "New root index: " << t->root_index << std::endl;

		++count;
		delete[] verts, faces, norms;
	}
	std::cout << "Done initialising triangle buffer" << std::endl;
}


void MainWindow::setup_camera()
{
	std::cout << "Camera initial setup." << std::endl;
	SceneCam = new RCamera();
	movable_camera = new RMovableCamera();
	movable_camera->build_camera(SceneCam);
}


void MainWindow::build_scene()
{
	std::cout << "Building scene" << "\" .. " << std::endl;
	Scene = new RScene;
	Scene->update_camera(movable_camera);
	int i = 0;
	Tree = Scene->GetSceneTree();
	for (auto t : Tree)
	{
		RKDThreeGPU *gpu_tree = new RKDThreeGPU(t);
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



// helper to check and display for shader compiler errors
bool check_shader_compile_status(GLuint obj) {
	GLint status;
	glGetShaderiv(obj, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		GLint length;
		glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &length);
		std::vector<char> log(length);
		glGetShaderInfoLog(obj, length, &length, &log[0]);
		std::cerr << &log[0];
		return false;
	}
	return true;
}


// helper to check and display for shader linker error
bool check_program_link_status(GLuint obj) {
	GLint status;
	glGetProgramiv(obj, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		GLint length;
		glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &length);
		std::vector<char> log(length);
		glGetProgramInfoLog(obj, length, &length, &log[0]);
		std::cerr << &log[0];
		return false;
	}
	return true;
}


void processInput(float delta_time, GLFWwindow *window)
{
	main_window->processInput(delta_time, window);
}



// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);
}

static float2 oldMousePosition;

// mouse event handlers
int lastX = 0, lastY = 0;
int theButtonState = 0;
int theModifierState = 0;

void mouse_motion(double xPos, double yPos)
{
	int deltaX = lastX - xPos;
	int deltaY = lastY - yPos;

	if (deltaX != 0 || deltaY != 0)
	{
		movable_camera->change_yaw(-deltaX * 0.005);
		movable_camera->change_pitch(-deltaY * 0.005);
	}
	lastX = xPos;
	lastY = yPos;
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	movable_camera->change_altitude(yoffset * 0.01);
}

static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos)
{
	mouse_motion(xPos, yPos);
}
void cursorEnterCallback(GLFWwindow *widnow, int entered)
{

}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{

}

void render_thread(MainWindow *main_window, float delta_time)
{
	if (main_window->CUDATree.size() > 0)
	{
		//main_window->CUDATree.clear(); //clear content
		//main_window->CUDATree.resize(0); //resize it to 0
		//main_window->CUDATree.shrink_to_fit(); //reallocate memory
		//delete main_window->Tree;
	}
	main_window->Scene->Tick(delta_time);

	//main_window->Tree = main_window->Scene->GetSceneTree();
	for (auto t : main_window->Tree)
	{
		//main_window->CUDATree.push_back(new RKDThreeGPU(t));
	}
	//main_window->init_triangles();
}
bool point_aabb_collision(const GPUBoundingBox& tBox, const float3& vecPoint)
{

	//Check if the point is less than max and greater than min
	if (vecPoint.x > tBox.Min.x && vecPoint.x < tBox.Max.x &&
		vecPoint.y > tBox.Min.y && vecPoint.y < tBox.Max.y &&
		vecPoint.z > tBox.Min.z && vecPoint.z < tBox.Max.z)
	{
		return true;
	}

	//If not, then return false
	return false;

}


int main()
{
	main_window = new MainWindow();


	// glfw: initialize and configure
// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, true);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(1920, 1080, "RayEngine", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glfwSetCursorPosCallback(window, cursorPositionCallback);
	glfwSetCursorEnterCallback(window, cursorEnterCallback);
	glfwSetMouseButtonCallback(window, mouseButtonCallback);
	glfwSetScrollCallback(window, scroll_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	//main_window->Text = new TextRenderer(SCR_WIDTH, SCR_HEIGHT);
	//main_window->Text->Load("fonts/ocraext.TTF", 24);

	// shader source code
	std::string vertex_source =
		"#version 330\n"
		"layout(location = 0) in vec4 vposition;\n"
		"layout(location = 1) in vec2 vtexcoord;\n"
		"out vec2 ftexcoord;\n"
		"void main() {\n"
		"   ftexcoord = vtexcoord;\n"
		"   gl_Position = vposition;\n"
		"}\n";

	std::string fragment_source =
		"#version 330\n"
		"uniform sampler2D tex;\n" // texture uniform
		"in vec2 ftexcoord;\n"
		"layout(location = 0) out vec4 FragColor;\n"
		"void main() {\n"
		"   FragColor = texture(tex, ftexcoord);\n"
		"}\n";

	// program and shader handles
	GLuint shader_program, vertex_shader, fragment_shader;

	// we need these to properly pass the strings
	const char *source;
	int length;

	// create and compiler vertex shader
	vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	source = vertex_source.c_str();
	length = vertex_source.size();
	glShaderSource(vertex_shader, 1, &source, &length);
	glCompileShader(vertex_shader);

	if (!check_shader_compile_status(vertex_shader)) {
		glfwDestroyWindow(window);
		glfwTerminate();
		return 1;
	}

	// create and compiler fragment shader
	fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	source = fragment_source.c_str();
	length = fragment_source.size();
	glShaderSource(fragment_shader, 1, &source, &length);
	glCompileShader(fragment_shader);
	if (!check_shader_compile_status(fragment_shader)) {
		glfwDestroyWindow(window);
		glfwTerminate();
		return 1;
	}

	// create program
	shader_program = glCreateProgram();

	// attach shaders
	glAttachShader(shader_program, vertex_shader);
	glAttachShader(shader_program, fragment_shader);

	// link the program and check for errors
	glLinkProgram(shader_program);
	check_program_link_status(shader_program);

	// get texture uniform location
	GLint texture_location = glGetUniformLocation(shader_program, "tex");

	// vao and vbo handle
	GLuint vao, vbo, ibo;

	// generate and bind the vao
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// generate and bind the vertex buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	// data for a fullscreen quad (this time with texture coords)
	GLfloat vertexData[] = {
		//  X     Y     Z           U     V     
		   1.0f, 1.0f, 0.0f,       1.0f, 1.0f, // vertex 0
		  -1.0f, 1.0f, 0.0f,       0.0f, 1.0f, // vertex 1
		   1.0f,-1.0f, 0.0f,       1.0f, 0.0f, // vertex 2
		  -1.0f,-1.0f, 0.0f,       0.0f, 0.0f, // vertex 3
	}; // 4 vertices with 5 components (floats) each

	// fill with data
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * 4 * 5, vertexData, GL_STATIC_DRAW);


	// set up generic attrib pointers
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (char*)0 + 0 * sizeof(GLfloat));

	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (char*)0 + 3 * sizeof(GLfloat));


	// generate and bind the index buffer object
	glGenBuffers(1, &ibo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

	GLuint indexData[] = {
		0,1,2, // first triangle
		2,1,3, // second triangle
	};

	// fill with data
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * 2 * 3, indexData, GL_STATIC_DRAW);

	// "unbind" vao
	glBindVertexArray(0);

	// texture handle
	GLuint texture;

	// generate texture
	glGenTextures(1, &texture);

	// bind the texture
	glBindTexture(GL_TEXTURE_2D, texture);
	// set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	GLuint bufferID;
	glGenBuffers(1, &bufferID);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, bufferID);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, SCR_HEIGHT * SCR_WIDTH * sizeof(GLubyte) * 4, NULL, GL_STREAM_DRAW);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

	std::cout << "Welcome" << "\" .. " << std::endl;

	// Main render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		//lastFrame = currentFrame;
		frame_counter++;
		if (deltaTime >= 1.0)
		{
			printf("%f ms/frame\n", 1000.0 / double(frame_counter));
			frame_counter = 0;
			lastFrame++;

		}




		//main_window->Text->RenderText("Hello World!!", 5.0f, 5.0f, 2.0f);

		// render
		// ------
		glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		// use the shader program
		glUseProgram(shader_program);

		// bind texture to texture unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);

		// set texture uniform
		glUniform1i(texture_location, 0);

		// bind the vao
		glBindVertexArray(vao);

		// draw
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		// check for errors
		GLenum error = glGetError();
		if (error != GL_NO_ERROR) {
			std::cerr << error << std::endl;
			break;
		}

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		// -------------------------------------------------------------------------------
		glfwSwapBuffers(window);
		glfwPollEvents();

		main_window->RenderFrame();
		render_thread(main_window, deltaTime);


		// input
		processInput(deltaTime, window);
		if (should_spawn)
		{
		
			spawn_obj(*main_window->SceneCam);
			should_spawn = false;
		}

	}
	free_memory();

	delete main_window;
	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();

	return 0;
}
