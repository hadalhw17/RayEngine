#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "RayTracer.h"
#include "RayEngine.h"
#include "Scene.h"
#include "KDTree.h"
#include "Camera.h"
#include "KDThreeGPU.h"

#include "cutil_math.h"

#include <iostream>
#include <vector>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos);
void cursorEnterCallback(GLFWwindow *widnow, int entered);
void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods);

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

extern "C" float4 *Render(class RKDThreeGPU *tree, RCamera _sceneCam, std::vector<float4> h_triangles);

double x_pos = 0;
double y_pos;
double z_pos = 0;
double x_look_at = 0;
double y_look_at = 0;
double z_look_at = -0.8;
double xDelta = 0;
double yDelta = 0;
double oldMouseX = 0;
double oldMouseY = 0;
std::vector<float4> triangles;

RCamera sceneCam;
RKDThreeGPU *CUDATree;

float4 *pixels;
void RenderFrame(RKDThreeGPU *CUDATree)
{
	pixels = Render(CUDATree, sceneCam, triangles);
	// create some image data
	GLubyte *image = new GLubyte[4 * SCR_WIDTH * SCR_HEIGHT];
	for (int j = 0; j < SCR_HEIGHT; ++j) {
		for (int i = 0; i < SCR_WIDTH; ++i) {
			size_t index = j * SCR_WIDTH + i;
			image[4 * index + 0] = 0xFF * pixels[index].x; // R
			image[4 * index + 1] = 0xFF * pixels[index].y; // G
			image[4 * index + 2] = 0xFF * pixels[index].z; // B
			image[4 * index + 3] = 0xFF;
		}
	}

	// set texture parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	// set texture content
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, &image[0]);

	free(pixels);
	free(image);
}



int main()
{
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
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "RayEngine", NULL, NULL);
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

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

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
	RRayTracer *RayTracer;
	RScene *Scene;
	RKDTreeCPU *Tree;

	sceneCam.campos =  make_float3(10, 0, 10);


	//Vector between camera and the point where to look at
	float3 diffBtw = make_float3(sceneCam.campos.x - x_look_at,
		sceneCam.campos.y - y_look_at, 
		sceneCam.campos.z - z_look_at);
	float3 Y = make_float3(0.f, 1.f, 0.f);
	//Variables defining camera
	sceneCam.camdir = make_float3(1, 0, 1);
	sceneCam.camdown = Y;
	sceneCam.camright = cross(sceneCam.camdown, sceneCam.camdir);

	//sceneCam.lookAt(make_float3(x_look_at, y_look_at, z_look_at), 360);

	RayTracer = new RRayTracer;
	Scene = new RScene;
	Scene->BuildScene();
	Tree = Scene->GetSceneTree();
	CUDATree = new RKDThreeGPU(Tree);
	//RGBType *pixels = RayTracer->trace(Tree, 0.2, 0.f, -2, 0, 0);
	float3 *verts = CUDATree->get_verts();
	float3 *faces = CUDATree->get_faces();

	for (unsigned int i = 0; i < CUDATree->get_num_faces(); ++i)
	{
		// make a local copy of the triangle vertices
		float3 tri = faces[i];
		float3 v0 = verts[(int)tri.x];
		float3 v1 = verts[(int)tri.y];
		float3 v2 = verts[(int)tri.z];

		// store triangle data as float4
		// store two edges per triangle instead of vertices, to save some calculations in the
		// ray triangle intersection test
		triangles.push_back(make_float4(v0.x, v0.y, v0.z, 0));
		triangles.push_back(make_float4(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z, 0));
		triangles.push_back(make_float4(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z, 0));
	}

	// Main render loop
	// -----------
	while (!glfwWindowShouldClose(window))
	{
		// input
		// -----
		processInput(window);
		RenderFrame(CUDATree);

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
	}

	// glfw: terminate, clearing all previously allocated GLFW resources.
	// ------------------------------------------------------------------
	glfwTerminate();
	return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
	{
		sceneCam.move_forward();

	}
	else if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
	{
		sceneCam.move_backward();

	}
	else if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
	{
		sceneCam.strafe_left();
	}
	else if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
	{
		sceneCam.strafe_right();
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
		yDelta ++;


	}
	else if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
	{
		yDelta --;

	}
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


static void cursorPositionCallback(GLFWwindow *window, double xPos, double yPos)
{

}
void cursorEnterCallback(GLFWwindow *widnow, int entered)
{

}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
{

}

