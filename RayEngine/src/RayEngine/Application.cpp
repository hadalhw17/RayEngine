
#include <stdio.h>
#define NOMINMAX
#include <GL/gl3w.h>    // Initialize with gl3wInit()
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>    // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "../RayTracer.h"
#include "RayEngine.h"
#include "../Scene.h"
#include "../KDTree.h"
#include "../Camera.h"
#include "../KDThreeGPU.h"
#include "../SceneObject.h"

#include "helper_math.h"

#include <iostream>
#include <vector>
#include "Application.h"
#include "../MovableCamera.h"
#include <thread>
#include "../Grid.h"
#include <cuda_gl_interop.h>
#include <sstream>
#include "../Material.h"
#include "../PerlinNoise.h"
#include "Events/Event.h"
#include "Events/ApplicationEvent.h"
#include "Events/MouseEvent.h"
#include "Events/KeyEvent.h"
#include "Layers/SceneLayer.h"
#include "Layers/UILayer.h"


extern void cuda_render_frame(uint* output, const uint& width, const uint& heigth);
extern void free_memory();
extern void toggle_shadow();
extern bool sdf_collision(RCamera cam);
extern void spawn_obj(RCamera pos, TerrainBrush brush, int x, int y);


namespace RayEngine
{

	void framebuffer_size_callback(GLFWwindow* window, int width, int height);

	// helper to check and display for shader compiler errors
	bool Application::check_shader_compile_status(GLuint obj) {
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
	bool Application::check_program_link_status(GLuint obj) {
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

	// glfw: whenever the window size changed (by OS or user resize) this callback function executes
	// ---------------------------------------------------------------------------------------------
	void framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
		// make sure the viewport matches the new window dimensions; note that width and 
		// height will be significantly larger than specified on retina displays.
		glViewport(0, 0, width, height);
	}


	int frame_counter = 0;
	Application* Application::s_instance = nullptr;

	Application::Application()
	{
		s_instance = this;
		click_timer = 0.f;
		set_event_callback(BIND_EVENT_FN(Application::on_event));
		create_window();
		currentFrame = glfwGetTime();
		lastFrame = currentFrame;
	}

	Application::~Application()
	{
		//delete Scene, SceneCam;

		//if(distance_field) delete distance_field;
		free_memory();

		if (pbo)
		{
			cudaGraphicsUnregisterResource(cuda_pbo_resource);
			glDeleteBuffers(1, &pbo);
			glDeleteTextures(1, &tex);
		}
	}

	RSceneLayer& Application::get_scene_layer()
	{
		RSceneLayer* res = nullptr;
		for (auto layers : m_layer_stack)
		{
			RSceneLayer* sceneLayer = dynamic_cast<RSceneLayer*>(layers);
			if (sceneLayer)
			{
				return *sceneLayer;
			}
		}
		return *res;
	}

	void Application::initPixelBuffer()
	{
		if (pbo)
		{
			gpuErrchk(cudaDeviceSynchronize());
			// unregister this buffer object from CUDA C
			gpuErrchk(cudaGraphicsUnregisterResource(cuda_pbo_resource));

			// delete old buffer
			glDeleteBuffers(1, &pbo);
			glDeleteTextures(1, &tex);
		}

		// create pixel buffer object for display
		glGenBuffers(1, &pbo);
		glBindBuffer(0x88EC, pbo);
		glBufferData(0x88EC, SCR_WIDTH * SCR_HEIGHT * sizeof(GLubyte) * 4, 0, 0x88E0);
		glBindBuffer(0x88EC, 0);

		// register this buffer object with CUDA
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

		// create texture for display
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	int Application::initGL()
	{
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

		// we need these to properly pass the strings
		const char* source;
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
		texture_location = glGetUniformLocation(shader_program, "tex");



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

		return 0;
	}

	void Application::RenderFrame()
	{

		// map PBO to get CUDA device pointer
		gpuErrchk(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		size_t num_bytes;
		gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)& d_output, &num_bytes,
			cuda_pbo_resource));

		// call CUDA kernel, writing results to PBO
		cuda_render_frame(d_output, SCR_WIDTH, SCR_HEIGHT);

		gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

		// display results
		glClear(GL_COLOR_BUFFER_BIT);

		// draw image from PBO
		glDisable(GL_DEPTH_TEST);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// copy from pbo to texture
		glBindBuffer(0x88EC, pbo);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		glBindBuffer(0x88EC, 0);

	}


	// Register different events.
	void Application::on_event(Event& e)
	{
		EventDispatcher dispatcher(e);
		dispatcher.dipatch<MouseButtonPresedEvent>(BIND_EVENT_FN(Application::mouse_button_presssed));
		dispatcher.dipatch<MouseMovedEvent>(BIND_EVENT_FN(Application::on_mouse_move));
		dispatcher.dipatch<MouseScrolledEvent>(BIND_EVENT_FN(Application::on_mouse_scroll));
		dispatcher.dipatch<KeyPressedEvent>(BIND_EVENT_FN(Application::on_key_pressed));
		dispatcher.dipatch<KeyReleaseEvent>(BIND_EVENT_FN(Application::on_key_released));
		dispatcher.dipatch<WindowClosedEvent>(BIND_EVENT_FN(Application::on_window_closed));
		RE_LOG(e);

		for (auto it = m_layer_stack.end(); it != m_layer_stack.begin();)
		{
			(*--it)->on_event(e);
			if (e.get_handled())
				break;
		}
	}


	void Application::app_toggle_shadow()
	{
		toggle_shadow();
	}

	bool Application::app_sdf_collision(RCamera cam)
	{
		return sdf_collision(cam);
	}

	void Application::app_spawn_obj(RCamera pos, TerrainBrush brush, int x, int y)
	{
		spawn_obj(pos, brush, x, y);
	}

	void Application::push_layer(RLayer* layer)
	{
		m_layer_stack.push_layer(layer);
		layer->on_attach();
	}

	void Application::push_overlay(RLayer* overlay)
	{
		m_layer_stack.push_overlay(overlay);
		overlay->on_attach();
	}

	void Application::Run()
	{

		while (!glfwWindowShouldClose(window))
		{
			currentFrame = glfwGetTime();
			deltaTime = currentFrame - lastFrame;
			//lastFrame = currentFrame;
			frame_counter++;


			// render
			// ------
			glClearColor(0.02f, 0.02f, 0.02f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);

			// use the shader program
			glUseProgram(shader_program);

			// bind texture to texture unit 0
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, tex);

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

			// Iterate over the layer stack
			glfwPollEvents();
			for (auto layer : m_layer_stack)
			{
				layer->on_update();
			}

			if (show_mouse || edit_mode)
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
			else
			{
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			}

			glfwSwapBuffers(window);
			RenderFrame();

			click_timer += deltaTime;

			if (deltaTime >= 1.0)
			{
				std::stringstream ss;
				ss << "Ray Engine " << frame_counter << " FPS";
				glfwSetWindowTitle(window, ss.str().c_str());
				frame_counter = 0;
				lastFrame++;

			}
		}

		// Cleanup

		// glfw: terminate, clearing all previously allocated GLFW resources.
		// ------------------------------------------------------------------
		glfwTerminate();
	}

	bool Application::mouse_button_presssed(MouseButtonPresedEvent& e)
	{
		return false;
	}

	bool Application::on_mouse_released_callback(MouseButtonReleasedEvent& e)
	{
		return false;
	}

	bool Application::on_mouse_move(MouseMovedEvent& e)
	{
		return false;
	}

	bool Application::on_key_pressed(KeyPressedEvent& e)
	{
		if (e.get_key_code() == GLFW_KEY_ESCAPE) glfwSetWindowShouldClose(window, true);
		return true;
	}

	bool Application::on_key_released(KeyReleaseEvent& e)
	{
		if (!show_mouse)
		{
			switch (e.get_key_code())
			{
			case GLFW_KEY_LEFT_SHIFT:
				shift = false;
				break;
			default:
				break;
			}
		}
		switch (e.get_key_code())
		{
		case GLFW_KEY_TAB:
			show_mouse = false;
			break;
		case GLFW_KEY_LEFT_CONTROL:
			ctrl = false;
			break;
		default:
			break;
		}

		return true;
	}

	bool Application::on_mouse_scroll(MouseScrolledEvent& e)
	{
		return false;
	}

	bool Application::on_window_closed(WindowClosedEvent& e)
	{
		return false;
	}


	int Application::create_window()
	{
		// Setup window
		if (!glfwInit())
			return 1;

		window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Ray Engine", NULL, NULL);
		if (window == NULL)
		{
			RE_LOG("Failed to create GLFW window");
			glfwTerminate();
			return -1;
		}
		glfwMakeContextCurrent(window);

		//glfwSwapInterval(1); // Enable vsync


		// Initialize OpenGL loader
		RAY_ENGINE_ASSERT(!gl3wInit(), "Failed to initialize OpenGL loader!\n")

			glfwSetWindowUserPointer(window, &window_data);
		glfwMakeContextCurrent(window);


		glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
		glfwSetWindowSizeCallback(window, [](GLFWwindow* _window, int width, int height)
			{
				WindowData fun = *(WindowData*)glfwGetWindowUserPointer(_window);

				WindowResizedEvent event(width, height);
				fun.function_callback(event);
			});
		glfwSetCursorPosCallback(window, [](GLFWwindow* _window, double x_pos, double y_pos) {
			WindowData fun = *(WindowData*)glfwGetWindowUserPointer(_window);
			glfwGetCursorPos(_window, &x_pos, &y_pos);
			MouseMovedEvent event(x_pos, y_pos);
			fun.function_callback(event);
			});
		//glfwSetCursorEnterCallback(window, cursorEnterCallback);
		glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int button, int action, int mods) {
			WindowData fun = *(WindowData*)glfwGetWindowUserPointer(window);

			switch (action)
			{
			case GLFW_PRESS:
			{MouseButtonPresedEvent _event(button);
			fun.function_callback(_event); }
			break;
			case GLFW_RELEASE:
			{MouseButtonReleasedEvent event_(button);
			fun.function_callback(event_); }
			break;
			case GLFW_REPEAT:
			{MouseButtonPresedEvent _event_(button);
			fun.function_callback(_event_); }
			break;
			default:
				break;
			}
			});
		glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
			WindowData fun = *(WindowData*)glfwGetWindowUserPointer(window);

			MouseScrolledEvent event(xoffset, yoffset);
			fun.function_callback(event);
			});

		glfwSetKeyCallback(window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
			WindowData fun = *(WindowData*)glfwGetWindowUserPointer(window);
			switch (action)
			{
			case GLFW_PRESS:
			{KeyPressedEvent _event(key, 0);
			fun.function_callback(_event); }
			break;
			case GLFW_RELEASE:
			{KeyReleaseEvent event_(key);
			fun.function_callback(event_); }
			break;
			case GLFW_REPEAT:
			{KeyPressedEvent event(key, 1);
			fun.function_callback(event); }
			break;
			default:
				break;
			}
			});

		glfwSetCharCallback(window, [](GLFWwindow* window, unsigned int keycode) {
			WindowData fun = *(WindowData*)glfwGetWindowUserPointer(window);
			KeyTypedEvent event(keycode);
			fun.function_callback(event);
			});

		//while (!start_game)
		//{
		//	glfwPollEvents();
		//	main_window->main_ui->render_main_screen(&start_game);
		//	// Rendering
		//	ImGui::Render();
		//	int display_w, display_h;
		//	glfwMakeContextCurrent(window);
		//	glfwGetFramebufferSize(window, &display_w, &display_h);
		//	glViewport(0, 0, display_w, display_h);
		//	glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		//	glClear(GL_COLOR_BUFFER_BIT);
		//	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		//	glfwMakeContextCurrent(window);
		//	glfwSwapBuffers(window);
		//}
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		initGL();
		initPixelBuffer();
	}
}