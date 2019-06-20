#include "repch.h"


#include "Window.h"

#include <GL/gl3w.h>    // Initialize with gl3wInit()
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "Events/ApplicationEvent.h"
#include "Layers/SceneLayer.h"
#include "Layers/UILayer.h"
#include "RayEngine/Application.h"


extern uint *cuda_render_frame(uint* output, const uint& width, const uint& heigth);

namespace RayEngine
{
	void Window::RenderFrame()
	{
		glfwSwapBuffers(window);
		//// map PBO to get CUDA device pointer
		//gpuErrchk(cudaGraphicsMapResources(1, &cuda_pbo_resource, 0));
		//size_t num_bytes;
		//gpuErrchk(cudaGraphicsResourceGetMappedPointer((void**)& d_output, &num_bytes,
		//	cuda_pbo_resource));

		//// call CUDA kernel, writing results to PBO
		d_output = cuda_render_frame(d_output, m_data.m_width, m_data.m_heigth);

		// generate texture
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, &d_output[0]);
		//gpuErrchk(cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0));

		// display results
		glClear(GL_COLOR_BUFFER_BIT);

		// draw image from PBO
		glDisable(GL_DEPTH_TEST);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		// copy from pbo to texture
		//glBindBuffer(0x88EC, pbo);
		//glBindTexture(GL_TEXTURE_2D, tex);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_data.m_width, m_data.m_heigth, GL_RGBA, GL_UNSIGNED_BYTE, 0);
		//glBindBuffer(0x88EC, 0);

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

	void Window::initPixelBuffer()
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
		glBufferData(0x88EC, m_data.m_width * m_data.m_heigth * sizeof(GLubyte) * 4, 0, 0x88E0);
		glBindBuffer(0x88EC, 0);

		// register this buffer object with CUDA
		gpuErrchk(cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard));

		// create texture for display
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_data.m_width, m_data.m_heigth, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	int Window::initGL()
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
		glBufferData(GL_PIXEL_UNPACK_BUFFER, m_data.m_width * m_data.m_heigth * sizeof(GLubyte) * 4, NULL, GL_STREAM_DRAW);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

		return 0;
	}

	Window::Window()
	{
		d_output = new uint[SCR_WIDTH * SCR_HEIGHT * 4];
		click_timer = 0.f;
		create_window();
		currentFrame = glfwGetTime();
		lastFrame = currentFrame;
	}

	inline GLFWwindow* Window::get_window()
	{
		return window;
	}

	bool Window::updateGL()
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
			return false;;
		}

		glfwPollEvents();
		update_mouse();
		return true;
	}

	int Window::create_window()
	{
		// Setup window
		if (!glfwInit())
			return 1;

		window = glfwCreateWindow(m_data.m_width, m_data.m_heigth, "Ray Engine", NULL, NULL);
		if (window == NULL)
		{
			RE_LOG("Failed to create GLFW window");
			glfwTerminate();
			return -1;
		}
		glfwMakeContextCurrent(window);

		//glfwSwapInterval(1); // Enable vsync


		// Initialize OpenGL loader
		RAY_ENGINE_ASSERT(!gl3wInit(), "Failed to initialize OpenGL loader!\n");

		glfwSetWindowUserPointer(window, &m_data);
		glfwMakeContextCurrent(window);


		//glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
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
		//initPixelBuffer();
	}

	void Window::destroy_window()
	{
		glfwTerminate();
	}

	void Window::close_window()
	{
		window_shoud_close = true;
		glfwSetWindowShouldClose(window, true);
	}

	bool Window::check_shader_compile_status(unsigned int obj)
	{
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

	bool Window::check_program_link_status(unsigned int obj)
	{
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

	void Window::update_mouse()
	{
		Application& app = Application::get();
		if (app.show_mouse || app.edit_mode)
		{
			glfwSetInputMode(window, RE_CURSOR, RE_CURSOR_NORMAL);
		}
		else
		{
			glfwSetInputMode(window, RE_CURSOR, RE_CURSOR_DISABLED);
		}
	}
	void Window::framebuffer_size_callback(GLFWwindow* window, int width, int height)
	{
	}
}

