#pragma once



class GLFWwindow;
namespace RayEngine
{

	struct WindowData
	{
		size_t m_width, m_heigth;
		std::string m_title;
		bool vsync;


		std::function<void(class Event&)> function_callback;

		WindowData(const std::string& title = "RayEngine",
			size_t width = 1920,
			size_t heigth = 1080)
			: m_title(title), m_width(width), m_heigth(heigth) {}
	};

	class Window
	{
	public:
		Window();
		inline GLFWwindow* get_window();

		inline size_t get_width() const { return m_data.m_width; }
		inline size_t get_heigth() const { return m_data.m_heigth; }

		inline bool set_vsync(bool vsync) { m_data.vsync = vsync; }
		inline bool get_vsync() const { return m_data.vsync; }
		
		inline const bool should_close() const { return window_shoud_close; }
		bool updateGL();
		void RenderFrame();

		inline const double delta_time() const { return deltaTime; }
		
		void destroy_window();
		void close_window();

		void set_event_callback(std::function<void(Event&)> e) { m_data.function_callback = e; }

		GLFWwindow* window;
		double currentFrame;
		double lastFrame;
		int frame_counter = 0;
		double click_timer;
	private:

		//----------------------Window functions. Wille be moved to a separate class----
		void initPixelBuffer();
		int initGL();
		int create_window();
		bool check_shader_compile_status(unsigned int obj);
		bool check_program_link_status(unsigned int obj);
		void update_mouse();
		void framebuffer_size_callback(GLFWwindow* window, int width, int height);
		//------------------------------------------------------------------------------

	private:
		bool window_shoud_close = false;
		double deltaTime;
		// vao and vbo handle
		unsigned int vao, vbo, ibo;
		// texture handle
		unsigned int texture;
		unsigned int shader_program, vertex_shader, fragment_shader;
		int texture_location;
		unsigned int pbo = 0;     // OpenGL pixel buffer object
		unsigned int tex = 0;     // OpenGL texture object
		struct cudaGraphicsResource* cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)
		// map PBO to get CUDA device pointer
		unsigned int* d_output;
		WindowData m_data;
	};

}