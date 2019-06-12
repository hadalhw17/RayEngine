#include "repch.h"


#include "UILayer.h"
#include "../src/Primitives/Camera.h"
#include "../src/RayEngine/RayEngine.h"

#include "imgui.h"
#include "../../Platform/OpenGL/imgui_opengl_imlementation.h"
#include "../../Platform/OpenGL/imgui_glfw_implementation.h"
#include "GL/gl3w.h"

#include "GLFW/glfw3.h"
#include "SceneLayer.h"



extern
void save_map();
extern
void load_map(std::string filename);
extern
void update_render_settings(const RenderingSettings& render_settings, const SceneSettings& scene_settings);
extern
void generate_noise(const float3& pos);



namespace RayEngine
{


	RUILayer::RUILayer() :
		RLayer("UI Layer")
	{
		//main_ui = new RUserInterface();
	}

	RUILayer::~RUILayer()
	{
	}

	void RUILayer::on_attach()
	{
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

		ImGui::CreateContext();
		ImGui::StyleColorsDark();

		ImGuiIO& io = ImGui::GetIO(); //(void)io;
		io.BackendFlags |= ImGuiBackendFlags_HasMouseCursors;
		io.BackendFlags |= ImGuiBackendFlags_HasSetMousePos;

		// --------------------------TEMPORARY---------------------
		io.KeyMap[ImGuiKey_Tab] = RE_KEY_TAB;
		io.KeyMap[ImGuiKey_LeftArrow] = RE_KEY_LEFT;
		io.KeyMap[ImGuiKey_RightArrow] = RE_KEY_RIGHT;
		io.KeyMap[ImGuiKey_UpArrow] = RE_KEY_UP;
		io.KeyMap[ImGuiKey_DownArrow] = RE_KEY_DOWN;
		io.KeyMap[ImGuiKey_PageUp] = RE_KEY_PAGE_UP;
		io.KeyMap[ImGuiKey_PageDown] = RE_KEY_PAGE_DOWN;
		io.KeyMap[ImGuiKey_Home] = RE_KEY_HOME;
		io.KeyMap[ImGuiKey_End] = RE_KEY_END;
		io.KeyMap[ImGuiKey_Insert] = RE_KEY_INSERT;
		io.KeyMap[ImGuiKey_Delete] = RE_KEY_DELETE;
		io.KeyMap[ImGuiKey_Backspace] = RE_KEY_BACKSPACE;
		io.KeyMap[ImGuiKey_Space] = RE_KEY_SPACE;
		io.KeyMap[ImGuiKey_Enter] = RE_KEY_ENTER;
		io.KeyMap[ImGuiKey_Escape] = RE_KEY_ESCAPE;
		io.KeyMap[ImGuiKey_A] = RE_KEY_A;
		io.KeyMap[ImGuiKey_C] = RE_KEY_C;
		io.KeyMap[ImGuiKey_V] = RE_KEY_V;
		io.KeyMap[ImGuiKey_X] = RE_KEY_X;
		io.KeyMap[ImGuiKey_Y] = RE_KEY_Y;
		io.KeyMap[ImGuiKey_Z] = RE_KEY_Z;

		//Application& app = Application::get();
		ImGui_ImplOpenGL3_Init("#version 330 core");
	}

	void RUILayer::on_detach()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui::DestroyContext();
	}

	void RUILayer::on_update()
	{
		ImGuiIO& io = ImGui::GetIO();
		Application& app = Application::get();
		int width, heigth;
		glfwGetWindowSize(app.get_window().window, &width, &heigth);
		io.DisplaySize = ImVec2(width, heigth);

		float time = (float)glfwGetTime();
		io.DeltaTime = m_time > 0.f ? (time - m_time) : (1.f / 60.f);
		m_time = time;



		bool show = app.show_mouse;
		if (show)
		{
			m_is_visible = true;
			ImGui_ImplOpenGL3_NewFrame();
			ImGui::NewFrame();
			RSceneLayer& scene_layer = app.get_scene_layer();
			render_menu(scene_layer.brush, app.character_speed, app.render_settings, scene_layer.get_scene().scene_settings, scene_layer.get_scene().get_camera().campos);
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		}
		else
		{
			m_is_visible = false;
		}
	}

	void RUILayer::on_event(Event& event)
	{
		EventDispatcher dispatcher(event);

		dispatcher.dipatch< MouseButtonPresedEvent>(BIND_EVENT_FN(RUILayer::on_mouse_button_pressed));
		dispatcher.dipatch< MouseButtonReleasedEvent>(BIND_EVENT_FN(RUILayer::on_mouse_button_relseased));
		dispatcher.dipatch< MouseMovedEvent>(BIND_EVENT_FN(RUILayer::on_mouse_moved));
		dispatcher.dipatch< MouseScrolledEvent>(BIND_EVENT_FN(RUILayer::on_mouse_scrolled));
		dispatcher.dipatch< WindowResizedEvent>(BIND_EVENT_FN(RUILayer::on_window_reseized));
		dispatcher.dipatch< KeyReleaseEvent>(BIND_EVENT_FN(RUILayer::on_key_released));
		dispatcher.dipatch< KeyPressedEvent>(BIND_EVENT_FN(RUILayer::on_key_pressed));
		dispatcher.dipatch< KeyTypedEvent>(BIND_EVENT_FN(RUILayer::on_key_typed));
	}

	bool RUILayer::on_mouse_button_pressed(MouseButtonPresedEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.MouseDown[e.get_button()] = true;

		return m_is_visible;
	}

	bool RUILayer::on_mouse_button_relseased(MouseButtonReleasedEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.MouseDown[e.get_button()] = false;
		return m_is_visible;
	}

	bool RUILayer::on_mouse_moved(MouseMovedEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.MousePos = ImVec2(e.get_x(), e.get_y());
		return m_is_visible;
	}

	bool RUILayer::on_mouse_scrolled(MouseScrolledEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.MouseWheelH += e.get_x_offset();
		io.MouseWheel += e.get_y_offset();
		return m_is_visible;
	}

	bool RUILayer::on_window_reseized(WindowResizedEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.DisplaySize = ImVec2(e.get_width(), e.get_height());
		io.DisplayFramebufferScale = ImVec2(1.f, 1.f);
		glViewport(0, 0, e.get_width(), e.get_height());

		return m_is_visible;
	}

	bool RUILayer::on_key_released(KeyReleaseEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.KeysDown[e.get_key_code()] = false;

		return m_is_visible;
	}

	bool RUILayer::on_key_pressed(KeyPressedEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		io.KeysDown[e.get_key_code()] = true;

		io.KeyCtrl = io.KeysDown[RE_KEY_LEFT_CONTROL] || io.KeysDown[RE_KEY_RIGHT_CONTROL];
		io.KeyShift = io.KeysDown[RE_KEY_LEFT_SHIFT] || io.KeysDown[RE_KEY_RIGHT_SHIFT];
		io.KeyAlt = io.KeysDown[RE_KEY_LEFT_ALT] || io.KeysDown[RE_KEY_RIGHT_ALT];
		io.KeyShift = io.KeysDown[RE_KEY_LEFT_SUPER] || io.KeysDown[RE_KEY_RIGHT_SUPER];
		return m_is_visible;
	}

	bool RUILayer::on_key_typed(KeyTypedEvent& e)
	{
		ImGuiIO& io = ImGui::GetIO();
		int keycode = e.get_key_code();
		if (keycode > 0 && keycode < 0x10000)
			io.AddInputCharacter((unsigned short)keycode);
		return m_is_visible;
	}

	void RUILayer::render_menu(TerrainBrush& brush, float& character_speed, struct RenderingSettings& render_settings, struct SceneSettings& scene_settings, float3 curr_pos)
	{
		Application& app = Application::get();
		const float DISTANCE_X = 10;
		const float DISTANCE_Y = SCR_HEIGHT / 2 - 250;
		static int corner = 0;
		ImGuiIO& io = ImGui::GetIO();
		if (corner != -1)
		{
			ImVec2 window_pos = ImVec2((corner & 1) ? io.DisplaySize.x - DISTANCE_X : DISTANCE_X, (corner & 2) ? io.DisplaySize.y - DISTANCE_Y : DISTANCE_Y);
			ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
			ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
			ImGui::SetNextWindowSize(ImVec2(500, 500));
			ImGui::SetNextWindowContentSize(ImVec2(250, 250));
		}
		ImGui::SetNextWindowBgAlpha(.5f); // Transparent background
		bool show = app.show_mouse;
		if (ImGui::Begin("Menu", &show, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
		{
			ImGui::Text("Greetings my friend!");
			ImGui::BeginTabBar("TabBar");
			if (ImGui::BeginTabItem("Editor Settings"))
			{
				ImGui::SliderFloat("Brush size", &brush.brush_radius, 0.f, 5.0f, "%.4f", 2.0f);
				ImGui::SliderInt("Material index", &brush.material_index, 0.f, 2, "%.4f");
				ImGui::SliderFloat("Character speed", &character_speed, 0.f, 500.0f, "%.4f", 2.0f);
				ImGui::Checkbox("Enable gravity and collisions", &render_settings.gravity);

				if (ImGui::Button("Save changes"))
				{
					std::thread save_file(save_map);
					save_file.detach();
				}
				ImGui::SameLine();
				if (ImGui::Button("Load map"))
				{
					std::thread save_file(load_map, "SDFs/Edited.rsdf");
					save_file.detach();
				}
				if (ImGui::CollapsingHeader("Generator options"))
				{
					if (ImGui::SliderFloat("Noise frequency", &scene_settings.noise_freuency, 0.f, 30.f, "%.4f", 2.0f))
					{
						//generate_noise();
					}
					if (ImGui::SliderFloat("Noise amplitude", &scene_settings.noise_amplitude, 0.f, 200.f, "%.4f", 2.0f))
					{
						//generate_noise();
					}
					if (ImGui::SliderFloat("Noise redistribution", &scene_settings.noise_redistrebution, 0.f, 30.f, "%.4f", 2.0f))
					{
						//generate_noise();
					}
					if (ImGui::SliderInt("Terracing", (int*)& scene_settings.terracing, 1.f, 1000, "%.4f"))
					{
						//generate_noise();
					}
					if (ImGui::TreeNode("Volume resolution"))
					{
						if (ImGui::SliderInt("X", (int*)& scene_settings.volume_resolution.x, 1.f, 300, "%.4f"))
						{
							//generate_noise();
						}
						if (ImGui::SliderInt("Y", (int*)& scene_settings.volume_resolution.y, 1.f, 300, "%.4f"))
						{
							//generate_noise();
						}
						if (ImGui::SliderInt("Z", (int*)& scene_settings.volume_resolution.z, 1.f, 300, "%.4f"))
						{
							//generate_noise();
						}
						ImGui::TreePop();
						ImGui::Separator();
					}
					if (ImGui::TreeNode("World size"))
					{
						if (ImGui::SliderFloat("X##1", &scene_settings.world_size.x, 1.f, 300, "%.4f"))
						{
							//generate_noise();
						}
						if (ImGui::SliderFloat("Y##1", &scene_settings.world_size.y, 1.f, 300, "%.4f"))
						{
							//generate_noise();
						}
						if (ImGui::SliderFloat("Z##1", &scene_settings.world_size.z, 1.f, 300, "%.4f"))
						{
							//generate_noise();
						}
						ImGui::TreePop();
						ImGui::Separator();
					}

					if (ImGui::Button("Generate noise terrain"))
					{
						generate_noise(curr_pos);
					}
				}

				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Renderer Settings"))
			{
				ImGui::SliderFloat("Texture scale", &render_settings.texture_scale, 1.f, 250.0f, "%.4f", 2.0f);
				ImGui::SliderInt("Render quality", (int*)& render_settings.quality, 0, 2, "%.4f");
				ImGui::Checkbox("Apply gamma corrention", &render_settings.gamma);
				ImGui::Checkbox("Apply vignetting", &render_settings.vignetting);
				ImGui::SliderFloat("Vignetting k", &render_settings.vignetting_k, 0.f, 10.f, "%.4f");
				ImGui::EndTabItem();
			}
			if (ImGui::BeginTabItem("Scene Settings"))
			{
				ImGui::SliderFloat("Light angle", &scene_settings.light_pos.x, 0.f, 1000.f, "%.4f", 2.0f);
				ImGui::SliderFloat("Light Y", &scene_settings.light_pos.y, 0.f, 500.f, "%.4f", 2.0f);
				ImGui::SliderFloat("Light intensity", &scene_settings.light_intensity, 0.f, 100000, "%.4f", 2.0f);
				ImGui::SliderInt("Prelumbra size", &scene_settings.soft_shadow_k, 1, 128, "%.4f");
				ImGui::SliderFloat("Fog density", &scene_settings.fog_deisity, 0.f, 1.f, "%.4f", 2.0f); ImGui::SameLine();
				ImGui::Checkbox("Render Fog", &scene_settings.enable_fog);
				ImGui::EndTabItem();
			}
			ImGui::EndTabBar();
			update_render_settings(render_settings, scene_settings);
		}
		ImGui::End();

	}
}