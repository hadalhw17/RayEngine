#include "UserInterface.h"

#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>    // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>    // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>
#include <iostream>

#include "RayEngine.h"

void RUserInterface::inti_UI(GLFWwindow* window)
{
#if __APPLE__
	// GL 3.2 + GLSL 150
	const char* glsl_version = "#version 150";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 330 core";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
	//glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	//glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls

	// Setup Dear ImGui style
	IMGUI_CHECKVERSION();
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();

	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init(glsl_version);

	// Load Fonts
	// - If no fonts are loaded, dear imgui will use the default font. You can also load multiple fonts and use ImGui::PushFont()/PopFont() to select them.
	// - AddFontFromFileTTF() will return the ImFont* so you can store it if you need to select the font among multiple.
	// - If the file cannot be loaded, the function will return NULL. Please handle those errors in your application (e.g. use an assertion, or display an error and quit).
	// - The fonts will be rasterized at a given size (w/ oversampling) and stored into a texture when calling ImFontAtlas::Build()/GetTexDataAsXXXX(), which ImGui_ImplXXXX_NewFrame below will call.
	// - Read 'misc/fonts/README.txt' for more instructions and details.
	// - Remember that in C/C++ if you want to include a backslash \ in a string literal you need to write a double backslash \\ !
	//io.Fonts->AddFontDefault();
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf", 15.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../misc/fonts/ProggyTiny.ttf", 10.0f);
	//ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());
	//IM_ASSERT(font != NULL);
}

void RUserInterface::clean_UI()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	//start_game = false;
}



void RUserInterface::render_main_hud(bool show_menu, struct TerrainBrush& brush, float& character_speed, struct RenderingSettings& render_settings, struct SceneSettings& scene_settings)
{
	if (!show_menu)
	{
		bool overlay;
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ShowExampleAppSimpleOverlay(&overlay);
		//render_crosshair(&overlay);
		// Rendering
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}
	else
	{
		render_menu(brush, character_speed, render_settings, scene_settings);
	}

}

void RUserInterface::render_main_screen(bool* start_game)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	const float DISTANCE_X = SCR_WIDTH / 2 - 250;
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
	bool show = true;
	if (ImGui::Begin("WelcomeScreen", &show, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
	{
		ImGui::Text("Greetings my friend!");
		if (ImGui::Button("Start!"))
		{
			*start_game = true;
			show = false;
		}
		if (ImGui::Button("Quit"))
		{
			ImGui::End();
			ImGui::Render();
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			clean_UI();
			glfwTerminate();
			*start_game = true;
			return;
		}
	}
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

extern
void save_map();
extern
void update_render_settings(RenderingSettings render_settings, SceneSettings scene_settings);
extern
void generate_noise();
void RUserInterface::render_menu(TerrainBrush &brush, float &character_speed, struct RenderingSettings& render_settings, struct SceneSettings &scene_settings)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
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
	bool show = true;
	if (ImGui::Begin("Menu", &show, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
	{
		ImGui::Text("Greetings my friend!");
		ImGui::BeginTabBar("TabBar");
		if (ImGui::BeginTabItem("Editor Settings"))
		{
			ImGui::SliderFloat("Brush size", &brush.brush_radius, 0.f, 5.0f, "%.4f", 2.0f);
			ImGui::SliderInt("Material index", &brush.material_index, 0.f, 2, "%.4f");
			ImGui::SliderFloat("Character speed", &character_speed, 0.f, 2.0f, "%.4f", 2.0f);

			if (ImGui::Button("Save changes"))
			{

				save_map();
			}
			ImGui::SameLine();
			if (ImGui::Button("Load map"))
			{
				ImGui::End();
				ImGui::Render();
				ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
				clean_UI();
				glfwTerminate();
				return;
			}
			if (ImGui::CollapsingHeader("Generator options"))
			{
				if (ImGui::SliderFloat("Noise frequency", &scene_settings.noise_freuency, 0.f, 2.f, "%.4f", 2.0f))
				{
					generate_noise();
				}
				if (ImGui::SliderFloat("Noise amplitude", &scene_settings.noise_amplitude, 0.f, 10.f, "%.4f", 2.0f))
				{
					generate_noise();
				}
				if (ImGui::TreeNode("Volume resolution"))
				{
					if (ImGui::SliderInt("X", &scene_settings.volume_resolution.x, 1.f, 255.f, "%.4f"))
					{
						generate_noise();
					}
					if (ImGui::SliderInt("Y", &scene_settings.volume_resolution.y, 1.f, 255.f, "%.4f"))
					{
							generate_noise();
					}
					if (ImGui::SliderInt("Z", &scene_settings.volume_resolution.z, 1.f, 255.f, "%.4f"))
					{
						generate_noise();
					}
					ImGui::TreePop();
					ImGui::Separator();
				}
				if (ImGui::TreeNode("World size"))
				{
					if (ImGui::SliderFloat("X##1", &scene_settings.world_size.x, 1.f, 255.f, "%.4f"))
					{
						generate_noise();
					}
					if (ImGui::SliderFloat("Y##1", &scene_settings.world_size.y, 1.f, 255.f, "%.4f"))
					{
						generate_noise();
					}
					if (ImGui::SliderFloat("Z##1", &scene_settings.world_size.z, 1.f, 255.f, "%.4f"))
					{
						generate_noise();
					}
					ImGui::TreePop();
					ImGui::Separator();
				}

				if (ImGui::Button("Generate noise terrain"))
				{
					generate_noise();
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
			ImGui::SliderFloat("Light angle", &scene_settings.light_pos.x, 0.f, 1.f, "%.4f", 2.0f);
			ImGui::SliderFloat("Light Y", &scene_settings.light_pos.y, 0.f, 100.f, "%.4f", 2.0f);
			ImGui::SliderFloat("Light intensity", &scene_settings.light_intensity, 0.f, 100000, "%.4f", 2.0f);
			ImGui::SliderInt("Prelumbra size", &scene_settings.soft_shadow_k, 1.f, 128.0f, "%.4f");
			ImGui::SliderFloat("Fog density", &scene_settings.fog_deisity, 0.f, 1.f, "%.4f", 2.0f); ImGui::SameLine();
			ImGui::Checkbox("Render Fog", &scene_settings.enable_fog);
			ImGui::EndTabItem();
		}
		ImGui::EndTabBar();
		update_render_settings(render_settings, scene_settings);
	}
	ImGui::End();
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void RUserInterface::ShowExampleAppSimpleOverlay(bool* p_open)
{
	const float DISTANCE = 10.0f;
	static int corner = 0;
	ImGuiIO& io = ImGui::GetIO();
	if (corner != -1)
	{
		ImVec2 window_pos = ImVec2((corner & 1) ? io.DisplaySize.x - DISTANCE : DISTANCE, (corner & 2) ? io.DisplaySize.y - DISTANCE : DISTANCE);
		ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
		ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	}
	ImGui::SetNextWindowBgAlpha(0.35f); // Transparent background
	if (ImGui::Begin("RayCraft", p_open, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
	{
		ImGui::Text("Wait what is it here?\n" "This is UI in Ray Engine.\n" "Featuring RayCraft! World rendered on 2 triangles");
		ImGui::Separator();
		if (ImGui::IsMousePosValid())
			ImGui::Text("Mouse Position: (%.1f,%.1f)", io.MousePos.x, io.MousePos.y);
		else
			ImGui::Text("Mouse Position: <invalid>");
		/*if (ImGui::BeginPopupContextWindow())
		{
			if (ImGui::MenuItem("Custom", NULL, corner == -1)) corner = -1;
			if (ImGui::MenuItem("Top-left", NULL, corner == 0)) corner = 0;
			if (ImGui::MenuItem("Top-right", NULL, corner == 1)) corner = 1;
			if (ImGui::MenuItem("Bottom-left", NULL, corner == 2)) corner = 2;
			if (ImGui::MenuItem("Bottom-right", NULL, corner == 3)) corner = 3;
			if (p_open && ImGui::MenuItem("Close"))* p_open = false;
			ImGui::EndPopup();
		}*/
	}
	ImGui::End();
}


void RUserInterface::render_crosshair(bool* p_open)
{
	const float DISTANCE_X = SCR_WIDTH / 2 - 10;
	const float DISTANCE_Y = SCR_HEIGHT / 2 - 10;
	static int corner = 0;
	ImGuiIO& io = ImGui::GetIO();
	if (corner != -1)
	{
		ImVec2 window_pos = ImVec2((corner & 1) ? io.DisplaySize.x - DISTANCE_X : DISTANCE_X, (corner & 2) ? io.DisplaySize.y - DISTANCE_Y : DISTANCE_Y);
		ImVec2 window_pos_pivot = ImVec2((corner & 1) ? 1.0f : 0.0f, (corner & 2) ? 1.0f : 0.0f);
		ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	}
	ImGui::SetNextWindowBgAlpha(.7f); // Transparent background

	if (ImGui::Begin("Crosshair", p_open, (corner != -1 ? ImGuiWindowFlags_NoMove : 0) | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
	{
		ImGui::SameLine();
	}
	ImGui::End();
}

void RUserInterface::create_welcome_screen(bool *p_open)
{
	
}
