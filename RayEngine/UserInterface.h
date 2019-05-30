#pragma once
#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"

class RUserInterface
{

public:
	void inti_UI(GLFWwindow* window);
	void clean_UI();

	void render_main_hud();
	void render_main_screen(bool* start_game);

private:
	static void ShowExampleAppSimpleOverlay(bool* p_open);
	static void render_crosshair(bool* p_open);
	static void create_welcome_screen(bool* p_open);

	bool show_demo_window = true;
	bool show_another_window = true;
	ImVec4 clear_color;
};

