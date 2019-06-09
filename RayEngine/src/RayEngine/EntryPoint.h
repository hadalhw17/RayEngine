#pragma once

#ifdef RE_PLATFORM_WINDOWS

	extern RayEngine::Application* create_application();

	int main(int argc, char** argv)
	{
		printf("Starting client application...\n");
		auto app = create_application();
		app->Run();
		delete app;
	}
#endif // RE_PLATFORM_WINDOWS

