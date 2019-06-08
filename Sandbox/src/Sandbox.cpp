#include <RayCore.h>

class Sandbox : public RayEngine::MainWindow
{
public:
	Sandbox()
	{

	}

	~Sandbox()
	{

	}

};


RayEngine::MainWindow* create_application()
{
	return new Sandbox();
}