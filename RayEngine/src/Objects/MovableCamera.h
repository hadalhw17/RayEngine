#pragma once

#include "cuda_runtime_api.h"




struct RCamera;
class RAY_ENGINE_API RMovableCamera
{
public:
	RMovableCamera();
	~RMovableCamera();
	void strafe(const double& scale);
	void set_fovx(const double& fovx);
	void change_yaw(const double& scale);
	void move_forward(const double& scale);
	void change_pitch(const double& scale);
	void rotate_right(const double& scale);
	void change_radius(const double& scale);
	void change_altitude(const double& scale);
	void set_resolution(const double& x, const double& y);
	void change_focial__distance(const double& scale);
	void change_aperture_diameter(const double& scale);


	void build_camera(RCamera& camera);

public:
	float2 fov;


	float3 position;
	float3 view_direction;
	double yaw;
	double pitch;
	double radius;
	double apertude_radius;
	double focial_distance;
private:



	void fix_yaw();
	void fix_pitch();
	void fix_radius();
	void fix_aperture_radius();
	void fix_focial_distance();
};
