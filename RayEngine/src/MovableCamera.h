#pragma once

#include "cuda_runtime_api.h"

struct RCamera;


class RMovableCamera
{
public:
	RMovableCamera();
	~RMovableCamera();
	void strafe(const float &scale);
	void set_fovx(const float &fovx);
	void change_yaw(const float &scale);
	void move_forward(const float &scale);
	void change_pitch(const float &scale);
	void rotate_right(const float &scale);
	void change_radius(const float &scale);
	void change_altitude(const float &scale);
	void set_resolution(const float &x, const float &y);
	void change_focial__distance(const float &scale);
	void change_aperture_diameter(const float &scale);

	
	void build_camera(RCamera &camera);

public:
	float2 fov;
	

	float3 position;
	float3 view_direction;
	float yaw;
	float pitch;
private:


	float radius;
	float apertude_radius;
	float focial_distance;

	void fix_yaw();
	void fix_pitch();
	void fix_radius();
	void fix_aperture_radius();
	void fix_focial_distance();
};

