#pragma once

#include "cuda_runtime_api.h"

struct RCamera;


class RMovableCamera
{
public:
	RMovableCamera();
	~RMovableCamera();
	void strafe(float scale);
	void set_fovx(float fovx);
	void change_yaw(float scale);
	void move_forward(float scale);
	void change_pitch(float scale);
	void rotate_right(float scale);
	void change_radius(float scale);
	void change_altitude(float scale);
	void set_resolution(float x, float y);
	void change_focial__distance(float scale);
	void change_aperture_diameter(float scale);

	
	void build_camera(RCamera *camera);

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

