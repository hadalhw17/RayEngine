#pragma once

#include "cuda_runtime_api.h"
#include "RayEngine/RayEngine.h"



struct RCamera;
class RAY_ENGINE_API RMovableCamera
{
public:
	RMovableCamera();
	~RMovableCamera();
	void strafe(const float& scale);
	void set_fovx(const float& fovx);
	void change_yaw(const float& scale);
	void move_forward(const float& scale);
	void change_pitch(const float& scale);
	void rotate_right(const float& scale);
	void change_radius(const float& scale);
	void change_altitude(const float& scale);
	void set_resolution(const float& x, const float& y);
	void change_focial__distance(const float& scale);
	void change_aperture_diameter(const float& scale);


	void build_camera(RCamera& camera);

public:
	float2 fov;


	float3 position;
	float3 view_direction;
	float yaw;
	float pitch;
	float radius;
	float apertude_radius;
	float focial_distance;
private:



	void fix_yaw();
	void fix_pitch();
	void fix_radius();
	void fix_aperture_radius();
	void fix_focial_distance();
};

namespace meta {

	template <>
	inline auto registerMembers<RMovableCamera>()
	{
		return members(
			member("fov", &RMovableCamera::fov),
			member("position", &RMovableCamera::position),
			member("view_direction", &RMovableCamera::view_direction),
			member("yaw", &RMovableCamera::yaw),
			member("pitch", &RMovableCamera::pitch),
			member("radius", &RMovableCamera::radius),
			member("apertude_radius", &RMovableCamera::apertude_radius),
			member("focial_distance", &RMovableCamera::focial_distance)
		);
	}

} // end of namespace meta