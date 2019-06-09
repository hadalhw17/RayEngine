#include "MovableCamera.h"

#include "cuda_runtime_api.h"
#include "helper_math.h"

#include "RayEngine/RayEngine.h"
#include "Camera.h"




	RMovableCamera::RMovableCamera()
	{
		position = make_float3(0.f, 10.f, 0.f);
		yaw = 0.f;
		pitch = 0.3f;
		radius = 0.f;
		apertude_radius = .04f;
		focial_distance = 4.f;

		fov = make_float2(60, 60);
	}


	RMovableCamera::~RMovableCamera()
	{
	}

	void RMovableCamera::strafe(const float& scale)
	{
		float3 right = cross(view_direction, make_float3(0, 10, 0));
		right = normalize(right);
		position += right * scale;
	}

	float rad_to_deg(const float& radians) {
		float degrees = radians * 180.0f / M_PI;
		return degrees;
	}

	float deg_to_rad(const float& degrees) {
		float radians = degrees / 180.0f * M_PI;
		return radians;
	}

	void RMovableCamera::set_fovx(const float& fovx)
	{
		fov.x = fovx;
		fov.y = rad_to_deg(atanf(tanf(deg_to_rad(fovx) * 0.5f) * (SCR_HEIGHT / SCR_WIDTH)) * 2.f);
	}

	void RMovableCamera::change_yaw(const float& scale)
	{
		yaw += scale;
		fix_yaw();
	}


	void RMovableCamera::move_forward(const float& scale)
	{
		position += view_direction * scale;
	}

	void RMovableCamera::change_pitch(const float& scale)
	{
		pitch += scale;
		fix_pitch();
	}

	void RMovableCamera::rotate_right(const float& scale)
	{
		float yaw2 = yaw;
		yaw2 += scale;
		float pitch2 = pitch;
		const float xDirection = sin(yaw2) * cos(pitch2);
		const float yDirection = sin(pitch2);
		const float zDirection = cos(yaw2) * cos(pitch2);
		const float3 directionToCamera = make_float3(xDirection, yDirection, zDirection);
		view_direction = directionToCamera * (-1.0f);
	}

	void RMovableCamera::change_radius(const float& scale)
	{
		radius += scale;
		fix_radius();
	}

	void RMovableCamera::change_altitude(const float& scale)
	{
		position += view_direction * scale;
	}

	void RMovableCamera::set_resolution(const float& x, const float& y)
	{
	}

	void RMovableCamera::change_focial__distance(const float& scale)
	{
		focial_distance += scale;
		fix_focial_distance();
	}

	void RMovableCamera::change_aperture_diameter(const float& scale)
	{
		apertude_radius += (apertude_radius + 0.01f) * scale; // Change proportional to current apertureRadius.
		fix_aperture_radius();
	}

	void RMovableCamera::build_camera(RCamera& camera)
	{

		float xDirection = sin(yaw) * cos(pitch);
		float yDirection = sin(pitch);
		float zDirection = cos(yaw) * cos(pitch);
		float3 directionToCamera = make_float3(xDirection, yDirection, zDirection);
		view_direction = directionToCamera * (-1.0f);
		float3 eyePosition = position + directionToCamera * radius;


		camera.campos = eyePosition;
		camera.view = view_direction;
		camera.camdown = make_float3(0, -1, 0);
		camera.fov = make_float2(fov.x, fov.y);
		camera.apertude_radius = apertude_radius;
		camera.focial_distance = focial_distance;
	}

	float mod(float x, float y)
	{
		return x - y * floorf(x / y);
	}

	float clamp2(float n, float low, float high)
	{
		n = fminf(n, high);
		n = fmaxf(n, low);
		return n;
	}

	void RMovableCamera::fix_yaw()
	{
		yaw = mod(yaw, 2.f * M_PI);
	}

	void RMovableCamera::fix_pitch()
	{
		float padding = 0.05f;
		pitch = clamp2(pitch, -M_PI_2 + padding, M_PI_2 - padding); // Limit the pitch.
	}

	void RMovableCamera::fix_radius()
	{
		float minRadius = 0.2f;
		float maxRadius = 100.0f;
		radius = clamp2(radius, minRadius, maxRadius);
	}

	void RMovableCamera::fix_aperture_radius()
	{
		float minApertureRadius = 0.0f;
		float maxApertureRadius = 25.0f;
		apertude_radius = clamp2(apertude_radius, minApertureRadius, maxApertureRadius);
	}

	void RMovableCamera::fix_focial_distance()
	{
		float minFocalDist = 0.2f;
		float maxFocalDist = 100.0f;
		focial_distance = clamp2(focial_distance, minFocalDist, maxFocalDist);
	}

