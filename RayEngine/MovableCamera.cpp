#include "MovableCamera.h"

#include "cuda_runtime_api.h"
#include "cutil_math.h"

#include "RayEngine.h"
#include "Camera.h"


RMovableCamera::RMovableCamera()
{
	position = make_float3(0,1,0);
	yaw = 0;
	pitch = 0.3;
	radius = 4;
	apertude_radius = .04f;
	focial_distance = 4.f;

	fov = make_float2(45, 40);
}


RMovableCamera::~RMovableCamera()
{
}

void RMovableCamera::strafe(float scale)
{
	float3 right = cross(view_direction, make_float3(0, 10, 0));
	right = normalize(right);
	position += right * scale;
}

float rad_to_deg(float radians) {
	float degrees = radians * 180.0f / M_PI;
	return degrees;
}

float deg_to_rad(float degrees) {
	float radians = degrees / 180.0f * M_PI;
	return radians;
}

void RMovableCamera::set_fovx(float fovx)
{
	fov.x = fovx;
	fov.y = rad_to_deg(atanf(tanf(deg_to_rad(fovx) * 0.5f) * (SCR_HEIGHT / SCR_WIDTH)) * 2.f);
}

void RMovableCamera::change_yaw(float scale)
{
	yaw += scale;
	fix_yaw();
}


void RMovableCamera::move_forward(float scale)
{
	position += view_direction * scale;
}

void RMovableCamera::change_pitch(float scale)
{
	pitch += scale;
	fix_pitch();
}

void RMovableCamera::rotate_right(float scale)
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

void RMovableCamera::change_radius(float scale)
{
	radius += scale;
	fix_radius();
}

void RMovableCamera::change_altitude(float scale)
{
	position += view_direction * scale;
}

void RMovableCamera::set_resolution(float x, float y)
{
}

void RMovableCamera::change_focial__distance(float scale)
{
	focial_distance += scale;
	fix_focial_distance();
}

void RMovableCamera::change_aperture_diameter(float scale)
{
	apertude_radius += (apertude_radius + 0.01f) * scale; // Change proportional to current apertureRadius.
	fix_aperture_radius();
}

void RMovableCamera::build_camera(RCamera * camera)
{

	float xDirection = sin(yaw) * cos(pitch);
	float yDirection = sin(pitch);
	float zDirection = cos(yaw) * cos(pitch);
	float3 directionToCamera = make_float3(xDirection, yDirection, zDirection);
	view_direction = directionToCamera * (-1.0f);
	float3 eyePosition = position + directionToCamera * radius;
	//Vec3f eyePosition = centerPosition; // rotate camera from stationary viewpoint


	camera->campos = eyePosition;
	camera->view = view_direction;
	camera->camdown = make_float3(0,-1, 0);
	camera->fov = make_float2(fov.x, fov.y);
	camera->apertude_radius = apertude_radius;
	camera->focial_distance = focial_distance;
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
	pitch = clamp2(pitch, -PI_OVER_TWO + padding, PI_OVER_TWO - padding); // Limit the pitch.
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
