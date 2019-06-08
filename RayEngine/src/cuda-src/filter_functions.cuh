//#pragma once
//#include "RayEngine.h"
//#include "helper_math.h"
//
//
//
//__device__
//float4 bilinear_filter(HitResult primary_hit_results, float3* texture)
//{
//	float u = (primary_hit_results.uv.x * 1000.f) - 0.5f;
//	float v = (primary_hit_results.uv.y * 1000.f) - 0.5f;
//
//	int x = floor(u);
//	int y = floor(v);
//	int xy = x + 1000 * y;
//	int x1y = (x + 1) + 1000 * y;
//	int xy1 = x + 1000 * (y + 1);
//	int x1y1 = (x + 1) + 1000 * (y + 1);
//
//	float u_ratio = u - (float)x;
//	float v_ratio = v - (float)y;
//	float u_opposite = 1.f - u_ratio;
//	float v_opposite = 1.f - v_ratio;
//	float3 result = (texture[xy] * u_opposite + texture[x1y] * u_ratio) * v_opposite +
//		(texture[xy1] * u_opposite + texture[x1y1] * u_ratio) * v_ratio;
//	return make_float4(result);
//}
//
//__device__
//float4 nearest_neightbour_filter(HitResult primary_hit_results, float3 * texture)
//{
//	int tx = (int)(1000 * primary_hit_results.uv.x), ty = (int)(1000 * primary_hit_results.uv.y);
//	float3* _texture = (1000 * ty + tx) + texture;
//
//	return make_float4(*_texture);
//}