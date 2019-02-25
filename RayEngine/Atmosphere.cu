#include "Atmosphere.cuh"

#include "RayEngine.h"

#define M_PI 3.14156265

template <typename T>
__device__ __host__
void inline swap(T& a, T& b)
{
	T c(a); a = b; b = c;
}

