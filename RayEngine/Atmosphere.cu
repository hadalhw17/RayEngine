#include "Atmosphere.cuh"

#include "RayEngine.h"


template <typename T>
__device__ __host__
void inline swap(T& a, T& b)
{
	T c(a); a = b; b = c;
}

