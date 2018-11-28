#include <iostream>
#include <math.h>

//function to add elements of 2 arrays
__global__
void add(int n, float *x, float *y)
{
	int index = treadID.x;
	int stride = blockDim.x;
	for (int i = index; i < n; i+=stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	//1M elements
	int N = 1 << 20;

	float *x, float *y;

	//Allocate Unified memory(accessable from GPU or CPU)
	cudaMallocManaged(&x, N * sizeof(float));
	cudaMallocManaged(&y, N * sizeof(float));


	//Initialize x and y arrays on the host
	for (int i = 0; i < N; i++)
	{
		x[i] = 1.f;
		y[i] = 2.f;
	}

	//Run kernel of GPU
	add<<<1, 256>>>(N, x, y);

	//Wait for GPU to finish before moving to host
	cudaDeviceSynchronize();

	//Free memory
	cudaFree(x);
	cudaFree(y);

	return 0;
}

