#include "lesson_13.h"

__global__ void kernel_cudaWarmUpGPU()
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	ind = ind + 1;
}

cudaError_t cudaWarmUpGPU()
{
	kernel_cudaWarmUpGPU<<<1,1>>>();
	cudaDeviceSynchronize();
	return cudaGetLastError(); 
}

__global__ void kernel_cudaCompute_AtP(double *d_A, double *d_P, double *d_AtP, int rows, int columns )
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<rows*columns)
	{
		int row = ind%rows;
		int column = ind/rows;

		d_AtP[row + column * rows] = d_A[column + row * columns] * d_P[column];
	}
}

cudaError_t cudaCompute_AtP(int threads, double *d_A, double *d_P, double *d_AtP, int rows, int columns)
{
	cudaError_t err = ::cudaSuccess;

	kernel_cudaCompute_AtP<<<(rows*columns)/threads+1,threads>>>(d_A, d_P, d_AtP, rows, columns);

	err = cudaDeviceSynchronize();
	return err;
}

