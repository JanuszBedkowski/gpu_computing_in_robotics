#include "lesson_17.h"

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

typedef struct
{
	int counter[1024];
	int myCounter;
	int robotPosX;
	int robotPosY;
	int xg;
	int yg;

	double tempD;
	bool stop;
} __SharedData;

__SharedData __device__ SharedData;

__global__ void
kernel_InitRobotPos(int *d_robotPos, int sizeofmap1dim)
{
	 SharedData.robotPosX = d_robotPos[0];
	 SharedData.robotPosY = d_robotPos[1];
	 SharedData.xg = d_robotPos[2];
	 SharedData.yg = d_robotPos[3];

	 SharedData.tempD = 0.0f;
	 SharedData.stop = false;
}

__global__ void
kernelC(double *_dyf, double *_u,double *_dyfB, int sizeofmap1dim)
{
	double ap = 0.14f;
	double a = 0.1f;

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int index = tx + bx * sizeofmap1dim;

	int index1 = (tx-1) + (bx+1) * sizeofmap1dim;
	int index2 = (tx)   + (bx+1) * sizeofmap1dim;
	int index3 = (tx+1) + (bx+1) * sizeofmap1dim;

	int index4 = (tx-1) + (bx) * sizeofmap1dim;
	int index6 = (tx+1) + (bx) * sizeofmap1dim;

	int index7 = (tx-1) + (bx-1) * sizeofmap1dim;
	int index8 = (tx)   + (bx-1) * sizeofmap1dim;
	int index9 = (tx+1) + (bx-1) * sizeofmap1dim;

	double a1 = 0.0f, a2 = 0.0f ,a3 = 0.0f, a4 = 0.0f , a6 = 0.0f ,a7 = 0.0f ,a8 = 0.0f , a9 = 0.0f;
	double u = _u[index];
	double pom1, pom2, pom3, pom4, pom6, pom7, pom8, pom9;
	double max = 0.0f;


	if( tx > 0 && tx < (sizeofmap1dim-1) && bx > 0 && bx < (sizeofmap1dim-1) )
		{
			 //1 2 3
			 //4 5 6
			 //7 8 9
			 a1 = _dyf[index1];
			 a2 = _dyf[index2];
			 a3 = _dyf[index3];
			 a4 = _dyf[index4];
			 a6 = _dyf[index6];
			 a7 = _dyf[index7];
			 a8 = _dyf[index8];
			 a9 = _dyf[index9];
			 __syncthreads();

			 pom1 = a1 - ap - u;
			 if(pom1<0.0f)pom1 = 0.0f;
			 if(pom1 > max)max = pom1;

			 pom2 = a2 - a - u;
			 if(pom2<0.0f)pom2 = 0.0f;
			 if(pom2 > max)max = pom2;

			 pom3 = a3 - ap - u;
			 if(pom3<0.0f)pom3 = 0.0f;
			 if(pom3 > max)max = pom3;

			 pom4 = a4 - a - u;
			 if(pom4<0.0f)pom4 = 0.0f;
			 if(pom4 > max)max = pom4;

			 pom6 = a6 - a - u;
			 if(pom6<0.0f)pom6 = 0.0f;
			 if(pom6 > max)max = pom6;

			 pom7 = a7 - ap - u;
			 if(pom7<0.0f)pom7 = 0.0f;
			 if(pom7 > max)max = pom7;

			 pom8 = a8 - a - u;
			 if(pom8<0.0f)pom8 = 0.0f;
			 if(pom8 > max)max = pom8;

			 pom9 = a9 - ap - u;
			 if(pom9<0.0f)pom9 = 0.0f;
			 if(pom9 > max)max = pom9;
		}

		bool tempStop = SharedData.stop;
		int ind = SharedData.xg + SharedData.yg * sizeofmap1dim;
		__syncthreads();

		if(!tempStop)
		{
			if(ind!=index)
			_dyfB[index] = max;
		}
}

__global__ void
kernel_Check(double *d_dyf, int sizeofmap1dim)
{
	int index = SharedData.robotPosX + SharedData.robotPosY * sizeofmap1dim;
	if(d_dyf[index] != SharedData.tempD)SharedData.stop = true;
}

cudaError_t cudaPathPlanning(double *_h_dyf, double *_h_u, double *_h_diff,  int _robotX, int _robotY, int _xgoal, int _ygoal, int mem_size_double, int sizeofmap1dim, int steps)
{
	int *d_robotPos;(cudaMalloc((void**) &d_robotPos, sizeof(int)*4));
	int* h_robotPos = (int*) malloc(sizeof(int)*4);
	h_robotPos[0] = _robotX;
	h_robotPos[1] = _robotY;
	h_robotPos[2] = _xgoal;
	h_robotPos[3] = _ygoal;

	(cudaMemcpy(d_robotPos, h_robotPos, sizeof(int)*4, cudaMemcpyHostToDevice) );

	double* d_dyf; (cudaMalloc((void**) &d_dyf, mem_size_double));
	double* d_u; (cudaMalloc((void**) &d_u, mem_size_double));
	double* d_dyfB; (cudaMalloc((void**) &d_dyfB, mem_size_double));

	(cudaMemcpy(d_dyf, _h_dyf, mem_size_double, cudaMemcpyHostToDevice) );
	(cudaMemcpy(d_u, _h_u, mem_size_double, cudaMemcpyHostToDevice) );

	dim3 threads(sizeofmap1dim);
	dim3 grid(sizeofmap1dim);

	kernel_InitRobotPos<<<1, 1>>>(d_robotPos, sizeofmap1dim);

	for(int i = 0 ; i < steps ; i++)
	{
		kernelC<<<grid, threads>>>(d_dyf, d_u, d_dyfB, sizeofmap1dim);
		kernelC<<<grid, threads>>>(d_dyfB, d_u, d_dyf, sizeofmap1dim);
		if(i%10 == 0)kernel_Check<<<1, 1>>>(d_dyf,sizeofmap1dim);
	}

	(cudaMemcpy(_h_dyf, d_dyf, mem_size_double, cudaMemcpyDeviceToHost) );
	(cudaMemcpy(_h_u, d_u, mem_size_double, cudaMemcpyDeviceToHost) );

	cudaFree(d_robotPos);
	cudaFree(d_dyf);
	cudaFree(d_u);
	cudaFree(d_dyfB);

	free(h_robotPos);

	return cudaGetLastError();
}



