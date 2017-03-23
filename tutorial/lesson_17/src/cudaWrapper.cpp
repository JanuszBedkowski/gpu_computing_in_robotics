#include "cudaWrapper.h"


CCudaWrapper::CCudaWrapper()
{

}

CCudaWrapper::~CCudaWrapper()
{

}

void CCudaWrapper::warmUpGPU()
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return;

	err = cudaWarmUpGPU();
		if(err != ::cudaSuccess)return;

}

void CCudaWrapper::findMax(int tx, int bx, int &newIndexX, int &newIndexY, double *h_dyf, int sizeofmap1dim)
{
	if(tx > 0 && tx < sizeofmap1dim && bx > 0 && bx < sizeofmap1dim)
	{
		int index1 = (tx-1) + (bx+1) * sizeofmap1dim;
		int index2 = (tx)   + (bx+1) * sizeofmap1dim;
		int index3 = (tx+1) + (bx+1) * sizeofmap1dim;

		int index4 = (tx-1) + (bx) * sizeofmap1dim;
		int index6 = (tx+1) + (bx) * sizeofmap1dim;

		int index7 = (tx-1) + (bx-1) * sizeofmap1dim;
		int index8 = (tx)   + (bx-1) * sizeofmap1dim;
		int index9 = (tx+1) + (bx-1) * sizeofmap1dim;

		double max = h_dyf[index1];
		newIndexX = tx-1;
		newIndexY = bx+1;

		if(h_dyf[index2] > max)
		{
		max = h_dyf[index2];
		newIndexX = tx;
		newIndexY = bx+1;
		}

		if(h_dyf[index3] > max)
		{
		max = h_dyf[index3];
		newIndexX = tx+1;
		newIndexY = bx+1;
		}

		if(h_dyf[index4] > max)
		{
		max = h_dyf[index4];
		newIndexX = tx-1;
		newIndexY = bx;
		}

		if(h_dyf[index6] > max)
		{
		max = h_dyf[index6];
		newIndexX = tx+1;
		newIndexY = bx;
		}

		if(h_dyf[index7] > max)
		{
		max = h_dyf[index7];
		newIndexX = tx-1;
		newIndexY = bx-1;
		}

		if(h_dyf[index8] > max)
		{
		max = h_dyf[index8];
		newIndexX = tx;
		newIndexY = bx-1;
		}

		if(h_dyf[index9] > max)
		{
		max = h_dyf[index9];
		newIndexX = tx+1;
		newIndexY = bx-1;
		}
	}else
	{
		newIndexX = 0;
		newIndexY = 0;
	}
}


int CCudaWrapper::computePath(bool *map2D, int sizeofmap1dim, int xgoal, int ygoal, int robotXpos, int robotYpos,
	 int steps, char *_my_grid, int maxpathlength, int *path_x, int *path_y)
{
	unsigned int size = sizeofmap1dim * sizeofmap1dim;
	unsigned int mem_size_double = sizeof(double) * size;

	double* h_dyf = (double*) malloc(mem_size_double);
	double* h_u = (double*) malloc(mem_size_double);
	double* h_diff = (double*) malloc(mem_size_double);

	for(int i = 0 ; i < sizeofmap1dim; i++)
	{
		for(int j = 0 ; j < sizeofmap1dim; j++)
		{
			h_dyf[j + i * sizeofmap1dim] = 0.0f;
			h_u[j + i * sizeofmap1dim] = 0.0f;
		}
	}

	for(int i = 0 ; i < sizeofmap1dim*sizeofmap1dim; i++)
	{
		if(map2D[i]==1)h_u[i]=1000.0f;
	}

	double L=1000.0f;

	h_dyf[xgoal + ygoal * sizeofmap1dim ]=L;

	cudaPathPlanning(h_dyf, h_u, h_diff, robotXpos, robotYpos, xgoal, ygoal, mem_size_double, sizeofmap1dim, steps);

	double max = 0.0f;
	double min = 1000000000.0f;
	for(int i = 0 ; i < sizeofmap1dim; i++)
	{
		for(int j = 0 ; j < sizeofmap1dim; j++)
		{
			if(h_dyf[j + i * sizeofmap1dim] > max)max = h_dyf[j + i * sizeofmap1dim];
			if(h_dyf[j + i * sizeofmap1dim] < min && h_dyf[j + i * sizeofmap1dim]!=0)min = h_dyf[j + i * sizeofmap1dim];
		}
	}

	double interval = (max - min);

	for(int i = 0 ; i < sizeofmap1dim; i++)
	{
		for(int j = 0 ; j < sizeofmap1dim; j++)
		{
			if(h_dyf[j + i * sizeofmap1dim] > 0)
			{
				_my_grid[j + i * sizeofmap1dim] =  (unsigned char)(((h_dyf[j + i * sizeofmap1dim] - min)/interval) * 255);
			}else
			{
				_my_grid[j + i * sizeofmap1dim] = 0;
			}
		}
	}

	int indexX = robotXpos;
	int indexY = robotYpos;
	int newIndexX;
	int newIndexY;

	int counter =0;

	for(int i = 0 ; i < maxpathlength; i++)
	{
		int ind = indexX + indexY * sizeofmap1dim;
		if(i%2 ==0)	_my_grid[ind] = 0;else _my_grid[ind] = 255;
		findMax(indexX, indexY, newIndexX, newIndexY, h_dyf, sizeofmap1dim);

		if(newIndexX != xgoal || newIndexY!=ygoal)
		{
			indexX = newIndexX;
			indexY = newIndexY;
			path_x[counter]=indexX;
			path_y[counter]=indexY;
			counter++;
		}else
		{
			break;
		}
	}
	free(h_dyf);
	free(h_u);
	free(h_diff);
	return counter;
}
