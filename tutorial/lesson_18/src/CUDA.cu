#include "prototypes.h"
#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_texture_types.h>

#define TILE_SIZE 26
#define RADIUS 3
#define BLOCK_SIZE (TILE_SIZE+(2*RADIUS))

texture<unsigned char, 1, cudaReadModeElementType> texInImage;
texture<unsigned int, 1, cudaReadModeElementType> texIntegralImage;

__device__ unsigned int keypointsCount = 0;

__global__ void
kernel_findKeyPoints_invariance(int *_d_keypointsIndexX, int *_d_keypointsIndexY, int * _d_keypointsRotation, unsigned char *_d_image, bool *_d_lookup_table, int * _d_rotation_lookup, int _h_width, int _h_height, int _thresholdIntensity)
{
	__shared__ unsigned char smem[BLOCK_SIZE*BLOCK_SIZE];
	int x = blockIdx.x*(TILE_SIZE) + threadIdx.x;
	int y = blockIdx.y*(TILE_SIZE) + threadIdx.y;
	bool isValidBlockPx = x < _h_width && y < _h_height;
	bool isValidTilePx = threadIdx.x >= RADIUS && threadIdx.x<(BLOCK_SIZE-RADIUS) &&  threadIdx.y >= RADIUS && threadIdx.y<(BLOCK_SIZE-RADIUS) && x < _h_width - RADIUS && y < _h_height - RADIUS;
	if (isValidBlockPx)
	{
		int index = y*_h_width + x;
		int sindex = threadIdx.y*blockDim.y+threadIdx.x;

		smem[sindex] = _d_image[index];

		__syncthreads();
		if(isValidTilePx)
		{
			int id[16];
			id[0]  = (threadIdx.x-1) + (threadIdx.y-3)*blockDim.y;
			id[1]  =  threadIdx.x    + (threadIdx.y-3)*blockDim.y;
			id[2]  = (threadIdx.x+1) + (threadIdx.y-3)*blockDim.y;
			id[3]  = (threadIdx.x+2) + (threadIdx.y-2)*blockDim.y;
			id[4]  = (threadIdx.x+3) + (threadIdx.y-1)*blockDim.y;
			id[5]  = (threadIdx.x+3) +  threadIdx.y   *blockDim.y;
			id[6]  = (threadIdx.x+3) + (threadIdx.y+1)*blockDim.y;
			id[7]  = (threadIdx.x+2) + (threadIdx.y+2)*blockDim.y;
			id[8]  = (threadIdx.x+1) + (threadIdx.y+3)*blockDim.y;
			id[9]  =  threadIdx.x    + (threadIdx.y+3)*blockDim.y;
			id[10] = (threadIdx.x-1) + (threadIdx.y+3)*blockDim.y;
			id[11] = (threadIdx.x-2) + (threadIdx.y+2)*blockDim.y;
			id[12] = (threadIdx.x-3) + (threadIdx.y+1)*blockDim.y;
			id[13] = (threadIdx.x-3) +  threadIdx.y   *blockDim.y;
			id[14] = (threadIdx.x-3) + (threadIdx.y-1)*blockDim.y;
			id[15] = (threadIdx.x-2) + (threadIdx.y-2)*blockDim.y;

			int id1=0, id2=0;

			for (int i=0; i<16; i++)
			{
				id1 = ( id1 << 1 ) + (smem[sindex]+_thresholdIntensity < smem[id[i]]);
				id2 = ( id2 << 1 ) + (smem[sindex]-_thresholdIntensity > smem[id[i]]);
			}
			int v;
			if (_d_lookup_table[id1])
				v = id1;
			else
				v = id2;
			if (_d_lookup_table[id1] || _d_lookup_table[id2])
			{
				unsigned int k = atomicInc(&keypointsCount, 1024*1024);
				_d_keypointsIndexX[k] = x;
				_d_keypointsIndexY[k] = y;
				_d_keypointsRotation[k] = _d_rotation_lookup[v];
			}
		}
	}
}

__global__ void
kernel_initIntegrtalImage(unsigned int *_d_out_integralImage,  int _h_width, int _h_height)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int index = bx + tx*_h_width;
	_d_out_integralImage[index] = (unsigned int)0;
}

__global__ void
kernel_scanNaiveSumVertical(unsigned int *_d_out_integralImage, unsigned char *_d_in_image, int _h_width, int _h_height)
{
	unsigned int tmp[1024];

	int tx = threadIdx.x;



	for(int i = 0 ; i < _h_width; i++)
	{
		tmp[i] = (unsigned int )tex1Dfetch(texInImage, tx* _h_width + i);
	}

	for(int i = 1; i < _h_width; i++)
	{
		tmp[i] = tmp[i-1] + tmp[i];
	}

	for(int i = 0 ; i < _h_width; i++)
	{
		_d_out_integralImage[tx* _h_width + i] = tmp[i];
	}
}

__global__ void
kernel_scanNaiveSumHirizontal(unsigned int *_d_out_integralImage, int _h_width, int _h_height)
{
	unsigned int tmp[1024];

	int tx = threadIdx.x;



	for(int i = 0 ; i < _h_height; i++)
	{
		tmp[i] = (unsigned int )tex1Dfetch(texIntegralImage, tx + i*_h_width);
	}

	for(int i = 1; i < _h_height; i++)
	{
		tmp[i] = tmp[i-1] + tmp[i];
	}

	for(int i = 0 ; i < _h_height; i++)
	{
		_d_out_integralImage[tx + i * _h_width] = tmp[i];
	}
}

__global__ void
kernel_computeDesctriptorCUDARot(bool *_d_isdescriptor, char *_d_vdescriptor,
		int *_d_keypointsIndexX, int *_d_keypointsIndexY, int *_d_keypointsRotation, int _amountofkeypoints, unsigned int *_d_integralImage, int _d_width, int _d_height, float _scale)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int index = bx + tx*_d_height;
	_d_isdescriptor[index] = false;

	if(index < _amountofkeypoints)
	{
		float S[64];
		float _X[64];
		float _Y[64];
		float _Z[64];
		float r, phi;
		float pi = 3.1415926535f;

		for(int i = 0 ; i < 64; i++)
		{
			r = _scale*pow(2.0f, 2+(i%4));
			phi = (float)(i)/4.0f;
			_X[i] = (r * cos ((2.0f * pi *phi)/16.0f));
			_Y[i] = (r * sin ((2.0f * pi *phi)/16.0f));
			_Z[i] = _scale * 8;
		}

		int _xIndex = _d_keypointsIndexX[index];
		int _yIndex = _d_keypointsIndexY[index];
		int tau = 4*_d_keypointsRotation[index];

		bool check = true;
		int index0;
		int index1;
		int index2;
		int index3;

		int _h_width = _d_width;
		int _h_height = _d_height;

		for(int i = 0 ; i < 64; i++)
		{
			if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width) < 0)check = false;
			if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width) >= _h_width*_h_height)check = false;

			if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width ) < 0)check = false;
			if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width ) >= _h_width*_h_height)check = false;

			if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width )< 0)check = false;
			if(int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width )>= _h_width*_h_height)check = false;

			if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width )< 0)check = false;
			if(int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width )>= _h_width*_h_height)check = false;

			if(check)
			{
				index0 = int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width);
				index1 = int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width );
				index2 = int(_xIndex + floor(_X[i]) + floor(_Z[i]) + (_yIndex + floor(_Y[i]) - floor(_Z[i]))*_h_width );
				index3 = int(_xIndex + floor(_X[i]) - floor(_Z[i]) + (_yIndex + floor(_Y[i]) + floor(_Z[i]))*_h_width );

				unsigned int a1 = tex1Dfetch(texIntegralImage, index0);
				unsigned int a2 = tex1Dfetch(texIntegralImage, index1);
				unsigned int a3 = tex1Dfetch(texIntegralImage, index2);
				unsigned int a4 = tex1Dfetch(texIntegralImage, index3);

				S[i] = float(a1+a2-a3-a4);
			}
		}

		if(check)
		{
			_d_isdescriptor[index] = true;
			bool desc[256];

			for(int i = 0; i< 64; i++)
			{
				int id = (i+tau)%64;
				int index0 = (id + 8)%64;
				int index1 = (id + 24)%64;
				int	index2 = (id + 36)%64;
				int index3 = int((4.0f * id/4.0f  + 4.0f + (3 - (id%4))))%64;

				if(S[id] < S[index0])
				{
					desc[i * 4] = true;
				}else
				{
					desc[i * 4] = false;
				}

				if(S[id] < S[index1])
				{
					desc[i * 4 + 1] = true;
				}else
				{
					desc[i * 4 + 1] = false;
				}

				if(S[id] < S[index2])
				{
					desc[i * 4 + 2] = true;
				}else
				{
					desc[i * 4 + 2] = false;
				}

				if(S[id] < S[index3])
				{
					desc[i * 4 + 3] = true;
				}else
				{
					desc[i * 4 + 3] = false;
				}
			}

			for(int i = 0 ; i < 32; i++)
			{
				char wynik = 0;
				for(int j = 0; j < 8; j++)
				{
					wynik += (desc[i * 8 + j] * (1 << j));
				}
				_d_vdescriptor[index*32 + i]=wynik;
			}
		}
	}
}

cudaError_t setKeyPointsCount(int retCnt) {
	return cudaMemcpyToSymbol(keypointsCount, &retCnt, sizeof(unsigned int), 0, cudaMemcpyHostToDevice);
}

unsigned int getKeyPointsCount()
{
	unsigned int value;
	cudaMemcpyFromSymbol(&value, keypointsCount, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
	return value;
}

int funkcja_cuda()
{
return 0;
}

bool cuda_computeKeyPointsRot(int *_d_keypointsIndexX, int *_d_keypointsIndexY, int *_d_keypointsRotation,
							  unsigned char *_d_image, bool *_d_lookup_table, int * _d_rotation_lookup, int _h_width, int _h_height, int _thresholdIntensity)
{
	if(_h_width > 1024 || _h_height > 1024)
		return false;

	setKeyPointsCount(0);
	
	dim3 threads(32, 32);

	int blocks_x = std::ceil((float)_h_width/(float)(threads.x-6));
	int blocks_y = std::ceil((float)_h_height/(float)(threads.y-6));
	dim3 blocks(blocks_x, blocks_y);
		kernel_findKeyPoints_invariance<<<blocks, threads>>>(_d_keypointsIndexX, _d_keypointsIndexY, _d_keypointsRotation, _d_image, _d_lookup_table, _d_rotation_lookup, _h_width, _h_height, _thresholdIntensity);

	return true;
}

bool cuda_computeIntegralImage(unsigned int *_d_out_integralImage, unsigned char *_d_in_image, int _h_width, int _h_height, int _debugMode)
{
	if(_h_width > 1024 || _h_height > 1024)return false;

	cudaBindTexture(NULL, texInImage, _d_in_image);

	kernel_initIntegrtalImage<<<_h_width, _h_height>>>(_d_out_integralImage,  _h_width, _h_height);

	kernel_scanNaiveSumVertical<<<1, _h_height>>>(_d_out_integralImage, _d_in_image, _h_width, _h_height);

	cudaBindTexture(NULL, texIntegralImage, _d_out_integralImage);
	kernel_scanNaiveSumHirizontal<<<1, _h_width>>>(_d_out_integralImage, _h_width, _h_height);

	return true;
}

bool computeDesctriptorCUDARot(bool *_d_isdescriptor, char *_d_vdescriptor,
		int *_d_keypointsIndexX, int *_d_keypointsIndexY, int *_d_keypointsRotation, int _amountofkeypoints, unsigned int *_d_integralImage, int _d_width, int _d_height, float _scale)
{
	cudaBindTexture(NULL, texIntegralImage, _d_integralImage);
	kernel_computeDesctriptorCUDARot<<<_d_width, _d_height>>>(_d_isdescriptor, _d_vdescriptor,
		_d_keypointsIndexX, _d_keypointsIndexY, _d_keypointsRotation, _amountofkeypoints, _d_integralImage, _d_width, _d_height, _scale);

	return true;
}
