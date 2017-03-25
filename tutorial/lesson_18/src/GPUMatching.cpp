#include "GPUMatching.h"
#include "prototypes.h"

extern "C"
{
	int getIndex(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15)
	{
		return a0 | 
			a1<<1 |
			a2<<2 |
			a3<<3 |
			a4<<4 |
			a5<<5 |
			a6<<6 |
			a7<<7 |
			a8<<8 |
			a9<<9 |
			a10<<10 |
			a11<<11 |
			a12<<12 |
			a13<<13 |
			a14<<14 |
			a15<<15;
	}
}

CGPUMatching::CGPUMatching()
{
	this->d_debugLevel = 0;
	this->d_height = 0;
	this->d_width = 0;
	this->d_cudaDevice = 0;
}

CGPUMatching::~CGPUMatching()
{
}

bool CGPUMatching::Init(int _cudaDevice, int _d_width, int _d_height, int _thresholdAmountOfContiguousPoints)
{
	this->d_cudaDevice = _cudaDevice;
	this->d_width = _d_width;
	this->d_height = _d_height;

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if(deviceCount == 0)
	{	
		if(this->d_debugLevel >= 1)printf("There is no device supporting CUDA\n");
		return false;
	}else
	{
		if(this->d_debugLevel>=1)
			printf("There is %d CUDA device(s) on this machine\n", deviceCount);
	}

	if(_cudaDevice >= deviceCount)
	{
		if(this->d_debugLevel >= 1)printf("There is no CUDA device with index: %d on this machine\n", _cudaDevice);
		return false;
	}

	int runtimeVersion = 0;
	cudaRuntimeGetVersion(&runtimeVersion);
	if(this->d_debugLevel>=1)
		printf("CUDA runtime Version: %d\n", runtimeVersion); // >= 5000

	if(runtimeVersion < 5000)
	{
		if(this->d_debugLevel >= 1)printf("CUDA runtime Version has to be >= 5000\n");
		return false;
	}

	cudaGetDeviceProperties(&this->deviceProp, _cudaDevice);
	if(this->d_debugLevel>=1)
	{
		printf("Chosen GPGPU: %s\n", this->deviceProp.name);
	
		printf("totalGlobalMem %ld Gbytes\n", deviceProp.totalGlobalMem/(1024*1024*1024));
	
		printf("maxGridSize x:%d y:%d z:%d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

		printf("maxThreadsDim x:%d y:%d z:%d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

		printf("maxThreadsPerBlock: %d\n", deviceProp.maxThreadsPerBlock);
		printf("CUDA Compute Capability Major/Minor version number:  %d.%d\n", this->deviceProp.major, this->deviceProp.minor);
	}

	cudaMalloc((void **)&this->d_image, this->d_height*this->d_width*(sizeof(unsigned char)));
	cudaMalloc((void **)&this->d_integralImage, this->d_height*this->d_width*(sizeof(unsigned int)));

	cudaMalloc((void **)&this->d_keypoints, this->d_height*this->d_width*(sizeof(bool)));
	
	cudaMalloc((void **)&this->d_keypointsIndexX, this->d_height*this->d_width*(sizeof(int)));
	cudaMalloc((void **)&this->d_keypointsIndexY, this->d_height*this->d_width*(sizeof(int)));
	cudaMalloc((void **)&this->d_keypointsRotation, this->d_height*this->d_width*(sizeof(int)));

	cudaMalloc((void **)&this->d_X, 64*(sizeof(int)));
	cudaMalloc((void **)&this->d_Y, 64*(sizeof(int)));
	cudaMalloc((void **)&this->d_Z, 64*(sizeof(int)));
	cudaMalloc((void **)&this->d_n, 64*(sizeof(int)));

	cudaMalloc((void **)&this->d_isdescriptor, this->d_height*this->d_width*(sizeof(bool)));
	cudaMalloc((void **)&this->d_vdescriptor, this->d_height*this->d_width*(sizeof(bool))*256);
	cudaMalloc((void **)&this->d_vdescriptorCHAR, this->d_height*this->d_width*(sizeof(char))*32);

	
	cudaMalloc((void **)&this->d_lookup_table, 65536*(sizeof(bool)));
	cudaMalloc((void **)&this->d_rotation_lookup, 65536*sizeof(int));
	
	cudaMalloc((void **)&this->d_tempS, this->d_height*this->d_width*(sizeof(float))*64);

	this->thresholdAmountOfContiguousPoints = _thresholdAmountOfContiguousPoints;
	this->initLookupTableRot();

	cudaMemcpy(this->d_lookup_table, this->h_lookuptable, sizeof(bool)*65536, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_rotation_lookup, this->h_lookuptable_rot, sizeof(int)*65536, cudaMemcpyHostToDevice);

	if(this->d_debugLevel>=1)printf("CGPUMatching::Init ok\n");
	return true;
}

void CGPUMatching::Free()
{
	cudaFree(this->d_image);
	cudaFree(this->d_keypoints);
	cudaFree(this->d_lookup_table);
	cudaFree(this->d_rotation_lookup);
	cudaFree(this->d_integralImage);

	cudaFree(this->d_keypointsIndexX);
	cudaFree(this->d_keypointsIndexY);
	cudaFree(this->d_keypointsRotation);
	
	cudaFree(this->d_X);
	cudaFree(this->d_Y);
	cudaFree(this->d_Z);
	cudaFree(this->d_n);

	cudaFree(this->d_isdescriptor);
	cudaFree(this->d_vdescriptor);
	
	cudaFree(this->d_tempS);

	cudaFree(this->d_vdescriptorCHAR);

	if(this->d_debugLevel>=1)printf("CGPUMatching::Free ok\n");
}

void CGPUMatching::SetDebugLevel(int _debugLevel)
{
	this->d_debugLevel = _debugLevel;
return;
}

int CGPUMatching::ComputeKeyPointsRot(bool *_h_out_keypoints, unsigned char *_h_in_image, int _h_width, int _h_height, int _thresholdIntensity)
{
	int res = 0;
	if(this->d_width != _h_width || this->d_height != _h_height ) return 0;

	if(this->d_debugLevel >= 1)printf("CGPUMatching::ComputeKeyPointsRot start\n");
	
	cudaMemcpy(this->d_image, _h_in_image, sizeof(unsigned char)*_h_width*_h_height, cudaMemcpyHostToDevice);
	
	if(cuda_computeKeyPointsRot( 
		this->d_keypointsIndexX, 
		this->d_keypointsIndexY,
		this->d_keypointsRotation,
		this->d_image, 
		this->d_lookup_table, 
		this->d_rotation_lookup,
		_h_width, _h_height, _thresholdIntensity))
	{

		int *x = (int *)malloc(sizeof(int)*_h_width*_h_height);
		int *y = (int *)malloc(sizeof(int)*_h_width*_h_height);
		cudaMemcpy(x, this->d_keypointsIndexX, sizeof(int)*_h_width*_h_height, cudaMemcpyDeviceToHost);
		cudaMemcpy(y, this->d_keypointsIndexY, sizeof(int)*_h_width*_h_height, cudaMemcpyDeviceToHost);

		for(int i = 0 ; i < _h_width*_h_height; i++)_h_out_keypoints[i] = false;

		res = getKeyPointsCount();
		for(int i = 0 ; i < res; i++)
		{
			_h_out_keypoints[x[i] + y[i]*_h_width] = true;

		}
		delete x;
		delete y;
		
		if(this->d_debugLevel >= 2)printf("getKeyPointsCount(): %d\n",res);

		int counter = 0;
		for(int i = 0 ; i < _h_width*_h_height; i++)
		{
			if(_h_out_keypoints[i])counter++;
		}
		if(this->d_debugLevel >= 2)printf("number of keypoints: %d\n", counter);
	}
	else
	{
		if(this->d_debugLevel >=2)printf("error: cuda_computeKeyPoints\n");
	}

	if(this->d_debugLevel >= 1)printf("CGPUMatching::ComputeKeyPointsRot finished\n");
	return res;
}

void CGPUMatching::initLookupTableRot()
{
	unsigned short masks[16];
	masks[0] = 0;

	for (unsigned char i=0; i<this->thresholdAmountOfContiguousPoints; i++)
		masks[0] |= 1 << i;
	for (unsigned char i=1; i<16; i++)
	{
		masks[i] = masks[i-1] << 1;
		if (i > 16 - this->thresholdAmountOfContiguousPoints) masks[i]+=1;
	}

	for (int i=0; i<65536; i++)
	{
		this->h_lookuptable[i]=false;
		this->h_lookuptable_rot[i] = 0;
		for (unsigned char j=0; j<16; j++)
		{
			if (((unsigned short)i & masks[j]) == masks[j])
			{
				this->h_lookuptable[i]=true;
				this->h_lookuptable_rot[i] = get_rotation(i);
				break;
			}
		}
	}
}

int CGPUMatching::get_rot_index(int rot_mask)
{
	int index = 0;
	while(!(rot_mask & 1))
	{
		rot_mask = rot_mask >> 1;
		index++;
	}
	return index;
}

int CGPUMatching::get_rot_len(int a, int b)
{
	int len = b-a+1;
	if (len < 0)
		len += 16;
	return len;
}

int CGPUMatching::get_rotation(int i)
{
	if (i == 65535) return 0;

	unsigned short rot_mask = 1;
	int a, b, len;

	do
	{
		if (i&rot_mask)
		{
			do
			{
				if (rot_mask == 1)
					rot_mask = 32768;
				else
					rot_mask = rot_mask >> 1;
			}
			while(i&rot_mask);

			if (rot_mask == 32768)
				rot_mask = 1;
			else
				rot_mask = rot_mask << 1;
		}
		else
		{
			do
			{
				rot_mask = rot_mask << 1;
			}
			while(!(i&rot_mask));

		}

		a = get_rot_index(rot_mask);

		while(i&rot_mask)
		{
			if (rot_mask == 32768)
				rot_mask = 1;
			else
				rot_mask = rot_mask << 1;
		}
		if (rot_mask == 1)
			rot_mask = 32768;
		else
			rot_mask = rot_mask >> 1;

		b = get_rot_index(rot_mask);

		len = get_rot_len(a, b);

		if (rot_mask == 32768)
			rot_mask = 1;
		else
			rot_mask = rot_mask << 1;
	}
	while(len < this->thresholdAmountOfContiguousPoints);

	int rot = 0;
	if (a < b)
		rot = (a+b)/2;
	else
		rot = ((a+b+16)/2)%16;
	return rot;
}

bool CGPUMatching::checkiffeature(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12, int a13, int a14, int a15, int _debugLevel)
{
	int a[16] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15};
	bool res = false;
	
	int sum = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;

	if(sum >= this->thresholdAmountOfContiguousPoints)
	{
		bool found = false;
		int index = 0;
		for(int i = 0 ; i < 16; i++)
		{
			if(a[i] == 1)
			{
				index = i;
				found = true;
				break;
			}
		}
		int indexFound = index;
		if(found)
		{
			int counter = 0;
			for(int i = index; i<16; i++)
			{
				if(i == 16 || a[i] == 0)break;
				if(a[i] == 1)counter++;
			}
			int counter2 = 0;

			if(indexFound == 0 && counter < thresholdAmountOfContiguousPoints)
			{
				for(int i = 0 ; i < (16-counter) ; i++)
				{
					if(a[15 - i] == 1)counter2++;
					if(a[15 - i] == 0)break;
				}
			}
			if(counter + counter2 >= thresholdAmountOfContiguousPoints)res = true;
		}
	}
	return res;
}

void CGPUMatching::initLookupTable()
{
for(int a0 = 0 ; a0 <=1  ; a0++)
		for(int a1 = 0 ; a1 <=1  ; a1++)
			for(int a2 = 0 ; a2 <=1  ; a2++)
				for(int a3 = 0 ; a3 <=1  ; a3++)
					for(int a4 = 0 ; a4 <=1  ; a4++)
						for(int a5 = 0 ; a5 <=1  ; a5++)
							for(int a6 = 0 ; a6 <=1  ; a6++)
								for(int a7 = 0 ; a7 <=1  ; a7++)
									for(int a8 = 0 ; a8 <=1  ; a8++)
										for(int a9 = 0 ; a9 <=1  ; a9++)
											for(int a10 = 0 ; a10 <=1  ; a10++)
												for(int a11 = 0 ; a11 <=1  ; a11++)
													for(int a12 = 0 ; a12 <=1  ; a12++)
														for(int a13 = 0 ; a13 <=1  ; a13++)
															for(int a14 = 0 ; a14 <=1  ; a14++)
																for(int a15 = 0 ; a15 <=1  ; a15++)
																{
																	int sum = a0+a1+a2+a3+a4+a5+a6+a7+a8+a9+a10+a11+a12+a13+a14+a15;
																	int index = getIndex(a15,a14,a13,a12,a11,a10,a9,a8,a7,a6,a5,a4,a3,a2,a1,a0);
																	bool res = checkiffeature(a15,a14,a13,a12,a11,a10,a9,a8,a7,a6,a5,a4,a3,a2,a1,a0, 0);
																	this->h_lookuptable[index] = res;
																}
}

bool CGPUMatching::ComputeIntegralImageGPU(unsigned int *_h_out_integralImage, unsigned char *_h_in_image, int _h_width, int _h_height, int _debugMode)
{
	cudaMemcpy(this->d_image, _h_in_image, sizeof(unsigned char)*_h_width*_h_height, cudaMemcpyHostToDevice);
	bool res = cuda_computeIntegralImage(this->d_integralImage, this->d_image, _h_width, _h_height, _debugMode);

	return res;
}

bool CGPUMatching::ComputeSamplePattern(float *_X, float *_Y, float *_Z, float *_n, float _scale)
{
	float pi = 3.1415926535f;
	float r[64], phi[64];

	for(int i = 0 ; i < 64; i++)
	{
		r[i] = _scale*pow(2.0f, 2+(i%4));
		phi[i] = (float)(i)/4.0f;
		_X[i] = (r[i] * cos ((2.0f * pi *phi[i])/16.0f));
		_Y[i] = (r[i] * sin ((2.0f * pi *phi[i])/16.0f));
		_Z[i] = _scale * 8;
		_n[i] = 0;
	}
	return true;
}

bool CGPUMatching::ComputeSamplePatternGPU(float *_X, float *_Y, float *_Z, float *_n, float _scale)
{
	this->ComputeSamplePattern(_X, _Y, _Z, _n, _scale);
	
	cudaMemcpy(this->d_X, _X, sizeof(float)*64, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_Y, _Y, sizeof(float)*64, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_Z, _Z, sizeof(float)*64, cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_n, _n, sizeof(float)*64, cudaMemcpyHostToDevice);

	return true;
}

int CGPUMatching::distanceHamming(bool descriptor1[256], bool descriptor2[256])
{
	int res = 0;

	for(int i = 0 ; i < 256; i++)
	{
		if(descriptor1[i] != descriptor2[i])res++;
	}
	return res;
}

bool CGPUMatching::ComputeDescriptorGPURot(char *_h_vdesctriptor, bool *_h_isvdesctriptor,
		int _img1width, int _img1height,  int _amountofkeypoints, float _scale)
{
	computeDesctriptorCUDARot(this->d_isdescriptor, this->d_vdescriptorCHAR,
		this->d_keypointsIndexX, this->d_keypointsIndexY, this->d_keypointsRotation, _amountofkeypoints, 
		this->d_integralImage, this->d_width, this->d_height, _scale);
		
	cudaMemcpy( _h_isvdesctriptor, this->d_isdescriptor, this->d_height*this->d_width*(sizeof(bool)), cudaMemcpyDeviceToHost);
	cudaMemcpy( _h_vdesctriptor, this->d_vdescriptorCHAR, this->d_height*this->d_width*(sizeof(char)) * 32, cudaMemcpyDeviceToHost);
return true;
}

void CGPUMatching::getIndexesOfKeypoints(int *_h_keypointsIndexX, int *_h_keypointsIndexY,int _img1width, int _img1height)
{
	cudaMemcpy( _h_keypointsIndexX, this->d_keypointsIndexX, _img1width*_img1height*(sizeof(int)), cudaMemcpyDeviceToHost);
	cudaMemcpy( _h_keypointsIndexY, this->d_keypointsIndexY, _img1width*_img1height*(sizeof(int)), cudaMemcpyDeviceToHost);
}
