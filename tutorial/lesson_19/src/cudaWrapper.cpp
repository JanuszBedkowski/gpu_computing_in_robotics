#include "cudaWrapper.h"
#include <iostream>

CCudaWrapper::CCudaWrapper()
{
	this->threads = 0;
	this->d_vPlanes = NULL;
	this->d_distance = NULL;
}

CCudaWrapper::~CCudaWrapper()
{
	cudaFree(this->d_vPlanes);this->d_vPlanes = NULL;
	cudaFree(this->d_distance);this->d_distance = NULL;
}

void CCudaWrapper::warmUpGPU()
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return;

	err = cudaWarmUpGPU();
		if(err != ::cudaSuccess)return;

	getNumberOfAvailableThreads();
}

void CCudaWrapper::getNumberOfAvailableThreads()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	if(prop.major == 2)
	{
		this->threads=prop.maxThreadsPerBlock/2;
	}else if(prop.major > 2)
	{
		this->threads=prop.maxThreadsPerBlock;
	}else
	{
		this->threads=0;
		return;
	}

	return;
}

bool CCudaWrapper::copyPlaneDataToGPU(std::vector<plane> &_vPlanes)
{
	if(cudaMalloc((void**)&this->d_vPlanes, _vPlanes.size()*sizeof(plane)) != ::cudaSuccess)return false;

	if(cudaMemcpy(this->d_vPlanes, _vPlanes.data(),	_vPlanes.size()*sizeof(plane), cudaMemcpyHostToDevice) != ::cudaSuccess)return false;

	this->number_of_planes = (int)_vPlanes.size();

	if(cudaMalloc((void**)&this->d_distance, _vPlanes.size()* sizeof(float)) != ::cudaSuccess)return false;

return true;
}

void CCudaWrapper::computeDistance(laser_beam &_single_laser_beam)
{
	_single_laser_beam.distance = _single_laser_beam.range;
	cudaComputeDistance(this->threads, _single_laser_beam, this->d_vPlanes, this->number_of_planes, this->d_distance);
}




