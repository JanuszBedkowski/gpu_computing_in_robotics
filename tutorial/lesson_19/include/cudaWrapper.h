#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_19.h"
#include <vector>

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	void getNumberOfAvailableThreads();
	bool copyPlaneDataToGPU(std::vector<plane> &_vPlanes);

	void computeDistance(laser_beam &_single_laser_beam);

	int threads;

	plane *d_vPlanes;
	int number_of_planes;

	float *d_distance;
};



#endif
