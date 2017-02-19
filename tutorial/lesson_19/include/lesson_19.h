#ifndef __LESSON_19_H__
#define __LESSON_19_H__

#include <cuda_runtime.h>

#define TOLERANCE 0.01

struct simple_point3D{
	float x;
	float y;
	float z;
};

struct triangle{
	simple_point3D vertexA;
	simple_point3D vertexB;
	simple_point3D vertexC;
};

struct plane{
	float A;
	float B;
	float C;
	float D;
	triangle polygon;
};

struct laser_beam{
	simple_point3D position;
	simple_point3D direction;
	float distance;
	float range;
};

cudaError_t cudaWarmUpGPU();
cudaError_t cudaComputeDistance(int threads, laser_beam &_single_laser_beam, plane *d_vPlanes, int number_of_planes, float *d_distance);
#endif
