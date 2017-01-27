#ifndef __CUDA_STRUCTURES_H__
#define __CUDA_STRUCTURES_H__

#include <cuda.h>
#include <cuda_runtime.h>

struct hashElement{
	int index_of_point;
	int index_of_bucket;
};

bool compareHashElements(const hashElement& a, const hashElement& b);

struct bucket{
	int index_begin;
	int index_end;
	int number_of_points;
};

struct gridParameters{
	float bounding_box_min_X;
	float bounding_box_min_Y;
	float bounding_box_min_Z;
	float bounding_box_max_X;
	float bounding_box_max_Y;
	float bounding_box_max_Z;
	int number_of_buckets_X;
	int number_of_buckets_Y;
	int number_of_buckets_Z;
	long long int number_of_buckets;
	float resolution_X;
	float resolution_Y;
	float resolution_Z;
};

struct simple_point3D{
	float x;
	float y;
	float z;
};

struct minmax{
	double maxX;
	double maxY;
	double maxZ;
	double minX;
	double minY;
	double minZ;
};

#endif
