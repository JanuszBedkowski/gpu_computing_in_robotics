#ifndef __LESSON_1_H__
#define __LESSON_1_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>

struct hashElement{
	int index_of_point;
	int index_of_bucket;
};

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

cudaError_t cudaCalculateGridParams(pcl::PointXYZ* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, gridParameters &out_rgd_params);

cudaError_t cudaCalculateGrid(int threads, pcl::PointXYZ* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);

cudaError_t cudaDownSample(int threads, bool *d_markers,
		hashElement *d_hashTable, bucket *d_buckets, gridParameters rgd_params, int number_of_points);

cudaError_t cudaWarmUpGPU();
#endif
