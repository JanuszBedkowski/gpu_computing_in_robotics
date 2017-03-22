#ifndef __LESSON_16_H__
#define __LESSON_16_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>
#include "point_types.h"

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

struct simple_point3D{
	float x;
	float y;
	float z;
};

typedef struct obs_nn
{
	float x_diff;
	float y_diff;
	float z_diff;
	float x0;
	float y0;
	float z0;
}obs_nn_t;

cudaError_t cudaWarmUpGPU();

cudaError_t cudaCalculateGridParams(lidar_pointcloud::PointXYZIRNL* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params);

cudaError_t cudaCalculateGrid(int threads, lidar_pointcloud::PointXYZIRNL* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);

cudaError_t cudaCompute_AtP(int threads, double *d_A, double *d_P, double *d_AtP, int rows, int columns);

cudaError_t fill_A_l_cuda(int threads, double *d_A, double x, double y, double z, double om, double fi, double ka,
		obs_nn_t *d_obs_nn, int nop, double *d_P, double *d_l);

cudaError_t cudaNearestNeighborSearch(
		int threads,
		lidar_pointcloud::PointXYZIRNL *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		lidar_pointcloud::PointXYZIRNL *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		hashElement *d_hashTable,
		bucket *d_buckets,
		gridParameters rgd_params,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		int *d_nearest_neighbour_indexes);

cudaError_t cudaSemanticNearestNeighborSearch(
		int threads,
		lidar_pointcloud::PointXYZIRNL *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		lidar_pointcloud::PointXYZIRNL *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		hashElement *d_hashTable,
		bucket *d_buckets,
		gridParameters rgd_params,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		int *d_nearest_neighbour_indexes);
#endif
