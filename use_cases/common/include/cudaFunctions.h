#ifndef __CUDA_FUNCTIONS_H__
#define __CUDA_FUNCTIONS_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#include <thrust/count.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
#include <thrust/sort.h>
//#include <thrust/sequence.h>
#include <thrust/extrema.h>

#include <Eigen/Geometry>

#include "cudaStructures.h"
#include "point_types.h"


cudaError_t cudaWarmUpGPU();
cudaError_t cudaTransformPointCloud(int threads,
								pcl::PointXYZ *d_in_point_cloud,
								pcl::PointXYZ *d_out_point_cloud,
								int number_of_points,
								float *d_matrix);

cudaError_t cudaTransformPointCloud(int threads,
								Semantic::PointXYZL *d_in_point_cloud,
								Semantic::PointXYZL *d_out_point_cloud,
								int number_of_points,
								float *d_matrix);

//Regular Grid Decomposition
cudaError_t cudaCalculateGridParams(pcl::PointXYZ* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params);
cudaError_t cudaCalculateGrid(int threads, pcl::PointXYZ* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);

cudaError_t cudaCalculateGridParams(Semantic::PointXYZNL* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params);
cudaError_t cudaCalculateGrid(int threads, Semantic::PointXYZNL* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);

cudaError_t cudaCalculateGridParams(Semantic::PointXYZL* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params);
cudaError_t cudaCalculateGrid(int threads, Semantic::PointXYZL* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);

cudaError_t cudaCalculateGridParams(velodyne_pointcloud::PointXYZIR* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params);
cudaError_t cudaCalculateGrid(int threads, velodyne_pointcloud::PointXYZIR* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);

cudaError_t cudaCalculateGridParams(velodyne_pointcloud::PointXYZIRNL * d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params);
cudaError_t cudaCalculateGrid(int threads, velodyne_pointcloud::PointXYZIRNL * d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);


//Regular Grid Decomposition 2D
cudaError_t cudaCalculateGridParams2D(pcl::PointXYZ* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float bounding_box_extension, gridParameters &out_rgd_params);
cudaError_t cudaCalculateGrid2D(int threads, pcl::PointXYZ* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params);

cudaError_t cudaCalculateNumberOfNearestNeighbors(
			int threads,
			int *nearest_neighbour_indexes,
			unsigned int number_nearest_neighbour_indexes,
			unsigned int &number_of_nearest_neighbors);

cudaError_t cudaNearestNeighborSearch(
			int threads,
			pcl::PointXYZ *d_first_point_cloud,
			int number_of_points_first_point_cloud,
			pcl::PointXYZ *d_second_point_cloud,
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
			Semantic::PointXYZL *d_first_point_cloud,
			int number_of_points_first_point_cloud,
			Semantic::PointXYZL *d_second_point_cloud,
			int number_of_points_second_point_cloud,
			hashElement *d_hashTable,
			bucket *d_buckets,
			gridParameters rgd_params,
			float search_radius,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			int *d_nearest_neighbour_indexes);

cudaError_t cudaRemoveNoise(
			int threads,
			pcl::PointXYZ *d_point_cloud,
			int number_of_points,
			hashElement *d_hashTable,
			bucket *d_buckets,
			gridParameters rgd_params,
			float search_radius,
			int number_of_points_in_search_sphere_threshold,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			bool *d_markers_out);

cudaError_t cudaRemoveNoise(
			int threads,
			velodyne_pointcloud::PointXYZIR *d_point_cloud,
			int number_of_points,
			hashElement *d_hashTable,
			bucket *d_buckets,
			gridParameters rgd_params,
			float search_radius,
			int number_of_points_in_search_sphere_threshold,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			bool *d_markers_out);

cudaError_t cudaDownSample(int threads, bool *d_markers,
		hashElement *d_hashTable, bucket *d_buckets, gridParameters rgd_params, int number_of_points);

cudaError_t cudaSemanticLabelingPlaneEdges(
		int threads,
		Semantic::PointXYZNL * d_point_cloud,
		int number_of_points,
		hashElement* d_hashTable,
		bucket* d_buckets,
		simple_point3D* d_mean,
		gridParameters rgd_params,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		float curvature_threshold,
		int number_of_points_needed_for_plane_threshold);

cudaError_t cudaSemanticLabelingPlaneEdges(
		int threads,
		velodyne_pointcloud::PointXYZIRNL * d_point_cloud,
		int number_of_points,
		hashElement* d_hashTable,
		bucket* d_buckets,
		simple_point3D* d_mean,
		gridParameters rgd_params,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		float curvature_threshold,
		int number_of_points_needed_for_plane_threshold,
		float viepointX,
		float viepointY,
		float viepointZ);

cudaError_t cudaSemanticLabelingFloorCeiling(
		int threads,
		Semantic::PointXYZNL * d_point_cloud,
		int number_of_points,
		float ground_Z_coordinate_threshold);

cudaError_t cudaSemanticLabelingFloorCeiling(
		int threads,
		velodyne_pointcloud::PointXYZIRNL * d_point_cloud,
		int number_of_points,
		float ground_Z_coordinate_threshold);


cudaError_t cudaParticleFilterPrediction(int threads,
		float *d_vangle,
		float *d_vtrans,
		float distance_above_Z,
		float *d_vmatrix,
		int number_of_particles,
		float *d_odometryIncrement,
		hashElement *d_hashTable_2D,
		bucket *d_buckets_2D,
		gridParameters rgd_params_2D,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		pcl::PointXYZ *d_ground_points_from_map,
		int number_of_points_ground_points_from_map);

cudaError_t cudaInsertPointCloudToRGD(unsigned int threads,
	Semantic::PointXYZL *d_pc,
	int number_of_points,
	char *d_rgd,
	gridParameters rgdparams);

cudaError_t cudaComputeOverlap(
	unsigned int threads,
	Semantic::PointXYZL *d_pc,
	int number_of_points,
	char *d_rgd,
	gridParameters rgd_params,
	unsigned int *d_nn,
	float &overlap);


#endif
