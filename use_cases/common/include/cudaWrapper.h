#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "cudaFunctions.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

//common
#include "point_types.h"

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void setDevice(int cudaDevice);
	bool warmUpGPU();
	void printCUDAinfo(int _device_id);
	int getNumberOfAvailableThreads();
	bool getNumberOfAvailableThreads(int &threads, int &threadsNV);

	bool transformPointCloud(pcl::PointXYZ *d_in_point_cloud,
				   pcl::PointXYZ *d_out_point_cloud,
				   int number_of_points,
				   Eigen::Affine3f matrix);

	bool nearestNeighbourhoodSearch(
				pcl::PointXYZ *d_first_point_cloud,
				int number_of_points_first_point_cloud,
				pcl::PointXYZ *d_second_point_cloud,
				int number_of_points_second_point_cloud,
				float search_radius,
				float bounding_box_extension,
				int max_number_considered_in_INNER_bucket,
				int max_number_considered_in_OUTER_bucket,
				int *d_nearest_neighbour_indexes);

	bool removeNoise(
				pcl::PointCloud<pcl::PointXYZ> &point_cloud,
				float search_radius,
				float bounding_box_extension,
				int number_of_points_in_search_sphere_threshold,
				int max_number_considered_in_INNER_bucket,
				int max_number_considered_in_OUTER_bucket);

	bool removeNoise(
				pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &point_cloud,
				float search_radius,
				float bounding_box_extension,
				int number_of_points_in_search_sphere_threshold,
				int max_number_considered_in_INNER_bucket,
				int max_number_considered_in_OUTER_bucket);

	bool downsampling(pcl::PointCloud<pcl::PointXYZ> &point_cloud, float resolution);
	bool downsampling(pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &point_cloud, float resolution);
	bool downsampling(pcl::PointCloud<Semantic::PointXYZL> &point_cloud, float resolution);


	bool classify(  pcl::PointCloud<Semantic::PointXYZNL> &point_cloud,
						int number_of_points,
						float normal_vectors_search_radius,
						float curvature_threshold,
						float ground_Z_coordinate_threshold,
						int number_of_points_needed_for_plane_threshold,
						int max_number_considered_in_INNER_bucket,
						int max_number_considered_in_OUTER_bucket  );

	bool classify(  pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> &point_cloud,
							int number_of_points,
							float normal_vectors_search_radius,
							float curvature_threshold,
							float ground_Z_coordinate_threshold,
							int number_of_points_needed_for_plane_threshold,
							int max_number_considered_in_INNER_bucket,
							int max_number_considered_in_OUTER_bucket,
							float viepointX,
							float viepointY,
							float viepointZ);

private:
	int cudaDevice;
	int threads;
};



#endif
