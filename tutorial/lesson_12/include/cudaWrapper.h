#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_12.h"
#include <pcl/point_cloud.h>
#include "point_types.h"
#include "CCUDAAXBSolverWrapper.h"

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	bool warmUpGPU();
	int getNumberOfAvailableThreads();

	void coutMemoryStatus();

	bool compute_projections( 	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &first_point_cloud,
						pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &second_point_cloud,
						float projections_search_radius,
						float bounding_box_extension,
						int max_number_considered_in_INNER_bucket,
						int max_number_considered_in_OUTER_bucket,
						pcl::PointCloud<lidar_pointcloud::PointProjection> &projections	);

	bool registerLS3D(	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> first_point_cloud,
						pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> second_point_cloud,
						float projections_search_radius,
						double PforGround,
						double PforObstacles,
						int max_number_considered_in_INNER_bucket,
						int max_number_considered_in_OUTER_bucket,
						float bounding_box_extension,
						CCUDA_AX_B_SolverWrapper::Solver_Method solver_method,
						cudaError_t &errCUDA,
						Eigen::Affine3f &mLS3D);

	int threads;
};

#endif
