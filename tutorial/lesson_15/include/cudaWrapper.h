#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_15.h"
#include <pcl/point_cloud.h>
#include "point_types.h"
#include "CCUDAAXBSolverWrapper.h"

typedef struct observations_LS3D
{
	pcl::PointCloud<lidar_pointcloud::PointProjection> projections;
	Eigen::Affine3f m_pose;
	double om;
	double fi;
	double ka;
	double tx;
	double ty;
	double tz;
}observations_LS3D_t;

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	bool warmUpGPU();
	int getNumberOfAvailableThreads();

	void coutMemoryStatus();

	bool compute_projections(
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &first_point_cloud,
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &second_point_cloud,
			float projections_search_radius,
			float bounding_box_extension,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			pcl::PointCloud<lidar_pointcloud::PointProjection> &projections	);

	void Matrix4ToEuler(const double *alignxf, double *rPosTheta, double *rPos);
	void Matrix4ToEuler(Eigen::Affine3f m, Eigen::Vector3f &omfika, Eigen::Vector3f &xyz);
	void EulerToMatrix(Eigen::Vector3f omfika, Eigen::Vector3f xyz, Eigen::Affine3f &m);

	bool registerLS3D(observations_LS3D_t &obs);

	int threads;
};

#endif
