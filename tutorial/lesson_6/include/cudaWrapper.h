#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_6.h"
#include <pcl/point_cloud.h>

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();
	bool getNumberOfAvailableThreads(int &threads, int &threadsNV);
	void coutMemoryStatus();

	bool projections( 	pcl::PointCloud<pcl::PointNormal> &first_point_cloud,
						pcl::PointCloud<pcl::PointXYZ> &second_point_cloud,
						float normal_vectors_search_radius,
						float projections_search_radius,
						float bounding_box_extension,
						int max_number_considered_in_INNER_bucket,
						int max_number_considered_in_OUTER_bucket,
						pcl::PointCloud<pcl::PointXYZ> &second_point_cloud_projections,
						std::vector<char> &v_is_projection	);

	bool transform(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3f matrix);
	bool rotateXplus(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
	bool rotateXminus(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
};

#endif
