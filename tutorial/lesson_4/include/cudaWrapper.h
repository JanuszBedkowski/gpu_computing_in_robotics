#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_4.h"
#include <pcl/point_cloud.h>

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();
	void coutMemoryStatus();

	bool removeNoise(
			pcl::PointCloud<pcl::PointXYZ> &point_cloud,
			float search_radius,
			int number_of_points_in_search_sphere_threshold,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket);
};

#endif
