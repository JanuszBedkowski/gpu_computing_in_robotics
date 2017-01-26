#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_5.h"
#include <pcl/point_cloud.h>

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	bool getNumberOfAvailableThreads(int &threads, int &threadsNV);
	void coutMemoryStatus();

	bool normalVectorCalculation(pcl::PointCloud<pcl::PointNormal> &point_cloud,
			float normal_vector_radius,
	        int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket);

	bool normalVectorCalculationFast(pcl::PointCloud<pcl::PointNormal> &point_cloud,
			float normal_vector_radius,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket);

};



#endif
