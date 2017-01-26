#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_2.h"
#include <pcl/point_cloud.h>


class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();
	void coutMemoryStatus();

	bool removeNoiseNaive(pcl::PointCloud<pcl::PointXYZ> &point_cloud, float resolution, int number_of_points_in_bucket_threshold);
};



#endif
