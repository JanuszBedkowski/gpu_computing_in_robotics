#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_1.h"
#include <pcl/point_cloud.h>


class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();
	void coutMemoryStatus();

	bool downsampling(pcl::PointCloud<pcl::PointXYZ> &point_cloud, float resolution);
};

#endif
