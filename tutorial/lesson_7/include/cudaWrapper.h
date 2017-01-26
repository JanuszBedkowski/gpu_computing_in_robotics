#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_7.h"
#include <pcl/point_cloud.h>
#include "point_types.h"

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	bool getNumberOfAvailableThreads(int &threads, int &threadsNV);
	void coutMemoryStatus();

	bool classify(  pcl::PointCloud<VelodyneVLP16::PointXYZNL> &point_cloud,
					int number_of_points,
					float normal_vectors_search_radius,
					float curvature_threshold,
					float ground_Z_coordinate_threshold,
					int number_of_points_needed_for_plane_threshold,
					int max_number_considered_in_INNER_bucket,
					int max_number_considered_in_OUTER_bucket  );
};



#endif
