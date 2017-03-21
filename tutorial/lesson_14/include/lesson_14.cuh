#ifndef __LESSON_14_CUH__
#define __LESSON_14_CUH__

#include "lesson_14.h"

struct compareHashElements
{
	__host__ __device__
		bool operator()(hashElement l, hashElement r)
	{
		return l.index_of_bucket < r.index_of_bucket;
	}
};

struct compareX
{
	__host__ __device__
		bool operator()(lidar_pointcloud::PointXYZIRNL lp, lidar_pointcloud::PointXYZIRNL rp)
	{
		return lp.x < rp.x;
	}
};

struct compareY
{
	__host__ __device__
		bool operator()(lidar_pointcloud::PointXYZIRNL lp, lidar_pointcloud::PointXYZIRNL rp)
	{
		return lp.y < rp.y;
	}
};

struct compareZ
{
	__host__ __device__
		bool operator()(lidar_pointcloud::PointXYZIRNL lp, lidar_pointcloud::PointXYZIRNL rp)
	{
		return lp.z < rp.z;
	}
};

#endif
