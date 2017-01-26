#ifndef __LESSON_1_CUH__
#define __LESSON_1_CUH__

#include "lesson_1.h" 

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
		bool operator()(pcl::PointXYZ lp, pcl::PointXYZ rp)
	{
		return lp.x < rp.x;
	}
};

struct compareY
{
	__host__ __device__
		bool operator()(pcl::PointXYZ lp, pcl::PointXYZ rp)
	{
		return lp.y < rp.y;
	}
};

struct compareZ
{
	__host__ __device__
		bool operator()(pcl::PointXYZ lp, pcl::PointXYZ rp)
	{
		return lp.z < rp.z;
	}
};

#endif
