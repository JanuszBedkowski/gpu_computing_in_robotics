#ifndef __LESSON_7_CUH__
#define __LESSON_7_CUH__

#include "lesson_7.h"

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
		bool operator()(VelodyneVLP16::PointXYZNL lp, VelodyneVLP16::PointXYZNL rp)
	{
		return lp.x < rp.x;
	}
};

struct compareY
{
	__host__ __device__
		bool operator()(VelodyneVLP16::PointXYZNL lp, VelodyneVLP16::PointXYZNL rp)
	{
		return lp.y < rp.y;
	}
};

struct compareZ
{
	__host__ __device__
		bool operator()(VelodyneVLP16::PointXYZNL lp, VelodyneVLP16::PointXYZNL rp)
	{
		return lp.z < rp.z;
	}
};

#endif
