#ifndef __SEMANTIC_POINT_TYPES_H__
#define __SEMANTIC_POINT_TYPES_H__

#include <pcl/point_types.h>

#define PLANE 0
#define EDGE 1
#define CEILING 2
#define FLOOR_GROUND 3

namespace Semantic
{
	struct PointXYZL
	{
		float x;
		float y;
		float z;
		int label; // 0 - plane, 1 - edge, 2 - ceiling, 3 - floor/ground
	};

	struct PointXYZNL
	{
		float x;
		float y;
		float z;
		float normal_x;
		float normal_y;
		float normal_z;
		int label; // 0 - plane, 1 - edge, 2 - ceiling, 3 - floor/ground
	};
};


POINT_CLOUD_REGISTER_POINT_STRUCT(Semantic::PointXYZL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (int, label, label))

POINT_CLOUD_REGISTER_POINT_STRUCT(Semantic::PointXYZNL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, normal_x, normal_x)
								  (float, normal_y, normal_y)
								  (float, normal_z, normal_z)
								  (int, label, label))

namespace velodyne_pointcloud
{
	struct PointXYZIR
	{
		float x;
		float y;
		float z;
		float intensity;
		uint16_t   ring;
	};

	struct PointXYZIRNL
	{
		float x;
		float y;
		float z;
		float intensity;
		uint16_t   ring;
		float normal_x;
		float normal_y;
		float normal_z;
		int   label;
	};
};

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_pointcloud::PointXYZIR,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, intensity, intensity)
                                  (uint16_t, ring, ring))

POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_pointcloud::PointXYZIRNL,
								(float, x, x)
								(float, y, y)
								(float, z, z)
								(float, intensity, intensity)
								(uint16_t, ring, ring)
								(float, normal_x, normal_x)
								(float, normal_y, normal_y)
								(float, normal_z, normal_z)
								(int, label, label))

#endif


