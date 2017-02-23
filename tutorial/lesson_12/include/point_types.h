#ifndef __SEMANTIC_POINT_TYPES_H__
#define __SEMANTIC_POINT_TYPES_H__

#include <pcl/point_types.h>

namespace lidar_pointcloud
{
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
	struct PointProjection
	{
		float x_src;
		float y_src;
		float z_src;
		float x_dst;
		float y_dst;
		float z_dst;
		float normal_x;
		float normal_y;
		float normal_z;
		float distance;
		uint8_t isProjection;
	};
};

POINT_CLOUD_REGISTER_POINT_STRUCT(lidar_pointcloud::PointXYZIRNL,
								(float, x, x)
								(float, y, y)
								(float, z, z)
								(float, intensity, intensity)
								(uint16_t, ring, ring)
								(float, normal_x, normal_x)
								(float, normal_y, normal_y)
								(float, normal_z, normal_z)
								(int, label, label))

POINT_CLOUD_REGISTER_POINT_STRUCT(lidar_pointcloud::PointProjection,
								(float, x_src, x_src)
								(float, y_src, y_src)
								(float, z_src, z_src)
								(float, x_dst, x_dst)
								(float, y_dst, y_dst)
								(float, z_dst, z_dst)
								(float, normal_x, normal_x)
								(float, normal_y, normal_y)
								(float, normal_z, normal_z)
								(float, distance, distance)
								(uint8_t, isProjection, isProjection))

#endif


