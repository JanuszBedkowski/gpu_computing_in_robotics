#ifndef __VELODYNE_VLP16_POINTCLOUD_POINT_TYPES_H__
#define __VELODYNE_VLP16_POINTCLOUD_POINT_TYPES_H__

#include <pcl/point_types.h>

namespace VelodyneVLP16
{
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


POINT_CLOUD_REGISTER_POINT_STRUCT(VelodyneVLP16::PointXYZNL,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, normal_x, normal_x)
								  (float, normal_y, normal_y)
								  (float, normal_z, normal_z)
								  (int, label, label))

#endif

