#ifndef __BASIC_FUNCTIONS_H__
#define __BASIC_FUNCTIONS_H__


//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
//#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/filter.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/registration/transforms.h>
//#include <pcl/registration/ndt.h>
#include <pcl/console/parse.h>
//#include <pcl/registration/icp.h>
//#include <pcl/common/time.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/PCLPointCloud2.h>


#include "point_types.h"

void transformPointCloud(pcl::PointCloud<Semantic::PointXYZL> &in_point_cloud, pcl::PointCloud<Semantic::PointXYZL> &out_point_cloud, Eigen::Affine3f m);

#endif


