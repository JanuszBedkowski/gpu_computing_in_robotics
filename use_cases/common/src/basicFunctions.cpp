#include "basicFunctions.h"

void transformPointCloud(pcl::PointCloud<Semantic::PointXYZL> &in_point_cloud, pcl::PointCloud<Semantic::PointXYZL> &out_point_cloud, Eigen::Affine3f m)
{
	for(size_t i = 0 ; i < in_point_cloud.size(); i++)
	{
		Eigen::Vector3f v(in_point_cloud[i].x, in_point_cloud[i].y, in_point_cloud[i].z);
		Eigen::Vector3f vt = m*v;
		out_point_cloud[i].x = vt.x();
		out_point_cloud[i].y = vt.y();
		out_point_cloud[i].z = vt.z();
	}
}
