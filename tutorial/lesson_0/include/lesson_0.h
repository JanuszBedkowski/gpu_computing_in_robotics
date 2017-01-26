#ifndef __LESSON_0_H__
#define __LESSON_0_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>

cudaError_t cudaWarmUpGPU();
cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix);
cudaError_t cudaRemovePointsInsideSphere(int threads, pcl::PointXYZ *d_point_cloud, bool *d_markers,
		int number_of_points, float sphere_radius);


#endif
