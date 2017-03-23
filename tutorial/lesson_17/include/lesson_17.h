#ifndef __LESSON_17_H__
#define __LESSON_17_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <pcl/point_types.h>

cudaError_t cudaWarmUpGPU();
cudaError_t cudaPathPlanning(double *_h_dyf, double *_h_u, double *_h_diff,  int _robotX, int _robotY, int _xgoal, int _ygoal, int mem_size_double, int sizeofmap1dim, int steps);

#endif
