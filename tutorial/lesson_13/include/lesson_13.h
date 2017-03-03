#ifndef __LESSON_13_H__
#define __LESSON_13_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <Eigen/Geometry>
#include <vector>

typedef struct plane
{
	float nx;
	float ny;
	float nz;
	float rho;
}plane_t;

typedef struct local_observations
{
	std::vector<plane_t> planes;
	Eigen::Affine3f m_pose;
	double om;
	double fi;
	double ka;
	double tx;
	double ty;
	double tz;
}local_observations_t;

typedef struct pair_local_observations
{
	std::vector<plane_t> planes_reference;
	std::vector<plane_t> planes_to_register;
	Eigen::Affine3f m_pose;
	double om;
	double fi;
	double ka;
	double tx;
	double ty;
	double tz;
}pair_local_observations_t;


cudaError_t cudaWarmUpGPU();
cudaError_t cudaCompute_AtP(int threads, double *d_A, double *d_P, double *d_AtP, int rows, int columns);

#endif
