#ifndef __CUDA_ALGORITHMS_H__
#define __CUDA_ALGORITHMS_H__

#include <cuda.h>
#include <cuda_runtime.h>

#include "point_types.h"
#include "cudaStructures.h"

cudaError_t cuda_insertPointCloudToRGD(
	unsigned int threads, 
	velodyne_pointcloud::PointXYZIRNL *d_pc, 
	unsigned int nop, 
	double *d_m,
	char *d_rgd, 
	unsigned int particle_index,
	gridParameters rgdparams);
	
cudaError_t cuda_computeOverlap(
	unsigned int threads, 
	velodyne_pointcloud::PointXYZIRNL *d_pc, 
	unsigned int nop, 
	double *d_m,
	char *d_rgd, 
	unsigned int particle_index,
	gridParameters rgdparams, 
	float &overlap);
	
cudaError_t cuda_replikate_particle_in_rgd(
	unsigned int threads, 
	char *d_rgd, 
	gridParameters rgd_params, 
	unsigned int index_particle_to_eraze, 
	unsigned int index_best_particle,
	unsigned int number_of_particles);

#endif


