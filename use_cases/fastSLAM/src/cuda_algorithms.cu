#include "cuda_algorithms.cuh" 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>


__global__ void kernel_cuda_insertPointCloudToRGD(
		velodyne_pointcloud::PointXYZIRNL *d_pc, 
		unsigned int nop, 
		double *d_m,
		char *d_rgd, 
		unsigned int particle_index,
		gridParameters rgdparams)
{
	unsigned int ind=blockIdx.x*blockDim.x+threadIdx.x;
	
	if(ind<nop)
	{
		char label = d_pc[ind].label;
		
		double vSrcVector[3] = {d_pc[ind].x, d_pc[ind].y, d_pc[ind].z};
		double vOut[3];
		vOut[0]=d_m[0]*vSrcVector[0]+d_m[4]*vSrcVector[1]+d_m[8]*vSrcVector[2]+d_m[12];
   	 	vOut[1]=d_m[1]*vSrcVector[0]+d_m[5]*vSrcVector[1]+d_m[9]*vSrcVector[2]+d_m[13];
    	vOut[2]=d_m[2]*vSrcVector[0]+d_m[6]*vSrcVector[1]+d_m[10]*vSrcVector[2]+d_m[14];

		float x = vOut[0];
		float y = vOut[1];
		float z = vOut[2];
		
		if(x < rgdparams.bounding_box_min_X || x > rgdparams.bounding_box_max_X)
		{
			return;
		}
		if(y < rgdparams.bounding_box_min_Y || y > rgdparams.bounding_box_max_Y)
		{
			return;
		}
		if(z < rgdparams.bounding_box_min_Z || z > rgdparams.bounding_box_max_Z)
		{
			return;
		}
		
		unsigned int ix=(x-rgdparams.bounding_box_min_X)/rgdparams.resolution_X;
		unsigned int iy=(y-rgdparams.bounding_box_min_Y)/rgdparams.resolution_Y;
		unsigned int iz=(z-rgdparams.bounding_box_min_Z)/rgdparams.resolution_Z;
		unsigned int gr_index = 
			ix*rgdparams.number_of_buckets_Y*rgdparams.number_of_buckets_Z+iy*rgdparams.number_of_buckets_Z + iz
				+ rgdparams.number_of_buckets * particle_index;
		
		d_rgd[gr_index] = label;
	}
}
		
__global__ void kernel_cuda_semantic_nn(
		velodyne_pointcloud::PointXYZIRNL *d_pc, 
		unsigned int nop, 
		double *d_m,
		char *d_rgd, 
		unsigned int particle_index,
		gridParameters rgdparams,
		unsigned int *d_nn
		)
{
	unsigned int ind=blockIdx.x*blockDim.x+threadIdx.x;
	
	if(ind<nop)
	{
		d_nn[ind] = 0;
	
		double vSrcVector[3] = {d_pc[ind].x, d_pc[ind].y, d_pc[ind].z};
		double vOut[3];
		vOut[0]=d_m[0]*vSrcVector[0]+d_m[4]*vSrcVector[1]+d_m[8]*vSrcVector[2]+d_m[12];
   	 	vOut[1]=d_m[1]*vSrcVector[0]+d_m[5]*vSrcVector[1]+d_m[9]*vSrcVector[2]+d_m[13];
    	vOut[2]=d_m[2]*vSrcVector[0]+d_m[6]*vSrcVector[1]+d_m[10]*vSrcVector[2]+d_m[14];

		double x = vOut[0];
		double y = vOut[1];
		double z = vOut[2];
		
		char label = d_pc[ind].label;
		
		if(x < rgdparams.bounding_box_min_X || x > rgdparams.bounding_box_max_X)
		{
			return;
		}
		if(y < rgdparams.bounding_box_min_Y || y > rgdparams.bounding_box_max_Y)
		{
			return;
		}
		if(z < rgdparams.bounding_box_min_Z || z > rgdparams.bounding_box_max_Z)
		{
			return;
		}
		
		unsigned int ix=(x-rgdparams.bounding_box_min_X)/rgdparams.resolution_X;
		unsigned int iy=(y-rgdparams.bounding_box_min_Y)/rgdparams.resolution_Y;
		unsigned int iz=(z-rgdparams.bounding_box_min_Z)/rgdparams.resolution_Z;
		unsigned int bucket_index = ix*rgdparams.number_of_buckets_Y*rgdparams.number_of_buckets_Z+iy*rgdparams.number_of_buckets_Z
				+ iz + rgdparams.number_of_buckets * particle_index;
		
		char labelrgd = d_rgd[bucket_index];
		//if((labelrgd == label) && (label != 3))d_nn[ind] = 1;
		if(labelrgd == label)d_nn[ind] = 1;
	}
}

__global__ void kernel_cuda_replikate_particle_in_rgd(
	char *d_rgd, 
	gridParameters rgd_params, 
	unsigned int index_particle_to_eraze, 
	unsigned int index_best_particle,
	unsigned int number_of_particles)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	
	if(ind < rgd_params.number_of_buckets)
	{
		unsigned int _index_best_particle = ind + rgd_params.number_of_buckets * index_best_particle;
		unsigned int _index_particle_to_eraze = ind + rgd_params.number_of_buckets * index_particle_to_eraze;
		d_rgd[_index_particle_to_eraze] = d_rgd[_index_best_particle];
	}
}
	
cudaError_t cuda_insertPointCloudToRGD(unsigned int threads, 
	velodyne_pointcloud::PointXYZIRNL *d_pc, 
	unsigned int nop, 
	double *d_m,
	char *d_rgd, 
	unsigned int particle_index,
	gridParameters rgdparams)
{
	cudaError_t err = cudaGetLastError();
	unsigned int blocks=nop/threads+1;

	kernel_cuda_insertPointCloudToRGD<<<blocks,threads>>>(d_pc, nop, d_m, d_rgd, particle_index, rgdparams);
	err = cudaDeviceSynchronize();
return err;
}
	
cudaError_t cuda_computeOverlap(
	unsigned int threads, 
	velodyne_pointcloud::PointXYZIRNL *d_pc, 
	unsigned int nop, 
	double *d_m,
	char *d_rgd, 
	unsigned int particle_index,
	gridParameters rgdparams, float &overlap)
{
	cudaError_t err = cudaGetLastError();
	unsigned int blocks=nop/threads+1;
	
	unsigned int *d_nn = NULL;
	err = cudaMalloc((void**)&d_nn, nop*sizeof(unsigned int));
		if(err != ::cudaSuccess)return err;
	
	kernel_cuda_semantic_nn<<<blocks,threads>>>(d_pc, nop, d_m, d_rgd, particle_index, rgdparams, d_nn);
	err = cudaDeviceSynchronize();
		if(err != ::cudaSuccess)return err;

	thrust::device_ptr <unsigned int> dev_ptr_d_nn ( d_nn );
	unsigned int number_of_nearest_neighbors = thrust::reduce (dev_ptr_d_nn , dev_ptr_d_nn + nop);
	overlap = float(number_of_nearest_neighbors)/float(nop);

	err = cudaFree(d_nn);
	return err;
}
	
cudaError_t cuda_replikate_particle_in_rgd(
	unsigned int threads, 
	char *d_rgd, 
	gridParameters rgd_params, 
	unsigned int index_particle_to_eraze, 
	unsigned int index_best_particle,
	unsigned int number_of_particles)
{
	cudaError_t err = cudaGetLastError();

	kernel_cuda_replikate_particle_in_rgd<<<(rgd_params.number_of_buckets)/threads+1,threads>>>
			(d_rgd, rgd_params, index_particle_to_eraze, index_best_particle, number_of_particles);
	err = cudaDeviceSynchronize();
	return err;
}	
	


