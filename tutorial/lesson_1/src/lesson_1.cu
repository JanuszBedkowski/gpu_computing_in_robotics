#include "lesson_1.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#include <thrust/count.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
#include <thrust/sort.h>
//#include <thrust/sequence.h>
#include <thrust/extrema.h>


cudaError_t cudaCalculateGridParams(pcl::PointXYZ* d_point_cloud, int number_of_points, 
	float resolution_X, float resolution_Y, float resolution_Z, gridParameters &out_rgd_params)
{
	cudaError_t err = cudaGetLastError();
	try
	{
		thrust::device_ptr<pcl::PointXYZ> t_cloud(d_point_cloud);
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<pcl::PointXYZ>,thrust::device_ptr<pcl::PointXYZ> >
		 minmaxX=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareX());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<pcl::PointXYZ>,thrust::device_ptr<pcl::PointXYZ> >
		 minmaxY=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareY());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<pcl::PointXYZ>,thrust::device_ptr<pcl::PointXYZ> >
		 minmaxZ=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareZ());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
		
		pcl::PointXYZ minX,maxX,minZ,maxZ,minY,maxY;

		err = cudaMemcpy(&minX,minmaxX.first.get(),sizeof(pcl::PointXYZ),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxX,minmaxX.second.get(),sizeof(pcl::PointXYZ),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&minZ,minmaxZ.first.get(),sizeof(pcl::PointXYZ),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxZ,minmaxZ.second.get(),sizeof(pcl::PointXYZ),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&minY,minmaxY.first.get(),sizeof(pcl::PointXYZ),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxY,minmaxY.second.get(),sizeof(pcl::PointXYZ),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
	
		int number_of_buckets_X=((maxX.x-minX.x)/resolution_X)+1;
		int number_of_buckets_Y=((maxY.y-minY.y)/resolution_Y)+1;
		int number_of_buckets_Z=((maxZ.z-minZ.z)/resolution_Z)+1;

		out_rgd_params.number_of_buckets_X=number_of_buckets_X;
		out_rgd_params.number_of_buckets_Y=number_of_buckets_Y;
		out_rgd_params.number_of_buckets_Z=number_of_buckets_Z;
		out_rgd_params.number_of_buckets = number_of_buckets_X * number_of_buckets_Y * number_of_buckets_Z;
		
		out_rgd_params.bounding_box_max_X=maxX.x;
		out_rgd_params.bounding_box_min_X=minX.x;
		out_rgd_params.bounding_box_max_Y=maxY.y;
		out_rgd_params.bounding_box_min_Y=minY.y;
		out_rgd_params.bounding_box_max_Z=maxZ.z;
		out_rgd_params.bounding_box_min_Z=minZ.z;
		
		out_rgd_params.resolution_X=resolution_X;
		out_rgd_params.resolution_Y=resolution_Y;
		out_rgd_params.resolution_Z=resolution_Z;
	}
	catch(thrust::system_error &e)
	{
		err = cudaGetLastError();
		cudaDeviceReset();
		return err;
	}	
	catch(std::bad_alloc &e)
  	{
  	 	err = cudaGetLastError();
		cudaDeviceReset();
		return err;
  	}	
	return cudaGetLastError();
}

__global__ void kernel_initializeIndByKey(hashElement* d_hashTable, int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<number_of_points)
	{
		d_hashTable[ind].index_of_point=ind;
		d_hashTable[ind].index_of_bucket=0;
	}
}

__global__ void kernel_getIndexOfBucketForPoints(pcl::PointXYZ* d_point_cloud, hashElement* d_hashTable, int number_of_points, gridParameters rgd_params)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<number_of_points)
	{
		int ix=(d_point_cloud[ind].x-rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(d_point_cloud[ind].y-rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(d_point_cloud[ind].z-rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;
		d_hashTable[ind].index_of_bucket=ix*rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z+iy*rgd_params.number_of_buckets_Z+iz;
	}
}
__global__ void kernel_initializeBuckets(bucket* d_buckets, gridParameters rgd_params)
{
	long long int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < rgd_params.number_of_buckets)
	{
		d_buckets[ind].index_begin=-1;
		d_buckets[ind].index_end=-1;
		d_buckets[ind].number_of_points=0;
	}
}

__global__ void kernel_updateBuckets(hashElement* d_hashTable,
									 bucket* d_buckets,
									 gridParameters rgd_params,
									 int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(number_of_points > 0 && ind<number_of_points)
	{
		if(ind==0)
		{
			int index_bucket = d_hashTable[ind].index_of_bucket;
			int index_bucket_1 = d_hashTable[ind+1].index_of_bucket;

			d_buckets[index_bucket].index_begin=ind;
			if(index_bucket != index_bucket_1)
			{
				d_buckets[index_bucket].index_end=ind+1;
				d_buckets[index_bucket_1].index_end=ind+1;
			}
		}else if(ind==number_of_points-1)
		{
			d_buckets[d_hashTable[ind].index_of_bucket].index_end=ind+1;
		}else
		{
			int index_bucket = d_hashTable[ind].index_of_bucket;
			int index_bucket_1 = d_hashTable[ind+1].index_of_bucket;

			if(index_bucket!=index_bucket_1)
			{
				d_buckets[index_bucket].index_end=ind+1;
				d_buckets[index_bucket_1].index_begin=ind+1;
			}
		}
	}
}

__global__ void kernel_countNumberOfPointsForBuckets(bucket* d_buckets, gridParameters rgd_params)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<rgd_params.number_of_buckets)
	{
		int index_begin = d_buckets[ind].index_begin;
		int index_end = d_buckets[ind].index_end;

		if(index_begin != -1 && index_end !=-1)
		{
			d_buckets[ind].number_of_points=index_end-index_begin;
		}
	}
}

__global__ void kernel_copyKeys(hashElement* d_hashTable_in, hashElement* d_hashTable_out, int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<number_of_points)
	{
		d_hashTable_out[ind]=d_hashTable_in[ind];
	}
}

cudaError_t cudaCalculateGrid(int threads, pcl::PointXYZ* d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params)
{
	cudaError_t err = cudaGetLastError();
	hashElement* d_temp_hashTable;	cudaMalloc((void**)&d_temp_hashTable,number_of_points*sizeof(hashElement));
	int blocks=number_of_points/threads + 1;

	kernel_initializeIndByKey<<<blocks,threads>>>(d_temp_hashTable, number_of_points);
	err = cudaDeviceSynchronize();	if(err != ::cudaSuccess)return err;

	kernel_getIndexOfBucketForPoints<<<blocks,threads>>>(d_point_cloud, d_temp_hashTable, number_of_points, rgd_params);
	err = cudaDeviceSynchronize();	if(err != ::cudaSuccess)return err;

	try
	{
		thrust::device_ptr<hashElement> t_d_temp_hashTable(d_temp_hashTable);
		thrust::sort(t_d_temp_hashTable,t_d_temp_hashTable+number_of_points,compareHashElements());
	}
	catch(thrust::system_error &e)
	{
		err = cudaGetLastError();
		return err;
	}
	catch(std::bad_alloc &e)
	{
		err = cudaGetLastError();
		return err;
	}

	kernel_initializeBuckets<<<rgd_params.number_of_buckets/threads+1,threads>>>(d_buckets,rgd_params);
	err = cudaDeviceSynchronize();	if(err != ::cudaSuccess)return err;

	kernel_updateBuckets<<<blocks,threads>>>(d_temp_hashTable, d_buckets, rgd_params, number_of_points);
	err = cudaDeviceSynchronize();	if(err != ::cudaSuccess)return err;

	kernel_countNumberOfPointsForBuckets<<<rgd_params.number_of_buckets/threads+1,threads>>>(d_buckets, rgd_params);
	err = cudaDeviceSynchronize();	if(err != ::cudaSuccess)return err;

	kernel_copyKeys<<<blocks,threads>>>(d_temp_hashTable, d_hashTable, number_of_points);
	err = cudaDeviceSynchronize(); if(err != ::cudaSuccess)return err;

	err = cudaFree(d_temp_hashTable);
	return err;
}

__global__ void kernel_setAllPointsToRemove(bool *d_markers, int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<number_of_points)
	{
		d_markers[ind] = false;
	}
}

__global__ void kernel_markFirstPointInBuckets(bool *d_markers, hashElement *d_hashTable, bucket *d_buckets,
 gridParameters rgd_params)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<rgd_params.number_of_buckets)
	{
		int index_of_point = d_buckets[ind].index_begin;
		if(index_of_point != -1)
		{
			d_markers[d_hashTable[index_of_point].index_of_point] = true;
		}
	}
}

cudaError_t cudaDownSample(int threads, bool *d_markers,
		hashElement *d_hashTable, bucket *d_buckets, gridParameters rgd_params, int number_of_points)
{
	cudaError_t err = cudaGetLastError();
	kernel_setAllPointsToRemove<<<number_of_points/threads+1,threads>>>(d_markers, number_of_points);
	err = cudaDeviceSynchronize(); if(err != ::cudaSuccess)return err;

	kernel_markFirstPointInBuckets<<<rgd_params.number_of_buckets/threads+1,threads>>>
		(d_markers, d_hashTable, d_buckets, rgd_params);
	err = cudaDeviceSynchronize();

	return err;
}

__global__ void kernel_cudaWarmUpGPU()
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	ind = ind + 1;
}

cudaError_t cudaWarmUpGPU()
{
	kernel_cudaWarmUpGPU<<<1,1>>>();
	cudaDeviceSynchronize();
	return cudaGetLastError();
}
