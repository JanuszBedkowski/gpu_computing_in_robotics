/*
 * Software License Agreement (BSD License)
 *
 *  Data Registration Framework - Mobile Spatial Assistance System
 *  Copyright (c) 2014-2016, Institute of Mathematical Machines
 *  http://lider.zms.imm.org.pl/
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Institute of Mathematical Machines nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 */

#include "lesson_3.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#include <thrust/count.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
#include <thrust/sort.h>
//#include <thrust/sequence.h>
#include <thrust/extrema.h>

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

cudaError_t cudaCalculateGridParams(pcl::PointXYZ* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params)
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
	
		maxX.x += bounding_box_extension;
		minX.x -= bounding_box_extension;

		maxY.y += bounding_box_extension;
		minY.y -= bounding_box_extension;

		maxZ.z += bounding_box_extension;
		minZ.z -= bounding_box_extension;

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

__global__ void kernel_initializeHashTable(hashElement* d_hashTable, int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < number_of_points)
	{
		d_hashTable[ind].index_of_point=ind;
		d_hashTable[ind].index_of_bucket=0;
	}
}

__global__ void kernel_getIndexOfBucketForPoints(pcl::PointXYZ* cloud, hashElement* d_hashTable, int number_of_points, gridParameters rgd_params)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < number_of_points)
	{
		int ix=(cloud[ind].x-rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(cloud[ind].y-rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(cloud[ind].z-rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;
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

__global__ void kernel_updateBuckets(hashElement* d_hashTable, bucket* d_buckets,
		gridParameters rgd_params, int number_of_points)
{
	int ind = blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < number_of_points)
	{
		if(ind == 0)
		{
			int index_of_bucket = d_hashTable[ind].index_of_bucket;
			int index_of_bucket_1 = d_hashTable[ind+1].index_of_bucket;

			d_buckets[index_of_bucket].index_begin=ind;
			if(index_of_bucket != index_of_bucket_1)
			{
				d_buckets[index_of_bucket].index_end=ind+1;
				d_buckets[index_of_bucket_1].index_end=ind+1;
			}
		}else if(ind == number_of_points-1)
		{
			d_buckets[d_hashTable[ind].index_of_bucket].index_end=ind+1;
		}else
		{
			int index_of_bucket = d_hashTable[ind].index_of_bucket;
			int index_of_bucket_1 = d_hashTable[ind+1].index_of_bucket;

			if(index_of_bucket != index_of_bucket_1)
			{
				d_buckets[index_of_bucket].index_end=ind+1;
				d_buckets[index_of_bucket_1].index_begin=ind+1;
			}
		}
	}
}

__global__ void kernel_countNumberOfPointsForBuckets(bucket* d_buckets, gridParameters rgd_params)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < rgd_params.number_of_buckets)
	{
		int index_begin = d_buckets[ind].index_begin;
		int index_end = d_buckets[ind].index_end;

		if(index_begin != -1 && index_end !=-1)
		{
			d_buckets[ind].number_of_points = index_end - index_begin;
		}
	}
}

__global__ void kernel_copyHashTable(hashElement* d_hashTable_in, hashElement* d_hashTable_out, int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < number_of_points)
	{
		d_hashTable_out[ind] = d_hashTable_in[ind];
	}
}

cudaError_t cudaCalculateGrid(int threads, pcl::PointXYZ *d_point_cloud, bucket *d_buckets,
		hashElement *d_hashTable, int number_of_points, gridParameters rgd_params)
{
	cudaError_t err = cudaGetLastError();
	hashElement* d_temp_hashTable;	cudaMalloc((void**)&d_temp_hashTable,number_of_points*sizeof(hashElement));
	int blocks=number_of_points/threads + 1;
	
	kernel_initializeHashTable<<<blocks,threads>>>(d_temp_hashTable, number_of_points);
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
	
	kernel_copyHashTable<<<blocks,threads>>>(d_temp_hashTable, d_hashTable, number_of_points);
	err = cudaDeviceSynchronize(); if(err != ::cudaSuccess)return err;
	
	err = cudaFree(d_temp_hashTable);
	return err;
}

/////////////////////////////////nearest neighbourhood search////////////////////////////////
__global__ void kernel_nearestNeighborSearch(
		pcl::PointXYZ *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		pcl::PointXYZ *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		hashElement *d_hashTable,
		bucket *d_buckets,
		gridParameters rgd_params,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		int *d_nearest_neighbour_indexes)
{
	int index_of_point_second_point_cloud = blockIdx.x*blockDim.x+threadIdx.x;

	if(index_of_point_second_point_cloud < number_of_points_second_point_cloud)
	{
		bool isok = false;

		float x = d_second_point_cloud[index_of_point_second_point_cloud].x;
		float y = d_second_point_cloud[index_of_point_second_point_cloud].y;
		float z = d_second_point_cloud[index_of_point_second_point_cloud].z;

		if(x < rgd_params.bounding_box_min_X || x > rgd_params.bounding_box_max_X)
		{
			d_nearest_neighbour_indexes[index_of_point_second_point_cloud] = -1;
			return;
		}
		if(y < rgd_params.bounding_box_min_Y || y > rgd_params.bounding_box_max_Y)
		{
			d_nearest_neighbour_indexes[index_of_point_second_point_cloud] = -1;
			return;
		}
		if(z < rgd_params.bounding_box_min_Z || z > rgd_params.bounding_box_max_Z)
		{
			d_nearest_neighbour_indexes[index_of_point_second_point_cloud] = -1;
			return;
		}

		int ix=(x - rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(y - rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(z - rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;

		int index_bucket = ix*rgd_params.number_of_buckets_Y *
				rgd_params.number_of_buckets_Z + iy * rgd_params.number_of_buckets_Z + iz;
		int nn_index = -1;

		if(index_bucket >= 0 && index_bucket < rgd_params.number_of_buckets)
		{
			int sx, sy, sz, stx, sty, stz;
			if(ix == 0)sx = 0; else sx = -1;
			if(iy == 0)sy = 0; else sy = -1;
			if(iz == 0)sz = 0; else sz =- 1;

			if(ix == rgd_params.number_of_buckets_X - 1)stx = 1; else stx = 2;
			if(iy == rgd_params.number_of_buckets_Y - 1)sty = 1; else sty = 2;
			if(iz == rgd_params.number_of_buckets_Z - 1)stz = 1; else stz = 2;

			float _distance = 100000000.0f;
			int index_next_bucket;
			int iter;
			int number_of_points_in_bucket;
			int l_begin;
			int l_end;

			for(int i = sx; i < stx; i++)
			{
				for(int j = sy; j < sty; j++)
				{
					for(int k = sz; k < stz; k++)
					{
						index_next_bucket = index_bucket +
								i * rgd_params.number_of_buckets_Y * rgd_params.number_of_buckets_Z +
								j * rgd_params.number_of_buckets_Z + k;
						if(index_next_bucket >= 0 && index_next_bucket < rgd_params.number_of_buckets)
						{
							number_of_points_in_bucket = d_buckets[index_next_bucket].number_of_points;
							if(number_of_points_in_bucket <= 0)continue;

							int max_number_considered_in_bucket;
							if(index_next_bucket == index_bucket)
							{
								max_number_considered_in_bucket = max_number_considered_in_INNER_bucket;
							}else
							{
								max_number_considered_in_bucket = max_number_considered_in_OUTER_bucket;
							}
							if(max_number_considered_in_bucket <= 0)continue;

							if(max_number_considered_in_bucket >= number_of_points_in_bucket)
							{
								iter=1;
							}else
							{
								iter = number_of_points_in_bucket / max_number_considered_in_bucket;
								if(iter <= 0)iter = 1;
							}

							l_begin = d_buckets[index_next_bucket].index_begin;
							l_end = d_buckets[index_next_bucket].index_end;

							for(int l = l_begin; l < l_end; l += iter)
							{
								if(l >= 0 && l < number_of_points_first_point_cloud)
								{
									int hashed_index_of_point = d_hashTable[l].index_of_point;
									//inA[hashed_index_of_point].var = 1;

									float nn_x  = d_first_point_cloud[hashed_index_of_point].x;
									float nn_y  = d_first_point_cloud[hashed_index_of_point].y;
									float nn_z  = d_first_point_cloud[hashed_index_of_point].z;

									float dist  = (x - nn_x) * (x - nn_x) +
												  (y - nn_y) * (y - nn_y) +
												  (z - nn_z) * (z - nn_z);

									if(dist <= search_radius * search_radius)
									{
										if(dist < _distance)
										{
											isok = true;
											nn_index = hashed_index_of_point;
											_distance = dist;
										}
									}
								}
							}
						}
					}
				}
			}
		}

		if(isok)
		{
			d_nearest_neighbour_indexes[index_of_point_second_point_cloud] = nn_index;

		}else
		{
			d_nearest_neighbour_indexes[index_of_point_second_point_cloud] = -1;
		}
	}
}

cudaError_t cudaNearestNeighborSearch(
			int threads,
			pcl::PointXYZ *d_first_point_cloud,
			int number_of_points_first_point_cloud,
			pcl::PointXYZ *d_second_point_cloud,
			int number_of_points_second_point_cloud,
			hashElement *d_hashTable,
			bucket *d_buckets,
			gridParameters rgd_params,
			float search_radius,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			int *d_nearest_neighbour_indexes)
{
	cudaError_t err = cudaGetLastError();

	int blocks=number_of_points_second_point_cloud/threads+1;

	kernel_nearestNeighborSearch<<<blocks,threads>>>(
			d_first_point_cloud,
			number_of_points_first_point_cloud,
			d_second_point_cloud,
			number_of_points_second_point_cloud,
			d_hashTable,
			d_buckets,
			rgd_params,
			search_radius,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			d_nearest_neighbour_indexes);

	err = cudaDeviceSynchronize();
	return err;
}

__global__ void kernel_cudaTransformPoints(pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<number_of_points)
	{
		float vSrcVector[3] = {d_point_cloud[ind].x, d_point_cloud[ind].y, d_point_cloud[ind].z};
		float vOut[3];
		vOut[0]=d_matrix[0]*vSrcVector[0]+d_matrix[4]*vSrcVector[1]+d_matrix[8]*vSrcVector[2]+d_matrix[12];
   	 	vOut[1]=d_matrix[1]*vSrcVector[0]+d_matrix[5]*vSrcVector[1]+d_matrix[9]*vSrcVector[2]+d_matrix[13];
    	vOut[2]=d_matrix[2]*vSrcVector[0]+d_matrix[6]*vSrcVector[1]+d_matrix[10]*vSrcVector[2]+d_matrix[14];

		d_point_cloud[ind].x = vOut[0];
		d_point_cloud[ind].y = vOut[1];
		d_point_cloud[ind].z = vOut[2];
	}
}

cudaError_t cudaTransformPoints(int threads, pcl::PointXYZ *d_point_cloud, int number_of_points, float *d_matrix)
{
	kernel_cudaTransformPoints<<<number_of_points/threads+1,threads>>>
		(d_point_cloud, number_of_points, d_matrix);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}


