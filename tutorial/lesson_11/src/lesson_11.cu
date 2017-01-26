#include "lesson_11.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
//#include <thrust/count.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
#include <thrust/sort.h>
//#include <thrust/sequence.h>
#include <thrust/extrema.h>

#include "cuda_SVD.cu"

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

cudaError_t cudaCalculateGridParams(pcl::PointNormal* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params)
{
	cudaError_t err = cudaGetLastError();

	try
	{
		thrust::device_ptr<pcl::PointNormal> t_cloud(d_point_cloud);
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<pcl::PointNormal>,thrust::device_ptr<pcl::PointNormal> >
		 minmaxX=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareX());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<pcl::PointNormal>,thrust::device_ptr<pcl::PointNormal> >
		 minmaxY=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareY());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<pcl::PointNormal>,thrust::device_ptr<pcl::PointNormal> >
		 minmaxZ=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareZ());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
		
		pcl::PointNormal minX,maxX,minZ,maxZ,minY,maxY;

		err = cudaMemcpy(&minX,minmaxX.first.get(),sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxX,minmaxX.second.get(),sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&minZ,minmaxZ.first.get(),sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxZ,minmaxZ.second.get(),sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&minY,minmaxY.first.get(),sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxY,minmaxY.second.get(),sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
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


__global__ void kernel_initializeIndByKey(hashElement* d_hashTable, int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < number_of_points)
	{
		d_hashTable[ind].index_of_point=ind;
		d_hashTable[ind].index_of_bucket=0;
	}
}

__global__ void kernel_getIndexOfBucketForPoints(pcl::PointNormal* cloud, hashElement* d_hashTable, int number_of_points, gridParameters rgd_params)
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

__global__ void kernel_copyKeys(hashElement* d_hashTable_in, hashElement* d_hashTable_out, int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < number_of_points)
	{
		d_hashTable_out[ind] = d_hashTable_in[ind];
	}
}

cudaError_t cudaCalculateGrid(int threads, pcl::PointNormal *d_point_cloud, bucket *d_buckets,
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

__global__ void kernel_normalvectorcomputation_step1_fast(
		pcl::PointNormal * d_point_cloud,
		hashElement* d_hashTable,
		simple_point3D* d_mean,
		int number_of_points,
		bucket* d_buckets,
		gridParameters rgd_params,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket)
{
	int index_of_point = blockIdx.x * blockDim.x + threadIdx.x;

	if(index_of_point < number_of_points)
	{
		d_point_cloud[index_of_point].normal_x = 0;
		d_point_cloud[index_of_point].normal_y = 0;
		d_point_cloud[index_of_point].normal_z = 0;
		__syncthreads();

		int index_of_bucket = d_hashTable[index_of_point].index_of_bucket;

		if(index_of_bucket >= 0 && index_of_bucket < rgd_params.number_of_buckets)
		{
			int hashed_index_of_point = d_hashTable[index_of_point].index_of_point;

			if(hashed_index_of_point >= 0 && hashed_index_of_point < number_of_points)
			{
				float x = d_point_cloud[hashed_index_of_point].x;
				float y = d_point_cloud[hashed_index_of_point].y;
				float z = d_point_cloud[hashed_index_of_point].z;

				int ix = index_of_bucket/(rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z);
				int iy = (index_of_bucket%(rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z))/rgd_params.number_of_buckets_Z;
				int iz = (index_of_bucket%(rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z))%rgd_params.number_of_buckets_Z;

				int sx, sy, sz, stx, sty, stz;
				if(ix == 0) sx = 0; else sx = -1;
				if(iy == 0) sy = 0; else sy = -1;
				if(iz == 0) sz = 0; else sz = -1;

				if(ix == rgd_params.number_of_buckets_X - 1)stx = 1; else stx = 2;
				if(iy == rgd_params.number_of_buckets_Y - 1)sty = 1; else sty = 2;
				if(iz == rgd_params.number_of_buckets_Z - 1)stz = 1; else stz = 2;

				int number_of_nearest_neighbours = 0;
				simple_point3D mean;
				mean.x = 0.0f;
				mean.y = 0.0f;
				mean.z = 0.0f;

				float nearest_neighbour_x;
				float nearest_neighbour_y;
				float nearest_neighbour_z;

				for(int i = sx; i < stx; i++)
				{
					for(int j = sy; j < sty; j++)
					{
						for(int k = sz; k < stz; k++)
						{
							int index_of_neighbour_bucket=index_of_bucket+i*rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z+j*rgd_params.number_of_buckets_Z+k;

							if(index_of_neighbour_bucket >= 0 && index_of_neighbour_bucket < rgd_params.number_of_buckets)
							{
								int iter;
								int number_of_points_in_bucket = d_buckets[index_of_neighbour_bucket].number_of_points;
								if(number_of_points_in_bucket <= 0)continue;

								int max_number_considered_in_bucket;
								if(index_of_neighbour_bucket==index_of_bucket)
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

								int l_begin = d_buckets[index_of_neighbour_bucket].index_begin;
								int l_end = d_buckets[index_of_neighbour_bucket].index_end;

								for(int l = l_begin; l < l_end; l += iter)
								{
									if(l >= 0 && l < number_of_points)
									{
										int indexNextPointInBucket = d_hashTable[l].index_of_point;
										nearest_neighbour_x = d_point_cloud[indexNextPointInBucket].x;
										nearest_neighbour_y = d_point_cloud[indexNextPointInBucket].y;
										nearest_neighbour_z = d_point_cloud[indexNextPointInBucket].z;

										float dist=sqrtf((x - nearest_neighbour_x)*(x - nearest_neighbour_x)
														+(y - nearest_neighbour_y)*(y - nearest_neighbour_y)
														+(z - nearest_neighbour_z)*(z - nearest_neighbour_z));

										if(dist <= search_radius)
										{
											mean.x += nearest_neighbour_x;
											mean.y += nearest_neighbour_y;
											mean.z += nearest_neighbour_z;
											number_of_nearest_neighbours++;
										}
									}
								}
							}
						}
					}
				}

				if(number_of_nearest_neighbours >= 3)
				{
					d_mean[index_of_point].x = mean.x / number_of_nearest_neighbours;
					d_mean[index_of_point].y = mean.y / number_of_nearest_neighbours;
					d_mean[index_of_point].z = mean.z / number_of_nearest_neighbours;
				}else
				{
					d_mean[index_of_point].x = 0.0f;
					d_mean[index_of_point].y = 0.0f;
					d_mean[index_of_point].z = 0.0f;
				}
			}
		}
	}
}

__global__ void kernel_normalvectorcomputation_step2_fast(
	pcl::PointNormal *d_point_cloud,
	hashElement *d_hashTable,
	simple_point3D *d_mean,
	int number_of_points,
	bucket *d_buckets,
	gridParameters rgd_params,
	float search_radius,
	int max_number_considered_in_INNER_bucket,
	int max_number_considered_in_OUTER_bucket)
{
	int index_of_point = blockIdx.x * blockDim.x + threadIdx.x;
	if(index_of_point < number_of_points)
	{
		int index_of_bucket = d_hashTable[index_of_point].index_of_bucket;
		if(index_of_bucket >= 0 && index_of_bucket < rgd_params.number_of_buckets)
		{
			int hashed_index_of_point = d_hashTable[index_of_point].index_of_point;
			if(hashed_index_of_point >= 0 && hashed_index_of_point < number_of_points)
			{
				simple_point3D mean = d_mean[index_of_point];
				if(mean.x != 0.0f && mean.y != 0.0f && mean.z != 0.0f)
				{
					float x = d_point_cloud[hashed_index_of_point].x;
					float y = d_point_cloud[hashed_index_of_point].y;
					float z = d_point_cloud[hashed_index_of_point].z;

					int ix = index_of_bucket/(rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z);
					int iy = (index_of_bucket%(rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z))/rgd_params.number_of_buckets_Z;
					int iz = (index_of_bucket%(rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z))%rgd_params.number_of_buckets_Z;
					int sx, sy, sz, stx, sty, stz;
					if(ix == 0)sx = 0; else sx = -1;
					if(iy == 0)sy = 0; else sy = -1;
					if(iz == 0)sz = 0; else sz = -1;
					if(ix == rgd_params.number_of_buckets_X - 1)stx = 1; else stx = 2;
					if(iy == rgd_params.number_of_buckets_Y - 1)sty = 1; else sty = 2;
					if(iz == rgd_params.number_of_buckets_Z - 1)stz = 1; else stz = 2;

					int number_of_nearest_neighbours=0;

					double cov[3][3];
					cov[0][0]=cov[0][1]=cov[0][2]=cov[1][0]=cov[1][1]=cov[1][2]=cov[2][0]=cov[2][1]=cov[2][2]=0;

					float nearest_neighbour_x;
					float nearest_neighbour_y;
					float nearest_neighbour_z;

					for(int i = sx; i < stx; i++)
					{
						for(int j = sy; j < sty; j++)
						{
							for(int k = sz; k < stz; k++)
							{
								int index_of_neighbour_bucket=index_of_bucket+i*rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z+j*rgd_params.number_of_buckets_Z+k;
								if(index_of_neighbour_bucket >= 0 && index_of_neighbour_bucket < rgd_params.number_of_buckets)
								{
									int iter;
									int number_of_points_in_bucket = d_buckets[index_of_neighbour_bucket].number_of_points;
									if(number_of_points_in_bucket <= 0)continue;

									int max_number_considered_in_bucket;
									if(index_of_neighbour_bucket==index_of_bucket)
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

									int l_begin = d_buckets[index_of_neighbour_bucket].index_begin;
									int l_end = d_buckets[index_of_neighbour_bucket].index_end;

									for(int l = l_begin; l < l_end; l += iter)
									{
										if(l >= 0 && l < number_of_points)
										{
											int indexNextPointInBucket = d_hashTable[l].index_of_point;
											nearest_neighbour_x = d_point_cloud[indexNextPointInBucket].x;
											nearest_neighbour_y = d_point_cloud[indexNextPointInBucket].y;
											nearest_neighbour_z = d_point_cloud[indexNextPointInBucket].z;

											float dist = sqrtf((x - nearest_neighbour_x)*(x - nearest_neighbour_x)
															  +(y - nearest_neighbour_y)*(y - nearest_neighbour_y)
   															  +(z - nearest_neighbour_z)*(z - nearest_neighbour_z));

											if(dist <= search_radius)
											{
												cov[0][0]+=(mean.x - nearest_neighbour_x) * (mean.x - nearest_neighbour_x);
												cov[0][1]+=(mean.x - nearest_neighbour_x) * (mean.y - nearest_neighbour_y);
												cov[0][2]+=(mean.x - nearest_neighbour_x) * (mean.z - nearest_neighbour_z);
												cov[1][0]+=(mean.y - nearest_neighbour_y) * (mean.x - nearest_neighbour_x);
												cov[1][1]+=(mean.y - nearest_neighbour_y) * (mean.y - nearest_neighbour_y);
												cov[1][2]+=(mean.y - nearest_neighbour_y) * (mean.z - nearest_neighbour_z);
												cov[2][0]+=(mean.z - nearest_neighbour_z) * (mean.x - nearest_neighbour_x);
												cov[2][1]+=(mean.z - nearest_neighbour_z) * (mean.y - nearest_neighbour_y);
												cov[2][2]+=(mean.z - nearest_neighbour_z) * (mean.z - nearest_neighbour_z);
												number_of_nearest_neighbours++;
											}
										}
									}
								}
							}
						}
					}

					if(number_of_nearest_neighbours >= 3)
					{
						cov[0][0]/=number_of_nearest_neighbours;
						cov[0][1]/=number_of_nearest_neighbours;
						cov[0][2]/=number_of_nearest_neighbours;
						cov[1][0]/=number_of_nearest_neighbours;
						cov[1][1]/=number_of_nearest_neighbours;
						cov[1][2]/=number_of_nearest_neighbours;
						cov[2][0]/=number_of_nearest_neighbours;
						cov[2][1]/=number_of_nearest_neighbours;
						cov[2][2]/=number_of_nearest_neighbours;

						double U[3][3], V[3][3];
						double SS[9];
						gpuSVD((double *)cov, (double *)U, (double *)SS, (double *)V);
						double _nx = (float)(U[0][1]*U[1][2] - U[0][2]*U[1][1]);
						double _ny = (float)(-(U[0][0]*U[1][2] - U[0][2]*U[1][0] ));
						double _nz = (float)(U[0][0]*U[1][1] - U[0][1]*U[1][0]);

						double lenght = sqrt(_nx*_nx + _ny*_ny + _nz*_nz);
						if(lenght==0)
						{
							d_point_cloud[hashed_index_of_point].normal_x = 0.0f;
							d_point_cloud[hashed_index_of_point].normal_y = 0.0f;
							d_point_cloud[hashed_index_of_point].normal_z = 0.0f;
							d_point_cloud[hashed_index_of_point].curvature = 0.0f;
						}else
						{
							d_point_cloud[hashed_index_of_point].normal_x = _nx/lenght;
							d_point_cloud[hashed_index_of_point].normal_y = _ny/lenght;
							d_point_cloud[hashed_index_of_point].normal_z = _nz/lenght;
							d_point_cloud[hashed_index_of_point].curvature = SS[0]/(SS[0] + SS[4] + SS[8]);
						}
					}
					else
					{
						d_point_cloud[hashed_index_of_point].normal_x = 0.0f;
						d_point_cloud[hashed_index_of_point].normal_y = 0.0f;
						d_point_cloud[hashed_index_of_point].normal_z = 0.0f;
						d_point_cloud[hashed_index_of_point].curvature = 0.0f;
					}
				}
			}
		}
	}
}


cudaError_t cudaCalculateNormalVectorsFast(
	int threads,
	pcl::PointNormal * d_point_cloud,
	int number_of_points,
	hashElement* d_hashTable,
	bucket* d_buckets,
	simple_point3D* d_mean,
	gridParameters rgd_params,
	float search_radius,
	int max_number_considered_in_INNER_bucket,
	int max_number_considered_in_OUTER_bucket)
{
	cudaError_t err = cudaGetLastError();

	int blocks=number_of_points/threads+1;

	kernel_normalvectorcomputation_step1_fast<<<blocks,threads>>>(
		d_point_cloud,
		d_hashTable,
		d_mean,
		number_of_points,
		d_buckets,
		rgd_params,
		search_radius,
		max_number_considered_in_INNER_bucket,
		max_number_considered_in_OUTER_bucket);
	err = cudaDeviceSynchronize();
	if(err != ::cudaSuccess)return err;

	kernel_normalvectorcomputation_step2_fast<<<blocks,threads>>>(
		d_point_cloud,
		d_hashTable,
		d_mean,
		number_of_points,
		d_buckets,
		rgd_params,
		search_radius,
		max_number_considered_in_INNER_bucket,
		max_number_considered_in_OUTER_bucket);
	err = cudaDeviceSynchronize();

	return err;
}

__global__ void  kernel_calculateProjections (
		pcl::PointNormal *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		pcl::PointXYZ *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		hashElement *d_hashTable,
		bucket * d_buckets,
		gridParameters rgd_params,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		float projections_search_radius,
		char *d_v_is_projection,
		pcl::PointXYZ *d_second_point_cloud_projections)
{
	int index_of_point_second_point_cloud = blockIdx.x*blockDim.x+threadIdx.x;

	if(index_of_point_second_point_cloud < number_of_points_second_point_cloud)
	{
		bool isok = false;
		float final_projection_distance = 0.0f;
		int   final_nn_index = -1;

		float x_second_point_cloud = d_second_point_cloud[index_of_point_second_point_cloud].x;
		float y_second_point_cloud = d_second_point_cloud[index_of_point_second_point_cloud].y;
		float z_second_point_cloud = d_second_point_cloud[index_of_point_second_point_cloud].z;

		if(x_second_point_cloud < rgd_params.bounding_box_min_X || x_second_point_cloud > rgd_params.bounding_box_max_X)
		{
			d_v_is_projection[index_of_point_second_point_cloud] = false;
			return;
		}
		if(y_second_point_cloud < rgd_params.bounding_box_min_Y || y_second_point_cloud > rgd_params.bounding_box_max_Y)
		{
			d_v_is_projection[index_of_point_second_point_cloud] = false;
			return;
		}
		if(z_second_point_cloud < rgd_params.bounding_box_min_Z || z_second_point_cloud > rgd_params.bounding_box_max_Z)
		{
			d_v_is_projection[index_of_point_second_point_cloud] = false;
			return;
		}

		int ix=(x_second_point_cloud - rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(y_second_point_cloud - rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(z_second_point_cloud - rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;

		int index_bucket = ix*rgd_params.number_of_buckets_Y *
				rgd_params.number_of_buckets_Z + iy * rgd_params.number_of_buckets_Z + iz;

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

									float nx_first_point_cloud = d_first_point_cloud[hashed_index_of_point].normal_x;
									float ny_first_point_cloud = d_first_point_cloud[hashed_index_of_point].normal_y;
									float nz_first_point_cloud = d_first_point_cloud[hashed_index_of_point].normal_z;
									float x_first_point_cloud = d_first_point_cloud[hashed_index_of_point].x;
									float y_first_point_cloud = d_first_point_cloud[hashed_index_of_point].y;
									float z_first_point_cloud = d_first_point_cloud[hashed_index_of_point].z;

									float dist  = (x_second_point_cloud - x_first_point_cloud) * (x_second_point_cloud - x_first_point_cloud) +
												  (y_second_point_cloud - y_first_point_cloud) * (y_second_point_cloud - y_first_point_cloud) +
												  (z_second_point_cloud - z_first_point_cloud) * (z_second_point_cloud - z_first_point_cloud);

									float projection_distance = nx_first_point_cloud * x_second_point_cloud +
												ny_first_point_cloud * y_second_point_cloud +
												nz_first_point_cloud * z_second_point_cloud -
												nx_first_point_cloud * x_first_point_cloud -
												ny_first_point_cloud * y_first_point_cloud -
												nz_first_point_cloud * z_first_point_cloud;

									float abs_projection_distance = abs(projection_distance);

									if(dist <= projections_search_radius * projections_search_radius)
									{
										if(abs_projection_distance < _distance)
										{
											isok = true;
											_distance = abs_projection_distance;
											final_projection_distance = projection_distance;
											final_nn_index = hashed_index_of_point;
										}

										//if(dist < _distance)
										//{
										//	isok = true;
										//	nn_index = hashed_index_of_point;
										//	_distance = dist;
										//}
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
			d_v_is_projection[index_of_point_second_point_cloud] = 1;
			d_second_point_cloud_projections[index_of_point_second_point_cloud].x =
					d_second_point_cloud[index_of_point_second_point_cloud].x -
					d_first_point_cloud[final_nn_index].normal_x * final_projection_distance;

			d_second_point_cloud_projections[index_of_point_second_point_cloud].y =
					d_second_point_cloud[index_of_point_second_point_cloud].y -
					d_first_point_cloud[final_nn_index].normal_y * final_projection_distance;

			d_second_point_cloud_projections[index_of_point_second_point_cloud].z =
					d_second_point_cloud[index_of_point_second_point_cloud].z -
					d_first_point_cloud[final_nn_index].normal_z * final_projection_distance;

		}else
		{
			d_v_is_projection[index_of_point_second_point_cloud] = 0;
			d_second_point_cloud_projections[index_of_point_second_point_cloud].x = 0.0f;
			d_second_point_cloud_projections[index_of_point_second_point_cloud].y = 0.0f;
			d_second_point_cloud_projections[index_of_point_second_point_cloud].z = 0.0f;
		}
	}
}

cudaError_t cudaCalculateProjections(
		int threads,
		pcl::PointNormal *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		pcl::PointXYZ *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		hashElement *d_hashTable,
		bucket * d_buckets,
		gridParameters rgd_params,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		float projections_search_radius,
		char *d_v_is_projection,
		pcl::PointXYZ *d_second_point_cloud_projections)
{
	cudaError_t err = cudaGetLastError();
	if(err != ::cudaSuccess)return err;

	int blocks=number_of_points_second_point_cloud/threads+1;

	kernel_calculateProjections<<<blocks, threads>>> (
			d_first_point_cloud,
			number_of_points_first_point_cloud,
			d_second_point_cloud,
			number_of_points_second_point_cloud,
			d_hashTable,
			d_buckets,
			rgd_params,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			projections_search_radius,
			d_v_is_projection,
			d_second_point_cloud_projections);
	err = cudaDeviceSynchronize();

	return err;
}
/*
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
}*/

__global__ void kernel_cudaTransformPointCloud(pcl::PointXYZ *d_in_point_cloud,
												pcl::PointXYZ *d_out_point_cloud,
											   int number_of_points,
											   double *d_matrix)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<number_of_points)
	{
		double vSrcVector[3] = {d_in_point_cloud[ind].x, d_in_point_cloud[ind].y, d_in_point_cloud[ind].z};
		double vOut[3];
		vOut[0]=d_matrix[0]*vSrcVector[0]+d_matrix[4]*vSrcVector[1]+d_matrix[8]*vSrcVector[2]+d_matrix[12];
   	 	vOut[1]=d_matrix[1]*vSrcVector[0]+d_matrix[5]*vSrcVector[1]+d_matrix[9]*vSrcVector[2]+d_matrix[13];
    	vOut[2]=d_matrix[2]*vSrcVector[0]+d_matrix[6]*vSrcVector[1]+d_matrix[10]*vSrcVector[2]+d_matrix[14];

    	d_out_point_cloud[ind].x = vOut[0];
    	d_out_point_cloud[ind].y = vOut[1];
    	d_out_point_cloud[ind].z = vOut[2];
	}
}

cudaError_t cudaTransformPointCloud(int threads,
								pcl::PointXYZ *d_in_point_cloud,
								pcl::PointXYZ *d_out_point_cloud,
								int number_of_points,
								double *d_matrix)
{
	kernel_cudaTransformPointCloud<<<number_of_points/threads+1,threads>>>
			(d_in_point_cloud, d_out_point_cloud, number_of_points, d_matrix);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}


__global__ void
kernel_cudaPrepareProjectionIndexes(char *d_v_is_projection, int  *d_nearest_neighbour_indexes,	int number_of_points)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<number_of_points)
	{
		if(d_v_is_projection[ind] == 0)
		{
			d_nearest_neighbour_indexes[ind] = -1;
		}else
		{
			d_nearest_neighbour_indexes[ind] = ind;
		}
	}
}

cudaError_t cudaPrepareProjectionIndexes(
		int threads,
		char *d_v_is_projection,
		int  *d_nearest_neighbour_indexes,
		int number_of_points)
{
	kernel_cudaPrepareProjectionIndexes<<<number_of_points/threads+1,threads>>>
				(d_v_is_projection, d_nearest_neighbour_indexes, number_of_points);

	cudaDeviceSynchronize();

	return cudaGetLastError();
}



__global__ void kernel_copy_NN_with_NN_assuption(double *d_temp_double_mem, int *d_nearest_neighbour_indexes, int number_of_points)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index < number_of_points)
	{
		int i = d_nearest_neighbour_indexes[index];
		if(i != -1)
		{
			d_temp_double_mem[index] = 1.0f;
		}else
		{
			d_temp_double_mem[index] = 0.0f;
		}
	}
}

__global__ void
kernel_copy_ref_with_NN_assuption(
		double *d_temp_double_mem,
		int _case,
		int number_of_points,
		pcl::PointXYZ *_Ref,
		pcl::PointXYZ *_ToAlign,
		int *d_nearest_neighbour_indexes
		)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	double Ref;

	if(index < number_of_points)
	{
		d_temp_double_mem[index] = 0.0f;
		int i = d_nearest_neighbour_indexes[index];
		if(i != -1)
		{
			if(_case==0)Ref = _Ref[i].x;
			if(_case==1)Ref = _Ref[i].y;
			if(_case==2)Ref = _Ref[i].z;
			if(_case==3)Ref = _ToAlign[index].x;
			if(_case==4)Ref = _ToAlign[index].y;
			if(_case==5)Ref = _ToAlign[index].z;
			d_temp_double_mem[index] = Ref;
		}
	}
}

__global__ void
kernel_d_table_m_d_divide_NN(double *_d_table_m_d)
{
	int bx = blockIdx.x;
	int index = bx;

	if(index > 0 && index < 7)
	{
		double N = _d_table_m_d[0];
		double t = _d_table_m_d[index];
		t = t/N;
		_d_table_m_d[index] = t;
	}
}

__global__ void
kernel_copy_ref_with_NN_assuption_to_C(
		double *_d_temp_double_mem,
		double *_d_table_m_d,
		int _case,
		int _amountOfPoints,
		pcl::PointXYZ *_Ref,
		pcl::PointXYZ *_ToAlign,
		int *d_nearest_neighbour_indexes)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index < _amountOfPoints)
	{
	double dm1 = _d_table_m_d[1];
	double dm2 = _d_table_m_d[2];
	double dm3 = _d_table_m_d[3];
	double dm4 = _d_table_m_d[4];
	double dm5 = _d_table_m_d[5];
	double dm6 = _d_table_m_d[6];

		switch(_case)
		{
			case 0: //Cxx
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double cxx = 0.0f;
					double xref = _Ref[ind].x;
					double xtoalign = _ToAlign[index].x;
					xref -= dm1;
					xtoalign -= dm4;
					cxx = xref * xtoalign;
					_d_temp_double_mem[index] = cxx;
				}
				break;
			}
			case 1: //Cxy
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double cxy = 0.0f;
					double xref =  _Ref[ind].x;
					double ytoalign = _ToAlign[index].y;
					xref -= dm1;// cmx
					ytoalign -= dm5;//cdy
					cxy = xref * ytoalign;
					_d_temp_double_mem[index] = cxy;
				}
				break;
			}
			case 2: //Cxz
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double cxz = 0.0f;
					double xref =  _Ref[ind].x;
					double ztoalign = _ToAlign[index].z;
					xref -= dm1;// cmx
					ztoalign -= dm6;//cdz
					cxz = xref * ztoalign;
					_d_temp_double_mem[index] = cxz;
				}
				break;
			}
			case 3: //Cyx
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double cyx = 0.0f;
					double yref =  _Ref[ind].y;
					double xtoalign = _ToAlign[index].x;
					yref -= dm2;// cmy
					xtoalign -= dm4;//cdx
					cyx = yref * xtoalign;
					_d_temp_double_mem[index] = cyx;
				}
				break;
			}
			case 4: //Cyy
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double cyy = 0.0f;
					double yref =  _Ref[ind].y;
					double ytoalign = _ToAlign[index].y;
					yref -= dm2;// cmy
					ytoalign -= dm5;//cdy
					cyy = yref * ytoalign;
					_d_temp_double_mem[index] = cyy;
				}
				break;
			}
			case 5: //Cyz
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double cyz = 0.0f;
					double yref =  _Ref[ind].y;
					double ztoalign = _ToAlign[index].z;
					yref -= dm2;// cmy
					ztoalign -= dm6;//cdz
					cyz = yref * ztoalign;
					_d_temp_double_mem[index] = cyz;
				}
				break;
			}
			case 6: //Czx
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double czx = 0.0f;
					double zref =  _Ref[ind].z;
					double xtoalign = _ToAlign[index].x;
					zref -= dm3;// cmz
					xtoalign -= dm4;//cdx
					czx = zref * xtoalign;
					_d_temp_double_mem[index] = czx;
				}
				break;
			}
			case 7: //Czy
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double czy = 0.0f;
					double zref =  _Ref[ind].z;
					double ytoalign = _ToAlign[index].y;
					zref -= dm3;// cmz
					ytoalign -= dm5;//cdy
					czy = zref * ytoalign;
					_d_temp_double_mem[index] = czy;
				}
				break;
			}
			case 8: //Czz
			{
				int ind = d_nearest_neighbour_indexes[index];
				if(ind != -1)
				{
					double czz = 0.0f;
					double zref =  _Ref[ind].z;
					double ztoalign = _ToAlign[index].z;
					zref -= dm3;// cmz
					ztoalign -= dm6;//cdz
					czz = zref * ztoalign;
					_d_temp_double_mem[index] = czz;
				}
				break;
			}
		}
	}
}


__host__ __device__
void
kernel_ata3(double *  AA, double *  A)
{
	AA[3*0+0] = A[3*0+0]*A[3*0+0] + A[3*0+1]*A[3*0+1] + A[3*0+2]*A[3*0+2];
	AA[3*1+0] = A[3*0+0]*A[3*1+0] + A[3*0+1]*A[3*1+1] + A[3*0+2]*A[3*1+2];
	AA[3*2+0] = A[3*0+0]*A[3*2+0] + A[3*0+1]*A[3*2+1] + A[3*0+2]*A[3*2+2];

	AA[3*0+1] = AA[3*1+0];
	AA[3*1+1] = A[3*1+0]*A[3*1+0] + A[3*1+1]*A[3*1+1] + A[3*1+2]*A[3*1+2];
	AA[3*2+1] = A[3*1+0]*A[3*2+0] + A[3*1+1]*A[3*2+1] + A[3*1+2]*A[3*2+2];

	AA[3*0+2] = AA[3*2+0];
	AA[3*1+2] = AA[3*2+1];
	AA[3*2+2] = A[3*2+0]*A[3*2+0] + A[3*2+1]*A[3*2+1] + A[3*2+2]*A[3*2+2];
}


__host__ __device__
void kernel_solvecubic(double *  c)
{
	double sq3d2 = 0.86602540378443864676, c2d3 = c[2]/3,
		c2sq = c[2]*c[2], Q = (3*c[1]-c2sq)/9,
		R = (c[2]*(9*c[1]-2*c2sq)-27*c[0])/54;
	double tmp, t, sint, cost;

	if (Q < 0) {
		/*
		 * Instead of computing
		 * c_0 = A cos(t) - B
		 * c_1 = A cos(t + 2 pi/3) - B
		 * c_2 = A cos(t + 4 pi/3) - B
		 * Use cos(a+b) = cos(a) cos(b) - sin(a) sin(b)
		 * Keeps t small and eliminates 1 function call.
		 * cos(2 pi/3) = cos(4 pi/3) = -0.5
		 * sin(2 pi/3) = sqrtf(3.0f)/2
		 * sin(4 pi/3) = -sqrtf(3.0f)/2
		 */

		tmp = 2*sqrt(-Q);
		t = acos(R/sqrt(-Q*Q*Q))/3;
		cost = tmp*cos(t);
		sint = tmp*sin(t);

		c[0] = cost - c2d3;

		cost = -0.5*cost - c2d3;
		sint = sq3d2*sint;

		c[1] = cost - sint;
		c[2] = cost + sint;
	}
	else {
		tmp = cbrt(R);
		c[0] = -c2d3 + 2*tmp;
		c[1] = c[2] = -c2d3 - tmp;
	}
}


__host__ __device__
void kernel_sort3(double *  x)
{
	double tmp;

	if (x[0] < x[1]) {
		tmp = x[0];
		x[0] = x[1];
		x[1] = tmp;
	}
	if (x[1] < x[2]) {
		if (x[0] < x[2]) {
			tmp = x[2];
			x[2] = x[1];
			x[1] = x[0];
			x[0] = tmp;
		}
		else {
			tmp = x[1];
			x[1] = x[2];
			x[2] = tmp;
		}
	}
}

__host__ __device__
void kernel_ldu3(double *  A, int *  P)
{
	int tmp;

	P[1] = 1;
	P[2] = 2;

	P[0] = abs(A[3*1+0]) > abs(A[3*0+0]) ?
		(abs(A[3*2+0]) > abs(A[3*1+0]) ? 2 : 1) :
		(abs(A[3*2+0]) > abs(A[3*0+0]) ? 2 : 0);
	P[P[0]] = 0;

	if (abs(A[3*P[2]+1]) > abs(A[3*P[1]+1])) {
		tmp = P[1];
		P[1] = P[2];
		P[2] = tmp;
	}

	if (A[3*P[0]+0] != 0) {
		A[3*P[1]+0] = A[3*P[1]+0]/A[3*P[0]+0];
		A[3*P[2]+0] = A[3*P[2]+0]/A[3*P[0]+0];
		A[3*P[0]+1] = A[3*P[0]+1]/A[3*P[0]+0];
		A[3*P[0]+2] = A[3*P[0]+2]/A[3*P[0]+0];
	}

	A[3*P[1]+1] = A[3*P[1]+1] - A[3*P[0]+1]*A[3*P[1]+0]*A[3*P[0]+0];

	if (A[3*P[1]+1] != 0) {
		A[3*P[2]+1] = (A[3*P[2]+1] - A[3*P[0]+1]*A[3*P[2]+0]*A[3*P[0]+0])/A[3*P[1]+1];
		A[3*P[1]+2] = (A[3*P[1]+2] - A[3*P[0]+2]*A[3*P[1]+0]*A[3*P[0]+0])/A[3*P[1]+1];
	}

	A[3*P[2]+2] = A[3*P[2]+2] - A[3*P[0]+2]*A[3*P[2]+0]*A[3*P[0]+0] - A[3*P[1]+2]*A[3*P[2]+1]*A[3*P[1]+1];
}


__host__ __device__
void kernel_ldubsolve3(double *  x,  double *  y, double *  LDU, const int *  P)
{
	x[P[2]] = y[2];
	x[P[1]] = y[1] - LDU[3*P[2]+1]*x[P[2]];
	x[P[0]] = y[0] - LDU[3*P[2]+0]*x[P[2]] - LDU[3*P[1]+0]*x[P[1]];
}


__host__ __device__
void kernel_cross(double *  z, double *  x, double *  y)
{
	z[0] = x[1]*y[2]-x[2]*y[1];
	z[1] = -(x[0]*y[2]-x[2]*y[0]);
	z[2] = x[0]*y[1]-x[1]*y[0];
}


__host__ __device__
void kernel_matvec3(double *  y, double *  A, double *  x)
{
	y[0] = A[3*0+0]*x[0] + A[3*1+0]*x[1] + A[3*2+0]*x[2];
	y[1] = A[3*0+1]*x[0] + A[3*1+1]*x[1] + A[3*2+1]*x[2];
	y[2] = A[3*0+2]*x[0] + A[3*1+2]*x[1] + A[3*2+2]*x[2];
}


__host__ __device__
void kernel_matmul3(double *  C, double *  A, double *  B)
{
	C[3*0+0] = A[3*0+0]*B[3*0+0] + A[3*1+0]*B[3*0+1] + A[3*2+0]*B[3*0+2];
	C[3*1+0] = A[3*0+0]*B[3*1+0] + A[3*1+0]*B[3*1+1] + A[3*2+0]*B[3*1+2];
	C[3*2+0] = A[3*0+0]*B[3*2+0] + A[3*1+0]*B[3*2+1] + A[3*2+0]*B[3*2+2];

	C[3*0+1] = A[3*0+1]*B[3*0+0] + A[3*1+1]*B[3*0+1] + A[3*2+1]*B[3*0+2];
	C[3*1+1] = A[3*0+1]*B[3*1+0] + A[3*1+1]*B[3*1+1] + A[3*2+1]*B[3*1+2];
	C[3*2+1] = A[3*0+1]*B[3*2+0] + A[3*1+1]*B[3*2+1] + A[3*2+1]*B[3*2+2];

	C[3*0+2] = A[3*0+2]*B[3*0+0] + A[3*1+2]*B[3*0+1] + A[3*2+2]*B[3*0+2];
	C[3*1+2] = A[3*0+2]*B[3*1+0] + A[3*1+2]*B[3*1+1] + A[3*2+2]*B[3*1+2];
	C[3*2+2] = A[3*0+2]*B[3*2+0] + A[3*1+2]*B[3*2+1] + A[3*2+2]*B[3*2+2];
}


__host__ __device__
void kernel_unit3(double *  x)
{
	double tmp = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
	x[0] /= tmp;
	x[1] /= tmp;
	x[2] /= tmp;
}

__host__ __device__
void kernel_trans3(double *  A)
{
	double tmp;

	tmp = A[3*1+0];
	A[3*1+0] = A[3*0+1];
	A[3*0+1] = tmp;

	tmp = A[3*2+0];
	A[3*2+0] = A[3*0+2];
	A[3*0+2] = tmp;

	tmp = A[3*2+1];
	A[3*2+1] = A[3*1+2];
	A[3*1+2] = tmp;
}

__global__ void
kernel_compute_R_T(double *_d_table_m_d, double *_d_U, double *_d_S, double *_d_V, double *_d_R_T)
{
	double R[3][3];
	double tx;
	double ty;
	double tz;
	kernel_trans3((double *)_d_U);
	kernel_matmul3((double *)R, (double *)_d_V, (double *)_d_U);

	//mx, my, mz, dx, dy, dz;-> table_m_d
	tx = _d_table_m_d[1] - (R[0][0] * _d_table_m_d[4] + R[0][1] * _d_table_m_d[5] + R[0][2] * _d_table_m_d[6]);
	ty = _d_table_m_d[2] - (R[1][0] * _d_table_m_d[4] + R[1][1] * _d_table_m_d[5] + R[1][2] * _d_table_m_d[6]);
	tz = _d_table_m_d[3] - (R[2][0] * _d_table_m_d[4] + R[2][1] * _d_table_m_d[5] + R[2][2] * _d_table_m_d[6]);

	if(tx != tx)tx = 0;
	if(ty != ty)ty = 0;
	if(tz != tz)tz = 0;

	for(int i = 0 ; i < 16; i++)_d_R_T[i]=0;

			_d_R_T[0] = R[0][0];
			_d_R_T[1] = R[0][1];
			_d_R_T[2] = R[0][2];
			_d_R_T[4] = R[1][0];
			_d_R_T[5] = R[1][1];
			_d_R_T[6] = R[1][2];
			_d_R_T[8] = R[2][0];
			_d_R_T[9] = R[2][1];
			_d_R_T[10] = R[2][2];

			_d_R_T[12] = tx;
			_d_R_T[13] = ty;
			_d_R_T[14] = tz;
			_d_R_T[15] = 1.0;
}


__global__ void
kernel_SVD(double *_A, double *U, double *_S, double *V)
{
	double A[9];
	for(int i = 0 ; i< 9;i++)A[i]= (double)_A[i];

	double thr = 1e-10;
	int P[3], k;
	double y[3], AA[3][3], LDU[3][3];


	//new
	double S[3];
	/*
	 * Steps:
	 * 1) Use eigendecomposition on A^T A to compute V.
	 * Since A = U S V^T then A^T A = V S^T S V^T with D = S^T S and V the
	 * eigenvalues and eigenvectors respectively (V is orthogonal).
	 * 2) Compute U from A and V.
	 * 3) Normalize columns of U and V and root the eigenvalues to obtain
	 * the singular values.
	 */

	/* Compute AA = A^T A */
	kernel_ata3((double *)AA, A);

	/* Form the monic characteristic polynomial */
	S[2] = -AA[0][0] - AA[1][1] - AA[2][2];
	S[1] = AA[0][0]*AA[1][1] + AA[2][2]*AA[0][0] + AA[2][2]*AA[1][1] -
		AA[2][1]*AA[1][2] - AA[2][0]*AA[0][2] - AA[1][0]*AA[0][1];
	S[0] = AA[2][1]*AA[1][2]*AA[0][0] + AA[2][0]*AA[0][2]*AA[1][1] + AA[1][0]*AA[0][1]*AA[2][2] -
		AA[0][0]*AA[1][1]*AA[2][2] - AA[1][0]*AA[2][1]*AA[0][2] - AA[2][0]*AA[0][1]*AA[1][2];

	/* Solve the cubic equation. */
	kernel_solvecubic(S);

	/* All roots should be positive */
	if (S[0] < 0)
		S[0] = 0;
	if (S[1] < 0)
		S[1] = 0;
	if (S[2] < 0)
		S[2] = 0;

	/* Sort from greatest to least */
	kernel_sort3(S);

	/* Form the eigenvector system for the first (largest) eigenvalue */
	//memcpy(LDU,AA,sizeof(LDU));
	for(int i = 0 ; i < 3; i++)
		for(int j = 0 ; j < 3; j++)
		{
			LDU[i][j] = AA[i][j];
		}

	LDU[0][0] -= S[0];
	LDU[1][1] -= S[0];
	LDU[2][2] -= S[0];

	/* Perform LDUP decomposition */
	kernel_ldu3((double *)LDU, P);


	/*
	 * Write LDU = AA-I*lambda.  Then an eigenvector can be
	 * found by solving LDU x = LD y = L z = 0
	 * L is invertible, so L z = 0 implies z = 0
	 * D is singular since det(AA-I*lambda) = 0 and so
	 * D y = z = 0 has a non-unique solution.
	 * Pick k so that D_kk = 0 and set y = e_k, the k'th column
	 * of the identity matrix.
	 * U is invertible so U x = y has a unique solution for a given y.
	 * The solution for U x = y is an eigenvector.
	 */

	/* Pick the component of D nearest to 0 */
	y[0] = y[1] = y[2] = 0;
	k = abs(LDU[P[1]][1]) < abs(LDU[P[0]][0]) ?
		(abs(LDU[P[2]][2]) < abs(LDU[P[1]][1]) ? 2 : 1) :
		(abs(LDU[P[2]][2]) < abs(LDU[P[0]][0]) ? 2 : 0);
	y[k] = 1;

	/* Do a backward solve for the eigenvector */
	kernel_ldubsolve3(V+(3*0+0), y, (double *)LDU, P);

	/* Form the eigenvector system for the last (smallest) eigenvalue */
	//memcpy(LDU,AA,sizeof(LDU));
	for(int i = 0 ; i < 3; i++)
		for(int j = 0 ; j < 3; j++)
		{
			LDU[i][j] = AA[i][j];
		}

	LDU[0][0] -= S[2];
	LDU[1][1] -= S[2];
	LDU[2][2] -= S[2];

	/* Perform LDUP decomposition */
	kernel_ldu3((double *)LDU, P);

	/*
	 * NOTE: The arrangement of the ternary operator output is IMPORTANT!
	 * It ensures a different system is solved if there are 3 repeat eigenvalues.
	 */

	/* Pick the component of D nearest to 0 */
	y[0] = y[1] = y[2] = 0;
	k = abs(LDU[P[0]][0]) < abs(LDU[P[2]][2]) ?
		(abs(LDU[P[0]][0]) < abs(LDU[P[1]][1]) ? 0 : 1) :
		(abs(LDU[P[1]][1]) < abs(LDU[P[2]][2]) ? 1 : 2);
	y[k] = 1;

	/* Do a backward solve for the eigenvector */
	kernel_ldubsolve3(V+(3*2+0), y, (double *)LDU, P);

	 /* The remaining column must be orthogonal (AA is symmetric) */
	kernel_cross(V+(3*1+0), V+(3*2+0), V+(3*0+0));

	/* Count the rank */
	k = (S[0] > thr) + (S[1] > thr) + (S[2] > thr);

	switch (k) {
		case 0:
			/*
			 * Zero matrix.
			 * Since V is already orthogonal, just copy it into U.
			 */
			//memcpy(U,V,9*sizeof(double));
			for(int i = 0 ; i < 9 ; i++)U[i]=V[i];

			break;
		case 1:
			/*
			 * The first singular value is non-zero.
			 * Since A = U S V^T, then A V = U S.
			 * A V_1 = S_11 U_1 is non-zero. Here V_1 and U_1 are
			 * column vectors. Since V_1 is known, we may compute
			 * U_1 = A V_1.  The S_11 factor is not important as
			 * U_1 will be normalized later.
			 */
			kernel_matvec3(U+(3*0+0), A, V+(3*0+0));

			/*
			 * The other columns of U do not contribute to the expansion
			 * and we may arbitrarily choose them (but they do need to be
			 * orthogonal). To ensure the first cross product does not fail,
			 * pick k so that U_k1 is nearest 0 and then cross with e_k to
			 * obtain an orthogonal vector to U_1.
			 */
			y[0] = y[1] = y[2] = 0;
			k = abs(U[3*0+0]) < abs(U[3*0+2]) ?
				(abs(U[3*0+0]) < abs(U[3*0+1]) ? 0 : 1) :
				(abs(U[3*0+1]) < abs(U[3*0+2]) ? 1 : 2);
			y[k] = 1;

			kernel_cross(U+(3*1+0), y, U+(3*0+0));

			/* Cross the first two to obtain the remaining column */
			kernel_cross(U+(3*2+0), U+(3*0+0), U+(3*1+0));
			break;
		case 2:
			/*
			 * The first two singular values are non-zero.
			 * Compute U_1 = A V_1 and U_2 = A V_2. See case 1
			 * for more information.
			 */
			kernel_matvec3(U+(3*0+0), A, V+(3*0+0));
			kernel_matvec3(U+(3*1+0), A, V+(3*1+0));

			/* Cross the first two to obtain the remaining column */
			kernel_cross(U+(3*2+0), U+(3*0+0), U+(3*1+0));
			break;
		case 3:
			/*
			 * All singular values are non-zero.
			 * We may compute U = A V. See case 1 for more information.
			 */
			kernel_matmul3(U, A, V);
			break;
	}



/* Normalize the columns of U and V */
	kernel_unit3(V+(3*0+0));
	kernel_unit3(V+(3*1+0));
	kernel_unit3(V+(3*2+0));

	kernel_unit3(U+(3*0+0));
	kernel_unit3(U+(3*1+0));
	kernel_unit3(U+(3*2+0));

	/* S was initially the eigenvalues of A^T A = V S^T S V^T which are squared. */
	S[0] = sqrt(S[0]);
	S[1] = sqrt(S[1]);
	S[2] = sqrt(S[2]);

	for(int i = 0 ; i < 9 ; i++)_S[i]=0.0;

	_S[0] = S[0];
	_S[4] = S[1];
	_S[8] = S[2];
}

cudaError_t cudaICP(
				int threads,
				pcl::PointXYZ *d_first_point_cloud,
				int number_of_points_first_point_cloud,
				pcl::PointXYZ *d_second_point_cloud,
				int *d_nearest_neighbour_indexes,
				int number_of_points_second_point_cloud,
				double *d_mICP,
				bool &icp_computationsucced)
{
	icp_computationsucced = false;

	cudaError_t errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess)return errCUDA;

	double *_d_temp_double_mem;
	errCUDA  = cudaMalloc((void**)&_d_temp_double_mem, number_of_points_second_point_cloud*sizeof(double) );
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	double *_d_table_m_d;
	errCUDA  = cudaMalloc((void**)&_d_table_m_d, 7* sizeof(double) );
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	double *_d_table_C;
	errCUDA  = cudaMalloc((void**)&_d_table_C, 9* sizeof(double) );
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	double *d_U;
	errCUDA  = cudaMalloc((void**)&d_U, 9* sizeof(double) );
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	double *d_S;
	errCUDA  = cudaMalloc((void**)&d_S, 9* sizeof(double) );
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	double *d_V;
	errCUDA  = cudaMalloc((void**)&d_V, 9* sizeof(double) );
	if(errCUDA != ::cudaSuccess){return errCUDA;}


	kernel_copy_NN_with_NN_assuption<<<number_of_points_second_point_cloud/threads+1,threads>>>(
			_d_temp_double_mem,
			d_nearest_neighbour_indexes,
			number_of_points_second_point_cloud);
	errCUDA = cudaDeviceSynchronize();
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	try
	{
		thrust::device_ptr <double> dev_ptr_d_temp_double_mem ( _d_temp_double_mem );
		thrust::device_ptr <double> dev_ptr_d_table_m_d ( _d_table_m_d );
		thrust::device_ptr <double> dev_ptr_d_table_C ( _d_table_C );
		double sum = thrust::reduce (dev_ptr_d_temp_double_mem , dev_ptr_d_temp_double_mem + number_of_points_second_point_cloud);
		thrust::fill ( dev_ptr_d_table_m_d , dev_ptr_d_table_m_d + 1, ( double ) sum);

		for(int kk=0;kk<6;kk++)
		{
			kernel_copy_ref_with_NN_assuption<<<number_of_points_second_point_cloud/threads+1,threads>>>
				(_d_temp_double_mem,
						kk,
						number_of_points_second_point_cloud,
						d_first_point_cloud,
						d_second_point_cloud,
						d_nearest_neighbour_indexes);
			errCUDA = cudaDeviceSynchronize();
			if(errCUDA != ::cudaSuccess){return errCUDA;}

			double sum = thrust::reduce (dev_ptr_d_temp_double_mem , dev_ptr_d_temp_double_mem + number_of_points_second_point_cloud);
			if(sum == 0.0)
			{
				icp_computationsucced = false;
				return cudaGetLastError();
			}

			thrust::fill ( dev_ptr_d_table_m_d+kk+1 , dev_ptr_d_table_m_d + kk + 2, ( double ) sum);
		}
		kernel_d_table_m_d_divide_NN<<<8, 1>>>(_d_table_m_d);
		errCUDA = cudaDeviceSynchronize();
		if(errCUDA != ::cudaSuccess) return errCUDA;

		for(int kk=0;kk<9;kk++)
		{
			kernel_copy_ref_with_NN_assuption_to_C<<<number_of_points_second_point_cloud/threads+1,threads>>>(
					_d_temp_double_mem,
					_d_table_m_d,
					kk,
					number_of_points_second_point_cloud,
					d_first_point_cloud,
					d_second_point_cloud,
					d_nearest_neighbour_indexes);
			errCUDA = cudaDeviceSynchronize();
			if(errCUDA != ::cudaSuccess) return errCUDA;

			double sum = thrust::reduce (dev_ptr_d_temp_double_mem , dev_ptr_d_temp_double_mem + number_of_points_second_point_cloud);
			if(sum == 0.0)
			{
				icp_computationsucced = false;
				return cudaGetLastError();
			}

			thrust::fill ( dev_ptr_d_table_C+kk , dev_ptr_d_table_C + kk + 1, ( double ) sum);
		}
	}//try
	catch(thrust::system_error &e)
	{
		errCUDA = cudaGetLastError();
		return errCUDA;
	}
	catch(std::bad_alloc &e)
	{
		errCUDA = cudaGetLastError();

		return errCUDA;
	}


	kernel_SVD<<<1, 1>>>(_d_table_C, d_U, d_S, d_V);
	errCUDA = cudaDeviceSynchronize();
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	kernel_compute_R_T<<<1, 1>>>(_d_table_m_d, d_U, d_S, d_V, d_mICP);
	errCUDA = cudaDeviceSynchronize();
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	kernel_cudaTransformPointCloud<<<number_of_points_second_point_cloud/threads+1,threads>>>	(d_second_point_cloud, d_second_point_cloud, number_of_points_second_point_cloud, d_mICP);
	errCUDA = cudaDeviceSynchronize();
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	errCUDA = cudaFree(_d_temp_double_mem); _d_temp_double_mem = 0;
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	errCUDA = cudaFree(_d_table_m_d); _d_table_m_d = 0;
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	errCUDA = cudaFree(_d_table_C); _d_table_C = 0;
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	errCUDA = cudaFree(d_U); d_U = 0;
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	errCUDA = cudaFree(d_S); d_S = 0;
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	errCUDA = cudaFree(d_V); d_V = 0;
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	errCUDA = cudaGetLastError();
	if(errCUDA == ::cudaSuccess)icp_computationsucced = true;

	return errCUDA;
}


__global__ void
kernel_transform_points_via_centroid(pcl::PointXYZ *cudaData, int  number_of_points, pcl::PointXYZ centroid)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index < number_of_points)
	{
		float x = cudaData[index].x - centroid.x;
		float y = cudaData[index].y - centroid.y;
		float z = cudaData[index].z - centroid.z;

		cudaData[index].x = x;
		cudaData[index].y = y;
		cudaData[index].z = z;
	}
}

__global__ void
kernel_transform_points_via_centroid(pcl::PointNormal *cudaData, int  number_of_points, pcl::PointXYZ centroid)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index < number_of_points)
	{
		float x = cudaData[index].x - centroid.x;
		float y = cudaData[index].y - centroid.y;
		float z = cudaData[index].z - centroid.z;

		cudaData[index].x = x;
		cudaData[index].y = y;
		cudaData[index].z = z;
	}
}

struct add_value
{
	__host__ __device__
	float operator()(float lhs, float rhs)
	{
		return lhs + rhs;
	}
};

__global__ void
kernel_copy_x(float *_d_temp_table, pcl::PointXYZ *cudaData, int nop)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index < nop)
	{
		_d_temp_table[index] = cudaData[index].x;
	}
}

__global__ void
kernel_copy_y(float *_d_temp_table, pcl::PointXYZ *cudaData, int nop)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index < nop)
	{
		_d_temp_table[index] = cudaData[index].y;
	}
}

__global__ void
kernel_copy_z(float *_d_temp_table, pcl::PointXYZ *cudaData, int nop)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;

	if(index < nop)
	{
		_d_temp_table[index] = cudaData[index].z;
	}
}


cudaError_t cudaTransformViaCentroidOfSecondCloud(int threads, pcl::PointNormal *cudaDataA, int nopA, pcl::PointXYZ *cudaDataB, int nopB, pcl::PointXYZ &centroid)
{
	cudaError_t errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess)return errCUDA;

	try
	{
		float p_x = 0.0f;
		float p_y = 0.0f;
		float p_z = 0.0f;

		float *_d_temp_table;
		errCUDA  = cudaMalloc((void**)&_d_temp_table, nopB*sizeof(float) );
		if(errCUDA != ::cudaSuccess){return errCUDA;}

		thrust::device_ptr <float> dev_ptr_temp_table( _d_temp_table );
		add_value op;

		kernel_copy_x<<<nopB/threads+1,threads>>>(_d_temp_table, cudaDataB, nopB);
		errCUDA = cudaDeviceSynchronize();
			if(errCUDA != ::cudaSuccess){return errCUDA;}
		p_x = thrust::reduce (dev_ptr_temp_table , dev_ptr_temp_table + nopB, p_x, op);

		kernel_copy_y<<<nopB/threads+1,threads>>>(_d_temp_table, cudaDataB, nopB);
		errCUDA = cudaDeviceSynchronize();
			if(errCUDA != ::cudaSuccess){return errCUDA;}
		p_y = thrust::reduce (dev_ptr_temp_table , dev_ptr_temp_table + nopB, p_y, op);

		kernel_copy_z<<<nopB/threads+1,threads>>>(_d_temp_table, cudaDataB, nopB);
		errCUDA = cudaDeviceSynchronize();
			if(errCUDA != ::cudaSuccess){return errCUDA;}
		p_z = thrust::reduce (dev_ptr_temp_table , dev_ptr_temp_table + nopB, p_z, op);


		errCUDA = cudaFree(_d_temp_table); _d_temp_table = 0;
		if(errCUDA != ::cudaSuccess){return errCUDA;}


		centroid.x = p_x / nopB;
		centroid.y = p_y / nopB;
		centroid.z = p_z / nopB;

	}//try
	catch(thrust::system_error &e)
	{
		return cudaGetLastError();
	}
	catch(std::bad_alloc &e)
  	{
  	 	return cudaGetLastError();
  	}

	kernel_transform_points_via_centroid<<<nopA/threads+1,threads>>>(cudaDataA, nopA, centroid);
	errCUDA = cudaDeviceSynchronize();
	if(errCUDA != ::cudaSuccess){return errCUDA;}

	kernel_transform_points_via_centroid<<<nopB/threads+1,threads>>>(cudaDataB, nopB, centroid);
	errCUDA = cudaDeviceSynchronize();
	return errCUDA;
}


