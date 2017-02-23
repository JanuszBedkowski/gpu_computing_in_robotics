#include "lesson_12.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
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

cudaError_t cudaCalculateGridParams(lidar_pointcloud::PointXYZIRNL* d_point_cloud, int number_of_points,
	float resolution_X, float resolution_Y, float resolution_Z, float bounding_box_extension, gridParameters &out_rgd_params)
{
	cudaError_t err = cudaGetLastError();

	try
	{
		thrust::device_ptr<lidar_pointcloud::PointXYZIRNL> t_cloud(d_point_cloud);
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<lidar_pointcloud::PointXYZIRNL>,thrust::device_ptr<lidar_pointcloud::PointXYZIRNL> >
		 minmaxX=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareX());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<lidar_pointcloud::PointXYZIRNL>,thrust::device_ptr<lidar_pointcloud::PointXYZIRNL> >
		 minmaxY=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareY());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
	
		thrust::pair<thrust::device_ptr<lidar_pointcloud::PointXYZIRNL>,thrust::device_ptr<lidar_pointcloud::PointXYZIRNL> >
		 minmaxZ=thrust::minmax_element(t_cloud,t_cloud+number_of_points,compareZ());
		err = cudaGetLastError();
		if(err != ::cudaSuccess)return err;
		
		lidar_pointcloud::PointXYZIRNL minX,maxX,minZ,maxZ,minY,maxY;

		err = cudaMemcpy(&minX,minmaxX.first.get(),sizeof(lidar_pointcloud::PointXYZIRNL),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxX,minmaxX.second.get(),sizeof(lidar_pointcloud::PointXYZIRNL),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&minZ,minmaxZ.first.get(),sizeof(lidar_pointcloud::PointXYZIRNL),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxZ,minmaxZ.second.get(),sizeof(lidar_pointcloud::PointXYZIRNL),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&minY,minmaxY.first.get(),sizeof(lidar_pointcloud::PointXYZIRNL),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return err;
		err = cudaMemcpy(&maxY,minmaxY.second.get(),sizeof(lidar_pointcloud::PointXYZIRNL),cudaMemcpyDeviceToHost);
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

__global__ void kernel_getIndexOfBucketForPoints(lidar_pointcloud::PointXYZIRNL* cloud, hashElement* d_hashTable, int number_of_points, gridParameters rgd_params)
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

cudaError_t cudaCalculateGrid(int threads, lidar_pointcloud::PointXYZIRNL *d_point_cloud, bucket *d_buckets,
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

__global__ void  kernel_calculateProjections (
		lidar_pointcloud::PointXYZIRNL *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		lidar_pointcloud::PointXYZIRNL *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		hashElement *d_hashTable,
		bucket * d_buckets,
		gridParameters rgd_params,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		float projections_search_radius,
		lidar_pointcloud::PointProjection *d_projections)
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

		d_projections[index_of_point_second_point_cloud].x_src = x_second_point_cloud;
		d_projections[index_of_point_second_point_cloud].y_src = y_second_point_cloud;
		d_projections[index_of_point_second_point_cloud].z_src = z_second_point_cloud;
		d_projections[index_of_point_second_point_cloud].isProjection = 0;
		d_projections[index_of_point_second_point_cloud].x_dst = 0.0f;
		d_projections[index_of_point_second_point_cloud].y_dst = 0.0f;
		d_projections[index_of_point_second_point_cloud].z_dst = 0.0f;
		d_projections[index_of_point_second_point_cloud].normal_x = 0.0f;
		d_projections[index_of_point_second_point_cloud].normal_y = 0.0f;
		d_projections[index_of_point_second_point_cloud].normal_z = 0.0f;



		if(x_second_point_cloud < rgd_params.bounding_box_min_X || x_second_point_cloud > rgd_params.bounding_box_max_X)
		{
			d_projections[index_of_point_second_point_cloud].isProjection = 0;
			return;
		}
		if(y_second_point_cloud < rgd_params.bounding_box_min_Y || y_second_point_cloud > rgd_params.bounding_box_max_Y)
		{
			d_projections[index_of_point_second_point_cloud].isProjection = 0;
			return;
		}
		if(z_second_point_cloud < rgd_params.bounding_box_min_Z || z_second_point_cloud > rgd_params.bounding_box_max_Z)
		{
			d_projections[index_of_point_second_point_cloud].isProjection = 0;
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
			d_projections[index_of_point_second_point_cloud].isProjection = 1;

			d_projections[index_of_point_second_point_cloud].x_dst = d_second_point_cloud[index_of_point_second_point_cloud].x -
					d_first_point_cloud[final_nn_index].normal_x * final_projection_distance;

			d_projections[index_of_point_second_point_cloud].y_dst = d_second_point_cloud[index_of_point_second_point_cloud].y -
					d_first_point_cloud[final_nn_index].normal_y * final_projection_distance;

			d_projections[index_of_point_second_point_cloud].z_dst = d_second_point_cloud[index_of_point_second_point_cloud].z -
					d_first_point_cloud[final_nn_index].normal_z * final_projection_distance;

			d_projections[index_of_point_second_point_cloud].distance = final_projection_distance;

			d_projections[index_of_point_second_point_cloud].normal_x = d_first_point_cloud[final_nn_index].normal_x;
			d_projections[index_of_point_second_point_cloud].normal_y = d_first_point_cloud[final_nn_index].normal_y;
			d_projections[index_of_point_second_point_cloud].normal_z = d_first_point_cloud[final_nn_index].normal_z;


		}else
		{
			d_projections[index_of_point_second_point_cloud].isProjection = 0;
			d_projections[index_of_point_second_point_cloud].x_dst = 0.0f;
			d_projections[index_of_point_second_point_cloud].y_dst = 0.0f;
			d_projections[index_of_point_second_point_cloud].z_dst = 0.0f;
			d_projections[index_of_point_second_point_cloud].normal_x = 0.0f;
			d_projections[index_of_point_second_point_cloud].normal_y = 0.0f;
			d_projections[index_of_point_second_point_cloud].normal_z = 0.0f;
		}
	}
}

cudaError_t cudaCalculateProjections(
		int threads,
		lidar_pointcloud::PointXYZIRNL *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		lidar_pointcloud::PointXYZIRNL *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		hashElement *d_hashTable,
		bucket * d_buckets,
		gridParameters rgd_params,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		float projections_search_radius,
		lidar_pointcloud::PointProjection *d_projections)
{
	//cudaError_t err = cudaGetLastError();
	//if(err != ::cudaSuccess)return err;
	cudaError_t err = ::cudaSuccess;

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
			d_projections);
	err = cudaDeviceSynchronize();

	return err;
}

__global__ void kernel_cudaCompute_AtP(double *d_A, double *d_P, double *d_AtP, int rows, int columns )
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<rows*columns)
	{
		int row = ind%rows;
		int column = ind/rows;

		d_AtP[row + column * rows] = d_A[column + row * columns] * d_P[column];
	}
}

cudaError_t cudaCompute_AtP(int threads, double *d_A, double *d_P, double *d_AtP, int rows, int columns)
{
	cudaError_t err = ::cudaSuccess;

	kernel_cudaCompute_AtP<<<(rows*columns)/threads+1,threads>>>(d_A, d_P, d_AtP, rows, columns);

	err = cudaDeviceSynchronize();
	return err;
}


#define _11 0
#define _12 1
#define _13 2
#define _21 3
#define _22 4
#define _23 5
#define _31 6
#define _32 7
#define _33 8

__device__ void computeR(double om, double fi, double ka, double *R)
{
	//R[11 12 13; 21 22 23; 31 32 33]
	//R[0  1  2 ; 3  4  5 ; 6  7  8]
	R[_11] = cos(fi) * cos(ka);
	R[_12] = -cos(fi) * sin(ka);
	R[_13] = sin(fi);

	R[_21] = cos(om)*sin(ka) + sin(om)*sin(fi)*cos(ka);
	R[_22] = cos(om) *cos(ka) - sin(om)*sin(fi)*sin(ka);
	R[_23] = -sin(om) * cos(fi);

	R[_31] = sin(om) * sin(ka) - cos(om)*sin(fi)*cos(ka);
	R[_32] = sin(om) * cos(ka) + cos(om)*sin(fi)*sin(ka);
	R[_33] = cos(om) * cos(fi);
}

__device__ double compute_a10(double *r, double x0, double y0, double z0)
{
	return r[_11]*x0 + r[_12] * y0 + r[_13] * z0;
}

__device__ double compute_a20(double *r, double x0, double y0, double z0)
{
	return r[_21]*x0 + r[_22] * y0 + r[_23] * z0;
}

__device__ double compute_a30(double *r, double x0, double y0, double z0)
{
	return r[_31] * x0 + r[_32] * y0 + r[_33]*z0;
}

__device__ double compute_a11()
{
	return 0.0;
}

__device__ double compute_a12(double m, double om, double fi, double ka, double x0, double y0, double z0)
{
	return m*(-sin(fi)*cos(ka)*x0 + sin(fi)*sin(ka)*y0 + cos (fi) *z0);
}

__device__ double compute_a13(double m, double *r, double x0, double y0)
{
	return m*(r[_12]*x0-r[_11]*y0);
}

__device__ double compute_a21(double m, double *r, double x0, double y0, double z0)
{
	return m*(-r[_31]*x0-r[_32]*y0-r[_33]*z0);
}

__device__ double compute_a22(double m, double om, double fi, double ka, double x0, double y0, double z0)
{
	return m*(sin(om)*cos(fi)*cos(ka)*x0 - sin(om)*cos(fi)*sin(ka)*y0+sin(om)*sin(fi)*z0);
}

__device__ double compute_a23(double m, double *r, double x0, double y0)
{
	return m*(r[_22]*x0-r[_21]*y0);
}

__device__ double compute_a31(double m, double *r, double x0, double y0, double z0)
{
	return m*(r[_21]*x0+r[_22]*y0 +r[_23]*z0);
}

__device__ double compute_a32(double m, double om, double fi, double ka, double x0, double y0, double z0)
{
	return m * (-cos(om)*cos(fi)*cos(ka)*x0 + cos(om)*cos(fi)*sin(ka)*y0 - cos(om)*sin(fi)*z0);
}

__device__ double compute_a33(double m, double *r, double x0, double y0)
{
	return m*(r[_32]*x0 - r[_31]*y0);
}

__global__ void  kernel_fill_A_l_cuda(double *d_A, double x, double y, double z, double m, double om, double fi, double ka,
		lidar_pointcloud::PointProjection* d_projections, int nop, double *d_P, double PforGround, double PforObstacles, double *d_l)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<nop)
	{
		lidar_pointcloud::PointProjection p = d_projections[ind];

		double r[9];
		computeR(om, fi, ka, r);

		double d_m[16];
		d_m[0] = r[0];
		d_m[1] = r[3];
		d_m[2] = r[6];
		d_m[3] = 0.0;

		d_m[4] = r[1];
		d_m[5] = r[4];
		d_m[6] = r[7];
		d_m[7] = 0.0;

		d_m[8] = r[2];
		d_m[9] = r[5];
		d_m[10] = r[8];
		d_m[11] = 0.0;

		d_m[12] = x;
		d_m[13] = y;
		d_m[14] = z;
		d_m[15] = 1.0;


		if(p.isProjection == 1)
		{
			//printf("la\n");
			//[dtx  dty  dtz   dm                      dom                    dfi                     dka                   ]
			//[gx   gy   gz   (gx*a10+gy*a20+gz*a30)  (gx*a11+gy*a21+gz*a31)  (gx*a12+gy*a22+gz*a32)  (gx*a13+gy*a23+gz*a33)]

			double gx = p.normal_x;
			double gy = p.normal_y;
			double gz = p.normal_z;

			double v[3] = {p.x_src, p.y_src, p.z_src};
			double vt[3];
			vt[0]=d_m[0]*v[0]+d_m[4]*v[1]+d_m[8]*v[2]+d_m[12];
			vt[1]=d_m[1]*v[0]+d_m[5]*v[1]+d_m[9]*v[2]+d_m[13];
			vt[2]=d_m[2]*v[0]+d_m[6]*v[1]+d_m[10]*v[2]+d_m[14];

			double x0 = vt[0] - p.normal_x * p.distance;
			double y0 = vt[1] - p.normal_y * p.distance;
			double z0 = vt[2] - p.normal_z * p.distance;

			double a10 = compute_a10(r, x0, y0, z0);
			double a20 = compute_a20(r, x0, y0, z0);
			double a30 = compute_a30(r, x0, y0, z0);
			double a11 = compute_a11();
			double a12 = compute_a12(m, om, fi, ka, x0, y0, z0);
			double a13 = compute_a13(m, r, x0, y0);
			double a21 = compute_a21(m, r, x0, y0, z0);
			double a22 = compute_a22(m, om, fi, ka, x0, y0, z0);
			double a23 = compute_a23(m, r, x0, y0);
			double a31 = compute_a31(m, r, x0, y0, z0);
			double a32 = compute_a32(m, om, fi, ka, x0, y0, z0);
			double a33 = compute_a33(m, r, x0, y0);

			double for_dm  = gx*a10+gy*a20+gz*a30;
			double for_dom = gx*a11+gy*a21+gz*a31;
			double for_dfi = gx*a12+gy*a22+gz*a32;
			double for_dka = gx*a13+gy*a23+gz*a33;

			d_A[ind + 0 * nop] = gx;
			d_A[ind + 1 * nop] = gy;
			d_A[ind + 2 * nop] = gz;
			d_A[ind + 3 * nop] = for_dm;
			d_A[ind + 4 * nop] = for_dom;
			d_A[ind + 5 * nop] = for_dfi;
			d_A[ind + 6 * nop] = for_dka;

			if(fabs(p.normal_z) > 0.7)
			{
				d_P[ind] = PforGround;
			}else
			{
				d_P[ind] = PforObstacles;
			}

			d_l[ind] = p.distance;
		}else
		{
			d_A[ind + 0 * nop] = 0.0;
			d_A[ind + 1 * nop] = 0.0;
			d_A[ind + 2 * nop] = 0.0;
			d_A[ind + 3 * nop] = 0.0;
			d_A[ind + 4 * nop] = 0.0;
			d_A[ind + 5 * nop] = 0.0;
			d_A[ind + 6 * nop] = 0.0;

			d_P[ind] = 0.0;
			d_l[ind] = 0.0;
		}
	}
}

cudaError_t fill_A_l_cuda(int threads, double *d_A, double x, double y, double z, double m, double om, double fi, double ka,
		lidar_pointcloud::PointProjection* d_projections, int nop, double *d_P, double PforGround, double PforObstacles, double *d_l)
{
	cudaError_t err = ::cudaSuccess;
	int blocks=nop/threads+1;
	kernel_fill_A_l_cuda<<<blocks,threads>>>(d_A, x, y, z, m, om, fi, ka, d_projections, nop, d_P, PforGround, PforObstacles, d_l);

	err = cudaDeviceSynchronize();
	return err;
}
