#include "lesson_14.cuh"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
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

__global__ void  kernel_fill_A_l_cuda(double *d_A, double x, double y, double z, double om, double fi, double ka,
		obs_nn_t *d_obs_nn, int nop, double *d_P, double *d_l)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<nop)
	{
		obs_nn_t obs_nn = d_obs_nn[ind];
		double r[9];
		computeR(om, fi, ka, r);
		double x0 = obs_nn.x0;
		double y0 = obs_nn.y0;
		double z0 = obs_nn.z0;

		double a11 = -1.0;
		double a12 =  0.0;
		double a13 =  0.0;

		double a21 =  0.0;
		double a22 = -1.0;
		double a23 =  0.0;

		double a31 =  0.0;
		double a32 =  0.0;
		double a33 = -1.0;

		double m = 1.0;

		double a14 = compute_a11();
		double a15 = compute_a12(m, om, fi, ka, x0, y0, z0);
		double a16 = compute_a13(m, r, x0, y0);

		double a24 = compute_a21(m, r, x0, y0, z0);
		double a25 = compute_a22(m, om, fi, ka, x0, y0, z0);
		double a26 = compute_a23(m, r, x0, y0);

		double a34 = compute_a31(m, r, x0, y0, z0);
		double a35 = compute_a32(m, om, fi, ka, x0, y0, z0);
		double a36 = compute_a33(m, r, x0, y0);

		d_A[ind * 3 + 0 + 0 * nop * 3] = a11;
		d_A[ind * 3 + 1 + 0 * nop * 3] = a21;
		d_A[ind * 3 + 2 + 0 * nop * 3] = a31;

		d_A[ind * 3 + 0 + 1 * nop * 3] = a12;
		d_A[ind * 3 + 1 + 1 * nop * 3] = a22;
		d_A[ind * 3 + 2 + 1 * nop * 3] = a32;

		d_A[ind * 3 + 0 + 2 * nop * 3] = a13;
		d_A[ind * 3 + 1 + 2 * nop * 3] = a23;
		d_A[ind * 3 + 2 + 2 * nop * 3] = a33;

		d_A[ind * 3 + 0 + 3 * nop * 3] = -a14;
		d_A[ind * 3 + 1 + 3 * nop * 3] = -a24;
		d_A[ind * 3 + 2 + 3 * nop * 3] = -a34;

		d_A[ind * 3 + 0 + 4 * nop * 3] = -a15;
		d_A[ind * 3 + 1 + 4 * nop * 3] = -a25;
		d_A[ind * 3 + 2 + 4 * nop * 3] = -a35;

		d_A[ind * 3 + 0 + 5 * nop * 3] = -a16;
		d_A[ind * 3 + 1 + 5 * nop * 3] = -a26;
		d_A[ind * 3 + 2 + 5 * nop * 3] = -a36;

		d_P[ind * 3    ] = 1.0;
		d_P[ind * 3 + 1] = 1.0;
		d_P[ind * 3 + 2] = 1.0;

		d_l[ind * 3    ] = obs_nn.x_diff;
		d_l[ind * 3 + 1] = obs_nn.y_diff;
		d_l[ind * 3 + 2] = obs_nn.z_diff;

	}
}

cudaError_t fill_A_l_cuda(int threads, double *d_A, double x, double y, double z, double om, double fi, double ka,
		obs_nn_t *d_obs_nn, int nop, double *d_P, double *d_l)
{
	cudaError_t err = ::cudaSuccess;
	int blocks=nop/threads+1;
	kernel_fill_A_l_cuda<<<blocks,threads>>>(d_A, x, y, z, om, fi, ka, d_obs_nn, nop, d_P, d_l);

	err = cudaDeviceSynchronize();
	return err;
}

__global__ void kernel_nearestNeighborSearch(
		lidar_pointcloud::PointXYZIRNL *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		lidar_pointcloud::PointXYZIRNL *d_second_point_cloud,
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
		lidar_pointcloud::PointXYZIRNL *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		lidar_pointcloud::PointXYZIRNL *d_second_point_cloud,
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

