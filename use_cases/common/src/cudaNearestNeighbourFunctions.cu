#include "cudaFunctions.h"
#include "cudaStructures.cuh"

__global__ void  kernel_fill_nn_cuda(unsigned int *d_nn, int *nearest_neighbour_indexes, unsigned int number_nearest_neighbour_indexes)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind < number_nearest_neighbour_indexes)
	{
		if(nearest_neighbour_indexes[ind] < 0)
		{
			d_nn[ind] = 0;
		}else
		{
			d_nn[ind] = 1;
		}
	}
}

cudaError_t cudaCalculateNumberOfNearestNeighbors(
			int threads,
			int *nearest_neighbour_indexes,
			unsigned int number_nearest_neighbour_indexes,
			unsigned int &number_of_nearest_neighbors)
{
	int blocks = number_nearest_neighbour_indexes/threads+1;

	try
	{
		unsigned int *d_nn = NULL;
		if(cudaMalloc((void**)&d_nn, number_nearest_neighbour_indexes*sizeof(unsigned int)) != ::cudaSuccess)
		{
			cudaDeviceReset();
			return cudaGetLastError();
		}

		kernel_fill_nn_cuda<<<blocks,threads>>>(d_nn, nearest_neighbour_indexes, number_nearest_neighbour_indexes);

		thrust::device_ptr <unsigned int> dev_ptr_d_nn ( d_nn );
		number_of_nearest_neighbors = thrust::reduce (dev_ptr_d_nn , dev_ptr_d_nn + number_nearest_neighbour_indexes);

		if(cudaFree(d_nn) !=::cudaSuccess )
		{
			//cudaDeviceReset();
			return cudaGetLastError();
		}else
		{
			d_nn = 0;
		}
	}//try
	catch(thrust::system_error &e)
	{
		//cudaDeviceReset();
		return cudaGetLastError();
	}
	catch(std::bad_alloc &e)
	{
		//cudaDeviceReset();
		return cudaGetLastError();
	}
	return cudaGetLastError();
}


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

__global__ void kernel_semanticNearestNeighborSearch(
		Semantic::PointXYZL *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		Semantic::PointXYZL *d_second_point_cloud,
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
		int label_second_point_cloud =  d_second_point_cloud[index_of_point_second_point_cloud].label;

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

									int label_first_point_cloud = d_first_point_cloud[hashed_index_of_point].label;

									float dist  = (x - nn_x) * (x - nn_x) +
												  (y - nn_y) * (y - nn_y) +
												  (z - nn_z) * (z - nn_z);

									if(label_first_point_cloud == label_second_point_cloud)
									{
										if(dist <= search_radius * search_radius )
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

cudaError_t cudaSemanticNearestNeighborSearch(
			int threads,
			Semantic::PointXYZL *d_first_point_cloud,
			int number_of_points_first_point_cloud,
			Semantic::PointXYZL *d_second_point_cloud,
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

	kernel_semanticNearestNeighborSearch<<<blocks,threads>>>(
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

