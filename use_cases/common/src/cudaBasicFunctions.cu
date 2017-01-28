#include "cudaFunctions.h"
#include "cudaStructures.cuh"

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


__global__ void kernel_cudaTransformPointCloud(pcl::PointXYZ *d_in_point_cloud,
											   pcl::PointXYZ *d_out_point_cloud,
											   int number_of_points,
											   float *d_matrix)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<number_of_points)
	{
		float vSrcVector[3] = {d_in_point_cloud[ind].x, d_in_point_cloud[ind].y, d_in_point_cloud[ind].z};
		float vOut[3];
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
								float *d_matrix)
{
	kernel_cudaTransformPointCloud<<<number_of_points/threads+1,threads>>>
			(d_in_point_cloud, d_out_point_cloud, number_of_points, d_matrix);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

__global__ void kernel_cudaTransformPointCloud(Semantic::PointXYZL *d_in_point_cloud,
											   Semantic::PointXYZL *d_out_point_cloud,
											   int number_of_points,
											   float *d_matrix)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind<number_of_points)
	{
		float vSrcVector[3] = {d_in_point_cloud[ind].x, d_in_point_cloud[ind].y, d_in_point_cloud[ind].z};
		float vOut[3];
		vOut[0]=d_matrix[0]*vSrcVector[0]+d_matrix[4]*vSrcVector[1]+d_matrix[8]*vSrcVector[2]+d_matrix[12];
   	 	vOut[1]=d_matrix[1]*vSrcVector[0]+d_matrix[5]*vSrcVector[1]+d_matrix[9]*vSrcVector[2]+d_matrix[13];
    	vOut[2]=d_matrix[2]*vSrcVector[0]+d_matrix[6]*vSrcVector[1]+d_matrix[10]*vSrcVector[2]+d_matrix[14];

    	d_out_point_cloud[ind].x = vOut[0];
    	d_out_point_cloud[ind].y = vOut[1];
    	d_out_point_cloud[ind].z = vOut[2];

    	d_out_point_cloud[ind].label = d_in_point_cloud[ind].label;
	}
}

cudaError_t cudaTransformPointCloud(int threads,
								Semantic::PointXYZL *d_in_point_cloud,
								Semantic::PointXYZL *d_out_point_cloud,
								int number_of_points,
								float *d_matrix)
{
	kernel_cudaTransformPointCloud<<<number_of_points/threads+1,threads>>>
			(d_in_point_cloud, d_out_point_cloud, number_of_points, d_matrix);

	cudaDeviceSynchronize();
	return cudaGetLastError();
}

__global__ void kernel_setAllPointsToRemove(int number_of_points, bool *d_markers_out)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind<number_of_points)
	{
		d_markers_out[ind] = false;
	}
}

__global__ void kernel_markPointsToRemain(pcl::PointXYZ *d_point_cloud,
		int number_of_points,
		hashElement *d_hashTable,
		bucket *d_buckets,
		gridParameters rgd_params,
		float search_radius,
		int number_of_points_in_search_sphere_threshold,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		bool *d_markers_out)
{
	int index_of_point = blockIdx.x*blockDim.x+threadIdx.x;

	if(index_of_point < number_of_points)
	{
		int number_of_found_points_in_search_sphere_threshold = 0;

		float x = d_point_cloud[index_of_point].x;
		float y = d_point_cloud[index_of_point].y;
		float z = d_point_cloud[index_of_point].z;

		if(x < rgd_params.bounding_box_min_X || x > rgd_params.bounding_box_max_X)
		{
			d_markers_out[index_of_point] = false;
			return;
		}
		if(y < rgd_params.bounding_box_min_Y || y > rgd_params.bounding_box_max_Y)
		{
			d_markers_out[index_of_point] = false;
			return;
		}
		if(z < rgd_params.bounding_box_min_Z || z > rgd_params.bounding_box_max_Z)
		{
			d_markers_out[index_of_point] = false;
			return;
		}

		int ix=(x - rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(y - rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(z - rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;

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
								if(l >= 0 && l < number_of_points)
								{
									int hashed_index_of_point = d_hashTable[l].index_of_point;

									float nn_x  = d_point_cloud[hashed_index_of_point].x;
									float nn_y  = d_point_cloud[hashed_index_of_point].y;
									float nn_z  = d_point_cloud[hashed_index_of_point].z;

									float dist  = (x - nn_x) * (x - nn_x) +
												  (y - nn_y) * (y - nn_y) +
												  (z - nn_z) * (z - nn_z);

									if(dist <= search_radius * search_radius)
									{
										number_of_found_points_in_search_sphere_threshold++;
									}
								}
							}
						}
					}
				}
			}
		}

		if(number_of_found_points_in_search_sphere_threshold >= number_of_points_in_search_sphere_threshold + 1)
		{
			d_markers_out[index_of_point] = true;
		}else
		{
			d_markers_out[index_of_point] = false;
		}
	}
}

cudaError_t cudaRemoveNoise(
			int threads,
			pcl::PointXYZ *d_point_cloud,
			int number_of_points,
			hashElement *d_hashTable,
			bucket *d_buckets,
			gridParameters rgd_params,
			float search_radius,
			int number_of_points_in_search_sphere_threshold,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			bool *d_markers_out)
{
	cudaError_t err = cudaGetLastError();

	kernel_setAllPointsToRemove<<<number_of_points/threads+1,threads>>>(number_of_points, d_markers_out);
	err = cudaDeviceSynchronize();	if(err != ::cudaSuccess)return err;

	kernel_markPointsToRemain<<<number_of_points/threads+1,threads>>>
				   (d_point_cloud,
					number_of_points,
					d_hashTable,
					d_buckets,
					rgd_params,
					search_radius,
					number_of_points_in_search_sphere_threshold,
					max_number_considered_in_INNER_bucket,
					max_number_considered_in_OUTER_bucket,
					d_markers_out);
	err = cudaDeviceSynchronize();

	return err;
}


__global__ void kernel_markPointsToRemain(velodyne_pointcloud::PointXYZIR *d_point_cloud,
		int number_of_points,
		hashElement *d_hashTable,
		bucket *d_buckets,
		gridParameters rgd_params,
		float search_radius,
		int number_of_points_in_search_sphere_threshold,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		bool *d_markers_out)
{
	int index_of_point = blockIdx.x*blockDim.x+threadIdx.x;

	if(index_of_point < number_of_points)
	{
		int number_of_found_points_in_search_sphere_threshold = 0;

		float x = d_point_cloud[index_of_point].x;
		float y = d_point_cloud[index_of_point].y;
		float z = d_point_cloud[index_of_point].z;

		if(x < rgd_params.bounding_box_min_X || x > rgd_params.bounding_box_max_X)
		{
			d_markers_out[index_of_point] = false;
			return;
		}
		if(y < rgd_params.bounding_box_min_Y || y > rgd_params.bounding_box_max_Y)
		{
			d_markers_out[index_of_point] = false;
			return;
		}
		if(z < rgd_params.bounding_box_min_Z || z > rgd_params.bounding_box_max_Z)
		{
			d_markers_out[index_of_point] = false;
			return;
		}

		int ix=(x - rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(y - rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(z - rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;

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
								if(l >= 0 && l < number_of_points)
								{
									int hashed_index_of_point = d_hashTable[l].index_of_point;

									float nn_x  = d_point_cloud[hashed_index_of_point].x;
									float nn_y  = d_point_cloud[hashed_index_of_point].y;
									float nn_z  = d_point_cloud[hashed_index_of_point].z;

									float dist  = (x - nn_x) * (x - nn_x) +
												  (y - nn_y) * (y - nn_y) +
												  (z - nn_z) * (z - nn_z);

									if(dist <= search_radius * search_radius)
									{
										number_of_found_points_in_search_sphere_threshold++;
									}
								}
							}
						}
					}
				}
			}
		}

		if(number_of_found_points_in_search_sphere_threshold >= number_of_points_in_search_sphere_threshold + 1)
		{
			d_markers_out[index_of_point] = true;
		}else
		{
			d_markers_out[index_of_point] = false;
		}
	}
}

cudaError_t cudaRemoveNoise(
			int threads,
			velodyne_pointcloud::PointXYZIR *d_point_cloud,
			int number_of_points,
			hashElement *d_hashTable,
			bucket *d_buckets,
			gridParameters rgd_params,
			float search_radius,
			int number_of_points_in_search_sphere_threshold,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			bool *d_markers_out)
{
	cudaError_t err = cudaGetLastError();

	kernel_setAllPointsToRemove<<<number_of_points/threads+1,threads>>>(number_of_points, d_markers_out);
	err = cudaDeviceSynchronize();	if(err != ::cudaSuccess)return err;

	kernel_markPointsToRemain<<<number_of_points/threads+1,threads>>>
				   (d_point_cloud,
					number_of_points,
					d_hashTable,
					d_buckets,
					rgd_params,
					search_radius,
					number_of_points_in_search_sphere_threshold,
					max_number_considered_in_INNER_bucket,
					max_number_considered_in_OUTER_bucket,
					d_markers_out);
	err = cudaDeviceSynchronize();

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

/////////////////////////////////computeNormalVectors/////////////////////////////////////
__global__ void kernel_normalvectorcomputation_step1_fast(
		Semantic::PointXYZNL * d_point_cloud,
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

__global__ void kernel_normalvectorcomputation_step2_fast_with_classification(
	Semantic::PointXYZNL *d_point_cloud,
	hashElement *d_hashTable,
	simple_point3D *d_mean,
	int number_of_points,
	bucket *d_buckets,
	gridParameters rgd_params,
	float search_radius,
	int max_number_considered_in_INNER_bucket,
	int max_number_considered_in_OUTER_bucket,
	float curvature_threshold,
	int number_of_points_needed_for_plane_threshold)
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
				d_point_cloud[hashed_index_of_point].label = EDGE;

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

					if(number_of_nearest_neighbours >= number_of_points_needed_for_plane_threshold)
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
						}else
						{
							d_point_cloud[hashed_index_of_point].normal_x = _nx/lenght;
							d_point_cloud[hashed_index_of_point].normal_y = _ny/lenght;
							d_point_cloud[hashed_index_of_point].normal_z = _nz/lenght;
							if( (SS[4]/SS[8]) > curvature_threshold)
							{
								d_point_cloud[hashed_index_of_point].label = PLANE;
							}
						}
					}
					else
					{
						d_point_cloud[hashed_index_of_point].normal_x = 0.0f;
						d_point_cloud[hashed_index_of_point].normal_y = 0.0f;
						d_point_cloud[hashed_index_of_point].normal_z = 0.0f;
					}
				}
			}
		}
	}
}

cudaError_t cudaSemanticLabelingPlaneEdges(
	int threads,
	Semantic::PointXYZNL * d_point_cloud,
	int number_of_points,
	hashElement* d_hashTable,
	bucket* d_buckets,
	simple_point3D* d_mean,
	gridParameters rgd_params,
	float search_radius,
	int max_number_considered_in_INNER_bucket,
	int max_number_considered_in_OUTER_bucket,
	float curvature_threshold,
	int number_of_points_needed_for_plane_threshold)
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

	kernel_normalvectorcomputation_step2_fast_with_classification<<<blocks,threads>>>(
		d_point_cloud,
		d_hashTable,
		d_mean,
		number_of_points,
		d_buckets,
		rgd_params,
		search_radius,
		max_number_considered_in_INNER_bucket,
		max_number_considered_in_OUTER_bucket,
		curvature_threshold,
		number_of_points_needed_for_plane_threshold);
	err = cudaDeviceSynchronize();

	return err;
}

__global__ void kernel_normalvectorcomputation_step1_fast(
		velodyne_pointcloud::PointXYZIRNL * d_point_cloud,
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

__global__ void kernel_normalvectorcomputation_step2_fast_with_classification(
	velodyne_pointcloud::PointXYZIRNL *d_point_cloud,
	hashElement *d_hashTable,
	simple_point3D *d_mean,
	int number_of_points,
	bucket *d_buckets,
	gridParameters rgd_params,
	float search_radius,
	int max_number_considered_in_INNER_bucket,
	int max_number_considered_in_OUTER_bucket,
	float curvature_threshold,
	int number_of_points_needed_for_plane_threshold)
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
				d_point_cloud[hashed_index_of_point].label = EDGE;

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

					if(number_of_nearest_neighbours >= number_of_points_needed_for_plane_threshold)
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
						}else
						{
							d_point_cloud[hashed_index_of_point].normal_x = _nx/lenght;
							d_point_cloud[hashed_index_of_point].normal_y = _ny/lenght;
							d_point_cloud[hashed_index_of_point].normal_z = _nz/lenght;
							if( (SS[4]/SS[8]) > curvature_threshold)
							{
								d_point_cloud[hashed_index_of_point].label = PLANE;
							}
						}
					}
					else
					{
						d_point_cloud[hashed_index_of_point].normal_x = 0.0f;
						d_point_cloud[hashed_index_of_point].normal_y = 0.0f;
						d_point_cloud[hashed_index_of_point].normal_z = 0.0f;
					}
				}
			}
		}
	}
}

__global__ void kernel_FlipNormalsTowardsViewpoint(velodyne_pointcloud::PointXYZIRNL* cudaData, int number_of_points, float viepointX, float viepointY, float viepointZ)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;
	if(ind < number_of_points)
	{
		velodyne_pointcloud::PointXYZIRNL p = cudaData[ind];
		if(p.normal_x * (viepointX - p.x) + p.normal_y * (viepointY - p.y) + p.normal_z * (viepointZ - p.z)  < 0.0)
		{
			cudaData[ind].normal_x = -p.normal_x;
			cudaData[ind].normal_y = -p.normal_y;
			cudaData[ind].normal_z = -p.normal_z;
		}
	}
}

cudaError_t cudaSemanticLabelingPlaneEdges(
	int threads,
	velodyne_pointcloud::PointXYZIRNL * d_point_cloud,
	int number_of_points,
	hashElement* d_hashTable,
	bucket* d_buckets,
	simple_point3D* d_mean,
	gridParameters rgd_params,
	float search_radius,
	int max_number_considered_in_INNER_bucket,
	int max_number_considered_in_OUTER_bucket,
	float curvature_threshold,
	int number_of_points_needed_for_plane_threshold,
	float viepointX,
	float viepointY,
	float viepointZ
	)
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

	kernel_normalvectorcomputation_step2_fast_with_classification<<<blocks,threads>>>(
		d_point_cloud,
		d_hashTable,
		d_mean,
		number_of_points,
		d_buckets,
		rgd_params,
		search_radius,
		max_number_considered_in_INNER_bucket,
		max_number_considered_in_OUTER_bucket,
		curvature_threshold,
		number_of_points_needed_for_plane_threshold);
	err = cudaDeviceSynchronize();
	if(err != ::cudaSuccess)return err;

	kernel_FlipNormalsTowardsViewpoint<<<blocks,threads>>>(d_point_cloud, number_of_points, viepointX, viepointY, viepointZ);
	err = cudaDeviceSynchronize();

	return err;
}

__global__ void kernel_semanticLabelingFloorCeiling(
			Semantic::PointXYZNL * d_point_cloud,
			int number_of_points,
			float ground_Z_coordinate_threshold)
{
	int index_of_point = blockIdx.x * blockDim.x + threadIdx.x;
	if(index_of_point < number_of_points)
	{
		if(d_point_cloud[index_of_point].label == 0)
		{
			if(d_point_cloud[index_of_point].normal_z > 0.7 || d_point_cloud[index_of_point].normal_z < -0.7)
			{
				if(d_point_cloud[index_of_point].z < ground_Z_coordinate_threshold)
				{
					d_point_cloud[index_of_point].label = FLOOR_GROUND;
				}else
				{
					d_point_cloud[index_of_point].label = CEILING;
				}
			}
		}
	}
}

cudaError_t cudaSemanticLabelingFloorCeiling(
		int threads,
		Semantic::PointXYZNL * d_point_cloud,
		int number_of_points,
		float ground_Z_coordinate_threshold)
{
	cudaError_t err = cudaGetLastError();
	int blocks=number_of_points/threads+1;

	kernel_semanticLabelingFloorCeiling<<<blocks,threads>>>(
			d_point_cloud,
			number_of_points,
			ground_Z_coordinate_threshold);

	err = cudaDeviceSynchronize();
	return err;
}


__global__ void kernel_semanticLabelingFloorCeiling(
			velodyne_pointcloud::PointXYZIRNL * d_point_cloud,
			int number_of_points,
			float ground_Z_coordinate_threshold)
{
	int index_of_point = blockIdx.x * blockDim.x + threadIdx.x;
	if(index_of_point < number_of_points)
	{
		if(d_point_cloud[index_of_point].label == 0)
		{
			if(d_point_cloud[index_of_point].normal_z > 0.7 || d_point_cloud[index_of_point].normal_z < -0.7)
			{
				if(d_point_cloud[index_of_point].z < ground_Z_coordinate_threshold)
				{
					d_point_cloud[index_of_point].label = FLOOR_GROUND;
				}else
				{
					d_point_cloud[index_of_point].label = CEILING;
				}
			}
		}
	}
}

cudaError_t cudaSemanticLabelingFloorCeiling(
		int threads,
		velodyne_pointcloud::PointXYZIRNL * d_point_cloud,
		int number_of_points,
		float ground_Z_coordinate_threshold)
{
	cudaError_t err = cudaGetLastError();
	int blocks=number_of_points/threads+1;

	kernel_semanticLabelingFloorCeiling<<<blocks,threads>>>(
			d_point_cloud,
			number_of_points,
			ground_Z_coordinate_threshold);

	err = cudaDeviceSynchronize();
	return err;
}

inline __host__ __device__ void gltLoadIdentityMatrix(float *m)
{
	m[0] = 1.0f;
	m[1] = 0.0f;
	m[2] = 0.0f;
	m[3] = 0.0f;

	m[4] = 0.0f;
	m[5] = 1.0f;
	m[6] = 0.0f;
	m[7] = 0.0f;

	m[8] = 0.0f;
	m[9] = 0.0f;
	m[10] = 1.0f;
	m[11] = 0.0f;

	m[12] = 0.0f;
	m[13] = 0.0f;
	m[14] = 0.0f;
	m[15] = 1.0f;
}

inline __host__ __device__ void gltRotationMatrix(double angle, double x, double y, double z, float *mMatrix)
{
	double vecLength, sinSave, cosSave, oneMinusCos;
	double xx, yy, zz, xy, yz, zx, xs, ys, zs;

	// If NULL vector passed in, this will blow up...
	if(x==0.0f && y==0.0f && z==0.0f)
		{
		gltLoadIdentityMatrix(mMatrix);
		return;
		}

	// Scale vector
	vecLength=(double)sqrt( x*x+y*y+z*z );

	// Rotation matrix is normalized
	x /=vecLength;
	y /=vecLength;
	z /=vecLength;

	sinSave=(double)sin(angle);
	cosSave=(double)cos(angle);
	oneMinusCos=1.0f-cosSave;

	xx=x*x;
	yy=y*y;
	zz=z*z;
	xy=x*y;
	yz=y*z;
	zx=z*x;
	xs=x*sinSave;
	ys=y*sinSave;
	zs=z*sinSave;

	mMatrix[0]=(oneMinusCos*xx)+cosSave;
	mMatrix[4]=(oneMinusCos*xy)-zs;
	mMatrix[8]=(oneMinusCos*zx)+ys;
	mMatrix[12]=0.0f;

	mMatrix[1]=(oneMinusCos*xy)+zs;
	mMatrix[5]=(oneMinusCos*yy)+cosSave;
	mMatrix[9]=(oneMinusCos*yz)-xs;
	mMatrix[13]=0.0f;

	mMatrix[2]=(oneMinusCos*zx)-ys;
	mMatrix[6]=(oneMinusCos*yz)+xs;
	mMatrix[10]=(oneMinusCos*zz)+cosSave;
	mMatrix[14]=0.0f;

	mMatrix[3]=0.0f;
	mMatrix[7]=0.0f;
	mMatrix[11]=0.0f;
	mMatrix[15]=1.0f;
}

inline __host__ __device__ void gltMultiplyMatrix(float * m1, float * m2, float * mProduct )
{
    mProduct[0]=m1[0]*m2[0]+m1[4]*m2[1]+m1[8]*m2[2]+m1[12]*m2[3];
    mProduct[4]=m1[0]*m2[4]+m1[4]*m2[5]+m1[8]*m2[6]+m1[12]*m2[7];
    mProduct[8]=m1[0]*m2[8]+m1[4]*m2[9]+m1[8]*m2[10]+m1[12]*m2[11];
    mProduct[12]=m1[0]*m2[12]+m1[4]*m2[13]+m1[8]*m2[14]+m1[12]*m2[15];

    mProduct[1]=m1[1]*m2[0]+m1[5]*m2[1]+m1[9]*m2[2]+m1[13]*m2[3];
    mProduct[5]=m1[1]*m2[4]+m1[5]*m2[5]+m1[9]*m2[6]+m1[13]*m2[7];
    mProduct[9]=m1[1]*m2[8]+m1[5]*m2[9]+m1[9]*m2[10]+m1[13]*m2[11];
    mProduct[13]=m1[1]*m2[12]+m1[5]*m2[13]+m1[9]*m2[14]+m1[13]*m2[15];

    mProduct[2]=m1[2]*m2[0]+m1[6]*m2[1]+m1[10]*m2[2]+m1[14]*m2[3];
    mProduct[6]=m1[2]*m2[4]+m1[6]*m2[5]+m1[10]*m2[6]+m1[14]*m2[7];
    mProduct[10]=m1[2]*m2[8]+m1[6]*m2[9]+m1[10]*m2[10]+m1[14]*m2[11];
    mProduct[14]=m1[2]*m2[12]+m1[6]*m2[13]+m1[10]*m2[14]+m1[14]*m2[15];

    mProduct[3]=m1[3]*m2[0]+m1[7]*m2[1]+m1[11]*m2[2]+m1[15]*m2[3];
    mProduct[7]=m1[3]*m2[4]+m1[7]*m2[5]+m1[11]*m2[6]+m1[15]*m2[7];
    mProduct[11]=m1[3]*m2[8]+m1[7]*m2[9]+m1[11]*m2[10]+m1[15]*m2[11];
    mProduct[15]=m1[3]*m2[12]+m1[7]*m2[13]+m1[11]*m2[14]+m1[15]*m2[15];
}

__global__ void kernel_particleFilterPrediction(float *d_vangle,
		float *d_vtrans,
		float distance_above_Z,
		float *d_vmatrix,
		int number_of_particles,
		float *d_odometryIncrement,
		hashElement *d_hashTable_2D,
		bucket *d_buckets_2D,
		gridParameters rgd_params_2D,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		pcl::PointXYZ *d_ground_points_from_map,
		int number_of_points_ground_points_from_map)
{
	int index_of_particle = blockIdx.x * blockDim.x + threadIdx.x;
	if(index_of_particle < number_of_particles)
	{
		float m[16];
		float mTemp[16];
		float mTemp2[16];

		for(int i = 0 ; i < 16; i++)
			m[i] = d_vmatrix[index_of_particle * 16 + i];

		float angle = d_vangle[index_of_particle];
		float mr[16];
		float mt[16];
		gltRotationMatrix(angle, 0.0f, 0.0f, 1.0f, mr);
		gltLoadIdentityMatrix(mt);
		mt[12] = d_vtrans[index_of_particle];

		//mt[14] += distance_above_Z;

		gltMultiplyMatrix(m, mr, mTemp);
		gltMultiplyMatrix(mTemp, mt, mTemp2);
		gltMultiplyMatrix(mTemp2, d_odometryIncrement, m);

		//gltMultiplyMatrix(m, mt, mTemp);
		//gltMultiplyMatrix(mTemp, d_odometryIncrement, m);

		/////////////////////////////////////////////////////////////////////////////////////

		bool isok = false;
		//float xRes = 0.0f;
		//float yRes = 0.0f;
		float zRes = 0.0f;

		float x = m[12];
		float y = m[13];

		if(x < rgd_params_2D.bounding_box_min_X || x > rgd_params_2D.bounding_box_max_X)
		{
			for(int i = 0 ; i < 16; i++)d_vmatrix[index_of_particle * 16 + i] = m[i];
			return;
		}
		if(y < rgd_params_2D.bounding_box_min_Y || y > rgd_params_2D.bounding_box_max_Y)
		{
			for(int i = 0 ; i < 16; i++)d_vmatrix[index_of_particle * 16 + i] = m[i];
			return;
		}

		int ix=(x - rgd_params_2D.bounding_box_min_X)/rgd_params_2D.resolution_X;
		int iy=(y - rgd_params_2D.bounding_box_min_Y)/rgd_params_2D.resolution_Y;

		int index_bucket = ix*rgd_params_2D.number_of_buckets_Y + iy;

		if(index_bucket >= 0 && index_bucket < rgd_params_2D.number_of_buckets)
		{
			int sx, sy, stx, sty;
			if(ix == 0)sx = 0; else sx = -1;
			if(iy == 0)sy = 0; else sy = -1;

			if(ix == rgd_params_2D.number_of_buckets_X - 1)stx = 1; else stx = 2;
			if(iy == rgd_params_2D.number_of_buckets_Y - 1)sty = 1; else sty = 2;

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
					index_next_bucket = index_bucket + i * rgd_params_2D.number_of_buckets_Y +	j;
					if(index_next_bucket >= 0 && index_next_bucket < rgd_params_2D.number_of_buckets)
					{
						number_of_points_in_bucket = d_buckets_2D[index_next_bucket].number_of_points;
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

						l_begin = d_buckets_2D[index_next_bucket].index_begin;
						l_end = d_buckets_2D[index_next_bucket].index_end;

						for(int l = l_begin; l < l_end; l += iter)
						{
							if(l >= 0 && l < number_of_points_ground_points_from_map)
							{
								int hashed_index_of_point = d_hashTable_2D[l].index_of_point;

								float nn_x  = d_ground_points_from_map[hashed_index_of_point].x;
								float nn_y  = d_ground_points_from_map[hashed_index_of_point].y;
								float nn_z  = d_ground_points_from_map[hashed_index_of_point].z;


								float dist  = (x - nn_x) * (x - nn_x) +
											  (y - nn_y) * (y - nn_y);


								if(dist <= search_radius * search_radius )
								{
									if(dist < _distance)
									{
										isok = true;
										//xRes = nn_x;
										//yRes = nn_y;
										zRes = nn_z;


										_distance = dist;
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
			m[14] = zRes + distance_above_Z;
			for(int i = 0 ; i < 16; i++)d_vmatrix[index_of_particle * 16 + i] = m[i];
		}else
		{
			for(int i = 0 ; i < 16; i++)d_vmatrix[index_of_particle * 16 + i] = m[i];
		}
	}
}


cudaError_t cudaParticleFilterPrediction(int threads,
		float *d_vangle,
		float *d_vtrans,
		float distance_above_Z,
		float *d_vmatrix,
		int number_of_particles,
		float *d_odometryIncrement,
		hashElement *d_hashTable_2D,
		bucket *d_buckets_2D,
		gridParameters rgd_params_2D,
		float search_radius,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		pcl::PointXYZ *d_ground_points_from_map,
		int number_of_points_ground_points_from_map)
{
	cudaError_t err = cudaGetLastError();
	int blocks=number_of_particles/threads+1;

	kernel_particleFilterPrediction<<<blocks, threads>>>(d_vangle,
			d_vtrans,
			distance_above_Z,
			d_vmatrix,
			number_of_particles,
			d_odometryIncrement,
			d_hashTable_2D,
			d_buckets_2D,
			rgd_params_2D,
			search_radius,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			d_ground_points_from_map,
			number_of_points_ground_points_from_map
	);

	err = cudaDeviceSynchronize();
	return err;
}


__global__ void kernel_cudaInsertPointCloudToRGD(
		Semantic::PointXYZL *d_pc,
		int number_of_points,
		char *d_rgd,
		gridParameters rgd_params)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind < number_of_points)
	{
		char label = d_pc[ind].label;

		float x = d_pc[ind].x;
		float y = d_pc[ind].y;
		float z = d_pc[ind].z;

		if(x < rgd_params.bounding_box_min_X || x > rgd_params.bounding_box_max_X)
		{
			return;
		}
		if(y < rgd_params.bounding_box_min_Y || y > rgd_params.bounding_box_max_Y)
		{
			return;
		}
		if(z < rgd_params.bounding_box_min_Z || z > rgd_params.bounding_box_max_Z)
		{
			return;
		}

		int ix=(x - rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(y - rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(z - rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;

		int index_bucket = ix*rgd_params.number_of_buckets_Y *
				rgd_params.number_of_buckets_Z + iy * rgd_params.number_of_buckets_Z + iz;

		d_rgd[index_bucket] = label;
	}
}

cudaError_t cudaInsertPointCloudToRGD(unsigned int threads,
	Semantic::PointXYZL *d_pc,
	int number_of_points,
	char *d_rgd,
	gridParameters rgdparams)
{
	cudaError_t err = ::cudaSuccess;
		unsigned int blocks=number_of_points/threads+1;
		kernel_cudaInsertPointCloudToRGD<<<blocks,threads>>>(d_pc, number_of_points, d_rgd, rgdparams);
		err = cudaDeviceSynchronize();
	return err;
}

__global__ void kernel_cudaSemanticNN(
		Semantic::PointXYZL *d_pc,
		int number_of_points,
		char *d_rgd,
		gridParameters rgd_params,
		unsigned int *d_nn
		)
{
	unsigned int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind < number_of_points)
	{
		d_nn[ind] = 0;

		double x = d_pc[ind].x;
		double y = d_pc[ind].y;
		double z = d_pc[ind].z;

		char label = d_pc[ind].label;

		if(x < rgd_params.bounding_box_min_X || x > rgd_params.bounding_box_max_X)
		{
			return;
		}
		if(y < rgd_params.bounding_box_min_Y || y > rgd_params.bounding_box_max_Y)
		{
			return;
		}
		if(z < rgd_params.bounding_box_min_Z || z > rgd_params.bounding_box_max_Z)
		{
			return;
		}

		int ix=(x - rgd_params.bounding_box_min_X)/rgd_params.resolution_X;
		int iy=(y - rgd_params.bounding_box_min_Y)/rgd_params.resolution_Y;
		int iz=(z - rgd_params.bounding_box_min_Z)/rgd_params.resolution_Z;

		int index_bucket = ix*rgd_params.number_of_buckets_Y *
				rgd_params.number_of_buckets_Z + iy * rgd_params.number_of_buckets_Z + iz;


		char labelrgd = d_rgd[index_bucket];
		//if((labelrgd == label) && (label != 3))d_nn[ind] = 1;
		if( labelrgd == label )d_nn[ind] = 1;
	}
}


cudaError_t cudaComputeOverlap(
	unsigned int threads,
	Semantic::PointXYZL *d_pc,
	int number_of_points,
	char *d_rgd,
	gridParameters rgd_params,
	unsigned int *d_nn,
	float &overlap)
{
	cudaError_t err = ::cudaSuccess;
	unsigned int blocks = number_of_points/threads+1;

	kernel_cudaSemanticNN<<<blocks,threads>>>(d_pc, number_of_points, d_rgd, rgd_params, d_nn);

	thrust::device_ptr <unsigned int> dev_ptr_d_nn ( d_nn );
	unsigned int number_of_nearest_neighbors = thrust::reduce (dev_ptr_d_nn , dev_ptr_d_nn + number_of_points);
	overlap = float(number_of_nearest_neighbors)/float(number_of_points);

	return err;
}


