#include "lesson_19.h"
#include <stdio.h>

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


__global__ void kernel_cudaComputeDistance(laser_beam _single_laser_beam, plane *d_vPlanes, int number_of_planes, float *d_distance)
{
	int ind=blockIdx.x*blockDim.x+threadIdx.x;

	if(ind < number_of_planes)
	{
		d_distance[ind] = _single_laser_beam.range;
		plane d_plane = d_vPlanes[ind];
		simple_point3D p;
		p.x = _single_laser_beam.position.x;
		p.y = _single_laser_beam.position.y;
		p.z = _single_laser_beam.position.z;

		float a = d_plane.A * _single_laser_beam.direction.x + d_plane.B * _single_laser_beam.direction.y + d_plane.C * _single_laser_beam.direction.z	;

		if ( a > -TOLERANCE && a < TOLERANCE)
		{

		}else
		{
			float distance = d_plane.A * p.x + d_plane.B * p.y + d_plane.C * p.z + d_plane.D;
			p.x = _single_laser_beam.position.x - _single_laser_beam.direction.x * (distance/a);
			p.y = _single_laser_beam.position.y - _single_laser_beam.direction.y * (distance/a);
			p.z = _single_laser_beam.position.z - _single_laser_beam.direction.z * (distance/a);


			bool isOK = false;

			simple_point3D v0 = d_plane.polygon.vertexA;
			simple_point3D v1 = d_plane.polygon.vertexB;
			simple_point3D v2 = d_plane.polygon.vertexC;

			simple_point3D segment1, segment2;
			double sumAngles = 0.0;
			double cosAngle = 0.0;

			segment1.x = v0.x - p.x;
			segment1.y = v0.y - p.y;
			segment1.z = v0.z - p.z;
			segment2.x = v1.x - p.x;
			segment2.y = v1.y - p.y;
			segment2.z = v1.z - p.z;

			float l1 = sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z);
			float l2 = sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z);


			if (l1 * l2 <=  TOLERANCE)
			{
				isOK = true;
			}

			cosAngle = (segment1.x*segment2.x + segment1.y*segment2.y + segment1.z*segment2.z)/
					(sqrt(segment1.x*segment1.x + segment1.y*segment1.y + segment1.z*segment1.z) *
					 sqrt(segment2.x*segment2.x + segment2.y*segment2.y + segment2.z*segment2.z));
			sumAngles += acos(cosAngle);

			segment1.x = v1.x - p.x;
			segment1.y = v1.y - p.y;
			segment1.z = v1.z - p.z;
			segment2.x = v2.x - p.x;
			segment2.y = v2.y - p.y;
			segment2.z = v2.z - p.z;
			l1 = sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z);
			l2 = sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z);


			if (l1 * l2 <=  TOLERANCE)
			{
				isOK = true;
			}

			cosAngle = (segment1.x*segment2.x + segment1.y*segment2.y + segment1.z*segment2.z)/
					(sqrt(segment1.x*segment1.x + segment1.y*segment1.y + segment1.z*segment1.z) *
					 sqrt(segment2.x*segment2.x + segment2.y*segment2.y + segment2.z*segment2.z));
			sumAngles += acos(cosAngle);

			segment1.x = v2.x - p.x;
			segment1.y = v2.y - p.y;
			segment1.z = v2.z - p.z;
			segment2.x = v0.x - p.x;
			segment2.y = v0.y - p.y;
			segment2.z = v0.z - p.z;
			l1 = sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z);
			l2 = sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z);

			if (l1 * l2 <=  TOLERANCE)
			{
				isOK = true;
			}

			cosAngle = (segment1.x * segment2.x + segment1.y * segment2.y + segment1.z * segment2.z)/
					(sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z) *
					 sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z));
			sumAngles += acos(cosAngle);

			if((sumAngles <= (6.283185307179586476925287 + TOLERANCE)) && (sumAngles >= (6.283185307179586476925287 - TOLERANCE)))
			{
				isOK = true;
			}
			__syncthreads();


			if(isOK)
			{
				float dist = sqrt((_single_laser_beam.position.x - p.x) * (_single_laser_beam.position.x - p.x) +
						(_single_laser_beam.position.y - p.y) * (_single_laser_beam.position.y - p.y) +
						(_single_laser_beam.position.z - p.z) * (_single_laser_beam.position.z - p.z));

				simple_point3D p_temp;
				p_temp.x = _single_laser_beam.position.x + _single_laser_beam.direction.x * dist;
				p_temp.y = _single_laser_beam.position.y + _single_laser_beam.direction.y * dist;
				p_temp.z = _single_laser_beam.position.z + _single_laser_beam.direction.z * dist;

				if( p_temp.x == p.x && p_temp.y == p.y && p_temp.z == p.z)
				{
					d_distance[ind] = dist;
				}
			}
		}
	}
}


cudaError_t cudaComputeDistance(int threads, laser_beam &_single_laser_beam, plane *d_vPlanes, int number_of_planes,  float *d_distance)
{
	int blocks=number_of_planes/threads + 1;
	kernel_cudaComputeDistance<<<blocks,threads>>>(_single_laser_beam, d_vPlanes, number_of_planes, d_distance);
	cudaDeviceSynchronize();

	thrust::device_ptr<float> dp = thrust::device_pointer_cast(d_distance);
	thrust::device_ptr<float> pos = thrust::min_element(dp, dp + number_of_planes);
	unsigned int pos_index = thrust::distance(dp, pos);
	float min_val;
	cudaMemcpy(&min_val, &d_distance[pos_index], sizeof(float), cudaMemcpyDeviceToHost);
	_single_laser_beam.distance = min_val;

	return cudaGetLastError();
}






