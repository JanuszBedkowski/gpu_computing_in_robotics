#include "cudaWrapper.h"


CCudaWrapper::CCudaWrapper()
{

}

CCudaWrapper::~CCudaWrapper()
{

}

void CCudaWrapper::warmUpGPU()
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return;

	err = cudaWarmUpGPU();
		if(err != ::cudaSuccess)return;
}

bool CCudaWrapper::getNumberOfAvailableThreads(int &threads, int &threadsNV)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	threads = 0;
	threadsNV = 0;
	if(prop.major == 2)
	{
		threads=prop.maxThreadsPerBlock/2;
		threadsNV=prop.maxThreadsPerBlock/8;
	}else if(prop.major > 2)
	{
		threads=prop.maxThreadsPerBlock;
		threadsNV=prop.maxThreadsPerBlock/4;
	}else
	{
		return false;
	}
	return true;
}

void CCudaWrapper::coutMemoryStatus()
{
	size_t free_byte ;
    size_t total_byte ;

    cudaError_t err = cudaMemGetInfo( &free_byte, &total_byte ) ;

    if(err != ::cudaSuccess)
	{
		std::cout << "Error: cudaMemGetInfo fails: " << cudaGetErrorString(err) << std::endl;
		return;
	}
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;

    std::cout << "GPU memory usage: used = " <<
    		used_db/1024.0/1024.0 <<
			"(MB), free = " <<
			free_db/1024.0/1024.0 <<
			"(MB), total = " <<
			total_db/1024.0/1024.0 <<
			"(MB)" << std::endl;
}

bool CCudaWrapper::normalVectorCalculation(pcl::PointCloud<pcl::PointNormal> &point_cloud, float normal_vector_radius,
	        			int max_number_considered_in_INNER_bucket, int max_number_considered_in_OUTER_bucket)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	std::cout << "Before cudaMalloc" << std::endl;
	coutMemoryStatus();

	gridParameters rgd_params;
	pcl::PointNormal * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	simple_point3D *d_mean=NULL;
	int threads = 0;
	int threadsNV = 0;
	if(!getNumberOfAvailableThreads(threads, threadsNV))return false;
	std::cout << "CUDA code will use " << threads << " device threads for flip normal vectors vie viepoint" << std::endl;
	std::cout << "CUDA code will use " << threadsNV << " device threads for normal vector calculation" << std::endl;
	if(threads == 0)return false;

	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointNormal) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(pcl::PointNormal), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			normal_vector_radius, normal_vector_radius, normal_vector_radius, rgd_params);
		if(err != ::cudaSuccess)return false;

	std::cout << "regular grid parameters:" << std::endl;
	std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
	std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
	std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
	std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
	std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
	std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
	std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
	std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
	std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
	std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
	std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
	std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_mean, point_cloud.points.size()*sizeof(simple_point3D) );
		if(err != ::cudaSuccess)return false;

	std::cout << "After cudaMalloc" << std::endl;
	coutMemoryStatus();

	err = cudaCalculateNormalVectors(
			threadsNV,
			d_point_cloud,
			point_cloud.size(),
			d_hashTable,
			d_buckets,
			d_mean,
			rgd_params,
			normal_vector_radius,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket);
	if(err != ::cudaSuccess){return false;}

	err = cudaFlipNormalsTowardsViewpoint(threads,
			d_point_cloud,
			point_cloud.size(),
			0.0f,
			0.0f,
			10.0f);

	err = cudaMemcpy(point_cloud.points.data(), d_point_cloud, point_cloud.size()*sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
	if(err != ::cudaSuccess){return false;}

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_mean); d_mean = NULL;
		if(err != ::cudaSuccess)return false;

	std::cout << "After cudaFree" << std::endl;
	coutMemoryStatus();

	return true;
}

bool CCudaWrapper::normalVectorCalculationFast(pcl::PointCloud<pcl::PointNormal> &point_cloud, float normal_vector_radius,
	        			int max_number_considered_in_INNER_bucket, int max_number_considered_in_OUTER_bucket)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	std::cout << "Before cudaMalloc" << std::endl;
	coutMemoryStatus();

	gridParameters rgd_params;
	pcl::PointNormal * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	simple_point3D *d_mean=NULL;
	int threads = 0;
	int threadsNV = 0;
	if(!getNumberOfAvailableThreads(threads, threadsNV))return false;

	std::cout << "CUDA code will use " << threads << " device threads for flip normal vectors via viepoint" << std::endl;
	std::cout << "CUDA code will use " << threadsNV << " device threads for normal vector calculation" << std::endl;
	if(threads == 0)return false;

	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointNormal) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(pcl::PointNormal), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			normal_vector_radius, normal_vector_radius, normal_vector_radius, rgd_params);
		if(err != ::cudaSuccess)return false;

	std::cout << "regular grid parameters:" << std::endl;
	std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
	std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
	std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
	std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
	std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
	std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
	std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
	std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
	std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
	std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
	std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
	std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_mean, point_cloud.points.size()*sizeof(simple_point3D) );
		if(err != ::cudaSuccess)return false;

	std::cout << "After cudaMalloc" << std::endl;
	coutMemoryStatus();

	err = cudaCalculateNormalVectorsFast(
			threadsNV,
			d_point_cloud,
			point_cloud.size(),
			d_hashTable,
			d_buckets,
			d_mean,
			rgd_params,
			normal_vector_radius,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket);
	if(err != ::cudaSuccess){return false;}

	err = cudaFlipNormalsTowardsViewpoint(threads,
			d_point_cloud,
			point_cloud.size(),
			0.0f,
			0.0f,
			10.0f);

	err = cudaMemcpy(point_cloud.points.data(), d_point_cloud, point_cloud.size()*sizeof(pcl::PointNormal),cudaMemcpyDeviceToHost);
	if(err != ::cudaSuccess){return false;}

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_mean); d_mean = NULL;
		if(err != ::cudaSuccess)return false;

	std::cout << "After cudaFree" << std::endl;
	coutMemoryStatus();

	return true;
}
