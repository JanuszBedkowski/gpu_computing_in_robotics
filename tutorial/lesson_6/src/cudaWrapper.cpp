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

int CCudaWrapper::getNumberOfAvailableThreads()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	int threads = 0;
	if(prop.major == 2)
	{
		threads=prop.maxThreadsPerBlock/2;
	}else if(prop.major > 2)
	{
		threads=prop.maxThreadsPerBlock;
	}else
	{
		return 0;
	}

	return threads;
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

bool CCudaWrapper::projections(	pcl::PointCloud<pcl::PointNormal> &first_point_cloud,
								pcl::PointCloud<pcl::PointXYZ> &second_point_cloud,
								float normal_vectors_search_radius,
								float projections_search_radius,
								float bounding_box_extension,
								int max_number_considered_in_INNER_bucket,
								int max_number_considered_in_OUTER_bucket,
								pcl::PointCloud<pcl::PointXYZ> &second_point_cloud_projections,
								std::vector<char> &v_is_projection )
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	std::cout << "Memory before cudaMalloc" << std::endl;
	coutMemoryStatus();

	gridParameters rgd_params;
	pcl::PointNormal *d_first_point_cloud = NULL;
	pcl::PointXYZ    *d_second_point_cloud = NULL;
	pcl::PointXYZ    *d_second_point_cloud_projections = NULL;
	char             *d_v_is_projection = NULL;

	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	simple_point3D *d_mean=NULL;
	int threads, threadsNV;
	if(!getNumberOfAvailableThreads(threads, threadsNV))return false;

	std::cout << "CUDA code will use " << threads << " device threads for flip normal vectors via viepoint" << std::endl;
	std::cout << "CUDA code will use " << threadsNV << " device threads for normal vector calculation" << std::endl;

	err = cudaMalloc((void**)&d_first_point_cloud, first_point_cloud.points.size()*sizeof(pcl::PointNormal) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_first_point_cloud, first_point_cloud.points.data(), first_point_cloud.points.size()*sizeof(pcl::PointNormal), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_second_point_cloud, second_point_cloud.points.size()*sizeof(pcl::PointXYZ) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_second_point_cloud, second_point_cloud.points.data(), second_point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(
			d_first_point_cloud,
			first_point_cloud.points.size(),
			normal_vectors_search_radius,
			normal_vectors_search_radius,
			normal_vectors_search_radius,
			bounding_box_extension,
			rgd_params);
		if(err != ::cudaSuccess)return false;

	std::cout << "regular grid parameters for normal vectors:" << std::endl;
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

	err = cudaMalloc((void**)&d_hashTable, first_point_cloud.points.size()*sizeof(hashElement));
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_mean, first_point_cloud.points.size()*sizeof(simple_point3D) );
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_first_point_cloud, d_buckets, d_hashTable, first_point_cloud.points.size(), rgd_params);
		if(err != ::cudaSuccess)return false;

	std::cout << "Memory status after cudaMalloc normal vector computation" << std::endl;
	coutMemoryStatus();

	err = cudaCalculateNormalVectorsFast(
			threadsNV,
			d_first_point_cloud,
			first_point_cloud.size(),
			d_hashTable,
			d_buckets,
			d_mean,
			rgd_params,
			normal_vectors_search_radius,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket);
	if(err != ::cudaSuccess){return false;}

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_mean); d_mean = NULL;
		if(err != ::cudaSuccess)return false;

//normal vector computation finished

	err = cudaCalculateGridParams(
				d_first_point_cloud,
				first_point_cloud.points.size(),
				projections_search_radius,
				projections_search_radius,
				projections_search_radius,
				bounding_box_extension,
				rgd_params);
			if(err != ::cudaSuccess)return false;

	std::cout << "regular grid parameters for projections:" << std::endl;
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

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_first_point_cloud, d_buckets, d_hashTable, first_point_cloud.points.size(), rgd_params);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_v_is_projection, second_point_cloud.points.size() * sizeof(char));
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_second_point_cloud_projections, second_point_cloud.points.size()*sizeof(pcl::PointXYZ) );
		if(err != ::cudaSuccess)return false;


	std::cout << "Memory status after cudaMalloc projections" << std::endl;
	coutMemoryStatus();


	err = cudaCalculateProjections(
			threads,
			d_first_point_cloud,
			first_point_cloud.points.size(),
			d_second_point_cloud,
			second_point_cloud.points.size(),
			d_hashTable,
			d_buckets,
			rgd_params,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			projections_search_radius,
			d_v_is_projection,
			d_second_point_cloud_projections);
		if(err != ::cudaSuccess)return false;


	err = cudaMemcpy(v_is_projection.data(),
			d_v_is_projection,
			second_point_cloud.points.size()/**sizeof(bool)*/,
			cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess){return false;}

	err = cudaMemcpy(second_point_cloud_projections.points.data(), d_second_point_cloud_projections, second_point_cloud.points.size()*sizeof(pcl::PointXYZ),cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess){return false;}

	err = cudaFree(d_first_point_cloud); d_first_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_second_point_cloud); d_second_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_v_is_projection); d_v_is_projection = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_second_point_cloud_projections); d_second_point_cloud_projections = NULL;
		if(err != ::cudaSuccess)return false;

	std::cout << "After cudaFree" << std::endl;
	coutMemoryStatus();

	return true;
}

bool CCudaWrapper::transform(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3f matrix)
{
	int threads;
	pcl::PointXYZ * d_point_cloud;

	float h_m[16];
	float *d_m;

	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	threads = getNumberOfAvailableThreads();

	h_m[0] = matrix.matrix()(0,0);
	h_m[1] = matrix.matrix()(1,0);
	h_m[2] = matrix.matrix()(2,0);
	h_m[3] = matrix.matrix()(3,0);

	h_m[4] = matrix.matrix()(0,1);
	h_m[5] = matrix.matrix()(1,1);
	h_m[6] = matrix.matrix()(2,1);
	h_m[7] = matrix.matrix()(3,1);

	h_m[8] = matrix.matrix()(0,2);
	h_m[9] = matrix.matrix()(1,2);
	h_m[10] = matrix.matrix()(2,2);
	h_m[11] = matrix.matrix()(3,2);

	h_m[12] = matrix.matrix()(0,3);
	h_m[13] = matrix.matrix()(1,3);
	h_m[14] = matrix.matrix()(2,3);
	h_m[15] = matrix.matrix()(3,3);

	err = cudaMalloc((void**)&d_m, 16*sizeof(float) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_m, h_m, 16*sizeof(float), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointXYZ) );
			if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaTransformPoints(threads, d_point_cloud, point_cloud.points.size(), d_m);
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(point_cloud.points.data(), d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyDeviceToHost);
		if(err != ::cudaSuccess)return false;


	err = cudaFree(d_m);
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;


	return true;
}

bool CCudaWrapper::rotateXplus(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	float anglaRad = 1.0f*M_PI/180.0;

	Eigen::Affine3f mr;
			mr = Eigen::AngleAxisf(anglaRad, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ());

	if(!transform(point_cloud, mr))
	{
		std::cout << "Problem with transform" << std::endl;
		cudaDeviceReset();
		return false;
	}
	return true;
}

bool CCudaWrapper::rotateXminus(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	float anglaRad = -1.0f*M_PI/180.0;

	Eigen::Affine3f mr;
			mr = Eigen::AngleAxisf(anglaRad, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ());

	if(!transform(point_cloud, mr))
	{
		std::cout << "Problem with transform" << std::endl;
		cudaDeviceReset();
		return false;
	}
	return true;
}
