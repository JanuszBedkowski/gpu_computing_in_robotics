#include "cudaWrapper.h"
#include <stdio.h>



CCudaWrapper::CCudaWrapper()
{
	this->cudaDevice = 0;
}

CCudaWrapper::~CCudaWrapper()
{

}

void CCudaWrapper::setDevice(int cudaDevice)
{
	this->cudaDevice = cudaDevice;
}

bool CCudaWrapper::warmUpGPU()
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(this->cudaDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaWarmUpGPU();
		if(err != ::cudaSuccess)return false;

	this->threads = getNumberOfAvailableThreads();

return true;
}

void CCudaWrapper::printCUDAinfo(int _device_id)
{
	//todo change into CUDA 8.0!!! PASCAL info is missing

	int dev=_device_id, driverVersion = 0, runtimeVersion = 0;

		cudaSetDevice(_device_id);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

		// Console log
		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

		char msg[256];
		sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
				(float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
		printf("%s", msg);

		//printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
		//	   deviceProp.multiProcessorCount,
		//	   _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
		//	   _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
		printf("  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
		// This is supported in CUDA 5.0 (runtime API device properties)
		printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
		}

#else
		// This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
		int memoryClock;
		getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
		printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
		int memBusWidth;
		getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
		printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
		int L2CacheSize;
		getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

		if (L2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
		}

#endif

		printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
			   deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
			   deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
		printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
			   deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
		printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
			   deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
			   deviceProp.maxThreadsDim[0],
			   deviceProp.maxThreadsDim[1],
			   deviceProp.maxThreadsDim[2]);
		printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
			   deviceProp.maxGridSize[0],
			   deviceProp.maxGridSize[1],
			   deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
		printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
		printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
		printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n", deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

		const char *sComputeMode[] =
		{
			"Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
			"Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
			"Prohibited (no host thread can use ::cudaSetDevice() with this device)",
			"Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
			"Unknown",
			NULL
		};
		printf("  Compute Mode:\n");
		printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);

}

int CCudaWrapper::getNumberOfAvailableThreads()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, this->cudaDevice);

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
	cudaGetDeviceProperties(&prop, this->cudaDevice);

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

bool CCudaWrapper::transformPointCloud(pcl::PointXYZ *d_in_point_cloud,
				   pcl::PointXYZ *d_out_point_cloud,
				   int number_of_points,
				   Eigen::Affine3f matrix)
{
	cudaError_t err = ::cudaSuccess;

	//float h_m[16];
	float *d_m;

	/*h_m[0] = matrix.matrix()(0,0);
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
	h_m[15] = matrix.matrix()(3,3);*/

	err = cudaMalloc((void**)&d_m, 16*sizeof(float) );
		if(err != ::cudaSuccess)return false;

	//err = cudaMemcpy(d_m, h_m, 16*sizeof(float), cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_m, matrix.data(), 16*sizeof(float), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaTransformPointCloud(threads,
			d_in_point_cloud,
			d_out_point_cloud,
			number_of_points,
			d_m);
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_m);
		if(err != ::cudaSuccess)return false;

return true;
}

bool CCudaWrapper::nearestNeighbourhoodSearch(
		pcl::PointXYZ *d_first_point_cloud,
		int number_of_points_first_point_cloud,
		pcl::PointXYZ *d_second_point_cloud,
		int number_of_points_second_point_cloud,
		float search_radius,
		float bounding_box_extension,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket,
		int *d_nearest_neighbour_indexes)
{
	cudaError_t err = ::cudaSuccess;
	gridParameters rgd_params;
	hashElement *d_hashTable = NULL;
	bucket *d_buckets = NULL;

	err = cudaCalculateGridParams(d_first_point_cloud, number_of_points_first_point_cloud,
				search_radius, search_radius, search_radius, bounding_box_extension, rgd_params);
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

	err = cudaMalloc((void**)&d_hashTable, number_of_points_first_point_cloud*sizeof(hashElement));
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_first_point_cloud, d_buckets, d_hashTable, number_of_points_first_point_cloud, rgd_params);
			if(err != ::cudaSuccess)return false;


			err = cudaNearestNeighborSearch(
						threads,
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
			   	   if(err != ::cudaSuccess)return false;


	err = cudaFree(d_hashTable); d_hashTable = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

return true;
}

bool CCudaWrapper::removeNoise(
		pcl::PointCloud<pcl::PointXYZ> &point_cloud,
		float search_radius,
		float bounding_box_extension,
		int number_of_points_in_search_sphere_threshold,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	gridParameters rgd_params;
	pcl::PointXYZ * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	bool* d_markers;
	bool* h_markers;
	int threads = getNumberOfAvailableThreads();

	//std::cout << "CUDA code will use " << threads << " device threads" << std::endl;
	if(threads == 0)return false;


	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointXYZ) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			search_radius, search_radius, search_radius, bounding_box_extension,rgd_params);
		if(err != ::cudaSuccess)return false;

		//std::cout << "regular grid parameters:" << std::endl;
		//std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
		//std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
		//std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
		//std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
		//std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
		//std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
		//std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
		//std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
		//std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
		//std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
		//std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
		//std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
			if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_markers, point_cloud.points.size()*sizeof(bool) );
			if(err != ::cudaSuccess)return false;

	err = cudaRemoveNoise(
			threads,
			d_point_cloud,
			point_cloud.points.size(),
			d_hashTable,
			d_buckets,
			rgd_params,
			search_radius,
			number_of_points_in_search_sphere_threshold,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			d_markers);
			if(err != ::cudaSuccess)return false;

	h_markers = (bool *)malloc(point_cloud.points.size()*sizeof(bool));

	err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size()*sizeof(bool),cudaMemcpyDeviceToHost);
			if(err != ::cudaSuccess)return false;

	pcl::PointCloud<pcl::PointXYZ> filtered_point_cloud;
	for(size_t i = 0; i < point_cloud.points.size(); i++)
	{
		if(h_markers[i])filtered_point_cloud.push_back(point_cloud[i]);
	}

	//std::cout << "Number of points before filtering: " << point_cloud.size() << std::endl;

	point_cloud = filtered_point_cloud;
	//std::cout << "Number of points after filtering: " << point_cloud.size() << std::endl;

	//std::cout << "Before cudaFree" << std::endl;
	//coutMemoryStatus();

	free(h_markers);

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_markers); d_markers = NULL;
			if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaFree" << std::endl;
	//coutMemoryStatus();

	return true;
}

bool CCudaWrapper::removeNoise(
		pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &point_cloud,
		float search_radius,
		float bounding_box_extension,
		int number_of_points_in_search_sphere_threshold,
		int max_number_considered_in_INNER_bucket,
		int max_number_considered_in_OUTER_bucket)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	gridParameters rgd_params;
	velodyne_pointcloud::PointXYZIR *d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	bool* d_markers;
	bool* h_markers;
	int threads = getNumberOfAvailableThreads();

	//std::cout << "CUDA code will use " << threads << " device threads" << std::endl;
	if(threads == 0)return false;


	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(velodyne_pointcloud::PointXYZIR) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(velodyne_pointcloud::PointXYZIR), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			search_radius, search_radius, search_radius, bounding_box_extension,rgd_params);
		if(err != ::cudaSuccess)return false;

		//std::cout << "regular grid parameters:" << std::endl;
		//std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
		//std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
		//std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
		//std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
		//std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
		//std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
		//std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
		//std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
		//std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
		//std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
		//std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
		//std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
			if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_markers, point_cloud.points.size()*sizeof(bool) );
			if(err != ::cudaSuccess)return false;

	err = cudaRemoveNoise(
			threads,
			d_point_cloud,
			point_cloud.points.size(),
			d_hashTable,
			d_buckets,
			rgd_params,
			search_radius,
			number_of_points_in_search_sphere_threshold,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			d_markers);
			if(err != ::cudaSuccess)return false;

	h_markers = (bool *)malloc(point_cloud.points.size()*sizeof(bool));

	err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size()*sizeof(bool),cudaMemcpyDeviceToHost);
			if(err != ::cudaSuccess)return false;

	pcl::PointCloud<velodyne_pointcloud::PointXYZIR> filtered_point_cloud;
	for(size_t i = 0; i < point_cloud.points.size(); i++)
	{
		if(h_markers[i])filtered_point_cloud.push_back(point_cloud[i]);
	}

	//std::cout << "Number of points before filtering: " << point_cloud.size() << std::endl;

	point_cloud = filtered_point_cloud;
	//std::cout << "Number of points after filtering: " << point_cloud.size() << std::endl;

	//std::cout << "Before cudaFree" << std::endl;
	//coutMemoryStatus();

	free(h_markers);

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_markers); d_markers = NULL;
			if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaFree" << std::endl;
	//coutMemoryStatus();

	return true;
}

bool CCudaWrapper::downsampling(pcl::PointCloud<pcl::PointXYZ> &point_cloud, float resolution)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	gridParameters rgd_params;
	pcl::PointXYZ * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	bool* d_markers;
	bool* h_markers;
	int threads = getNumberOfAvailableThreads();

	//std::cout << "CUDA code will use " << threads << " device threads" << std::endl;
	if(threads == 0)return false;


	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(pcl::PointXYZ) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(pcl::PointXYZ), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			resolution, resolution, resolution, 0.0f, rgd_params);
		if(err != ::cudaSuccess)return false;

		//std::cout << "regular grid parameters:" << std::endl;
		//std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
		//std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
		//std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
		//std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
		//std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
		//std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
		//std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
		//std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
		//std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
		//std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
		//std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
		//std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
			if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_markers, point_cloud.points.size()*sizeof(bool) );
			if(err != ::cudaSuccess)return false;

	err = cudaDownSample(threads, d_markers, d_hashTable, d_buckets, rgd_params, point_cloud.points.size());
			if(err != ::cudaSuccess)return false;

	h_markers = (bool *)malloc(point_cloud.points.size()*sizeof(bool));

	err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size()*sizeof(bool),cudaMemcpyDeviceToHost);
			if(err != ::cudaSuccess)return false;

	pcl::PointCloud<pcl::PointXYZ> downsampled_point_cloud;
	for(size_t i = 0; i < point_cloud.points.size(); i++)
	{
		if(h_markers[i])downsampled_point_cloud.push_back(point_cloud[i]);
	}


	//std::cout << "Number of points before down-sampling: " << point_cloud.size() << std::endl;

	point_cloud = downsampled_point_cloud;
	//std::cout << "Number of points after down-sampling: " << point_cloud.size() << std::endl;

	//std::cout << "Before cudaFree" << std::endl;
	//coutMemoryStatus();

	free(h_markers);

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_markers); d_markers = NULL;
			if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaFree" << std::endl;
	//coutMemoryStatus();

	return true;
}

bool CCudaWrapper::downsampling(pcl::PointCloud<velodyne_pointcloud::PointXYZIR> &point_cloud, float resolution)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	gridParameters rgd_params;
	velodyne_pointcloud::PointXYZIR * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	bool* d_markers;
	bool* h_markers;
	int threads = getNumberOfAvailableThreads();

	//std::cout << "CUDA code will use " << threads << " device threads" << std::endl;
	if(threads == 0)return false;


	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(velodyne_pointcloud::PointXYZIR) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(velodyne_pointcloud::PointXYZIR), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			resolution, resolution, resolution, 0.0f, rgd_params);
		if(err != ::cudaSuccess)return false;

		//std::cout << "regular grid parameters:" << std::endl;
		//std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
		//std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
		//std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
		//std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
		//std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
		//std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
		//std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
		//std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
		//std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
		//std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
		//std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
		//std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
			if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_markers, point_cloud.points.size()*sizeof(bool) );
			if(err != ::cudaSuccess)return false;

	err = cudaDownSample(threads, d_markers, d_hashTable, d_buckets, rgd_params, point_cloud.points.size());
			if(err != ::cudaSuccess)return false;

	h_markers = (bool *)malloc(point_cloud.points.size()*sizeof(bool));

	err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size()*sizeof(bool),cudaMemcpyDeviceToHost);
			if(err != ::cudaSuccess)return false;

	pcl::PointCloud<velodyne_pointcloud::PointXYZIR> downsampled_point_cloud;
	for(size_t i = 0; i < point_cloud.points.size(); i++)
	{
		if(h_markers[i])downsampled_point_cloud.push_back(point_cloud[i]);
	}


	//std::cout << "Number of points before down-sampling: " << point_cloud.size() << std::endl;

	point_cloud = downsampled_point_cloud;
	//std::cout << "Number of points after down-sampling: " << point_cloud.size() << std::endl;

	//std::cout << "Before cudaFree" << std::endl;
	//coutMemoryStatus();

	free(h_markers);

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_markers); d_markers = NULL;
			if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaFree" << std::endl;
	//coutMemoryStatus();

	return true;
}

//bool downsampling(pcl::PointCloud<Semantic::PointXYZL> &point_cloud, float resolution);
bool CCudaWrapper::downsampling(pcl::PointCloud<Semantic::PointXYZL> &point_cloud, float resolution)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	gridParameters rgd_params;
	Semantic::PointXYZL * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	bool* d_markers;
	bool* h_markers;
	int threads = getNumberOfAvailableThreads();

	//std::cout << "CUDA code will use " << threads << " device threads" << std::endl;
	if(threads == 0)return false;


	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(Semantic::PointXYZL) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(Semantic::PointXYZL), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			resolution, resolution, resolution, 0.0f, rgd_params);
		if(err != ::cudaSuccess)return false;

		//std::cout << "regular grid parameters:" << std::endl;
		//std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
		//std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
		//std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
		//std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
		//std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
		//std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
		//std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
		//std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
		//std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
		//std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
		//std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
		//std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
			if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
			if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_markers, point_cloud.points.size()*sizeof(bool) );
			if(err != ::cudaSuccess)return false;

	err = cudaDownSample(threads, d_markers, d_hashTable, d_buckets, rgd_params, point_cloud.points.size());
			if(err != ::cudaSuccess)return false;

	h_markers = (bool *)malloc(point_cloud.points.size()*sizeof(bool));

	err = cudaMemcpy(h_markers, d_markers, point_cloud.points.size()*sizeof(bool),cudaMemcpyDeviceToHost);
			if(err != ::cudaSuccess)return false;

	pcl::PointCloud<Semantic::PointXYZL> downsampled_point_cloud;
	for(size_t i = 0; i < point_cloud.points.size(); i++)
	{
		if(h_markers[i])downsampled_point_cloud.push_back(point_cloud[i]);
	}


	//std::cout << "Number of points before down-sampling: " << point_cloud.size() << std::endl;

	point_cloud = downsampled_point_cloud;
	//std::cout << "Number of points after down-sampling: " << point_cloud.size() << std::endl;

	//std::cout << "Before cudaFree" << std::endl;
	//coutMemoryStatus();

	free(h_markers);

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
			if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_markers); d_markers = NULL;
			if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaFree" << std::endl;
	//coutMemoryStatus();

	return true;
}

bool CCudaWrapper::classify(  pcl::PointCloud<Semantic::PointXYZNL>	&point_cloud,
					int number_of_points,
					float normal_vectors_search_radius,
					float curvature_threshold,
					float ground_Z_coordinate_threshold,
					int number_of_points_needed_for_plane_threshold,
					int max_number_considered_in_INNER_bucket,
					int max_number_considered_in_OUTER_bucket  )
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
	if(err != ::cudaSuccess)return false;

	//std::cout << "Before cudaMalloc" << std::endl;
	//coutMemoryStatus();

	gridParameters rgd_params;
	Semantic::PointXYZNL * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	simple_point3D *d_mean=NULL;

	int threads = 0;
	int threadsNV = 0;
	if(!getNumberOfAvailableThreads(threads, threadsNV))return false;


	//std::cout << "CUDA code will use " << threads << " device threads for flip normal vectors via viepoint" << std::endl;
	//std::cout << "CUDA code will use " << threadsNV << " device threads for normal vector calculation" << std::endl;
	if(threads == 0)return false;

	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(Semantic::PointXYZNL) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(Semantic::PointXYZNL), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;


	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			normal_vectors_search_radius, normal_vectors_search_radius, normal_vectors_search_radius, normal_vectors_search_radius, rgd_params);
		if(err != ::cudaSuccess)return false;

	//std::cout << "regular grid parameters:" << std::endl;
	//std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
	//std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
	//std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
	//std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
	//std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
	//std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
	//std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
	//std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
	//std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
	//std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
	//std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
	//std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threadsNV, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_mean, point_cloud.points.size()*sizeof(simple_point3D) );
		if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaMalloc" << std::endl;
	//coutMemoryStatus();

	err = cudaSemanticLabelingPlaneEdges(
			threadsNV,
			d_point_cloud,
			point_cloud.size(),
			d_hashTable,
			d_buckets,
			d_mean,
			rgd_params,
			normal_vectors_search_radius,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			curvature_threshold,
			number_of_points_needed_for_plane_threshold);
	if(err != ::cudaSuccess){return false;}

	err = cudaSemanticLabelingFloorCeiling(
			threads,
			d_point_cloud,
			point_cloud.size(),
			ground_Z_coordinate_threshold);
	if(err != ::cudaSuccess){return false;}

	err = cudaMemcpy(point_cloud.points.data(), d_point_cloud, point_cloud.size()*sizeof(Semantic::PointXYZNL), cudaMemcpyDeviceToHost);
	if(err != ::cudaSuccess){return false;}

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_mean); d_mean = NULL;
		if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaFree" << std::endl;
	//coutMemoryStatus();

	return true;
}


bool CCudaWrapper::classify(  pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL>	&point_cloud,
					int number_of_points,
					float normal_vectors_search_radius,
					float curvature_threshold,
					float ground_Z_coordinate_threshold,
					int number_of_points_needed_for_plane_threshold,
					int max_number_considered_in_INNER_bucket,
					int max_number_considered_in_OUTER_bucket,
					float viepointX,
					float viepointY,
					float viepointZ)
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
	if(err != ::cudaSuccess)return false;

	//std::cout << "Before cudaMalloc" << std::endl;
	//coutMemoryStatus();

	gridParameters rgd_params;
	velodyne_pointcloud::PointXYZIRNL * d_point_cloud;
	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;
	simple_point3D *d_mean=NULL;

	int threads = 0;
	int threadsNV = 0;
	if(!getNumberOfAvailableThreads(threads, threadsNV))return false;


	//std::cout << "CUDA code will use " << threads << " device threads for flip normal vectors via viepoint" << std::endl;
	//std::cout << "CUDA code will use " << threadsNV << " device threads for normal vector calculation" << std::endl;
	if(threads == 0)return false;

	err = cudaMalloc((void**)&d_point_cloud, point_cloud.points.size()*sizeof(velodyne_pointcloud::PointXYZIRNL) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_point_cloud, point_cloud.points.data(), point_cloud.points.size()*sizeof(velodyne_pointcloud::PointXYZIRNL), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;


	err = cudaCalculateGridParams(d_point_cloud, point_cloud.points.size(),
			normal_vectors_search_radius, normal_vectors_search_radius, normal_vectors_search_radius, normal_vectors_search_radius, rgd_params);
		if(err != ::cudaSuccess)return false;

	//std::cout << "regular grid parameters:" << std::endl;
	//std::cout << "bounding_box_min_X: " << rgd_params.bounding_box_min_X << std::endl;
	//std::cout << "bounding_box_min_Y: " << rgd_params.bounding_box_min_Y << std::endl;
	//std::cout << "bounding_box_min_Z: " << rgd_params.bounding_box_min_Z << std::endl;
	//std::cout << "bounding_box_max_X: " << rgd_params.bounding_box_max_X << std::endl;
	//std::cout << "bounding_box_max_Y: " << rgd_params.bounding_box_max_Y << std::endl;
	//std::cout << "bounding_box_max_Z: " << rgd_params.bounding_box_max_Z << std::endl;
	//std::cout << "number_of_buckets_X: " << rgd_params.number_of_buckets_X << std::endl;
	//std::cout << "number_of_buckets_Y: " << rgd_params.number_of_buckets_Y << std::endl;
	//std::cout << "number_of_buckets_Z: " << rgd_params.number_of_buckets_Z << std::endl;
	//std::cout << "resolution_X: " << rgd_params.resolution_X << std::endl;
	//std::cout << "resolution_Y: " << rgd_params.resolution_Y << std::endl;
	//std::cout << "resolution_Z: " << rgd_params.resolution_Z << std::endl;

	err = cudaMalloc((void**)&d_hashTable,point_cloud.points.size()*sizeof(hashElement));
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threadsNV, d_point_cloud, d_buckets, d_hashTable, point_cloud.points.size(), rgd_params);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_mean, point_cloud.points.size()*sizeof(simple_point3D) );
		if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaMalloc" << std::endl;
	//coutMemoryStatus();

	err = cudaSemanticLabelingPlaneEdges(
			threadsNV,
			d_point_cloud,
			point_cloud.size(),
			d_hashTable,
			d_buckets,
			d_mean,
			rgd_params,
			normal_vectors_search_radius,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			curvature_threshold,
			number_of_points_needed_for_plane_threshold,
			viepointX,
			viepointY,
			viepointZ);
	if(err != ::cudaSuccess){return false;}

	err = cudaSemanticLabelingFloorCeiling(
			threads,
			d_point_cloud,
			point_cloud.size(),
			ground_Z_coordinate_threshold);
	if(err != ::cudaSuccess){return false;}

	err = cudaMemcpy(point_cloud.points.data(), d_point_cloud, point_cloud.size()*sizeof(velodyne_pointcloud::PointXYZIRNL), cudaMemcpyDeviceToHost);
	if(err != ::cudaSuccess){return false;}

	err = cudaFree(d_point_cloud); d_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_mean); d_mean = NULL;
		if(err != ::cudaSuccess)return false;

	//std::cout << "After cudaFree" << std::endl;
	//coutMemoryStatus();

	return true;
}
