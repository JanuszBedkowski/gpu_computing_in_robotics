#include "cudaWrapper.h"


CCudaWrapper::CCudaWrapper()
{

}

CCudaWrapper::~CCudaWrapper()
{

}

bool CCudaWrapper::warmUpGPU()
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return false;

	err = cudaWarmUpGPU();
		if(err != ::cudaSuccess)return false;

	this->threads = getNumberOfAvailableThreads();

return true;
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

bool CCudaWrapper::compute_projections( 	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &first_point_cloud,
							pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &second_point_cloud,
							float projections_search_radius,
							float bounding_box_extension,
							int max_number_considered_in_INNER_bucket,
							int max_number_considered_in_OUTER_bucket,
							pcl::PointCloud<lidar_pointcloud::PointProjection> &projections	)
{
	cudaError_t err = ::cudaSuccess;

	std::cout << "Memory before cudaMalloc" << std::endl;
	coutMemoryStatus();

	gridParameters rgd_params;
	lidar_pointcloud::PointXYZIRNL    *d_first_point_cloud = NULL;
	lidar_pointcloud::PointXYZIRNL    *d_second_point_cloud = NULL;
	lidar_pointcloud::PointProjection *d_projections = NULL;

	hashElement* d_hashTable = NULL;
	bucket* d_buckets = NULL;

	err = cudaMalloc((void**)&d_first_point_cloud, first_point_cloud.points.size()*sizeof(lidar_pointcloud::PointXYZIRNL) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_first_point_cloud, first_point_cloud.points.data(), first_point_cloud.points.size()*sizeof(lidar_pointcloud::PointXYZIRNL), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_second_point_cloud, second_point_cloud.points.size()*sizeof(lidar_pointcloud::PointXYZIRNL) );
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(d_second_point_cloud, second_point_cloud.points.data(), second_point_cloud.points.size()*sizeof(lidar_pointcloud::PointXYZIRNL), cudaMemcpyHostToDevice);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_hashTable, first_point_cloud.points.size()*sizeof(hashElement));
			if(err != ::cudaSuccess)return false;

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
	std::cout << "number_of_buckets: " << rgd_params.number_of_buckets << std::endl;


	err = cudaMalloc((void**)&d_buckets, rgd_params.number_of_buckets*sizeof(bucket));
		if(err != ::cudaSuccess)return false;

	err = cudaCalculateGrid(threads, d_first_point_cloud, d_buckets, d_hashTable, first_point_cloud.points.size(), rgd_params);
		if(err != ::cudaSuccess)return false;

	err = cudaMalloc((void**)&d_projections, second_point_cloud.points.size()*sizeof(lidar_pointcloud::PointProjection) );
		if(err != ::cudaSuccess)return false;


	std::cout << "Memory status after cudaMalloc projections" << std::endl;
	coutMemoryStatus();

	err = cudaCalculateProjections(
			this->threads,
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
			d_projections);
		if(err != ::cudaSuccess)return false;

	err = cudaMemcpy(projections.points.data(), d_projections, second_point_cloud.points.size()*sizeof(lidar_pointcloud::PointProjection),cudaMemcpyDeviceToHost);
			if(err != ::cudaSuccess){return false;}

	err = cudaFree(d_first_point_cloud); d_first_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_second_point_cloud); d_second_point_cloud = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_hashTable); d_hashTable = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_buckets); d_buckets = NULL;
		if(err != ::cudaSuccess)return false;

	err = cudaFree(d_projections); d_projections = NULL;
		if(err != ::cudaSuccess)return false;

	std::cout << "After cudaFree" << std::endl;
	coutMemoryStatus();

return true;
}

void CCudaWrapper::Matrix4ToEuler(const double *alignxf, double *rPosTheta, double *rPos)
{
	double _trX, _trY;

	if(alignxf[0] > 0.0)
	{
		rPosTheta[1] = asin(alignxf[8]);
	}
	else
	{
		rPosTheta[1] = M_PI - asin(alignxf[8]);
	}

	double  C    =  cos( rPosTheta[1] );
	if ( fabs( C ) > 0.005 )
	{                 // Gimball lock?
		_trX      =  alignxf[10] / C;             // No, so get X-axis angle
		_trY      =  -alignxf[9] / C;
		rPosTheta[0]  = atan2( _trY, _trX );
		_trX      =  alignxf[0] / C;              // Get Z-axis angle
		_trY      = -alignxf[4] / C;
		rPosTheta[2]  = atan2( _trY, _trX );
	}
	else
	{                                    // Gimball lock has occurred
		rPosTheta[0] = 0.0;                       // Set X-axis angle to zero
		_trX      =  alignxf[5];  //1                // And calculate Z-axis angle
		_trY      =  alignxf[1];  //2
		rPosTheta[2]  = atan2( _trY, _trX );
	}

	rPosTheta[0] = rPosTheta[0];
	rPosTheta[1] = rPosTheta[1];
	rPosTheta[2] = rPosTheta[2];

	if (rPos != 0)
	{
		rPos[0] = alignxf[12];
		rPos[1] = alignxf[13];
		rPos[2] = alignxf[14];
	}
}

void CCudaWrapper::Matrix4ToEuler(Eigen::Affine3f m, Eigen::Vector3f &omfika, Eigen::Vector3f &xyz)
{
	double _trX, _trY;

	if(m(0,0) > 0.0)
	{
		omfika.y() = asin(m(0,2));
	}
	else
	{
		omfika.y() = M_PI - asin(m(0,2));
	}

	double  C    =  cos( omfika.y() );
	if ( fabs( C ) > 0.005 )
	{                 // Gimball lock?
		_trX      =  m(2,2) / C;             // No, so get X-axis angle
		_trY      =  -m(1,2) / C;
		omfika.x()  = atan2( _trY, _trX );
		_trX      =  m(0,0) / C;              // Get Z-axis angle
		_trY      = -m(0,1) / C;
		omfika.z()= atan2( _trY, _trX );
	}
	else
	{                                    // Gimball lock has occurred
		omfika.x() = 0.0;                       // Set X-axis angle to zero
		_trX      =  m(1,1);  //1                // And calculate Z-axis angle
		_trY      =  m(1,0);  //2
		omfika.z() = atan2( _trY, _trX );
	}

	xyz.x() = m(0,3);
	xyz.y() = m(1,3);
	xyz.z() = m(2,3);
}

void CCudaWrapper::EulerToMatrix(Eigen::Vector3f omfika, Eigen::Vector3f xyz, Eigen::Affine3f &m)
{
	Eigen::Affine3f mR;
	mR = Eigen::AngleAxisf(omfika.x(), Eigen::Vector3f::UnitX())
		  * Eigen::AngleAxisf(omfika.y(), Eigen::Vector3f::UnitY())
		  * Eigen::AngleAxisf(omfika.z(), Eigen::Vector3f::UnitZ());
	Eigen::Affine3f mT(Eigen::Translation3f(xyz.x(), xyz.y(), xyz.z()));
	m = mT * mR;
}

bool CCudaWrapper::registerLS3D(observations_LS3D_t &obs)
{
	std::cout << "CCudaWrapper::registerLS3D" << std::endl;
	cudaError_t errCUDA = ::cudaSuccess;
	CCUDA_AX_B_SolverWrapper::Solver_Method solver_method = CCUDA_AX_B_SolverWrapper::chol;

	double PforGround = 1.0;
	double PforObstacles = 1.0;

	double x[7];

	pcl::PointCloud<lidar_pointcloud::PointProjection> projections_confirmed;
	for(size_t i = 0 ; i < obs.projections.size(); i++)
	{
		if(obs.projections[i].isProjection == 1)projections_confirmed.push_back(obs.projections[i]);
	}

	std::cout << projections_confirmed.size() << std::endl;

	double *d_A = NULL;
	errCUDA  = cudaMalloc((void**)&d_A,  projections_confirmed.size()* 7 *sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}

	double *d_P = NULL;
	errCUDA  = cudaMalloc((void**)&d_P,  projections_confirmed.size() *sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}

	double *d_l=NULL;
	errCUDA  = cudaMalloc((void**)&d_l,  projections_confirmed.size() *sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}

	lidar_pointcloud::PointProjection *d_projections; d_projections = NULL;

	errCUDA = cudaMalloc((void**)&d_projections, projections_confirmed.size()*sizeof(lidar_pointcloud::PointProjection) );
		if(errCUDA != ::cudaSuccess)return false;

	errCUDA = cudaMemcpy(d_projections, projections_confirmed.points.data(), projections_confirmed.size()*sizeof(lidar_pointcloud::PointProjection), cudaMemcpyHostToDevice);
		if(errCUDA != ::cudaSuccess)return false;

	errCUDA =  fill_A_l_cuda(threads/2, d_A, obs.tx, obs.ty, obs.tz, 1.0, obs.om, obs.fi, obs.ka, d_projections, projections_confirmed.size(),
							d_P, PforGround, PforObstacles, d_l);
		if(errCUDA != ::cudaSuccess){return false;}

	CCUDA_AX_B_SolverWrapper * wr = new CCUDA_AX_B_SolverWrapper(false, 0);

	CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper_error errAXB =
					wr->Solve_ATPA_ATPl_x_data_on_GPU(this->threads, d_A, d_P, d_l, x, 7, projections_confirmed.size(), solver_method);
	if(errAXB!=CCUDA_AX_B_SolverWrapper::success)
	{
		std::cout << "problem with solving Ax=B" << std::endl;
		return false;
	}

	delete wr;

	errCUDA = cudaFree(d_projections); d_projections = 0;
		if(errCUDA != ::cudaSuccess){return false;}

	std::cout << "Ax=B solution:" << std::endl;
	for(int i = 0 ; i < 7; i++)
	{
		std::cout << "x[" << i <<"]: " << x[i] << std::endl;
	}

	errCUDA = cudaFree(d_A); d_A = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	errCUDA = cudaFree(d_P); d_P = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	errCUDA = cudaFree(d_l); d_l = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	obs.tx -= x[0];
	obs.ty -= x[1];
	obs.tz -= x[2];
	obs.om -= x[4];
    obs.fi -= x[5];
    obs.ka -= x[6];

	return true;
}
