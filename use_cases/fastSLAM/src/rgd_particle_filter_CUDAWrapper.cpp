#include "rgd_particle_filter_CUDAWrapper.h"
#include "cuda_algorithms.cuh"
#include <GL/freeglut.h>


CRGD_Particle_Filter_CUDAWrapper::CRGD_Particle_Filter_CUDAWrapper()
{
	rez = 0.5;
	number_of_particles = 500;
	d_cudaDevice = 0;
	motion_model_max_angle = 5.0f;
	motion_model_max_translation_X = 0.1f;
	motion_model_max_translation_Y = 0.0f;
	nW_threshold = 0.95;
}

CRGD_Particle_Filter_CUDAWrapper::~CRGD_Particle_Filter_CUDAWrapper()
{

}

void CRGD_Particle_Filter_CUDAWrapper::calculateBestTrajectory(
		std::vector<pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> > &vpc,
		std::vector<pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> > &vpc_raw,
		std::vector<Eigen::Affine3f> &vtransforms)
{
	std::cout << "calculate_rgd: begin" << std::endl;

	char *d_rgd = NULL;

	if(vpc[0].size() == 0)return;

	rgd_bounding_box.maxX = vpc[0][0].x;
	rgd_bounding_box.minX = vpc[0][0].x;
	rgd_bounding_box.maxY = vpc[0][0].y;
	rgd_bounding_box.minY = vpc[0][0].y;
	rgd_bounding_box.maxZ = vpc[0][0].z;
	rgd_bounding_box.minZ = vpc[0][0].z;

	for(size_t i = 0 ; i < vpc.size(); i++)
	{
		for(int j = 0; j < vpc[i].size(); j++)
		{
			if(vpc[i][j].x > rgd_bounding_box.maxX )rgd_bounding_box.maxX  = vpc[i][j].x;
			if(vpc[i][j].x < rgd_bounding_box.minX )rgd_bounding_box.minX  = vpc[i][j].x;

			if(vpc[i][j].y > rgd_bounding_box.maxY )rgd_bounding_box.maxY  = vpc[i][j].y;
			if(vpc[i][j].y < rgd_bounding_box.minY )rgd_bounding_box.minY  = vpc[i][j].y;

			if(vpc[i][j].z > rgd_bounding_box.maxZ )rgd_bounding_box.maxZ  = vpc[i][j].z;
			if(vpc[i][j].z < rgd_bounding_box.minZ )rgd_bounding_box.minZ  = vpc[i][j].z;
		}
	}

	std::cout << "rgd_bounding_box.maxX: " << rgd_bounding_box.maxX << std::endl;
	std::cout << "rgd_bounding_box.minX: " << rgd_bounding_box.minX << std::endl;
	std::cout << "rgd_bounding_box.maxY: " << rgd_bounding_box.maxY << std::endl;
	std::cout << "rgd_bounding_box.minY: " << rgd_bounding_box.minY << std::endl;
	std::cout << "rgd_bounding_box.maxZ: " << rgd_bounding_box.maxZ << std::endl;
	std::cout << "rgd_bounding_box.minZ: " << rgd_bounding_box.minZ << std::endl;

	std::cout << "bounding_box_resolution: " << rez << std::endl;

	rgd_params.resolution_X = rez;
	rgd_params.resolution_Y = rez;
	rgd_params.resolution_Z = rez;

	int number_of_buckets_X=((rgd_bounding_box.maxX-rgd_bounding_box.minX)/rez)+1;
	int number_of_buckets_Y=((rgd_bounding_box.maxY-rgd_bounding_box.minY)/rez)+1;
	int number_of_buckets_Z=((rgd_bounding_box.maxZ-rgd_bounding_box.minZ)/rez)+1;

	std::cout << "number_of_buckets_X: " << number_of_buckets_X << std::endl;
	std::cout << "number_of_buckets_Y: " << number_of_buckets_Y << std::endl;
	std::cout << "number_of_buckets_Z: " << number_of_buckets_Z << std::endl;

	rgd_params.bounding_box_min_X = rgd_bounding_box.minX;
	rgd_params.bounding_box_max_X = rgd_bounding_box.maxX;

	rgd_params.bounding_box_min_Y = rgd_bounding_box.minY;
	rgd_params.bounding_box_max_Y = rgd_bounding_box.maxY;

	rgd_params.bounding_box_min_Z = rgd_bounding_box.minZ;
	rgd_params.bounding_box_max_Z = rgd_bounding_box.maxZ;

	rgd_params.number_of_buckets_X = number_of_buckets_X;
	rgd_params.number_of_buckets_Y = number_of_buckets_Y;
	rgd_params.number_of_buckets_Z = number_of_buckets_Z;
	rgd_params.number_of_buckets = rgd_params.number_of_buckets_X * rgd_params.number_of_buckets_Y * rgd_params.number_of_buckets_Z;


	std::cout << "memory usage for RGD: " << (double(rgd_params.number_of_buckets * number_of_particles))/(1024.0 * 1024.0) << " MB" << std::endl;

	cudaSetDevice(this->d_cudaDevice);
	int threads = getNumberOfAvailableThreads();
	std::cout << "Number of Available Threads: " << threads << std::endl;

	cudaMalloc((void**)&d_rgd, rgd_params.number_of_buckets * this->number_of_particles*sizeof(char) );

	char *h_rgd = (char *)malloc(rgd_params.number_of_buckets * this->number_of_particles*sizeof(char));
	memset(h_rgd, -1, rgd_params.number_of_buckets * this->number_of_particles*sizeof(char));
	cudaMemcpy(d_rgd, h_rgd, rgd_params.number_of_buckets * this->number_of_particles*sizeof(char) ,cudaMemcpyHostToDevice);

	unsigned int scan_idx = 0;

	double *d_m;
	if(cudaMalloc((void**)&d_m, 16*sizeof(double) ) != ::cudaSuccess){return;}

	double h_m[16];
	copyMatrixToDobleTable(h_m, vtransforms[scan_idx]);
	if(cudaMemcpy(d_m, h_m, 16*sizeof(double),cudaMemcpyHostToDevice)!= ::cudaSuccess){return;}

	velodyne_pointcloud::PointXYZIRNL * d_pc;
	cudaMalloc((void**)&d_pc, vpc_raw[scan_idx].points.size()*sizeof(velodyne_pointcloud::PointXYZIRNL) );
	cudaMemcpy(d_pc, vpc_raw[scan_idx].points.data(), vpc_raw[scan_idx].points.size()*sizeof(velodyne_pointcloud::PointXYZIRNL), cudaMemcpyHostToDevice);

	for(unsigned int i = 0 ; i < this->number_of_particles ; i++)
	{
		cuda_insertPointCloudToRGD(threads, d_pc, (int)vpc_raw[scan_idx].points.size(), d_m, d_rgd, i, this->rgd_params);
	}

	cudaFree(d_pc); d_pc = NULL;

	vparticles.clear();

	for(int i = 0 ; i < number_of_particles; i++)
	{
		particle_state_t particle_state;
		particle_state.matrix = vtransforms[scan_idx];
		particle_state.overlap = 1.0f;
		particle_t particle;
		particle.isOverlapOK = true;
		particle.W = std::log(1);
		particle.nW = 0.0f;
		particle.v_particle_states.push_back(particle_state);
		vparticles.push_back(particle);
	}

	for( int scan_idx = 1; scan_idx < vpc_raw.size(); scan_idx++)
	{
		std::cout << "Processing: " << scan_idx << " of: " << vpc_raw.size() << std::endl;
		clock_t begin_time = clock();

		Eigen::Affine3f odometryIncrement = vtransforms[scan_idx-1].inverse()*vtransforms[scan_idx];

		prediction(odometryIncrement);

		cudaMalloc((void**)&d_pc, vpc_raw[scan_idx].points.size()*sizeof(velodyne_pointcloud::PointXYZIRNL) );
		cudaMemcpy(d_pc, vpc_raw[scan_idx].points.data(), vpc_raw[scan_idx].points.size()*sizeof(velodyne_pointcloud::PointXYZIRNL), cudaMemcpyHostToDevice);

		for(unsigned int i = 0 ; i < this->number_of_particles ; i++)
		{
			size_t size = vparticles[i].v_particle_states.size();
			Eigen::Affine3f matrix = vparticles[i].v_particle_states[size-1].matrix;
			copyMatrixToDobleTable(h_m, matrix);

			if(cudaMemcpy(d_m, h_m, 16*sizeof(double),cudaMemcpyHostToDevice)!= ::cudaSuccess){return;}

			float overlap;
			cuda_computeOverlap(threads, d_pc, (int)vpc_raw[scan_idx].points.size(), d_m, d_rgd, i, this->rgd_params, overlap);
			cuda_insertPointCloudToRGD(threads, d_pc, (int)vpc_raw[scan_idx].points.size(), d_m, d_rgd, i, this->rgd_params);

			if(overlap > 0.0f)
			{
				vparticles[i].isOverlapOK = true;
				vparticles[i].W += std::log(overlap);
			}else
			{
				vparticles[i].isOverlapOK = false;
			}
		}
		cudaFree(d_pc); d_pc = NULL;

		float lmax = vparticles[0].W;
		for(size_t i = 1; i < vparticles.size(); i++)
		{
			if(vparticles[i].W > lmax)lmax = vparticles[i].W;
		}

		float gain = 1.0f;
		float sum = 0.0f;
		for(size_t i = 0; i < vparticles.size(); i++)
		{
			vparticles[i].nW = std::exp((vparticles[i].W - lmax) * gain);
			sum += vparticles[i].nW;
		}

		float _max = vparticles[0].nW;
		index_best_particle = 0;

		for(size_t i = 1; i < vparticles.size(); i++)
		{
			if(vparticles[i].nW > _max)
			{
				_max = vparticles[i].nW;
				index_best_particle = i;
			}
		}

		int counter = 0;
		for(size_t i = 0; i < vparticles.size(); i++)
		{
			if(vparticles[i].isOverlapOK == false || vparticles[i].nW < nW_threshold)
			{
				cuda_replikate_particle_in_rgd(
						threads,
						d_rgd,
						this->rgd_params,
						i,
						index_best_particle,
						this->number_of_particles);
				vparticles[i] = vparticles[index_best_particle];
				counter++;
			}
		}
		std::cout << "solve_time: " << (double)( clock () - begin_time ) /  CLOCKS_PER_SEC << std::endl;
		std::cout << "Number of replicated particles: " << counter << std::endl;
	}

	float _max = vparticles[0].nW;
	index_best_particle = 0;

	for(size_t i = 1; i < vparticles.size(); i++)
	{
		if(vparticles[i].nW > _max)
		{
			_max = vparticles[i].nW;
			index_best_particle = i;
		}
	}

	std::cout << "data to render computation start" << std::endl;
	vpc_to_render.clear();
	for(size_t i = 0 ; i < vpc_raw.size(); i++ )
	{
		pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> pc = vpc_raw[i];
		transformPoints(pc, vparticles[index_best_particle].v_particle_states[i].matrix);
		vpc_to_render.push_back(pc);
	}
	std::cout << "data to render computation finished" << std::endl;

	cudaFree(d_m); d_m = NULL;

	cudaMemcpy(h_rgd, d_rgd, rgd_params.number_of_buckets * this->number_of_particles*sizeof(char), cudaMemcpyDeviceToHost);
	map_to_render.clear();

	for(int ix = 0 ; ix < this->rgd_params.number_of_buckets_X; ix++)
	{
		for(int iy = 0 ; iy < this->rgd_params.number_of_buckets_Y; iy++)
		{
			for(int iz = 0 ; iz < this->rgd_params.number_of_buckets_Z; iz++)
			{
				unsigned int gr_index =
					ix*rgd_params.number_of_buckets_Y*rgd_params.number_of_buckets_Z+
					iy*rgd_params.number_of_buckets_Z +
					iz + rgd_params.number_of_buckets * index_best_particle;

				if(h_rgd[gr_index] != -1)
				{
					velodyne_pointcloud::PointXYZIRNL p;
					p.label = h_rgd[gr_index];
					p.x = rgd_bounding_box.minX + ix * this->rez;
					p.y = rgd_bounding_box.minY + iy * this->rez;
					p.z = rgd_bounding_box.minZ + iz * this->rez;

					map_to_render.push_back(p);
				}
			}
		}
	}

	free(h_rgd);

	cudaFree(d_rgd); d_rgd = NULL;

	for(size_t i = 0 ; i < vparticles[index_best_particle].v_particle_states.size(); i++)
	{
		vtransforms[i] = vparticles[index_best_particle].v_particle_states[i].matrix;
	}
}

void CRGD_Particle_Filter_CUDAWrapper::transformPoints(pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> &points, Eigen::Affine3f matrix)
{
	for(size_t i = 0 ; i < points.size(); i++)
	{
		Eigen::Vector3f p(points[i].x, points[i].y, points[i].z);
		Eigen::Vector3f pt;

		pt = matrix * p;

		points[i].x = pt.x();
		points[i].y = pt.y();
		points[i].z = pt.z();

		Eigen::Vector3f vn(points[i].normal_x, points[i].normal_y, points[i].normal_z);
		Eigen::Vector3f vnt =  matrix.rotation() * vn;

		points[i].normal_x = vnt(0);
		points[i].normal_y = vnt(1);
		points[i].normal_z = vnt(2);
	}
	return;
}

void CRGD_Particle_Filter_CUDAWrapper::prediction(Eigen::Affine3f odometryIncrement)
{
	for(size_t i = 0; i < vparticles.size(); i++)
	{
		particle_state_t p;

		float angle = (float(rand()%100000))/100000.0f * motion_model_max_angle * 2.0f - motion_model_max_angle;
		float anglaRad = angle*M_PI/180.0;

		Eigen::Affine3f mr;
				mr = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitX())
					  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
					  * Eigen::AngleAxisf(anglaRad, Eigen::Vector3f::UnitZ());

		Eigen::Affine3f mt = Eigen::Affine3f::Identity();
		mt(0,3) = (float(rand()%100000))/100000.0f * motion_model_max_translation_X * 2.0f - motion_model_max_translation_X;
		mt(1,3) = (float(rand()%100000))/100000.0f * motion_model_max_translation_Y * 2.0f - motion_model_max_translation_Y;

		p.matrix = vparticles[i].v_particle_states[vparticles[i].v_particle_states.size()-1].matrix * mr * mt * odometryIncrement;
		p.overlap = 0.0f;

		vparticles[i].v_particle_states.push_back(p);
	}
}

void CRGD_Particle_Filter_CUDAWrapper::copyMatrixToDobleTable(double *h_m, Eigen::Affine3f matrix)
{
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
}

int CRGD_Particle_Filter_CUDAWrapper::getNumberOfAvailableThreads()
{
	cudaSetDevice(this->d_cudaDevice);
	cudaDeviceProp prop;
	cudaGetDevice(&this->d_cudaDevice);
	cudaGetDeviceProperties(&prop,this->d_cudaDevice);

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

void CRGD_Particle_Filter_CUDAWrapper::renderTrajectories(bool ispathrendered, bool isallpathsrendered)
{
	if(ispathrendered)
	{
		glColor3f(0.0f, 1.0f, 0.0f);
		glBegin(GL_LINE_STRIP);
		for(size_t j = 0; j < vparticles[index_best_particle].v_particle_states.size(); j++ )
		{
			glVertex3f(vparticles[index_best_particle].v_particle_states[j].matrix(0,3),
					vparticles[index_best_particle].v_particle_states[j].matrix(1,3),
					vparticles[index_best_particle].v_particle_states[j].matrix(2,3));
		}
		glEnd();
	}

	if(isallpathsrendered)
	{
		glColor3f(1.0f, 0.0f, 0.0f);
		for(size_t i = 0 ; i < vparticles.size(); i++)
		{
			if(i != index_best_particle)
			{
				glBegin(GL_LINE_STRIP);
				for(size_t j = 0; j < vparticles[i].v_particle_states.size(); j++ )
				{
					glVertex3f(vparticles[i].v_particle_states[j].matrix(0,3),
							vparticles[i].v_particle_states[j].matrix(1,3),
							vparticles[i].v_particle_states[j].matrix(2,3));
				}
			}
		glEnd();
		}
	}
}

void CRGD_Particle_Filter_CUDAWrapper::render(bool ispathrendered, bool isallpathsrendered)
{
	if(vparticles.size() == 0)return;

	renderTrajectories(ispathrendered, isallpathsrendered);

	glBegin(GL_POINTS);

	for(size_t j = 0 ; j < map_to_render.size() ; j++)
	{
		if(map_to_render[j].label == 0)glColor3f(1.0f, 0.0f, 0.0f);
		if(map_to_render[j].label == 1)glColor3f(0.0f, 1.0f, 0.0f);
		if(map_to_render[j].label == 2)glColor3f(0.0f, 0.0f, 1.0f);
		if(map_to_render[j].label == 3)glColor3f(1.0f, 1.0f, 1.0f);

		glVertex3f(map_to_render[j].x,
				   map_to_render[j].y,
				   map_to_render[j].z);
	}

	glEnd();
}

