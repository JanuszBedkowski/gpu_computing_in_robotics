#ifndef __RGD_PARTICLE_FILTER_CUDAWRAPPER__
#define __RGD_PARTICLE_FILTER_CUDAWRAPPER__

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "cudaStructures.h"
#include "data_model.hpp"
#include "point_types.h"
#include <vector>


class CRGD_Particle_Filter_CUDAWrapper
{
public:
	typedef struct particle_state
	{
		Eigen::Affine3f matrix;
		float overlap;
	}particle_state_t;

	typedef struct particle
	{
		bool isOverlapOK;
		float W;
		float nW;
		std::vector<particle_state_t> v_particle_states;
	}particle_t;

	CRGD_Particle_Filter_CUDAWrapper();
	~CRGD_Particle_Filter_CUDAWrapper();

	void calculateBestTrajectory(std::vector<pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> > &vpc,
			std::vector<pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> > &vpc_raw,
			std::vector<Eigen::Affine3f> &vtransforms);

	void transformPoints(pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> &points, Eigen::Affine3f matrix);
	void prediction(Eigen::Affine3f odometryIncrement);
	void copyMatrixToDobleTable(double *h_m, Eigen::Affine3f matrix);
	int getNumberOfAvailableThreads();
	void renderTrajectories(bool ispathrendered, bool isallpathsrendered);
	void render(bool ispathrendered, bool isallpathsrendered);

	double rez;
	unsigned int number_of_particles;
	minmax rgd_bounding_box;
	gridParameters rgd_params;
	int d_cudaDevice;

	std::vector<particle_t> vparticles;
	float motion_model_max_angle;
	float motion_model_max_translation_X;
	float motion_model_max_translation_Y;
	float nW_threshold;

	int index_best_particle;

	std::vector<pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> > vpc_to_render;
	pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL>  map_to_render;
};


#endif
