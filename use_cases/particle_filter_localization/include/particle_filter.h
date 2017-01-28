#ifndef __PARTICLE_FILTER__
#define __PARTICLE_FILTER__

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

//common
#include "point_types.h"
#include "cudaFunctions.h"

class CParticleFilter
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

	CParticleFilter();
	~CParticleFilter();

	bool init(int cuda_device,
			float _motion_model_max_angle,
			float _motion_model_max_translation,
			float _nearest_neighbor_radius,
			float _max_particle_size,
			float _max_particle_size_kidnapped_robot,
			float _distanceAboveZ,
			float _rgd_resolution,
			int _max_number_considered_in_INNER_bucket,
			int _max_number_considered_in_OUTER_bucket,
			float _overlap_threshold,
			float _propability_threshold,
			float _rgd_2D_res);
	bool prediction(Eigen::Affine3f odometryIncrement);
	bool update();

	bool setCUDADevice(int _cudaDeveice);
	bool setGroundPointsFromMap(pcl::PointCloud<pcl::PointXYZ> pc);
	bool computeRGD();
	bool copyReferenceModelToGPU(pcl::PointCloud<Semantic::PointXYZL> &reference_model);
	bool copyCurrentScanToGPU(pcl::PointCloud<Semantic::PointXYZL> &current_scan);
	bool transformCurrentScan(Eigen::Affine3f matrix);
	void genParticlesKidnappedRobot();
	bool findClosestParticle(Eigen::Vector3f _reference_pose,
			Eigen::Vector3f &out_particle);
	Eigen::Affine3f getWinningParticle();

	bool computeNN(float &_overlap,
						float nearest_neighbor_radius,
						int max_number_considered_in_INNER_bucket,
						int max_number_considered_in_OUTER_bucket);
	void render();

private:
	float distance_above_Z;
	pcl::PointCloud<pcl::PointXYZ> ground_points_from_map;
	pcl::PointXYZ *d_ground_points_from_map;
	int number_of_points_ground_points_from_map;
	std::vector<particle_t> vparticles;
	float motion_model_max_angle;
	float motion_model_max_translation;
	int max_particle_size;
	int max_particle_size_kidnapped_robot;
	float nearest_neighbor_radius;

	int cudaDevice;
	int number_of_threads;

	pcl::PointCloud<Semantic::PointXYZL> h_second_point_cloud;
	std::vector<int> h_nearest_neighbour_indexes;

	Semantic::PointXYZL *d_first_point_cloud;
	int number_of_points_first_point_cloud;
	Semantic::PointXYZL *d_second_point_cloud;
	Semantic::PointXYZL *d_second_point_cloudT;
	int *d_nearest_neighbour_indexes;
	int number_of_points_second_point_cloud;

	float bounding_box_extension;
	gridParameters rgd_params;
	hashElement* d_hashTable;
	bucket* d_buckets;
	float rgd_res;
	float rgd_2D_res;

	gridParameters rgd_params_2D;
	hashElement* d_hashTable_2D;
	bucket* d_buckets_2D;

	float *d_m;

	int max_number_considered_in_INNER_bucket;
	int max_number_considered_in_OUTER_bucket;
	float overlap_threshold;
	float propability_threshold;
};



#endif
