/*
 * Software License Agreement (BSD License)
 *
 *  Data Registration Framework - Mobile Spatial Assistance System
 *  Copyright (c) 2014-2016, Institute of Mathematical Machines
 *  http://lider.zms.imm.org.pl/
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Institute of Mathematical Machines nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 */

#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_10.h"
#include <pcl/point_cloud.h>

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();
	bool getNumberOfAvailableThreads(int &threads, int &threadsNV);
	void coutMemoryStatus();

	bool transformPointCloud(pcl::PointCloud<VelodyneVLP16::PointXYZNL> &point_cloud, Eigen::Affine3d matrix);

	bool rotateXplus(pcl::PointCloud<VelodyneVLP16::PointXYZNL> &point_cloud);
	bool rotateXminus(pcl::PointCloud<VelodyneVLP16::PointXYZNL> &point_cloud);

	bool dataRegistrationSemanticICP(
				pcl::PointCloud<VelodyneVLP16::PointXYZNL> &first_point_cloud,
				pcl::PointCloud<VelodyneVLP16::PointXYZNL> &second_point_cloud,
				float search_radius,
				float bounding_box_extension,
				int max_number_considered_in_INNER_bucket,
				int max_number_considered_in_OUTER_bucket,
				int number_of_iterations,
				std::vector<int> &nearest_neighbour_indexes,
				Eigen::Affine3d &mICP);

	bool classify(
				pcl::PointCloud<VelodyneVLP16::PointXYZNL> &point_cloud,
				int number_of_points,
				float normal_vectors_search_radius,
				float curvature_threshold,
				float ground_Z_coordinate_threshold,
				int number_of_points_needed_for_plane_threshold,
				float bounding_box_extension,
				int max_number_considered_in_INNER_bucket,
				int max_number_considered_in_OUTER_bucket  );

	bool semanticNearestNeighbourhoodSearch(
				pcl::PointCloud<VelodyneVLP16::PointXYZNL> &first_point_cloud,
				pcl::PointCloud<VelodyneVLP16::PointXYZNL> &second_point_cloud,
				float search_radius,
				float bounding_box_extension,
				int max_number_considered_in_INNER_bucket,
				int max_number_considered_in_OUTER_bucket,
				std::vector<int> &nearest_neighbour_indexes );

};

#endif
