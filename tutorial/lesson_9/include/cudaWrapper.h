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

#include "lesson_9.h"
#include <pcl/point_cloud.h>

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();
	void coutMemoryStatus();

	bool nearestNeighbourhoodSearch(
			pcl::PointCloud<pcl::PointXYZ> &first_point_cloud,
			pcl::PointCloud<pcl::PointXYZ> &second_point_cloud,
			float search_radius,
			float bounding_box_extension,
			int max_number_considered_in_INNER_bucket,
			int max_number_considered_in_OUTER_bucket,
			std::vector<int> &nearest_neighbour_indexes);

	bool transformPointCloud(pcl::PointCloud<pcl::PointXYZ> &point_cloud, Eigen::Affine3d matrix);

	bool rotateXplus(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
	bool rotateXminus(pcl::PointCloud<pcl::PointXYZ> &point_cloud);

	bool dataRegistrationICP(
				pcl::PointCloud<pcl::PointXYZ> &first_point_cloud,
				pcl::PointCloud<pcl::PointXYZ> &second_point_cloud,
				float search_radius,
				float bounding_box_extension,
				int max_number_considered_in_INNER_bucket,
				int max_number_considered_in_OUTER_bucket,
				int number_of_iterations,
				std::vector<int> &nearest_neighbour_indexes,
				Eigen::Affine3d &mICP);
};

#endif
