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

#include <GL/freeglut.h>

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
//#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/filter.h>
//#include <pcl/features/normal_3d.h>
//#include <pcl/registration/transforms.h>
//#include <pcl/registration/ndt.h>
#include <pcl/console/parse.h>
//#include <pcl/registration/icp.h>
//#include <pcl/common/time.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/statistical_outlier_removal.h>
//#include <pcl/PCLPointCloud2.h>

#include "cudaWrapper.h"
#include "point_types.h"

float colors[16][3] =
		{1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		1.0f, 1.0f, 1.0f,
		0.4f, 0.1f, 0.7f,
		0.6f, 0.1f, 0.8f,
		0.8f, 0.4f, 0.9f,
		0.1f, 0.6f, 0.1f,
		0.3f, 0.1f, 0.2f,
		0.4f, 0.5f, 0.9f,
		0.4f, 0.5f, 0.1f,
		0.4f, 0.1f, 0.9f,
		0.4f, 0.6f, 0.9f,
		0.1f, 0.5f, 0.9f,
		0.4f, 0.1f, 0.1f,
		0.4f, 0.2f, 0.4f};

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -20.0;
float translate_x, translate_y = 0.0;

float search_radius = 0.5f;
float normal_vectors_search_radius = 1.0f;
float curvature_threshold = 10.0;
float ground_Z_coordinate_threshold = -1.0f;
int   number_of_points_needed_for_plane_threshold = 15;
float bounding_box_extension = 1.0f;

int max_number_considered_in_INNER_bucket = 10000;
int max_number_considered_in_OUTER_bucket = 10000;

pcl::PointCloud<VelodyneVLP16::PointXYZNL> first_point_cloud;
pcl::PointCloud<VelodyneVLP16::PointXYZNL> second_point_cloud;
std::vector<int> nearest_neighbour_indexes;
CCudaWrapper cudaWrapper;


bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();

int main(int argc, char **argv)
{
	std::cout << "Lesson 10 - data registration semantic Iterative Closest Point" << std::endl;

	if(argc < 3)
	{
		std::cout << "Usage:\n";
		std::cout << argv[0] <<" first_point_cloud_file.pcd second_point_cloud_file.pcd parameters\n";
		std::cout << "Default:  ../../data/scan_Velodyne_VLP16.pcd ../../data/scan_Velodyne_VLP16_2.pcd\n";
		std::cout << "-sr search_radius: default " << search_radius << std::endl;
		std::cout << "-nvsr normal_vectors_search_radius default: " << normal_vectors_search_radius << std::endl;
		std::cout << "-ct curvature_threshold default: " << curvature_threshold << std::endl;
		std::cout << "-gzt ground_Z_coordinate_threshold default: " << ground_Z_coordinate_threshold << std::endl;
		std::cout << "-npt number_of_points_needed_for_plane_threshold: " << number_of_points_needed_for_plane_threshold << std::endl;
		std::cout << "-bbe bounding_box_extension: default " << bounding_box_extension << std::endl;


		if(pcl::io::loadPCDFile("../../data/scan_Velodyne_VLP16.pcd", first_point_cloud) == -1)
		{
			return -1;
		}

		if(pcl::io::loadPCDFile("../../data/scan_Velodyne_VLP16_2.pcd", second_point_cloud) == -1)
		{
			return -1;
		}

	}else
	{
		std::vector<int> ind_pcd;
		ind_pcd = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

		if(ind_pcd.size() != 2)
		{
			std::cout << "did you forget pcd files location? return" << std::endl;
			return -1;
		}

		if(pcl::io::loadPCDFile(argv[1], first_point_cloud) == -1)
		{
			return -1;
		}

		if(pcl::io::loadPCDFile(argv[2], second_point_cloud) == -1)
		{
			return -1;
		}
	}

	for(size_t i = 0 ; i < first_point_cloud.size() ; i++)first_point_cloud[i].label = -1;
	for(size_t i = 0 ; i < second_point_cloud.size() ; i++)second_point_cloud[i].label = -1;


	pcl::console::parse_argument (argc, argv, "-sr", search_radius);
	std::cout << "search_radius: " << search_radius << std::endl;

	pcl::console::parse_argument (argc, argv, "-nvsr", normal_vectors_search_radius);
	std::cout << "normal_vectors_search_radius: " << normal_vectors_search_radius << std::endl;

	pcl::console::parse_argument (argc, argv, "-ct", curvature_threshold);
	std::cout << "curvature_threshold: " << curvature_threshold << std::endl;

	pcl::console::parse_argument (argc, argv, "-gzt", ground_Z_coordinate_threshold);
	std::cout << "ground_Z_coordinate_threshold: " << ground_Z_coordinate_threshold << std::endl;

	pcl::console::parse_argument (argc, argv, "-npt", number_of_points_needed_for_plane_threshold);
	std::cout << "number_of_points_needed_for_plane_threshold: " << number_of_points_needed_for_plane_threshold << std::endl;

	pcl::console::parse_argument (argc, argv, "-bbe", bounding_box_extension);
	std::cout << "bounding_box_extension: " << bounding_box_extension << std::endl;

	if (false == initGL(&argc, argv))
	{
		return -1;
	}

	nearest_neighbour_indexes.resize(second_point_cloud.size());
	std::fill(nearest_neighbour_indexes.begin(), nearest_neighbour_indexes.end(), -1);

	printHelp();
	cudaWrapper.warmUpGPU();

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
    glutReshapeFunc(reshape);
	glutMainLoop();
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Lesson 10 - data registration semantic Iterative Closest Point");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.01, 10000.0);

    return true;
}

void reshape(int width, int height) {
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)width / (GLfloat) height, 0.01, 10000.0);
}

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(translate_x, translate_y, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 0.0, 1.0);

    glBegin(GL_LINES);
   	glColor3f(1.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(1.0f, 0.0f, 0.0f);

    glColor3f(0.0f, 1.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);

    glColor3f(0.0f, 0.0f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(0.0f, 0.0f, 1.0f);
    glEnd();

    glPointSize(2);

    glBegin(GL_POINTS);
	for(size_t i = 0; i < first_point_cloud.size(); i++)
	{
		if(first_point_cloud[i].label<16 && first_point_cloud[i].label>=0)
		{
			glColor3f(colors[first_point_cloud[i].label][0], colors[first_point_cloud[i].label][1], colors[first_point_cloud[i].label][2]);
		}else
		{

			glColor3f(0.5f, 0.2f, 0.2f);
		}
		glVertex3f(first_point_cloud[i].x, first_point_cloud[i].y, first_point_cloud[i].z);
	}
	glEnd();

	glBegin(GL_POINTS);
	for(size_t i = 0; i < second_point_cloud.size(); i++)
	{
		if(second_point_cloud[i].label<16 && second_point_cloud[i].label>=0)
		{
			glColor3f(colors[second_point_cloud[i].label][0] * 0.7f, colors[second_point_cloud[i].label][1] * 0.7f, colors[second_point_cloud[i].label][2] * 0.7f);
		}else
		{

			glColor3f(0.2f, 0.2f, 0.5f);
		}
		glVertex3f(second_point_cloud[i].x, second_point_cloud[i].y, second_point_cloud[i].z);
	}
	glEnd();

	glBegin(GL_LINES);
		for(size_t i = 0; i < second_point_cloud.size(); i++)
		{
			int index_nn = nearest_neighbour_indexes[i];
			if(index_nn != -1 && index_nn >= 0 && index_nn < first_point_cloud.size())
			{
				glColor3f(colors[second_point_cloud[i].label][0], colors[second_point_cloud[i].label][1], colors[second_point_cloud[i].label][2]);

				glVertex3f(second_point_cloud[i].x, second_point_cloud[i].y, second_point_cloud[i].z);
				glVertex3f(first_point_cloud[index_nn].x, first_point_cloud[index_nn].y, first_point_cloud[index_nn].z);
			}
		}
	glEnd();

    glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            glutDestroyWindow(glutGetWindow());
            return;
        case 'n' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

			if(!cudaWrapper.classify( first_point_cloud,
								first_point_cloud.points.size(),
								normal_vectors_search_radius,
								curvature_threshold,
								ground_Z_coordinate_threshold,
								number_of_points_needed_for_plane_threshold,
								bounding_box_extension,
								max_number_considered_in_INNER_bucket,
								max_number_considered_in_OUTER_bucket ) )
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.classify NOT SUCCESFULL" << std::endl;
			}

			if(!cudaWrapper.classify( second_point_cloud,
								second_point_cloud.points.size(),
								normal_vectors_search_radius,
								curvature_threshold,
								ground_Z_coordinate_threshold,
								number_of_points_needed_for_plane_threshold,
								bounding_box_extension,
								max_number_considered_in_INNER_bucket,
								max_number_considered_in_OUTER_bucket ) )
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.classify NOT SUCCESFULL" << std::endl;
			}


        	if(!cudaWrapper.semanticNearestNeighbourhoodSearch(
        			first_point_cloud,
        			second_point_cloud,
        			search_radius,
        			bounding_box_extension,
        			max_number_considered_in_INNER_bucket,
        			max_number_considered_in_OUTER_bucket,
        			nearest_neighbour_indexes))
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.nearestNeighbourhoodSearch NOT SUCCESFULL" << std::endl;
			}

        	computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.nearestNeighbourhoodSearch computation_time: " << computation_time << std::endl;

			break;
        }
        case 'a' :
		{
			clock_t begin_time;
			double computation_time;
			begin_time = clock();

			cudaWrapper.rotateXplus(second_point_cloud);

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.rotateXplus computation_time: " << computation_time << std::endl;
			std::fill(nearest_neighbour_indexes.begin(), nearest_neighbour_indexes.end(), -1);
			break;
		}
		case 'd' :
		{
			clock_t begin_time;
			double computation_time;
			begin_time = clock();

			cudaWrapper.rotateXminus(second_point_cloud);

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.rotateXminus computation_time: " << computation_time << std::endl;
			std::fill(nearest_neighbour_indexes.begin(), nearest_neighbour_indexes.end(), -1);
			break;
		}
		case 'r' :
		{
			clock_t begin_time;
			double computation_time;
			begin_time = clock();

			if(!cudaWrapper.classify( first_point_cloud,
											first_point_cloud.points.size(),
											normal_vectors_search_radius,
											curvature_threshold,
											ground_Z_coordinate_threshold,
											number_of_points_needed_for_plane_threshold,
											bounding_box_extension,
											max_number_considered_in_INNER_bucket,
											max_number_considered_in_OUTER_bucket ) )
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.classify NOT SUCCESFULL" << std::endl;
			}

			if(!cudaWrapper.classify( second_point_cloud,
								second_point_cloud.points.size(),
								normal_vectors_search_radius,
								curvature_threshold,
								ground_Z_coordinate_threshold,
								number_of_points_needed_for_plane_threshold,
								bounding_box_extension,
								max_number_considered_in_INNER_bucket,
								max_number_considered_in_OUTER_bucket ) )
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.classify NOT SUCCESFULL" << std::endl;
			}

			Eigen::Affine3d mICP;
			if(!cudaWrapper.dataRegistrationSemanticICP(
								first_point_cloud,
								second_point_cloud,
								search_radius,
								bounding_box_extension,
								max_number_considered_in_INNER_bucket,
								max_number_considered_in_OUTER_bucket,
								1,
								nearest_neighbour_indexes,
								mICP))
			{
				std::cout << "cudaWrapper.dataRegistrationSemanticICP NOT SUCCESFULL" << std::endl;
			}else
			{
				std::cout << mICP.matrix() << std::endl;
			}


			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.dataRegistrationSemanticICP computation_time: " << computation_time << std::endl;


			if(!cudaWrapper.transformPointCloud(second_point_cloud, mICP))
			{
				std::cout << "cudaWrapper.transformPointCloud NOT SUCCESFULL" << std::endl;
			}

			break;
		}
    }
    glutPostRedisplay();
    printHelp();
}

void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN)
    {
        mouse_buttons |= 1<<button;
    }
    else if (state == GLUT_UP)
    {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}

void motion(int x, int y)
{
    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1)
    {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;

    }
    else if (mouse_buttons & 4)
    {
        translate_z += dy * 0.05f;
    }
    else if (mouse_buttons & 3)
    {
            translate_x += dx * 0.05f;
            translate_y -= dy * 0.05f;
    }

    mouse_old_x = x;
    mouse_old_y = y;

    glutPostRedisplay();
}

void printHelp()
{
	std::cout << "----------------------" << std::endl;
	std::cout << "press 'r': data registration with semantic ICP - 1 iteration" << std::endl;
	std::cout << "press 'n': compute nearest neighbours" << std::endl;
	std::cout << "press 'a': rotate second_point_cloud via X axis (+1 degree)" << std::endl;
	std::cout << "press 'd': rotate second_point_cloud via X axis (-1 degree)" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}
