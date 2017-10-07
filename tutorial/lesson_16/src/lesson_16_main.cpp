#include <string>
#include <GL/freeglut.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/parse.h>
#include <pcl/registration/icp.h>
#include <pcl/common/time.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/PCLPointCloud2.h>

#include "data_model.hpp"
#include "point_types.h"
#include "cudaWrapper.h"

CCudaWrapper cudaWrapper;

#define RENDER_TYPE_ONLY_3D_POINTS 0
#define RENDER_TYPE_3D_POINTS_AND_NN 1

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);


const unsigned int window_width  = 1024;
const unsigned int window_height = 1024;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -100.0;
float translate_x, translate_y = 0.0;

int render_type = RENDER_TYPE_ONLY_3D_POINTS;
int pointSize = 1;
int solver_method = 0;

std::vector<Eigen::Affine3f> vtransform;
std::vector<pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> > vpointcloud;
std::vector<pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> > r_vpointcloud;

pcl::PointCloud<pcl::PointXYZ> pc_nn_1;
pcl::PointCloud<pcl::PointXYZ> pc_nn_2;
std::vector<char> nn_semantic_label;

float search_radius = 2.0f;
int max_number_considered_in_INNER_bucket = 100;
int max_number_considered_in_OUTER_bucket = 100;
float bounding_box_extension = 1.0f;
unsigned int index_begin = 0;
unsigned int index_end = 0;
unsigned int index_step = 1;

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

void printHelp()
{
	std::cout << "----------------------" << std::endl;
	std::cout << "1: render only 3D points with semantic labels as colours" << std::endl;
	std::cout << "2: render 3D points and semantic nearest neighbours" << std::endl;
	std::cout << "+: increase point size" << std::endl;
	std::cout << "-: decrease point size" << std::endl;
	std::cout << "r: register" << std::endl;
}

void transformPointCloud(pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &pointcloud, Eigen::Affine3f transform);
void transformAllpointcloudsForRender();
void registerAllScans();

int
main(int argc, char **argv)
{
	std::string model_file;
	if(argc<2)
	{
		std::cout << "Usage:\n";
		std::cout << argv[0] <<" inputModel.xml parameters\n";
		index_begin = 0;
		index_end = 10;
		index_step = 2;
		std::cout << "-ib index_begin: default " << index_begin << std::endl;
		std::cout << "-ie index_end: default " << index_end << std::endl;
		std::cout << "-is index_step: default " << index_step << std::endl;
		model_file = "../../../data_sets/IMM_pointXYZIRNL/model_processed_pointXYZIRNL.xml";
	}else
	{
		model_file = argv[1];
	}

	pcl::console::parse_argument (argc, argv, "-ib", index_begin);
	pcl::console::parse_argument (argc, argv, "-ie", index_end);
	pcl::console::parse_argument (argc, argv, "-is", index_step);

	data_model dSets;
	std::vector<std::string> indices;

	dSets.loadFile(model_file);
	dSets.getAllScansId(indices);

	if(index_begin == 0 && index_end == 0)
	{
		index_begin = 0;
		index_end = (int)indices.size();
		std::cout << "default parameters: index_begin " << index_begin << " index_end " << index_end << " index_step " << index_step << std::endl;
	}else
	{
		if(index_begin < index_end && index_end <= (int)indices.size())
		{
			std::cout << "user defined parameters: index_begin " << index_begin << " index_end " << index_end << " index_step " << index_step << std::endl;
		}else
		{
			std::cout << "ERROR: check params <index_begin, index_end, index_step>" << std::endl;
			std::cout << "conditions: index_step >= 0 and index_begin < index_end and index_end <= " << (int)indices.size() << std::endl;
			std::cout << "index_begin: " << index_begin << std::endl;
			std::cout << "index_end: " << index_end << std::endl;
			std::cout << "index_step: " << index_step << std::endl;
			std::cout << "exit(-1)" << std::endl;
			exit(-1);
		}
	}

	for (int i=0; i< indices.size(); i++)
	{
		std::string fn;
		dSets.getPointcloudName(indices[i], fn);
		std::cout << indices[i]<<"\t"<<fn<<"\n";
	}

	for (int i=index_begin; i< index_end; i+=index_step)
	{
		std::string fn;
		Eigen::Affine3f transform;
		fn = dSets.getFullPathOfPointcloud(indices[i]);
		bool isOkTr = dSets.getAffine(indices[i], transform.matrix());
		vtransform.push_back(transform);

		if (isOkTr)
		{
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> pointcloud;
			if(pcl::io::loadPCDFile(fn, pointcloud) == -1)
			{
				std::cout << "problem with pcl::io::loadPCDFile: " << fn << std::endl;
				exit(-1);
			}
			vpointcloud.push_back(pointcloud);
		}
	}

	transformAllpointcloudsForRender();

	cudaWrapper.warmUpGPU();

	if (false == initGL(&argc, argv))
	{
		return -1;
	}

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
    glutCreateWindow("Lesson 16 - multi scan registration (semantic point to point)");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.01, 10000.0);

    return true;
}

void display()
{
	glPointSize(pointSize);

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

    switch (render_type)
	{
    	case RENDER_TYPE_ONLY_3D_POINTS:
		{

			glBegin(GL_POINTS);
			for(size_t i = 0; i < r_vpointcloud.size(); i++)
			{
				for(size_t j = 0; j < r_vpointcloud[i].points.size(); j++)
				{
					if(r_vpointcloud[i].points[j].label<16 && r_vpointcloud[i].points[j].label>=0)
					{
						glColor3f(colors[r_vpointcloud[i].points[j].label][0], colors[r_vpointcloud[i].points[j].label][1], colors[r_vpointcloud[i].points[j].label][2]);
					}else
					{
						glColor3f(0.7f, 0.7f, 0.7f);
					}

					glVertex3f(r_vpointcloud[i].points[j].x, r_vpointcloud[i].points[j].y, r_vpointcloud[i].points[j].z);
				}
			}
			glEnd();

		break;
		}

    	case RENDER_TYPE_3D_POINTS_AND_NN:
    	{
    		glBegin(GL_POINTS);
    		glColor3f(0.3f, 0.3f, 0.3f);
			for(size_t i = 0; i < r_vpointcloud.size(); i++)
			{
				for(size_t j = 0; j < r_vpointcloud[i].points.size(); j++)
				{
					glVertex3f(r_vpointcloud[i].points[j].x, r_vpointcloud[i].points[j].y, r_vpointcloud[i].points[j].z);
				}
			}
			glEnd();

			glBegin(GL_LINES);
			for(size_t i = 0; i < pc_nn_1.size(); i++)
			{
				if(nn_semantic_label[i]<16 && nn_semantic_label[i]>=0)
				{
					glColor3f(colors[nn_semantic_label[i]][0], colors[nn_semantic_label[i]][1], colors[nn_semantic_label[i]][2]);
				}else
				{
					glColor3f(0.7f, 0.7f, 0.7f);
				}

				glVertex3f(pc_nn_1[i].x, pc_nn_1[i].y, pc_nn_1[i].z);
				glVertex3f(pc_nn_2[i].x, pc_nn_2[i].y, pc_nn_2[i].z);
			}
			glEnd();
    		break;
    	}
    }
    printHelp();
    glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            glutDestroyWindow(glutGetWindow());
            return;
        case '1' :
        {
        	render_type = RENDER_TYPE_ONLY_3D_POINTS;
        	break;
        }
        case '2':
        {
        	render_type = RENDER_TYPE_3D_POINTS_AND_NN;
        	break;
        }
        case 'r' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

				registerAllScans();

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "registerAllScans computation_time: " << computation_time << std::endl;
            break;
        }
        case '+':
        {
        	pointSize++;
        	break;
        }
        case '-':
        {
        	pointSize--;
        	if(pointSize < 1)pointSize = 1;
        	break;
        }
    }
    glutPostRedisplay();
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

void transformPointCloud(pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &pointcloud, Eigen::Affine3f transform)
{
	for(size_t i = 0; i < pointcloud.size(); i++)
	{
		Eigen::Vector3f p(pointcloud[i].x, pointcloud[i].y, pointcloud[i].z);
		Eigen::Vector3f pt;

		pt = transform * p;

		pointcloud[i].x = pt.x();
		pointcloud[i].y = pt.y();
		pointcloud[i].z = pt.z();

		Eigen::Affine3f tr = transform;

		tr(0,3) = 0.0f;
		tr(1,3) = 0.0f;
		tr(2,3) = 0.0f;

		Eigen::Vector3f n(pointcloud[i].normal_x, pointcloud[i].normal_y, pointcloud[i].normal_z);
		Eigen::Vector3f nt;

		nt = tr * n;
		pointcloud[i].normal_x = nt.x();
		pointcloud[i].normal_y = nt.y();
		pointcloud[i].normal_z = nt.z();
	}
	return;
}

void transformAllpointcloudsForRender()
{
	r_vpointcloud.clear();
	for(size_t i = 0 ; i < vpointcloud.size(); i++)
	{
		pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> pc = vpointcloud[i];
		transformPointCloud(pc, vtransform[i]);
		r_vpointcloud.push_back(pc);
	}
}

void registerAllScans()
{
	pc_nn_1.clear();
	pc_nn_2.clear();
	nn_semantic_label.clear();

	std::vector<Eigen::Affine3f> v_poses;

	for(size_t i = 0; i < vpointcloud.size(); i++)
	{
		Eigen::Vector3f omfika1;
		Eigen::Vector3f xyz1;
		Eigen::Affine3f pose1;

		cudaWrapper.Matrix4ToEuler(vtransform[i], omfika1, xyz1);
		cudaWrapper.EulerToMatrix(omfika1, xyz1, pose1);

		pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> _point_cloud_1 = vpointcloud[i];
		transformPointCloud(_point_cloud_1, pose1);

		observations_t obs;
		obs.om = omfika1.x();
		obs.fi = omfika1.y();
		obs.ka = omfika1.z();
		obs.tx = xyz1.x();
		obs.ty = xyz1.y();
		obs.tz = xyz1.z();

		for(size_t j = 0; j < vpointcloud.size(); j++)
		{
			if(i != j)
			{
				Eigen::Vector3f omfika2;
				Eigen::Vector3f xyz2;
				Eigen::Affine3f pose2;
				cudaWrapper.Matrix4ToEuler(vtransform[j], omfika2, xyz2);
				cudaWrapper.EulerToMatrix(omfika2, xyz2, pose2);

				pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> _point_cloud_2 = vpointcloud[j];
				transformPointCloud(_point_cloud_2, pose2);

				std::vector<int> nearest_neighbour_indexes;
				nearest_neighbour_indexes.resize(_point_cloud_2.size());
				std::fill(nearest_neighbour_indexes.begin(), nearest_neighbour_indexes.end(), -1);

				if(!cudaWrapper.semanticNearestNeighbourhoodSearch(
						_point_cloud_1,
						_point_cloud_2,
						search_radius,
						bounding_box_extension,
						max_number_considered_in_INNER_bucket,
						max_number_considered_in_OUTER_bucket,
						nearest_neighbour_indexes))
				{
					cudaDeviceReset();
					std::cout << "cudaWrapper.semanticNearestNeighbourhoodSearch NOT SUCCESFULL" << std::endl;
				}


				for(size_t ii = 0 ; ii < nearest_neighbour_indexes.size(); ii++)
				{
					if(nearest_neighbour_indexes[ii] != -1)
					{
						pcl::PointXYZ p1(_point_cloud_1[nearest_neighbour_indexes[ii]].x, _point_cloud_1[nearest_neighbour_indexes[ii]].y, _point_cloud_1[nearest_neighbour_indexes[ii]].z);
						pcl::PointXYZ p2(_point_cloud_2[ii].x, _point_cloud_2[ii].y, _point_cloud_2[ii].z);

						pc_nn_1.push_back(p1);
						pc_nn_2.push_back(p2);
						nn_semantic_label.push_back(_point_cloud_2[ii].label);

						obs_nn_t obs_nn;
						obs_nn.x0 = vpointcloud[i].points[nearest_neighbour_indexes[ii]].x;
						obs_nn.y0 = vpointcloud[i].points[nearest_neighbour_indexes[ii]].y;
						obs_nn.z0 = vpointcloud[i].points[nearest_neighbour_indexes[ii]].z;
						obs_nn.x_diff = p1.x - p2.x;
						obs_nn.y_diff = p1.y - p2.y;
						obs_nn.z_diff = p1.z - p2.z;
						obs.vobs_nn.push_back(obs_nn);
					}
				}
			}//if(i != j)
		}//for(size_t j = 0; j < vpointcloud.size(); j++)

		if(!cudaWrapper.registerLS(obs))
		{
			std::cout << "PROBLEM: cudaWrapper.registerLS(obs2to1)" << std::endl;
			return;
		}

		Eigen::Vector3f omfika1_res(obs.om, obs.fi, obs.ka);
		Eigen::Vector3f xyz1_res(obs.tx, obs.ty, obs.tz);
		Eigen::Affine3f pose1_res;
		cudaWrapper.EulerToMatrix(omfika1_res, xyz1_res, pose1_res);

		std::cout << pose1_res.matrix() << std::endl;

		v_poses.push_back(pose1_res);
	}

	vtransform = v_poses;
	transformAllpointcloudsForRender();
}


