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
#define RENDER_TYPE_NORMALS 1
#define RENDER_TYPE_3D_POINTS_AND_PROJECTIONS 2
#define RENDER_TYPE_3D_POINTS_AND_PROJECTIONS_V2 3

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
pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> point_cloud_1;
pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> point_cloud_2;
Eigen::Affine3f m_pose1;
Eigen::Affine3f m_pose2;
pcl::PointCloud<lidar_pointcloud::PointProjection> projections;

float projections_search_radius = 0.5f;
int max_number_considered_in_INNER_bucket = 10000;
int max_number_considered_in_OUTER_bucket = 10000;
float bounding_box_extension = 1.0f;

void printHelp()
{
	std::cout << "----------------------" << std::endl;
	std::cout << "1: RENDER_TYPE_ONLY_3D_POINTS" << std::endl;
	std::cout << "2: RENDER_TYPE_NORMALS" << std::endl;
	std::cout << "3: RENDER_TYPE_3D_POINTS_AND_PROJECTIONS" << std::endl;
	std::cout << "4: RENDER_TYPE_3D_POINTS_AND_PROJECTIONS_V2" << std::endl;
	std::cout << "+: increase point size" << std::endl;
	std::cout << "-: decrease point size" << std::endl;
	std::cout << "r: register with LS3D obs1 to obs2;" << std::endl;
	std::cout << "t: register with LS3D obs2 to obs1;" << std::endl;
	std::cout << "o: compute projections obs2 to obs1" << std::endl;
	std::cout << "p: compute projections obs1 to obs2" << std::endl;
}

void transformPointCloud(pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &pointcloud, Eigen::Affine3f transform);
bool loadData(std::string filename, std::string scan_name_1, std::string scan_name_2,
		pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &_point_cloud_1, pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &_point_cloud_2);

void register_obs1_to_obs2();
void register_obs2_to_obs1();

int
main(int argc, char **argv)
{
	if(argc<4)
	{
		std::cout << "Usage: " << std::endl;
		std::cout << argv[0] <<" inputModel.xml scan_name_1 scan_name_2 parameters" << std::endl;
		std::cout << "-sm solver_method 0-chol, 1-lu, 2-qr: default " << solver_method << std::endl;

		if(!loadData("../../../data_sets/IMM_pointXYZIRNL/model_processed_pointXYZIRNL.xml", "scan000", "scan001", point_cloud_1, point_cloud_2))
		{
			exit(-1);
		}

	}else
	{
		std::vector<int> xml_indices;
		xml_indices = pcl::console::parse_file_extension_argument (argc, argv, ".xml");

		if(xml_indices.size()!=1)
		{
			std::cout << "inputXML wrong extension, check if it is xxx.xml!" << std::endl;
			exit(-1);
		}

		if(!loadData(argv[1], argv[2], argv[3], point_cloud_1, point_cloud_2))
		{
			exit(-1);
		}
	}

	pcl::console::parse_argument (argc, argv, "-sm", solver_method);
	std::cout << "solver_method: " << solver_method << std::endl;

	projections.resize(point_cloud_2.size());
	for(size_t i = 0 ;i < projections.size(); i++)projections[i].isProjection = 0;

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
    glutCreateWindow("Lesson 12 - data registration Least Squre Surface Matching");
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
	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> r_point_cloud_1 = point_cloud_1;
	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> r_point_cloud_2 = point_cloud_2;
	transformPointCloud(r_point_cloud_1, m_pose1);
	transformPointCloud(r_point_cloud_2, m_pose2);

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
			glColor3f(1.0f, 0.0f, 0.0f);
			glBegin(GL_POINTS);
			for(size_t j = 0; j < r_point_cloud_1.size(); j++)
			{
					glVertex3f(r_point_cloud_1[j].x, r_point_cloud_1[j].y, r_point_cloud_1[j].z);
			}
			glEnd();

			glColor3f(0.0f, 0.0f, 1.0f);
			glBegin(GL_POINTS);
			for(size_t j = 0; j < r_point_cloud_2.size(); j++)
			{
					glVertex3f(r_point_cloud_2[j].x, r_point_cloud_2[j].y, r_point_cloud_2[j].z);
			}
			glEnd();
		break;
		}
    	case RENDER_TYPE_NORMALS:
		{
			glBegin(GL_LINES);
			for(int j = 0; j <  r_point_cloud_1.size(); j++)
			{
				glColor3f(fabs(r_point_cloud_1[j].normal_x), fabs(r_point_cloud_1[j].normal_y), fabs(r_point_cloud_1[j].normal_z)  );
				glVertex3f(r_point_cloud_1[j].x, r_point_cloud_1[j].y, r_point_cloud_1[j].z);
				glVertex3f(r_point_cloud_1[j].x + r_point_cloud_1[j].normal_x, r_point_cloud_1[j].y + r_point_cloud_1[j].normal_y, r_point_cloud_1[j].z + r_point_cloud_1[j].normal_z);
			}
			glEnd();

			glBegin(GL_LINES);
			for(int j = 0; j <  r_point_cloud_2.size(); j++)
			{
				glColor3f(fabs(r_point_cloud_2[j].normal_x), fabs(r_point_cloud_2[j].normal_y), fabs(r_point_cloud_2[j].normal_z)  );
				glVertex3f(r_point_cloud_2[j].x, r_point_cloud_2[j].y, r_point_cloud_2[j].z);
				glVertex3f(r_point_cloud_2[j].x + r_point_cloud_2[j].normal_x, r_point_cloud_2[j].y + r_point_cloud_2[j].normal_y, r_point_cloud_2[j].z + r_point_cloud_2[j].normal_z);
			}
			glEnd();
		break;
		}
    	case RENDER_TYPE_3D_POINTS_AND_PROJECTIONS:
		{
			glPointSize(3);
			glColor3f(0.0f, 0.0f, 1.0f);
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < projections.size(); i++)
			{
				glVertex3f(projections[i].x_src, projections[i].y_src, projections[i].z_src);
			}
			glEnd();

			glColor3f(1.0f, 0.0f, 0.0f);
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < projections.size(); i++)
			{
				if(projections[i].isProjection == 1)
					glVertex3f(projections[i].x_dst, projections[i].y_dst, projections[i].z_dst);
			}
			glEnd();

			glColor3f(0.0f, 1.0f, 0.0f);
			glBegin(GL_LINES);
			for(size_t i = 0 ; i < projections.size(); i++)
			{
				if(projections[i].isProjection == 1)
				{
					glVertex3f(projections[i].x_src, projections[i].y_src, projections[i].z_src);
					glVertex3f(projections[i].x_dst, projections[i].y_dst, projections[i].z_dst);
				}
			}
			glEnd();
			glPointSize(1);
		break;
		}
    	case RENDER_TYPE_3D_POINTS_AND_PROJECTIONS_V2:
    	{
    		glPointSize(3);
			glColor3f(0.0f, 0.0f, 1.0f);
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < projections.size(); i++)
			{
				glVertex3f(projections[i].x_src, projections[i].y_src, projections[i].z_src);
			}
			glEnd();

			glColor3f(1.0f, 0.0f, 0.0f);
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < projections.size(); i++)
			{
				if(projections[i].isProjection == 1)
					glVertex3f(projections[i].x_src - projections[i].normal_x * projections[i].distance,
							   projections[i].y_src - projections[i].normal_y * projections[i].distance,
							   projections[i].z_src - projections[i].normal_z * projections[i].distance);

			}
			glEnd();

			glColor3f(0.0f, 1.0f, 0.0f);
			glBegin(GL_LINES);
			for(size_t i = 0 ; i < projections.size(); i++)
			{
				if(projections[i].isProjection == 1)
				{
					glVertex3f(projections[i].x_src, projections[i].y_src, projections[i].z_src);
					glVertex3f(projections[i].x_src - projections[i].normal_x * projections[i].distance,
							   projections[i].y_src - projections[i].normal_y * projections[i].distance,
							   projections[i].z_src - projections[i].normal_z * projections[i].distance);
				}
			}
			glEnd();


			glPointSize(1);
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
        	render_type = RENDER_TYPE_NORMALS;
        	break;
        }
        case '3':
        {
        	render_type = RENDER_TYPE_3D_POINTS_AND_PROJECTIONS;
        	break;
        }
        case '4':
        {
        	render_type = RENDER_TYPE_3D_POINTS_AND_PROJECTIONS_V2;
        	break;
        }
        case 'r' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

        		register_obs1_to_obs2();

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.registerLS3D computation_time: " << computation_time << std::endl;

            break;
        }
        case 'o' :
        {
        	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> r_point_cloud_1 = point_cloud_1;
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> r_point_cloud_2 = point_cloud_2;
			transformPointCloud(r_point_cloud_1, m_pose1);
			transformPointCloud(r_point_cloud_2, m_pose2);

			projections.resize(point_cloud_2.size());
			for(size_t i = 0 ;i < projections.size(); i++)projections[i].isProjection = 0;

        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

			if(!cudaWrapper.compute_projections(
						r_point_cloud_1,
						r_point_cloud_2,
						projections_search_radius,
						bounding_box_extension,
						max_number_considered_in_INNER_bucket,
						max_number_considered_in_OUTER_bucket,
						projections))
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.projections NOT SUCCESFULL" << std::endl;
			}

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.projections computation_time: " << computation_time << std::endl;

			render_type = RENDER_TYPE_3D_POINTS_AND_PROJECTIONS;
          	break;
        }
        case 'p' :
		{
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> r_point_cloud_1 = point_cloud_1;
			pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> r_point_cloud_2 = point_cloud_2;
			transformPointCloud(r_point_cloud_1, m_pose1);
			transformPointCloud(r_point_cloud_2, m_pose2);

			projections.resize(point_cloud_1.size());
			for(size_t i = 0 ;i < projections.size(); i++)projections[i].isProjection = 0;

			clock_t begin_time;
			double computation_time;
			begin_time = clock();

			if(!cudaWrapper.compute_projections(
						r_point_cloud_2,
						r_point_cloud_1,
						projections_search_radius,
						bounding_box_extension,
						max_number_considered_in_INNER_bucket,
						max_number_considered_in_OUTER_bucket,
						projections))
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.projections NOT SUCCESFULL" << std::endl;
			}

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.projections computation_time: " << computation_time << std::endl;

			render_type = RENDER_TYPE_3D_POINTS_AND_PROJECTIONS;
			break;
		}
        case 't':
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

				register_obs2_to_obs1();

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.registerLS3D computation_time: " << computation_time << std::endl;
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

bool loadData(std::string filename, std::string scan_name_1, std::string scan_name_2,
		pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &_point_cloud_1, pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> &_point_cloud_2)
{
	std::string inputXMLFn = filename;
	data_model inputXML;
	std::vector<std::string> indices;

	if(!inputXML.loadFile(inputXMLFn))
	{
		std::cout << "ERROR: failure of loading file: " <<  inputXMLFn << std::endl;
		return false;
	}

	inputXML.getAllScansId(indices);

	if(!inputXML.getAffine(scan_name_1, m_pose1.matrix()))
	{
		std::cout << "Available scans:" << std::endl;
		for(int i = 0; i < indices.size(); i++)
			std::cout << indices[i] << " ";
		std::cout << std::endl;
		std::cout << "ERROR: failure of loading m_position for scan: " <<  scan_name_1 << std::endl;
		return false;
	}

	if(!inputXML.getAffine(scan_name_2, m_pose2.matrix()))
	{
		std::cout << "Available scans:" << std::endl;
		for(int i = 0; i < indices.size(); i++)
			std::cout << indices[i] << " ";
		std::cout << std::endl;
		std::cout << "ERROR: failure of loading m_position for scan: " <<  scan_name_2 << std::endl;
		return false;
	}

	std::string fn;
	fn = inputXML.getFullPathOfPointcloud(scan_name_1);
	std::string pcdFileNameOnly;
	inputXML.getPointcloudName(scan_name_1, pcdFileNameOnly);

	if(pcl::io::loadPCDFile(fn, _point_cloud_1) == -1)
			return false;

	fn = inputXML.getFullPathOfPointcloud(scan_name_2);
	inputXML.getPointcloudName(scan_name_2, pcdFileNameOnly);

	if(pcl::io::loadPCDFile(fn, _point_cloud_2) == -1)
				return false;

	return true;
}

void register_obs1_to_obs2()
{
	Eigen::Vector3f omfika1, omfika2;
	Eigen::Vector3f xyz1, xyz2;
	Eigen::Affine3f pose1;
	Eigen::Affine3f pose2;
	cudaWrapper.Matrix4ToEuler(m_pose1, omfika1, xyz1);
	cudaWrapper.Matrix4ToEuler(m_pose2, omfika2, xyz2);
	cudaWrapper.EulerToMatrix(omfika1, xyz1, pose1);
	cudaWrapper.EulerToMatrix(omfika2, xyz2, pose2);

	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> _point_cloud_1 = point_cloud_1;
	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> _point_cloud_2 = point_cloud_2;

	observations_LS3D_t obs;
	obs.om = omfika1.x();
	obs.fi = omfika1.y();
	obs.ka = omfika1.z();
	obs.tx = xyz1.x();
	obs.ty = xyz1.y();
	obs.tz = xyz1.z();

	obs.m_pose = pose1;
	transformPointCloud(_point_cloud_1, pose1);
	transformPointCloud(_point_cloud_2, pose2);
	obs.projections.resize(_point_cloud_1.size());

	if(!cudaWrapper.compute_projections(
			_point_cloud_2,
			_point_cloud_1,
			projections_search_radius,
			bounding_box_extension,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			obs.projections))
	{
		std::cout << "PROBLEM: cudaWrapper.compute_projection" << std::endl;
		return;
	}

	int number_of_projections = 0;
	for(size_t i = 0 ; i < obs.projections.size(); i++)
	{
		obs.projections[i].x0 = point_cloud_1[i].x;
		obs.projections[i].y0 = point_cloud_1[i].y;
		obs.projections[i].z0 = point_cloud_1[i].z;

		if(obs.projections[i].isProjection == 1)
		{
			number_of_projections++;
		}
	}

	std::cout << "number of projections: " << number_of_projections << std::endl;

	if(number_of_projections < 10)
	{
		std::cout << "number_of_projections < 10  return" << std::endl;
		return;
	}

	if(!cudaWrapper.registerLS3D(obs))
	{
		std::cout << "PROBLEM: cudaWrapper.registerLS3D(obs1to2)" << std::endl;
		return;
	}

	Eigen::Vector3f omfika1_res(obs.om, obs.fi, obs.ka);
	Eigen::Vector3f xyz1_res(obs.tx, obs.ty, obs.tz);
	cudaWrapper.EulerToMatrix(omfika1_res, xyz1_res, m_pose1);
}

void register_obs2_to_obs1()
{
	Eigen::Vector3f omfika1, omfika2;
	Eigen::Vector3f xyz1, xyz2;
	Eigen::Affine3f pose1;
	Eigen::Affine3f pose2;
	cudaWrapper.Matrix4ToEuler(m_pose1, omfika1, xyz1);
	cudaWrapper.Matrix4ToEuler(m_pose2, omfika2, xyz2);
	cudaWrapper.EulerToMatrix(omfika1, xyz1, pose1);
	cudaWrapper.EulerToMatrix(omfika2, xyz2, pose2);

	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> _point_cloud_1 = point_cloud_1;
	pcl::PointCloud<lidar_pointcloud::PointXYZIRNL> _point_cloud_2 = point_cloud_2;

	observations_LS3D_t obs;
	obs.om = omfika2.x();
	obs.fi = omfika2.y();
	obs.ka = omfika2.z();
	obs.tx = xyz2.x();
	obs.ty = xyz2.y();
	obs.tz = xyz2.z();

	obs.m_pose = pose1;
	transformPointCloud(_point_cloud_1, pose1);
	transformPointCloud(_point_cloud_2, pose2);
	obs.projections.resize(_point_cloud_2.size());

	if(!cudaWrapper.compute_projections(
			_point_cloud_1,
			_point_cloud_2,
			projections_search_radius,
			bounding_box_extension,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			obs.projections))
	{
		std::cout << "PROBLEM: cudaWrapper.compute_projection" << std::endl;
		return;
	}

	int number_of_projections = 0;
	for(size_t i = 0 ; i < obs.projections.size(); i++)
	{
		obs.projections[i].x0 = point_cloud_2[i].x;
		obs.projections[i].y0 = point_cloud_2[i].y;
		obs.projections[i].z0 = point_cloud_2[i].z;

		if(obs.projections[i].isProjection == 1)
		{
			number_of_projections++;
		}
	}

	std::cout << "number of projections: " << number_of_projections << std::endl;

	if(number_of_projections < 10)
	{
		std::cout << "number_of_projections < 10  return" << std::endl;
		return;
	}

	if(!cudaWrapper.registerLS3D(obs))
	{
		std::cout << "PROBLEM: cudaWrapper.registerLS3D(obs2to1)" << std::endl;
		return;
	}

	Eigen::Vector3f omfika2_res(obs.om, obs.fi, obs.ka);
	Eigen::Vector3f xyz2_res(obs.tx, obs.ty, obs.tz);
	cudaWrapper.EulerToMatrix(omfika2_res, xyz2_res, m_pose2);
}
