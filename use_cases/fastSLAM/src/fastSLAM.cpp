
#include <GL/freeglut.h>
//#include <string>
//#include <vector>
//#include <sstream>
//#include <fstream>
//#include <iostream>

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

#include "data_model.hpp"
#include "cudaStructures.h"
#include "rgd_particle_filter_CUDAWrapper.h"

#define RENDER_TYPE_ONLY_3D_POINTS 0
#define RENDER_TYPE_SEMANTIC_LABELS 1
#define RENDER_TYPE_RINGS 2
#define RENDER_TYPE_INTENSITY 3
#define RENDER_TYPE_NORMALS 4
#define RENDER_TYPE_PARTICLE_ESTIMATION 5

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
void motion(int x, int y);

const unsigned int window_width  = 1024;
const unsigned int window_height = 1024;

int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -200.0;
float translate_x, translate_y = 0.0;

int render_type = RENDER_TYPE_ONLY_3D_POINTS;
int pointSize = 1;
unsigned int index_begin = 0;
unsigned int index_end = 0;
unsigned int index_step = 1;

float rez = 1.0;
float number_of_particles = 100;
float motion_model_max_angle = 0.2f;
float motion_model_max_translation_X = 0.1f;
float motion_model_max_translation_Y = 0.0f;
float nW_threshold = 0.1;

bool ispathrendered = true;
bool isallpathsrendered = true;

void printHelp()
{
	std::cout << "----------------------" << std::endl;
	std::cout << "1: RENDER_TYPE_ONLY_3D_POINTS" << std::endl;
	std::cout << "2: RENDER_TYPE_SEMANTIC_LABELS" << std::endl;
	std::cout << "3: RENDER_TYPE_RINGS" << std::endl;
	std::cout << "4: RENDER_TYPE_INTENSITY" << std::endl;
	std::cout << "5: RENDER_TYPE_NORMALS" << std::endl;
	std::cout << "6: RENDER_TYPE_PARTICLE_ESTIMATION" << std::endl;
	std::cout << "7: ispathrendered =! ispathrendered" << std::endl;
	std::cout << "8: isallpathsrendered =! isallpathsrendered" << std::endl;
	std::cout << "+: increase point size" << std::endl;
	std::cout << "-: decrease point size" << std::endl;
	std::cout << "c: particle filter data registration CUDA" << std::endl;
}

std::vector<pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> > vpc;
std::vector<pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> > vpc_raw;
std::vector<Eigen::Affine3f> vtransforms;

CRGD_Particle_Filter_CUDAWrapper rgd_particle_filter_CUDAWrapper;

bool loadData(std::string filename, pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> &_pc);
void transformPointCloud(pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> &pointcloud, Eigen::Affine3f transform);
void particle_filter_calculations_CUDA();

int
main(int argc, char **argv)
{
	if(argc<2)
	{
		std::cout << "Usage:\n";
		std::cout << argv[0] <<" inputModel.xml parameters\n";
		std::cout << "-ib index_begin: default " << index_begin << std::endl;
		std::cout << "-ie index_end: default " << index_end << std::endl;
		std::cout << "-is index_step: default " << index_step << std::endl;
		std::cout << "-rez rezolution of RGD: default " << rez << std::endl;
		std::cout << "-nop number_of_particles: default " << number_of_particles << std::endl;
		std::cout << "-mma motion_model_max_angle: default " << motion_model_max_angle << std::endl;
		std::cout << "-mmX motion_model_max_translation_X: default " << motion_model_max_translation_X << std::endl;
		std::cout << "-mmY motion_model_max_translation_Y: default " << motion_model_max_translation_Y << std::endl;
		std::cout << "-nwt nW_threshold: default " << nW_threshold << std::endl;
		return -1;
	}

	pcl::console::parse_argument (argc, argv, "-ib", index_begin);
	pcl::console::parse_argument (argc, argv, "-ie", index_end);
	pcl::console::parse_argument (argc, argv, "-is", index_step);
	pcl::console::parse_argument (argc, argv, "-rez", rez);
	pcl::console::parse_argument (argc, argv, "-nop", number_of_particles);
	pcl::console::parse_argument (argc, argv, "-mma", motion_model_max_angle);
	pcl::console::parse_argument (argc, argv, "-mmX", motion_model_max_translation_X);
	pcl::console::parse_argument (argc, argv, "-mmY", motion_model_max_translation_Y);
	pcl::console::parse_argument (argc, argv, "-nwt", nW_threshold);

	rgd_particle_filter_CUDAWrapper.rez = rez;
	rgd_particle_filter_CUDAWrapper.number_of_particles = number_of_particles;
	rgd_particle_filter_CUDAWrapper.motion_model_max_angle = motion_model_max_angle;
	rgd_particle_filter_CUDAWrapper.motion_model_max_translation_X = motion_model_max_translation_X;
	rgd_particle_filter_CUDAWrapper.motion_model_max_translation_Y = motion_model_max_translation_Y;
	rgd_particle_filter_CUDAWrapper.nW_threshold = nW_threshold;

	std::vector<int> ind_pcd;
	ind_pcd = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

	data_model dSets;
	std::vector<std::string> indices;
	std::string model_file = argv[1];

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
		if (isOkTr)
		{
			pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> pointcloud;
			if(!loadData(fn, pointcloud))
			{
				std::cout << "Problem with opening: " << fn << " file ... exit(-1)" << std::endl;
				exit(-1);
			}else
			{
				vpc_raw.push_back(pointcloud);
				vtransforms.push_back(transform);

				transformPointCloud(pointcloud, transform);
				vpc.push_back(pointcloud);
				std::cout << "file " << fn << " loaded" << std::endl;
			}
		}
	}

	if (false == initGL(&argc, argv))
	{
		return -1;
	}

	printHelp();

	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutMainLoop();

}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("fastSLAM");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    glViewport(0, 0, window_width, window_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.01, 10000.0);

    return true;
}

void display()
{
	 switch (render_type)
	 {
	 	case RENDER_TYPE_ONLY_3D_POINTS:
		{
			glClearColor(0.0, 0.0, 0.0, 1.0);
			break;
		}
		case RENDER_TYPE_SEMANTIC_LABELS:
		{
			glClearColor(0.0, 0.0, 0.0, 1.0);
			break;
		}
		case RENDER_TYPE_RINGS:
		{
			glClearColor(0.0, 0.0, 0.0, 1.0);
			break;
		}
		case RENDER_TYPE_INTENSITY:
		{
			glClearColor(0.2, 0.2, 0.7, 1.0);
			break;
		}
		case RENDER_TYPE_NORMALS:
		{
			glClearColor(0.0, 0.0, 0.0, 1.0);
			break;
		}
	 }

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
			glColor3f(1.0f, 1.0f, 1.0f);
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < vpc.size(); i++)
			{
				for(size_t j = 0; j < vpc[i].size(); j++)
				{
					glVertex3f(vpc[i][j].x, vpc[i][j].y, vpc[i][j].z);
				}
			}
			glEnd();
			break;
		}
		case RENDER_TYPE_SEMANTIC_LABELS:
		{
			glColor3f(1.0f, 1.0f, 1.0f);
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < vpc.size(); i++)
			{
				for(size_t j = 0; j < vpc[i].size(); j++)
				{
					if(vpc[i][j].label<16 && vpc[i][j].label>=0)
					{
						glColor3f(colors[vpc[i][j].label][0], colors[vpc[i][j].label][1], colors[vpc[i][j].label][2]);
					}else
					{

						glColor3f(0.7f, 0.7f, 0.7f);
					}
					glVertex3f(vpc[i][j].x, vpc[i][j].y, vpc[i][j].z);
				}

			}
			glEnd();

			break;
		}
		case RENDER_TYPE_RINGS:
		{
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < vpc.size(); i++)
			{
				for(size_t j = 0; j < vpc[i].size(); j++)
				{
					if(vpc[i][j].ring<16 && vpc[i][j].ring>=0)
					{
						glColor3f(colors[vpc[i][j].ring][0], colors[vpc[i][j].ring][1], colors[vpc[i][j].ring][2]);
					}else
					{

						glColor3f(0.7f, 0.7f, 0.7f);
					}
					glVertex3f(vpc[i][j].x, vpc[i][j].y, vpc[i][j].z);
				}
			}
			glEnd();
			break;
		}
		case RENDER_TYPE_INTENSITY:
		{
			glBegin(GL_POINTS);
			for(size_t i = 0 ; i < vpc.size(); i++)
			{
				for(int j = 0; j < vpc[i].size(); j++)
				{
					glColor3ub(vpc[i][j].intensity, vpc[i][j].intensity, vpc[i][j].intensity);
					glVertex3f(vpc[i][j].x, vpc[i][j].y, vpc[i][j].z);
				}
			}
			glEnd();
			break;
		}
		case RENDER_TYPE_NORMALS:
		{
			glBegin(GL_LINES);
			for(size_t i = 0 ; i < vpc.size(); i++)
			{
				for(int j = 0; j < vpc[i].size(); j++)
				{
					glColor3f(fabs(vpc[i][j].normal_x), fabs(vpc[i][j].normal_y), fabs(vpc[i][j].normal_z)  );
					glVertex3f(vpc[i][j].x, vpc[i][j].y, vpc[i][j].z);
					glVertex3f(vpc[i][j].x + vpc[i][j].normal_x, vpc[i][j].y + vpc[i][j].normal_y, vpc[i][j].z + vpc[i][j].normal_z);
				}
			}
			glEnd();
			break;
		}
		case RENDER_TYPE_PARTICLE_ESTIMATION:
		{
			rgd_particle_filter_CUDAWrapper.render(ispathrendered, isallpathsrendered);
		}
	}

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
        	render_type = RENDER_TYPE_SEMANTIC_LABELS;
        	break;
        }
        case '3':
        {
        	render_type = RENDER_TYPE_RINGS;
        	break;
        }
        case '4':
        {
        	render_type = RENDER_TYPE_INTENSITY;
        	break;
        }
        case '5':
        {
        	render_type = RENDER_TYPE_NORMALS;
        	break;
        }
        case '6':
		{
			render_type = RENDER_TYPE_PARTICLE_ESTIMATION;
			break;
		}
        case '7':
        {
        	ispathrendered =! ispathrendered;
           	break;
        }
        case '8':
		{
			isallpathsrendered =! isallpathsrendered;
			break;
		}
        case 'c' :
        {
        	particle_filter_calculations_CUDA();
        	render_type = RENDER_TYPE_PARTICLE_ESTIMATION;
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

bool loadData(std::string filename, pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> &_pc)
{
	if(pcl::io::loadPCDFile(filename, _pc) == -1)
		return false;



return true;
}

void transformPointCloud(pcl::PointCloud<velodyne_pointcloud::PointXYZIRNL> &pointcloud, Eigen::Affine3f transform)
{
	std::cout << transform.matrix() << std::endl;

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

void particle_filter_calculations_CUDA()
{
	rgd_particle_filter_CUDAWrapper.calculateBestTrajectory(vpc, vpc_raw, vtransforms);
}




