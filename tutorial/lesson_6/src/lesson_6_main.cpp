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

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -20.0;
float translate_x, translate_y = 0.0;

float normal_vectors_search_radius = 1.0f;
float projections_search_radius = 0.5f;
float bounding_box_extension = 1.0f;

int max_number_considered_in_INNER_bucket = 10000;
int max_number_considered_in_OUTER_bucket = 10000;

pcl::PointCloud<pcl::PointNormal> first_point_cloud;
pcl::PointCloud<pcl::PointXYZ> second_point_cloud;
pcl::PointCloud<pcl::PointXYZ> second_point_cloud_projections;
std::vector<char> v_is_projection;

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
	std::cout << "Lesson 6 - projections" << std::endl;

	if(argc < 3)
	{
		std::cout << "Usage:\n";
		std::cout << argv[0] <<" first_point_cloud_file.pcd second_point_cloud_file.pcd parameters\n";
		std::cout << "Default:  ../../data/scan_Velodyne_VLP16.pcd ../../data/scan_Velodyne_VLP16_2.pcd\n";
		std::cout << "-nvsr normal_vectors_search_radius: default " << normal_vectors_search_radius << std::endl;
		std::cout << "-psr projections_search_radius: default " << projections_search_radius << std::endl;
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

	pcl::console::parse_argument (argc, argv, "-nvsr", normal_vectors_search_radius);
	std::cout << "normal_vectors_search_radius: " << normal_vectors_search_radius << std::endl;

	pcl::console::parse_argument (argc, argv, "-psr", projections_search_radius);
	std::cout << "projections_search_radius: " << projections_search_radius << std::endl;

	pcl::console::parse_argument (argc, argv, "-bbe", bounding_box_extension);
	std::cout << "bounding_box_extension: " << bounding_box_extension << std::endl;

	if (false == initGL(&argc, argv))
	{
		return -1;
	}

	v_is_projection.resize(second_point_cloud.size());
	std::fill(v_is_projection.begin(), v_is_projection.end(), 0);
	second_point_cloud_projections = second_point_cloud;

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
    glutCreateWindow("Lesson 6 - projections");
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

    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_POINTS);
    	for(size_t i = 0; i < first_point_cloud.size(); i++)
    	{
    		glVertex3f(first_point_cloud[i].x, first_point_cloud[i].y, first_point_cloud[i].z);
    	}
    glEnd();

    glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_POINTS);
		for(size_t i = 0; i < second_point_cloud.size(); i++)
		{
			glVertex3f(second_point_cloud[i].x, second_point_cloud[i].y, second_point_cloud[i].z);
		}
	glEnd();

	glColor3f(0.1f, 1.0f, 0.1f);
	glBegin(GL_LINES);
		for(size_t i = 0; i < second_point_cloud.size(); i++)
		{
			if(v_is_projection[i] == 1)
			{
				glVertex3f(second_point_cloud[i].x, second_point_cloud[i].y, second_point_cloud[i].z);
				glVertex3f(second_point_cloud_projections[i].x, second_point_cloud_projections[i].y, second_point_cloud_projections[i].z);
			}
		}
	glEnd();

	glColor3f(0.0f, 0.5f, 0.0f);
	glBegin(GL_POINTS);
		for(size_t i = 0; i < second_point_cloud.size(); i++)
		{
			if(v_is_projection[i] == 1)
			{
				glVertex3f(second_point_cloud_projections[i].x, second_point_cloud_projections[i].y, second_point_cloud_projections[i].z);
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
        case 'p' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

			if(!cudaWrapper.projections(
			        			first_point_cloud,
			        			second_point_cloud,
			        			normal_vectors_search_radius,
			        			projections_search_radius,
			        			bounding_box_extension,
			        			max_number_considered_in_INNER_bucket,
			        			max_number_considered_in_OUTER_bucket,
			        			second_point_cloud_projections,
			        			v_is_projection))
			{
				cudaDeviceReset();
				std::cout << "cudaWrapper.projections NOT SUCCESFULL" << std::endl;
			}

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.projections computation_time: " << computation_time << std::endl;

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
			std::fill(v_is_projection.begin(), v_is_projection.end(), 0);
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
			std::fill(v_is_projection.begin(), v_is_projection.end(), 0);
			break;
		}

    }
    printHelp();
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

void printHelp()
{
	std::cout << "----------------------" << std::endl;
	std::cout << "press 'p': compute projections" << std::endl;
	std::cout << "press 'a': rotate second_point_cloud via X axis (+1 degree)" << std::endl;
	std::cout << "press 'd': rotate second_point_cloud via X axis (-1 degree)" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}
