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

#include "basicFunctions.h"
#include "particle_filter_fast.h"

//common
#include "data_model.hpp"
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
float translate_z = -150.0;
float translate_x, translate_y = 0.0;

pcl::PointCloud<Semantic::PointXYZL> point_cloud_semantic_map;
pcl::PointCloud<pcl::PointXYZ> point_cloud_semantic_map_label_floor_ground;

pcl::PointCloud<Semantic::PointXYZL> winning_point_cloud;

typedef struct scan_with_odo
{
	pcl::PointCloud<Semantic::PointXYZL> scan;
	Eigen::Affine3f odo;
}scan_with_odo_t;

std::vector<scan_with_odo_t> vscan_with_odo;

float motion_model_max_angle = 10.0f;
float motion_model_max_translation = 0.5f;
float max_particle_size = 500;
float max_particle_size_kidnapped_robot = 100000;
float distance_above_Z = 1.0f;
float rgd_resolution = 1.0f;
int cuda_device = 0;
int max_number_considered_in_INNER_bucket = 5;
int max_number_considered_in_OUTER_bucket = 5;
float overlap_threshold = 0.01f;
float propability_threshold = 0.1;
float rgd_2D_res = 5.0f;
CParticleFilterFast particle_filter;

bool is_iteration = false;
bool show_winning_point_cloud = true;

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();
void singleIteration();

int main(int argc, char **argv)
{
	std::cout << "use case: particle filter localization fast" << std::endl;
	std::cout << "data format: Semantic::PointXYZL" << std::endl;

	if(argc < 3)
	{
		std::cout << "Usage:\n";
		std::cout << argv[0] <<" point_cloud_semantic_map.pcd laser_data_with_trajectory.xml parameters\n";

		std::cout << "Parameters:" << std::endl;
		std::cout << "-mmma: motion_model_max_angle  default: " << motion_model_max_angle << std::endl;
		std::cout << "-mmmt: motion_model_max_translation  default: " << motion_model_max_translation << std::endl;
		std::cout << "-mps: max_particle_size  default: " << max_particle_size << std::endl;
		std::cout << "-mpskr: max_particle_size_kidnapped_robot  default: " << max_particle_size_kidnapped_robot << std::endl;
		std::cout << "-daz: distance_above_Z  default: " << distance_above_Z << std::endl;
		std::cout << "-rgdr: rgd_resolution  default: " << rgd_resolution << std::endl;
		std::cout << "-cd: cuda_device  default: " << cuda_device << std::endl;
		std::cout << "-inner: max_number_considered_in_INNER_bucket  default: " << max_number_considered_in_INNER_bucket << std::endl;
		std::cout << "-outer: max_number_considered_in_OUTER_bucket  default: " << max_number_considered_in_OUTER_bucket << std::endl;
		std::cout << "-ot: overlap_threshold  default: " << overlap_threshold << std::endl;
		std::cout << "-pt: propability_threshold  default: " << propability_threshold << std::endl;
		std::cout << "-rgdr2D: rgd_2D_res  default: " << rgd_2D_res << std::endl;

		std::cout << "return -1" << std::endl;
		return -1;
	}else
	{
		std::vector<int> ind_pcd;
		ind_pcd = pcl::console::parse_file_extension_argument (argc, argv, ".pcd");

		if(ind_pcd.size()!=1)
		{
			std::cout << "did you forget pcd file location? return" << std::endl;
			std::cout << "return -1" << std::endl;
			return -1;
		}

		if(pcl::io::loadPCDFile(argv[1], point_cloud_semantic_map) == -1)
		{
			std::cout << "return -1" << std::endl;
			return -1;
		}

		ind_pcd = pcl::console::parse_file_extension_argument (argc, argv, ".xml");

		if(ind_pcd.size()!=1)
		{
			std::cout << "did you forget xml file location? return" << std::endl;
			std::cout << "return -1" << std::endl;
			return -1;
		}

		data_model dSets;
		std::vector<std::string> indices;
		std::string model_file = argv[2];

		dSets.loadFile(model_file);
		dSets.getAllScansId(indices);

		for (int i=0; i< indices.size(); i++)
		{
			std::string fn;
			fn = dSets.getFullPathOfPointcloud(indices[i]);

			scan_with_odo_t swo;
			bool isOkTr = dSets.getAffine(indices[i], swo.odo.matrix());

			if (isOkTr)
			{
				if(pcl::io::loadPCDFile(fn, swo.scan) == -1)
				{
					std::cout << "Problem with opening: " << fn << " file ... exit(-1)" << std::endl;
					exit(-1);
				}else
				{
					vscan_with_odo.push_back(swo);
					std::cout << "file " << fn << " loaded" << std::endl;
				}
			}
		}

		for(size_t i = 0 ; i < point_cloud_semantic_map.size(); i++)
		{
			if(point_cloud_semantic_map[i].label == FLOOR_GROUND)
			{
				pcl::PointXYZ p;
				p.x = point_cloud_semantic_map[i].x;
				p.y = point_cloud_semantic_map[i].y;
				p.z = point_cloud_semantic_map[i].z;
				point_cloud_semantic_map_label_floor_ground.push_back(p);
			}
		}

		pcl::console::parse_argument (argc, argv, "-mmma", motion_model_max_angle);
		std::cout << "motion_model_max_angle: " << motion_model_max_angle << std::endl;

		pcl::console::parse_argument (argc, argv, "-mmmt", motion_model_max_translation);
		std::cout << "motion_model_max_translation: " << motion_model_max_translation << std::endl;

		pcl::console::parse_argument (argc, argv, "-mps", max_particle_size);
		std::cout << "max_particle_size: " << max_particle_size << std::endl;

		pcl::console::parse_argument (argc, argv, "-mpskr", max_particle_size_kidnapped_robot);
		std::cout << "max_particle_size_kidnapped_robot: " << max_particle_size_kidnapped_robot << std::endl;

		pcl::console::parse_argument (argc, argv, "-daz", distance_above_Z);
		std::cout << "distance_above_Z: " << distance_above_Z << std::endl;

		pcl::console::parse_argument (argc, argv, "-rgdr", rgd_resolution);
		std::cout << "rgd_resolution: " << rgd_resolution << std::endl;

		pcl::console::parse_argument (argc, argv, "-cd", cuda_device);
		std::cout << "cuda_device: " << cuda_device << std::endl;

		pcl::console::parse_argument (argc, argv, "-inner", max_number_considered_in_INNER_bucket);
		std::cout << "max_number_considered_in_INNER_bucket: " << max_number_considered_in_INNER_bucket << std::endl;

		pcl::console::parse_argument (argc, argv, "-outer", max_number_considered_in_OUTER_bucket);
		std::cout << "max_number_considered_in_OUTER_bucket: " << max_number_considered_in_OUTER_bucket << std::endl;

		pcl::console::parse_argument (argc, argv, "-ot", overlap_threshold);
		std::cout << "overlap_threshold: " << overlap_threshold << std::endl;

		pcl::console::parse_argument (argc, argv, "-pt", propability_threshold);
		std::cout << "propability_threshold: " << propability_threshold << std::endl;

		pcl::console::parse_argument (argc, argv, "-rgdr2D", rgd_2D_res);
		std::cout << "rgd_2D_res: " << rgd_2D_res << std::endl;
	}

	if(!particle_filter.init(cuda_device,
			motion_model_max_angle,
			motion_model_max_translation,
			max_particle_size,
			max_particle_size_kidnapped_robot,
			distance_above_Z,
			rgd_resolution,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			overlap_threshold,
			propability_threshold,
			rgd_2D_res))
	{
		std::cout << "problem with particle_filter.init() exit(-1)" << std::endl;
		exit(-1);
	}
	if(!particle_filter.setGroundPointsFromMap(point_cloud_semantic_map_label_floor_ground))
	{
		std::cout << "problem with particle_filter.setGroundPointsFromMap() exit(-1)" << std::endl;
		exit(-1);
	}

	particle_filter.genParticlesKidnappedRobot();

	if(!particle_filter.copyReferenceModelToGPU(point_cloud_semantic_map))
	{
		std::cout << "problem with particle_filter.copyReferenceModelToGPU(point_cloud_semantic_map)  exit(-1)" << std::endl;
		exit(-1);
	}
	if(!particle_filter.computeRGD())
	{
		std::cout << "problem with particle_filter.computeRGD() exit(-1)" << std::endl;
		exit(-1);
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
    glutReshapeFunc(reshape);
	glutIdleFunc(singleIteration);
	glutMainLoop();
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Particle Filter Localization fast");
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

    glPointSize(1.0f);
    glBegin(GL_POINTS);
	for(size_t i = 0; i < point_cloud_semantic_map.size(); i++)
	{
		if(point_cloud_semantic_map[i].label<16 && point_cloud_semantic_map[i].label>=0)
		{
			glColor3f(colors[point_cloud_semantic_map[i].label][0], colors[point_cloud_semantic_map[i].label][1], colors[point_cloud_semantic_map[i].label][2]);
		}else
		{

			glColor3f(0.7f, 0.7f, 0.7f);
		}
		glVertex3f(point_cloud_semantic_map[i].x, point_cloud_semantic_map[i].y, point_cloud_semantic_map[i].z);
	}
	glEnd();

	glPointSize(5.0f);
	glBegin(GL_POINTS);
	for(size_t i = 0; i < winning_point_cloud.size(); i++)
	{
		if(winning_point_cloud[i].label<16 && winning_point_cloud[i].label>=0)
		{
			glColor3f(colors[winning_point_cloud[i].label][0], colors[winning_point_cloud[i].label][1], colors[winning_point_cloud[i].label][2]);
		}else
		{
			glColor3f(0.7f, 0.7f, 0.7f);
		}
		glVertex3f(winning_point_cloud[i].x, winning_point_cloud[i].y, winning_point_cloud[i].z);
	}
	glEnd();

	particle_filter.render();

    glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            glutDestroyWindow(glutGetWindow());
            return;
        case 'i':
        {
        	is_iteration =! is_iteration;
        	break;
        }
        case 'g':
        {
        	particle_filter.genParticlesKidnappedRobot();
        	break;
        }
        case 's':
        {
        	show_winning_point_cloud = !show_winning_point_cloud;
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

	std::cout << "press 'i': is_iteration =! is_iteration (start stop particle filter computations)" << std::endl;
	std::cout << "press 'g': genParticlesKidnappedRobot()" << std::endl;
	std::cout << "press 's': show_winning_point_cloud = !show_winning_point_cloud" << std::endl;

	std::cout << "press 'Esc': EXIT" << std::endl;
}

void singleIteration()
{
	if(is_iteration)
	{
		is_iteration=false;
		
		clock_t begin_time;
		double computation_time;
		begin_time = clock();

		static int counter = 1;

		Eigen::Affine3f odometryIncrement = vscan_with_odo[counter-1].odo.inverse() * vscan_with_odo[counter].odo;

		if(!particle_filter.copyCurrentScanToGPU(vscan_with_odo[counter].scan))
		{
			std::cout << "problem with cuda_nn.copyCurrentScanToGPU(current_scan) return" << std::endl;
			return;
		}

		if(!particle_filter.update())return;

		Eigen::Affine3f winM = particle_filter.getWinningParticle();

		if(show_winning_point_cloud)
		{
			winning_point_cloud = vscan_with_odo[counter].scan;
			transformPointCloud(winning_point_cloud, winning_point_cloud, winM);
		}
		particle_filter.prediction(odometryIncrement);


		counter++;
		if(counter >= vscan_with_odo.size())
		{
			counter = 1;
			particle_filter.genParticlesKidnappedRobot();
		}

		computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
		std::cout << "particle filter singleIteration computation_time: " << computation_time << " counter: "<< counter << " of: "<< vscan_with_odo.size() << std::endl;

		glutPostRedisplay();
	}
}
