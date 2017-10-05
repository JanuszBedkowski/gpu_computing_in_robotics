#include <GL/freeglut.h>

//PCL
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include "cudaWrapper.h"

typedef struct plane
{
	float nx;
	float ny;
	float nz;
	float rho;
}plane_t;

typedef struct color
{
	float r;
	float g;
	float b;
}color_t;


const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -20.0;
float translate_x, translate_y = 0.0;

float search_radius = 0.5f;
float bounding_box_extension = 1.0f;
int max_number_considered_in_INNER_bucket = 10000;
int max_number_considered_in_OUTER_bucket = 10000;

pcl::PointCloud<pcl::PointXYZ> point_cloud_element;
pcl::PointCloud<pcl::PointXYZ> second_point_cloud;

pcl::PointCloud<pcl::PointXYZ> pc_nn_1;
pcl::PointCloud<pcl::PointXYZ> pc_nn_2;

CCudaWrapper cudaWrapper;
std::vector<color_t> vColors;
std::vector<float> vZAngles;
std::vector<Eigen::Affine3f> vPoses;
float R = 1;
float H = 0.5f;
float angle = 15.0f * M_PI/180.0;
float density = 0.2f;
float step_angle = 30.0f * M_PI/180.0;
int index_active_element = 1;
int number_of_elements = 20;

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();
void generateSingleElement(pcl::PointCloud<pcl::PointXYZ> &point_cloud);
plane_t transformPlane(plane_t plane, Eigen::Affine3f m);
float distanceToPlane(pcl::PointXYZ p, plane_t plane);
void genElements();
void genSecondPointCloud();
void updatePoses();
int countNN(pcl::PointCloud<pcl::PointXYZ> _point_cloud_element, pcl::PointCloud<pcl::PointXYZ> _second_point_cloud, std::vector<float> _vZAngles, float _R, float _H, float _angle, float _density);

int main(int argc, char **argv)
{
	cudaWrapper.warmUpGPU();
	genSecondPointCloud();
	generateSingleElement(point_cloud_element);
	genElements();

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
	glutMainLoop();
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Use case - robotic arm collision detection");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutMotionFunc(motion);
    glutReshapeFunc(reshape);

    glClearColor(0.0, 0.0, 0.0, 1.0);
    glEnable(GL_DEPTH_TEST);

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

    glPointSize(1);

    for(size_t i = 0 ; i < vPoses.size(); i++)
    {
    	if(i != index_active_element)
    	{
    		glColor3f(vColors[i].r, vColors[i].g, vColors[i].b);
    	}else
    	{
    		glColor3f(1.0f, 1.0f, 1.0f);
    	}
    	pcl::PointCloud<pcl::PointXYZ> point_cloud = point_cloud_element;
    	cudaWrapper.transform(point_cloud, vPoses[i]);

    	glBegin(GL_POINTS);
		for(size_t i = 0; i < point_cloud.size(); i++)
		{
			glVertex3f(point_cloud[i].x, point_cloud[i].y, point_cloud[i].z);
		}
		glEnd();
    }

    glColor3f(0.0f, 0.0f, 1.0f);
	glBegin(GL_POINTS);
		for(size_t i = 0; i < second_point_cloud.size(); i++)
		{
			glVertex3f(second_point_cloud[i].x, second_point_cloud[i].y, second_point_cloud[i].z);
		}
	glEnd();

	glColor3f(0.0f, 1.0f, 0.0f);
	glBegin(GL_LINES);
		for(size_t i = 0; i < pc_nn_1.size(); i++)
		{
			glVertex3f(pc_nn_1[i].x, pc_nn_1[i].y, pc_nn_1[i].z);
			glVertex3f(pc_nn_2[i].x, pc_nn_2[i].y, pc_nn_2[i].z);
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
        case '=':
        {
        	index_active_element++;
        	if(index_active_element >= vPoses.size()) index_active_element = vPoses.size()-1;
          	break;
        }
        case '-':
		{
			index_active_element--;
			if(index_active_element < 1) index_active_element = 1;
			break;
		}
        case 'n' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

			int nn = countNN(point_cloud_element, second_point_cloud, vZAngles, R, H, angle, density);

			std::cout << "Number of NN: " << nn << std::endl;

        	computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "cudaWrapper.nearestNeighbourhoodSearch computation_time: " << computation_time << std::endl;

			break;
        }
        case 'a' :
		{
			vZAngles[index_active_element] += step_angle;
			updatePoses();
			break;
		}
		case 's' :
		{
			vZAngles[index_active_element] -= step_angle;
			updatePoses();
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
	std::cout << "press '=': index active element++" << std::endl;
	std::cout << "press '-': index active element--" << std::endl;
	std::cout << "press 'a': rotate left" << std::endl;
	std::cout << "press 's': rotate right" << std::endl;
	std::cout << "press 'n': compute collisions" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}

void generateSingleElement(pcl::PointCloud<pcl::PointXYZ> &point_cloud)
{
	pcl::PointCloud<pcl::PointXYZ> out_point_cloud;

	plane_t planeUp;
	plane_t planeBottom;

	planeUp.nx = 0.0f;
	planeUp.ny = 0.0f;
	planeUp.nz = -1.0f;
	planeUp.rho = H;

	planeBottom.nx = 0.0f;
	planeBottom.ny = 0.0f;
	planeBottom.nz = 1.0f;
	planeBottom.rho = H;

	Eigen::Affine3f mt = Eigen::Affine3f::Identity();
	mt(2,3) = H;

	planeUp = transformPlane(planeUp, mt);

	Eigen::Affine3f mr;
	mr = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitX())
		  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
		  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ());

	planeUp = transformPlane(planeUp, mr);
	mt(2,3) = -H;
	planeUp = transformPlane(planeUp, mt);

	mt(2,3) = -H;
	planeBottom = transformPlane(planeBottom, mt);

	Eigen::Affine3f mr2;
	mr2 = Eigen::AngleAxisf(-angle, Eigen::Vector3f::UnitX())
		  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
		  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ());

	planeBottom = transformPlane(planeBottom, mr2);
	mt(2,3) = H;
	planeBottom = transformPlane(planeBottom, mt);


	for(float x = -4.0f; x <= 4.0f; x+= density)
		for(float y = -4.0f; y <= 4.0f; y+= density)
			for(float z = -4.0f; z <= 4.0f; z+= density)
			{
				float dist = sqrt(x*x + y*y);
				if(dist < R)
				{
					pcl::PointXYZ p;
					p.x = x;
					p.y = y;
					p.z = z;
					if(distanceToPlane(p, planeUp) > 0.0f && distanceToPlane(p, planeBottom) > 0.0f)
						out_point_cloud.push_back(p);
				}
			}

	point_cloud = out_point_cloud;

}

plane_t transformPlane(plane_t plane, Eigen::Affine3f m)
{
	plane_t new_plane;

	Eigen::Vector3f O(plane.nx * plane.rho, plane.ny * plane.rho, plane.nz * plane.rho);
	Eigen::Vector3f N(plane.nx , plane.ny , plane.nz );
	Eigen::Vector3f _O = m * O;
	Eigen::Affine3f mInv = m.inverse();
	Eigen::Affine3f mT;

	mT(0,0) = mInv(0,0); mT(0,1) = mInv(1,0); mT(0,2) = mInv(2,0); mT(0,3) = mInv(3,0);
	mT(1,0) = mInv(0,1); mT(1,1) = mInv(1,1); mT(1,2) = mInv(2,1); mT(1,3) = mInv(3,1);
	mT(2,0) = mInv(0,2); mT(2,1) = mInv(1,2); mT(2,2) = mInv(2,2); mT(2,3) = mInv(3,2);
	mT(3,0) = mInv(0,3); mT(3,1) = mInv(1,3); mT(3,2) = mInv(2,3); mT(3,3) = mInv(3,3);

	Eigen::Vector3f _N =  mT * N;

	float d = _O.dot(_N);

	new_plane.nx = _N.x();
	new_plane.ny = _N.y();
	new_plane.nz = _N.z();
	new_plane.rho = d;
	return new_plane;
}

float distanceToPlane(pcl::PointXYZ p, plane_t plane)
{
	float d = (p.x * plane.nx + p.y * plane.ny + p.z * plane.nz + plane.rho)/(sqrt(plane.nx * plane.nx + plane.ny * plane.ny + plane.nz * plane.nz));
	return d;
}

void genElements()
{
	color_t color;
	float zangle = 0.0f;

	color.b = (float(rand()%10000)/10000.0f);
	color.g = (float(rand()%10000)/10000.0f);
	color.r = (float(rand()%10000)/10000.0f);
	vColors.push_back(color);

	zangle = 0;
	vZAngles.push_back(zangle);

	Eigen::Affine3f m = Eigen::Affine3f::Identity();
	vPoses.push_back(m);

	for(int i = 0 ; i < number_of_elements ; i++)
	{
		color.b = (float(rand()%10000)/10000.0f);
		color.g = (float(rand()%10000)/10000.0f);
		color.r = (float(rand()%10000)/10000.0f);
		vColors.push_back(color);
		vZAngles.push_back(zangle);

		Eigen::Affine3f mt = Eigen::Affine3f::Identity();
		mt(2,3) = H;
		Eigen::Affine3f mr;
		mr = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ());
		Eigen::Affine3f mRes = mt * mr;
		Eigen::Affine3f mrZ;
		mrZ = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitX())
				  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
				  * Eigen::AngleAxisf(zangle, Eigen::Vector3f::UnitZ());
		mRes = mRes * mrZ;
		mRes = mRes * mr;
		mRes = mRes * mt;
		mRes = vPoses[i] * mRes;

		vPoses.push_back(mRes);
		zangle += step_angle;
	}
}

void genSecondPointCloud()
{
	pcl::PointCloud<pcl::PointXYZ> out_point_cloud_1;
	pcl::PointCloud<pcl::PointXYZ> out_point_cloud_2;

	for(float x = -4.0f; x <= 4.0f; x+= density)
		for(float y = -4.0f; y <= 4.0f; y+= density)
			for(float z = -4.0f; z <= 4.0f; z+= density)
			{
				pcl::PointXYZ p;
				p.x = x;
				p.y = y;
				p.z = z;
				out_point_cloud_1.push_back(p);
			}

	Eigen::Affine3f m = Eigen::Affine3f::Identity();
	m(0,3) = 6.0f;
	m(2,3) = 10.0f;
	cudaWrapper.transform(out_point_cloud_1, m);

	for(float x = -4.0f; x <= 4.0f; x+= density)
		for(float y = -4.0f; y <= 4.0f; y+= density)
			for(float z = -4.0f; z <= 4.0f; z+= density)
			{
				pcl::PointXYZ p;
				p.x = x;
				p.y = y;
				p.z = z;
				out_point_cloud_2.push_back(p);
			}

	m(0,3) = -6.0f;
	m(2,3) = 10.0f;
	cudaWrapper.transform(out_point_cloud_2, m);

	out_point_cloud_1 += out_point_cloud_2;

	second_point_cloud = out_point_cloud_1;
}

void updatePoses()
{
	for(int i = 1 ; i < vPoses.size() ; i++)
	{
		Eigen::Affine3f mt = Eigen::Affine3f::Identity();
		mt(2,3) = H;
		Eigen::Affine3f mr;
		mr = Eigen::AngleAxisf(angle, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ());
		Eigen::Affine3f mRes = mt * mr;
		Eigen::Affine3f mrZ;
		mrZ = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitX())
				  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
				  * Eigen::AngleAxisf(vZAngles[i], Eigen::Vector3f::UnitZ());
		mRes = mRes * mrZ;
		mRes = mRes * mr;
		mRes = mRes * mt;
		mRes = vPoses[i-1] * mRes;

		vPoses[i] = mRes;
	}
}

int countNN(pcl::PointCloud<pcl::PointXYZ> _point_cloud_element, pcl::PointCloud<pcl::PointXYZ> _second_point_cloud, std::vector<float> _vZAngles, float _R, float _H, float _angle, float _density)
{
	std::vector<Eigen::Affine3f> _vPoses;
	_vPoses.push_back(Eigen::Affine3f::Identity());

	for(int i = 1 ; i < _vZAngles.size() ; i++)
	{
		Eigen::Affine3f mt = Eigen::Affine3f::Identity();
		mt(2,3) = _H;
		Eigen::Affine3f mr;
		mr = Eigen::AngleAxisf(_angle, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitZ());
		Eigen::Affine3f mRes = mt * mr;
		Eigen::Affine3f mrZ;
		mrZ = Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitX())
				  * Eigen::AngleAxisf(0.0f, Eigen::Vector3f::UnitY())
				  * Eigen::AngleAxisf(_vZAngles[i], Eigen::Vector3f::UnitZ());
		mRes = mRes * mrZ;
		mRes = mRes * mr;
		mRes = mRes * mt;
		mRes = _vPoses[i-1] * mRes;

		_vPoses.push_back(mRes);
	}

	pcl::PointCloud<pcl::PointXYZ> pc_element;

	for(size_t i = 0 ; i < vPoses.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZ> point_cloud = _point_cloud_element;
		cudaWrapper.transform(point_cloud, _vPoses[i]);

		pc_element += point_cloud;
	}

	std::vector<int> nearest_neighbour_indexes;
	nearest_neighbour_indexes.resize(pc_element.size());
	std::fill(nearest_neighbour_indexes.begin(), nearest_neighbour_indexes.end(), -1);

	if(!cudaWrapper.nearestNeighbourhoodSearch(
			_second_point_cloud,
			pc_element,
			search_radius,
			bounding_box_extension,
			max_number_considered_in_INNER_bucket,
			max_number_considered_in_OUTER_bucket,
			nearest_neighbour_indexes))
	{
		cudaDeviceReset();
		std::cout << "cudaWrapper.nearestNeighbourhoodSearch NOT SUCCESFULL" << std::endl;
	}


	pc_nn_1.clear();
	pc_nn_2.clear();


	int counter = 0 ;
	for(size_t i = 0 ; i < nearest_neighbour_indexes.size(); i++)
	{
		if(nearest_neighbour_indexes[i] != -1)
		{
			counter++;
			pc_nn_1.push_back(_second_point_cloud[nearest_neighbour_indexes[i]]);
			pc_nn_2.push_back(pc_element[i]);
		}
	}
return counter;
}
