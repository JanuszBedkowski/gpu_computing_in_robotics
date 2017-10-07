#include <GL/freeglut.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Dense>

#include "lesson_19.h"
#include "cudaWrapper.h"

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -20.0;
float translate_x, translate_y = 0.0;

std::vector<triangle> vTriangles;
plane winningPlane;
std::vector<Eigen::Vector3f> vNormals;
std::vector<plane> vPlanes;
laser_beam single_laser_beam_CPU;
laser_beam single_laser_beam_GPU;
std::vector<simple_point3D> vRanfomMeasurements;

CCudaWrapper cudaWrapper;

GLfloat LightAmbient[]=		{ 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat LightDiffuse[]=		{ 0.2f, 0.2f, 0.2f, 1.0f };
GLfloat LightPosition[]=	{ 0.0f, -100.0f, 100.0f, 1.0f };

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();


bool loadTrianglesFromHeightmap(std::string _filename, std::vector<triangle> &_vtriangles, std::vector<Eigen::Vector3f> &_vNormals);
std::vector<plane> computePlanes(std::vector<triangle> &_vTriangles, std::vector<Eigen::Vector3f> &_vNormals);
void computeDistance(laser_beam &_single_laser_beam_CPU, std::vector<plane> &_vPlanes);
float distanceToPlane(plane _plane, simple_point3D _point3D);
simple_point3D rayIntersection(laser_beam &_laser_beam, plane &_plane);
bool pointInPolygon(simple_point3D p, simple_point3D v0, simple_point3D v1, simple_point3D v2);

int main(int argc, char **argv)
{
	if(argc < 2)
	{
		std::cout << "Usage:\n";
		std::cout << argv[0] <<" heightmap.png\n";

		std::cout << "Default: ../heightmaps/heightmap.png\n";

		if(!loadTrianglesFromHeightmap(std::string("../heightmaps/heightmap.png"), vTriangles, vNormals))
		{
			std::cout << "Problem with loadTrianglesFromHeightmap for filename: " << argv[1] << std::endl;
			return -1;
		}
	}else
	{
		if(!loadTrianglesFromHeightmap(std::string(argv[1]), vTriangles, vNormals))
		{
			std::cout << "Problem with loadTrianglesFromHeightmap for filename: " << argv[1] << std::endl;
			return -1;
		}
	}

	cudaWrapper.warmUpGPU();

	vPlanes = computePlanes(vTriangles, vNormals);
	if(!cudaWrapper.copyPlaneDataToGPU(vPlanes))
	{
		std::cout << "problem with cudaWrapper.copyPlaneDataToGPU(vPlanes)" << std::endl;
		return -1;
	}

	single_laser_beam_CPU.distance = 100.0f;
	single_laser_beam_CPU.range = 100.0f;
	single_laser_beam_CPU.position.x = 50.0f;
	single_laser_beam_CPU.position.y = 50.0f;
	single_laser_beam_CPU.position.z = 20.0f;
	single_laser_beam_CPU.direction.x = 0.0f;
	single_laser_beam_CPU.direction.y = 0.0f;
	single_laser_beam_CPU.direction.z = -1.0f;

	single_laser_beam_GPU = single_laser_beam_CPU;
	single_laser_beam_GPU.position.x += 0.1;

	for(int i = 0; i < 1000; i++)
	{

		std::cout << i << " of: " << 1000 << std::endl;
		laser_beam random_laser_beam;
		random_laser_beam.direction.x = 0.0f;
		random_laser_beam.direction.y = 0.0f;
		random_laser_beam.direction.z = -1.0f;

		random_laser_beam.range = 100.0f;
		random_laser_beam.distance = 100.0f;
		random_laser_beam.position.x = (float(rand()%10000))/10000.0f * 100.0f;
		random_laser_beam.position.y = (float(rand()%10000))/10000.0f * 100.0f;
		random_laser_beam.position.z = 20.0f;

		cudaWrapper.computeDistance(random_laser_beam);

		if(random_laser_beam.distance != random_laser_beam.range)
		{
			simple_point3D rm;
			rm.x = random_laser_beam.position.x + random_laser_beam.direction.x * random_laser_beam.distance;
			rm.y = random_laser_beam.position.y + random_laser_beam.direction.y * random_laser_beam.distance;
			rm.z = random_laser_beam.position.z + random_laser_beam.direction.z * random_laser_beam.distance;

			vRanfomMeasurements.push_back(rm);
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
    glutReshapeFunc(reshape);
	glutMainLoop();
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Lesson 19 - laser range finder simulation");
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

    glEnable ( GL_COLOR_MATERIAL );
    glColorMaterial ( GL_FRONT, GL_AMBIENT_AND_DIFFUSE );
    glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
   	glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
   	glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);
   	glEnable(GL_LIGHT1);
   	glEnable(GL_LIGHTING);

   	glEnable(GL_DEPTH_TEST);
    return true;
}

void display()
{
	clock_t begin_time;
	double computation_time;
	begin_time = clock();

		computeDistance(single_laser_beam_CPU, vPlanes);

	computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
	std::cout << "single ray computation on CPU time: " << computation_time << std::endl;

	clock_t begin_time_CUDA;
	double computation_time_CUDA;
	begin_time_CUDA = clock();

		cudaWrapper.computeDistance(single_laser_beam_GPU);

	computation_time_CUDA=(double)( clock () - begin_time_CUDA ) /  CLOCKS_PER_SEC;
	std::cout << "single ray computation on GPU time: " << computation_time_CUDA << std::endl;

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

    glColor3f(0.0f, 0.7f, 0.0f);

    glBegin(GL_TRIANGLES);
    for(size_t i = 0; i < vTriangles.size(); i++)
    {
    	glNormal3f(vNormals[i].x(), vNormals[i].y(), vNormals[i].z());
    	glVertex3f(vTriangles[i].vertexA.x, vTriangles[i].vertexA.y, vTriangles[i].vertexA.z);
    	glVertex3f(vTriangles[i].vertexB.x, vTriangles[i].vertexB.y, vTriangles[i].vertexB.z);
    	glVertex3f(vTriangles[i].vertexC.x, vTriangles[i].vertexC.y, vTriangles[i].vertexC.z);
    }
    glEnd();

    glColor3f(1.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    	glVertex3f(single_laser_beam_CPU.position.x, single_laser_beam_CPU.position.y, single_laser_beam_CPU.position.z);
    	glVertex3f(single_laser_beam_CPU.position.x + single_laser_beam_CPU.direction.x * single_laser_beam_CPU.distance,
    			   single_laser_beam_CPU.position.y + single_laser_beam_CPU.direction.y * single_laser_beam_CPU.distance,
    			   single_laser_beam_CPU.position.z + single_laser_beam_CPU.direction.z * single_laser_beam_CPU.distance);
    glEnd();

    glColor3f(0.0f, 1.0f, 0.0f);
   glBegin(GL_LINES);
	glVertex3f(single_laser_beam_GPU.position.x, single_laser_beam_GPU.position.y, single_laser_beam_GPU.position.z);
	glVertex3f(single_laser_beam_GPU.position.x + single_laser_beam_GPU.direction.x * single_laser_beam_GPU.distance,
			   single_laser_beam_GPU.position.y + single_laser_beam_GPU.direction.y * single_laser_beam_GPU.distance,
			   single_laser_beam_GPU.position.z + single_laser_beam_GPU.direction.z * single_laser_beam_GPU.distance);
   glEnd();


    glPointSize(5);
	glDisable(GL_LIGHTING);
    glColor3f(1.0f ,0.0f ,0.0f);
    glBegin(GL_POINTS);
    for(size_t i = 0 ; i < vRanfomMeasurements.size(); i++)
    {
    	glVertex3f(vRanfomMeasurements[i].x, vRanfomMeasurements[i].y, vRanfomMeasurements[i].z);
    }
    glEnd();

    glLineWidth(5);

    glBegin(GL_LINES);
    	glVertex3f(winningPlane.polygon.vertexA.x, winningPlane.polygon.vertexA.y, winningPlane.polygon.vertexA.z);
    	glVertex3f(winningPlane.polygon.vertexB.x, winningPlane.polygon.vertexB.y, winningPlane.polygon.vertexB.z);

    	glVertex3f(winningPlane.polygon.vertexB.x, winningPlane.polygon.vertexB.y, winningPlane.polygon.vertexB.z);
		glVertex3f(winningPlane.polygon.vertexC.x, winningPlane.polygon.vertexC.y, winningPlane.polygon.vertexC.z);

		glVertex3f(winningPlane.polygon.vertexA.x, winningPlane.polygon.vertexA.y, winningPlane.polygon.vertexA.z);
		glVertex3f(winningPlane.polygon.vertexC.x, winningPlane.polygon.vertexC.y, winningPlane.polygon.vertexC.z);
    glEnd();

    glEnable(GL_LIGHTING);

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
        case 'a' :
        {
        	single_laser_beam_CPU.position.x -= 0.1;
        	single_laser_beam_GPU.position.x -= 0.1;
        	break;
        }
        case 'd' :
        {
        	single_laser_beam_CPU.position.x += 0.1;
        	single_laser_beam_GPU.position.x += 0.1;
          	break;
        }
        case 'w':
        {
        	single_laser_beam_CPU.position.y += 0.1;
        	single_laser_beam_GPU.position.y += 0.1;
           	break;
        }
        case 's':
		{
			single_laser_beam_CPU.position.y -= 0.1;
			single_laser_beam_GPU.position.y -= 0.1;
			break;
		}
        case 'r':
		{
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
	std::cout << "press 'awsd': to move ray" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}

bool loadTrianglesFromHeightmap(std::string _filename, std::vector<triangle> &_vtriangles, std::vector<Eigen::Vector3f> &_vNormals)
{
	cv::Mat image;
	image = cv::imread( _filename.c_str(), 1 );

	for(int row = 0; row < image.rows-1; row++)
	{
		for(int col = 0; col < image.cols-1; col++)
		{
			int temp00 = image.at<char>(row, col);
			int temp01 = image.at<char>(row, col+1);
			int temp10 = image.at<char>(row+1, col);
			int temp11 = image.at<char>(row+1, col+1);

			triangle tr;
			tr.vertexA.x = row;
			tr.vertexA.y = col;
			tr.vertexA.z = (float(temp00))/10.0f;

			tr.vertexB.x = row;
			tr.vertexB.y = col+1;
			tr.vertexB.z = (float(temp01))/10.0f;

			tr.vertexC.x = row+1;
			tr.vertexC.y = col;
			tr.vertexC.z = (float(temp10))/10.0f;

			_vtriangles.push_back(tr);
			Eigen::Vector3f nv;
			Eigen::Vector3f va(tr.vertexB.x - tr.vertexA.x, tr.vertexB.y - tr.vertexA.y, tr.vertexB.z - tr.vertexA.z);
			Eigen::Vector3f vb(tr.vertexC.x - tr.vertexA.x, tr.vertexC.y - tr.vertexA.y, tr.vertexC.z - tr.vertexA.z);
			nv = vb.cross(va);

			nv/=nv.norm();
			vNormals.push_back(nv);


			tr.vertexA.x = row;
			tr.vertexA.y = col+1;
			tr.vertexA.z = (float(temp01))/10.0f;

			tr.vertexB.x = row+1;
			tr.vertexB.y = col;
			tr.vertexB.z = (float(temp10))/10.0f;

			tr.vertexC.x = row+1;
			tr.vertexC.y = col+1;
			tr.vertexC.z = (float(temp11))/10.0f;

			_vtriangles.push_back(tr);
			va = Eigen::Vector3f(tr.vertexB.x - tr.vertexA.x, tr.vertexB.y - tr.vertexA.y, tr.vertexB.z - tr.vertexA.z);
			vb = Eigen::Vector3f(tr.vertexC.x - tr.vertexA.x, tr.vertexC.y - tr.vertexA.y, tr.vertexC.z - tr.vertexA.z);
			nv = va.cross(vb);

			nv/=nv.norm();
			vNormals.push_back(nv);
		}
	}
return true;
}

std::vector<plane> computePlanes(std::vector<triangle> &_vTriangles, std::vector<Eigen::Vector3f> &_vNormals)
{
	std::vector<plane> out_planes;

	for(size_t i = 0 ; i < _vTriangles.size(); i++)
	{
		plane out_plane;
		out_plane.A = _vNormals[i].x();
		out_plane.B = _vNormals[i].y();
		out_plane.C = _vNormals[i].z();
		out_plane.polygon.vertexA = _vTriangles[i].vertexA;
		out_plane.polygon.vertexB = _vTriangles[i].vertexB;
		out_plane.polygon.vertexC = _vTriangles[i].vertexC;
		out_plane.D = - (out_plane.polygon.vertexA.x * out_plane.A + out_plane.polygon.vertexA.y * out_plane.B + out_plane.polygon.vertexA.z * out_plane.C);
		out_planes.push_back(out_plane);
	}
	return out_planes;
}

void computeDistance(laser_beam &_single_laser_beam_CPU, std::vector<plane> &_vPlanes)
{
	float min_distance = _single_laser_beam_CPU.range;
	winningPlane = _vPlanes[0];

	for(size_t i = 0; i < _vPlanes.size(); i ++)
	{
		_single_laser_beam_CPU.distance = _single_laser_beam_CPU.range;
		simple_point3D p_rayIntersection = rayIntersection(_single_laser_beam_CPU, _vPlanes[i]);

		if(pointInPolygon(p_rayIntersection, _vPlanes[i].polygon.vertexA, _vPlanes[i].polygon.vertexB, _vPlanes[i].polygon.vertexC))
		{
			float dist = sqrt((_single_laser_beam_CPU.position.x - p_rayIntersection.x) * (_single_laser_beam_CPU.position.x - p_rayIntersection.x) +
									(_single_laser_beam_CPU.position.y - p_rayIntersection.y) * (_single_laser_beam_CPU.position.y - p_rayIntersection.y) +
									(_single_laser_beam_CPU.position.z - p_rayIntersection.z) * (_single_laser_beam_CPU.position.z - p_rayIntersection.z));

			simple_point3D p_temp;
			p_temp.x = _single_laser_beam_CPU.position.x + _single_laser_beam_CPU.direction.x * dist;
			p_temp.y = _single_laser_beam_CPU.position.y + _single_laser_beam_CPU.direction.y * dist;
			p_temp.z = _single_laser_beam_CPU.position.z + _single_laser_beam_CPU.direction.z * dist;

			if( p_temp.x == p_rayIntersection.x && p_temp.y == p_rayIntersection.y && p_temp.z == p_rayIntersection.z)
			{
				if(dist < min_distance)
				{
					min_distance = dist;
					winningPlane = _vPlanes[i];
				}
			}
		}
	}
	_single_laser_beam_CPU.distance = min_distance;
}

float distanceToPlane(plane _plane, simple_point3D _point3D)
{
	return (_plane.A * _point3D.x + _plane.B * _point3D.y + _plane.C * _point3D.z + _plane.D);
}

simple_point3D rayIntersection(laser_beam &_laser_beam, plane &_plane)
{
	simple_point3D out_point;
	out_point.x = _laser_beam.position.x;
	out_point.y = _laser_beam.position.y;
	out_point.z = _laser_beam.position.z;

	float a = _plane.A * _laser_beam.direction.x + _plane.B * _laser_beam.direction.y + _plane.C * _laser_beam.direction.z	;

	if ( a > -TOLERANCE && a < TOLERANCE)
	{
		return out_point;
	}

	float distance = distanceToPlane(_plane, out_point);

	out_point.x = _laser_beam.position.x - _laser_beam.direction.x * (distance/a);
	out_point.y = _laser_beam.position.y - _laser_beam.direction.y * (distance/a);
	out_point.z = _laser_beam.position.z - _laser_beam.direction.z * (distance/a);

	return out_point;
}


bool pointInPolygon(simple_point3D p, simple_point3D v0, simple_point3D v1, simple_point3D v2)
{
	simple_point3D segment1, segment2;
	double sumAngles = 0.0;
	double cosAngle = 0.0;

	segment1.x = v0.x - p.x;
	segment1.y = v0.y - p.y;
	segment1.z = v0.z - p.z;
	segment2.x = v1.x - p.x;
	segment2.y = v1.y - p.y;
	segment2.z = v1.z - p.z;

	float l1 = sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z);
	float l2 = sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z);


	if (l1 * l2 <=  TOLERANCE)
	{
		return true;
	}

	cosAngle = (segment1.x*segment2.x + segment1.y*segment2.y + segment1.z*segment2.z)/
			(sqrt(segment1.x*segment1.x + segment1.y*segment1.y + segment1.z*segment1.z) *
			 sqrt(segment2.x*segment2.x + segment2.y*segment2.y + segment2.z*segment2.z));
	sumAngles += acos(cosAngle);

	segment1.x = v1.x - p.x;
	segment1.y = v1.y - p.y;
	segment1.z = v1.z - p.z;
	segment2.x = v2.x - p.x;
	segment2.y = v2.y - p.y;
	segment2.z = v2.z - p.z;
	l1 = sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z);
	l2 = sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z);

	if (l1 * l2 <=  TOLERANCE)
	{
		return true;
	}

	cosAngle = (segment1.x*segment2.x + segment1.y*segment2.y + segment1.z*segment2.z)/
			(sqrt(segment1.x*segment1.x + segment1.y*segment1.y + segment1.z*segment1.z) *
			 sqrt(segment2.x*segment2.x + segment2.y*segment2.y + segment2.z*segment2.z));
	sumAngles += acos(cosAngle);

	segment1.x = v2.x - p.x;
	segment1.y = v2.y - p.y;
	segment1.z = v2.z - p.z;
	segment2.x = v0.x - p.x;
	segment2.y = v0.y - p.y;
	segment2.z = v0.z - p.z;
	l1 = sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z);
	l2 = sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z);

	if (l1 * l2 <=  TOLERANCE)
	{
		return true;
	}

	cosAngle = (segment1.x * segment2.x + segment1.y * segment2.y + segment1.z * segment2.z)/
			(sqrt(segment1.x * segment1.x + segment1.y * segment1.y + segment1.z * segment1.z) *
			 sqrt(segment2.x * segment2.x + segment2.y * segment2.y + segment2.z * segment2.z));
	sumAngles += acos(cosAngle);

	if((sumAngles <= (6.283185307179586476925287 + TOLERANCE)) && (sumAngles >= (6.283185307179586476925287 - TOLERANCE)))
	{
		return true;
	}
	else
		return false;
}


