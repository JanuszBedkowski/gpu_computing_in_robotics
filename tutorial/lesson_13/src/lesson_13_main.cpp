#include <GL/freeglut.h>
#include <iostream>

#include <Eigen/Geometry>

#include "cudaWrapper.h"

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -20.0;
float translate_x, translate_y = 0.0;

CCudaWrapper cudaWrapper;

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();

local_observations_t observations_reference;
local_observations_t observations_to_register;
local_observations_t observations_to_register_temp;

void renderPose(Eigen::Affine3f &pose);

void initData();
void renderData();
void transformPose(float om, float fi, float ka, float x, float y, float z, local_observations_t &out);
void registerDataCUDA();
plane_t transformPlane(plane_t plane, Eigen::Affine3f m);

int main(int argc, char **argv)
{
	cudaWrapper.warmUpGPU();

	initData();

	if (false == initGL(&argc, argv))
	{
		return -1;
	}

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
    glutCreateWindow("Lesson 13 - data registration Plane To Plane");
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

    renderData();

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
        	transformPose(0.0f, 0.0f, 1.0f* (M_PI/180.0f), 0.0f, 0.0f, 0.0f, observations_to_register);
        	break;
        }
        case 'd' :
        {
        	transformPose(0.0f, 0.0f, -1.0f* (M_PI/180.0f), 0.0f, 0.0f, 0.0f, observations_to_register);
        	break;
        }
        case 'w':
        {
        	transformPose(0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, observations_to_register);
           	break;
        }
        case 's':
		{
			transformPose(0.0f, 0.0f, 0.0f, -0.1f, 0.0f, 0.0f, observations_to_register);
			break;
		}
        case 'f':
		{
			transformPose(0.0f, 0.0f, 0.0f, 0.1f, -0.1f, 0.0f, observations_to_register);
			break;
		}
        case 'g':
		{
			transformPose(0.0f, 0.0f, 0.0f, 0.f, 0.1f, 0.0f, observations_to_register);
			break;
		}
        case 'c':
		{
			clock_t begin_time;
			double computation_time;
			begin_time = clock();
				registerDataCUDA();
			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "registerdata GPU computation_time: " << computation_time << std::endl;
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
	std::cout << "press 'a': rotate left observations_to_register (1 degree)" << std::endl;
	std::cout << "press 'd': rotate right observations_to_register (1 degree)" << std::endl;
	std::cout << "press 'w': translate forward observations_to_register (0.1 meter)" << std::endl;
	std::cout << "press 's': translate backward observations_to_register (0.1 meter)" << std::endl;
	std::cout << "press 'f': translate left observations_to_register (0.1 meter)" << std::endl;
	std::cout << "press 'g': translate right observations_to_register (0.1 meter)" << std::endl;
	std::cout << "press 'c': register data GPU" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}

void renderPose(Eigen::Affine3f &pose)
{
	Eigen::Vector3f _v0(0.0f, 0.0f, 0.0f);
	Eigen::Vector3f _vX(1.0f, 0.0f, 0.0f);
	Eigen::Vector3f _vY(0.0f, 1.0f, 0.0f);
	Eigen::Vector3f _vZ(0.0f, 0.0f, 1.0f);

	Eigen::Vector3f v0t, vXt, vYt, vZt;
	v0t = pose * _v0;
	vXt = pose * _vX;
	vYt = pose * _vY;
	vZt = pose * _vZ;

	glBegin(GL_LINES);
		glColor3f(1.0f , 0.0f, 0.0f);
		glVertex3f(v0t.x(), v0t.y(), v0t.z());
		glVertex3f(vXt.x(), vXt.y(), vXt.z());

		glColor3f(0.0f , 1.0f, 0.0f);
		glVertex3f(v0t.x(), v0t.y(), v0t.z());
		glVertex3f(vYt.x(), vYt.y(), vYt.z());

		glColor3f(0.0f , 0.0f, 1.0f);
		glVertex3f(v0t.x(), v0t.y(), v0t.z());
		glVertex3f(vZt.x(), vZt.y(), vZt.z());
	glEnd();
}

void initData()
{
	Eigen::Affine3f mR, mT;

	observations_reference.om = 0.0f * (M_PI/180.0f);
	observations_reference.fi = 0.0f * (M_PI/180.0f);
	observations_reference.ka = 15.0f * (M_PI/180.0f);
	observations_reference.tx = 1.0f;
	observations_reference.ty = 0.0f;
	observations_reference.tz = 0.0f;
	mR = Eigen::AngleAxisf(observations_reference.om, Eigen::Vector3f::UnitX())
		  * Eigen::AngleAxisf(observations_reference.fi, Eigen::Vector3f::UnitY())
		  * Eigen::AngleAxisf(observations_reference.ka, Eigen::Vector3f::UnitZ());
	mT = Eigen::Translation3f(observations_reference.tx, observations_reference.ty, observations_reference.tz);
	observations_reference.m_pose = mT * mR;

	observations_to_register.om = 0.0f * (M_PI/180.0f);
	observations_to_register.fi = 0.0f * (M_PI/180.0f);
	observations_to_register.ka = 25.0f * (M_PI/180.0f);
	observations_to_register.tx = 3.0f;
	observations_to_register.ty = 0.0f;
	observations_to_register.tz = 0.0f;
	mR = Eigen::AngleAxisf(observations_to_register.om, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(observations_to_register.fi, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(observations_to_register.ka, Eigen::Vector3f::UnitZ());
	mT = Eigen::Translation3f(observations_to_register.tx, observations_to_register.ty, observations_to_register.tz);
	observations_to_register.m_pose = mT * mR;

	std::vector<plane_t> temp_planes;

	for(int i = 0 ; i < 100; i++)
	{
		plane_t temp_plane;
		float angle1 = (float(rand()%1000000))/1000000.0f * (M_PI);
		float angle2 = (float(rand()%1000000))/1000000.0f * (M_PI);
		float angle3 = (float(rand()%1000000))/1000000.0f * (M_PI);

		Eigen::Affine3f mR;

		mR = Eigen::AngleAxisf(angle1, Eigen::Vector3f::UnitX())
		  * Eigen::AngleAxisf(angle2, Eigen::Vector3f::UnitY())
		  * Eigen::AngleAxisf(angle3, Eigen::Vector3f::UnitZ());

		temp_plane.nx = mR.matrix()(0,0);
		temp_plane.ny = mR.matrix()(1,0);
		temp_plane.nz = mR.matrix()(2,0);
		temp_plane.rho = (float(rand()%1000000))/1000000.0f * 100;

		temp_planes.push_back(temp_plane);
	}

	Eigen::Affine3f m_pose = observations_reference.m_pose;
	Eigen::Affine3f m_poseInv = observations_reference.m_pose.inverse();


	for(size_t i = 0 ; i < temp_planes.size(); i++)
	{
		plane_t temp_plane = temp_planes[i];
		plane_t plane = transformPlane(temp_plane, m_poseInv);
		observations_reference.planes.push_back(plane);
	}

	m_poseInv = observations_to_register.m_pose.inverse();
	for(size_t i = 0 ; i < temp_planes.size(); i++)
	{
		plane_t temp_plane = temp_planes[i];
		plane_t plane = transformPlane(temp_plane, m_poseInv);
		observations_to_register.planes.push_back(plane);
	}

	std::cout << "initial pose:" << std::endl;
	std::cout << observations_to_register.om << " " << observations_to_register.fi << " " << observations_to_register.ka << " " << observations_to_register.tx << " " << observations_to_register.ty << " " << observations_to_register.tz << std::endl;

	observations_to_register_temp = observations_to_register;
}

void renderData()
{
	renderPose(observations_reference.m_pose);
	renderPose(observations_to_register.m_pose);

	Eigen::Affine3f m_ref = observations_reference.m_pose;
	Eigen::Affine3f m_reg = observations_to_register.m_pose;

	glBegin(GL_LINES);
	for(size_t i = 0 ; i < observations_reference.planes.size(); i++)
	{
		plane_t plane_ref = observations_reference.planes[i];
		plane_t plane_reg = observations_to_register.planes[i];
		plane_t plane_ref_t = transformPlane(plane_ref, m_ref);
		plane_t plane_reg_t = transformPlane(plane_reg, m_reg);

		float v1[3];
		float v2[3];
		v1[0] = plane_ref_t.rho * plane_ref_t.nx;
		v1[1] = plane_ref_t.rho * plane_ref_t.ny;
		v1[2] = plane_ref_t.rho * plane_ref_t.nz;

		v2[0] = (plane_ref_t.rho + 1.0f) * plane_ref_t.nx;
		v2[1] = (plane_ref_t.rho + 1.0f) * plane_ref_t.ny;
		v2[2] = (plane_ref_t.rho + 1.0f) * plane_ref_t.nz;

		glColor3f(fabs(plane_ref_t.nx), fabs(plane_ref_t.ny), fabs(plane_ref_t.nz));
		glVertex3fv(v1);
		glVertex3fv(v2);

		float v1a[3];
		float v2a[3];
		v1a[0] = plane_reg_t.rho * plane_reg_t.nx;
		v1a[1] = plane_reg_t.rho * plane_reg_t.ny;
		v1a[2] = plane_reg_t.rho * plane_reg_t.nz;

		v2a[0] = (plane_reg_t.rho + 1.0f) * plane_reg_t.nx;
		v2a[1] = (plane_reg_t.rho + 1.0f) * plane_reg_t.ny;
		v2a[2] = (plane_reg_t.rho + 1.0f) * plane_reg_t.nz;

		glColor3f(fabs(plane_reg_t.nx) *0.5f, fabs(plane_reg_t.ny) *0.5f, fabs(plane_reg_t.nz) * 0.5f);
		glVertex3fv(v1a);
		glVertex3fv(v2a);

		glColor3f(0.0f, 1.0f , 0.0f);
		glVertex3fv(v1);
		glVertex3fv(v1a);
	}
	glEnd();
}

void transformPose(float om, float fi, float ka, float x, float y, float z, local_observations_t &out)
{
	out.om += om;
	out.fi += fi;
	out.ka += ka;
	out.tx += x;
	out.ty += y;
	out.tz += z;

	Eigen::Affine3f mR, mT;

	mR = Eigen::AngleAxisf(out.om, Eigen::Vector3f::UnitX())
			  * Eigen::AngleAxisf(out.fi, Eigen::Vector3f::UnitY())
			  * Eigen::AngleAxisf(out.ka, Eigen::Vector3f::UnitZ());

	mT = Eigen::Translation3f(out.tx, out.ty, out.tz);

	out.m_pose = mT * mR;

	std::cout << "current pose:" << std::endl;
	std::cout << out.om << " " << out.fi << " " << out.ka << " " << out.tx << " " << out.ty << " " << out.tz << std::endl;

}

void registerDataCUDA()
{
	std::cout << "initial pose:" << std::endl;
	std::cout << observations_to_register_temp.om << " " << observations_to_register_temp.fi << " " << observations_to_register_temp.ka << " " << observations_to_register_temp.tx << " " << observations_to_register_temp.ty << " " << observations_to_register_temp.tz << std::endl;

	std::cout << "current pose:" << std::endl;
	std::cout << observations_to_register.om << " " << observations_to_register.fi << " " << observations_to_register.ka << " " << observations_to_register.tx << " " << observations_to_register.ty << " " << observations_to_register.tz << std::endl;

	std::vector<plane_t> planes_ref;
	Eigen::Affine3f m_ref = observations_reference.m_pose;
	for(size_t i = 0 ; i < observations_reference.planes.size(); i++)
	{
		plane_t plane_ref = observations_reference.planes[i];
		plane_t plane_ref_t = transformPlane(plane_ref, m_ref);
		planes_ref.push_back(plane_ref_t);
	}

	Eigen::Affine3f mTotal = Eigen::Affine3f::Identity();
	bool computation_succeed = false;

	if(cudaWrapper.computeSingleIterationOfPlanarFeatureMatchingCUDA(
			planes_ref,
			observations_to_register.planes,
			observations_to_register.om,
			observations_to_register.fi,
			observations_to_register.ka,
			observations_to_register.tx,
			observations_to_register.ty,
			observations_to_register.tz
			))
	{
		Eigen::Affine3f mR, mT;
		mR = Eigen::AngleAxisf(observations_to_register.om, Eigen::Vector3f::UnitX())
					  * Eigen::AngleAxisf(observations_to_register.fi, Eigen::Vector3f::UnitY())
					  * Eigen::AngleAxisf(observations_to_register.ka, Eigen::Vector3f::UnitZ());

		mT = Eigen::Translation3f(observations_to_register.tx, observations_to_register.ty, observations_to_register.tz);
		observations_to_register.m_pose = mT * mR;
		std::cout << "computed pose:" << std::endl;
		std::cout << observations_to_register.om << " " << observations_to_register.fi << " " << observations_to_register.ka << " " << observations_to_register.tx << " " << observations_to_register.ty << " " << observations_to_register.tz << std::endl;
	}
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


