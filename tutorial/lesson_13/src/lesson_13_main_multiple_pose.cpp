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
void motion(int x, int y);
void printHelp();

std::vector<local_observations_t> v_observations;

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
	glutMainLoop();
}

bool initGL(int *argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Lesson 13 - data registration Plane To Plane multiple pose");
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
        case 'n' :
        {
        	for(size_t i = 0 ; i < v_observations.size() ; i++)
			{
        		double om = v_observations[i].om + ((rand()%10000)/10000.0f)* (M_PI/180.0f);
        		double fi = v_observations[i].fi +((rand()%10000)/10000.0f)* (M_PI/180.0f);
        		double ka = v_observations[i].ka +((rand()%10000)/10000.0f)* (M_PI/180.0f);
        		double tx = v_observations[i].tx +((rand()%10000)/10000.0f) * 2.0;
        		double ty = v_observations[i].ty +((rand()%10000)/10000.0f) * 2.0;
        		double tz = v_observations[i].tz +((rand()%10000)/10000.0f) * 2.0;

        		Eigen::Affine3f mR, mT, m_pose;
				mR = Eigen::AngleAxisf(om, Eigen::Vector3f::UnitX())
							  * Eigen::AngleAxisf(fi, Eigen::Vector3f::UnitY())
							  * Eigen::AngleAxisf(ka, Eigen::Vector3f::UnitZ());

				mT = Eigen::Translation3f(tx, ty, tz);

				m_pose = mT * mR;

				v_observations[i].m_pose = m_pose;
				v_observations[i].om = om;
				v_observations[i].fi = fi;
				v_observations[i].ka = ka;
				v_observations[i].tx = tx;
				v_observations[i].ty = ty;
				v_observations[i].tz = tz;

			}
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
	std::cout << "press 'n': add noise to poses" << std::endl;
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
	for(int i = -10; i < 10; i+=2)
	{
		Eigen::Affine3f mR, mT;
		local_observations_t obs;
		obs.om = 0.0;
		obs.fi = 0.0;
		obs.ka = 0.0;
		obs.tx = i;
		obs.ty = -1.0;
		obs.tz = 0.0;
		mR = Eigen::AngleAxisf(obs.om, Eigen::Vector3f::UnitX())
				  * Eigen::AngleAxisf(obs.fi, Eigen::Vector3f::UnitY())
				  * Eigen::AngleAxisf(obs.ka, Eigen::Vector3f::UnitZ());
		mT = Eigen::Translation3f(obs.tx, obs.ty, obs.tz);
		obs.m_pose = mT * mR;

		v_observations.push_back(obs);
	}

	std::vector<plane_t> temp_planes;

	for(int i = 0 ; i < 10; i++)
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

	for(size_t i = 0 ; i < v_observations.size() ; i++)
	{
		for(size_t j = 0 ; j < temp_planes.size(); j++)
		{
			Eigen::Affine3f m_poseInv = v_observations[i].m_pose.inverse();

			plane_t temp_plane = temp_planes[j];
			plane_t plane = transformPlane(temp_plane, m_poseInv);
			v_observations[i].planes.push_back(plane);
		}
	}
}

void renderData()
{
	for(size_t i = 0 ; i < v_observations.size(); i++)
	{
		renderPose(v_observations[i].m_pose);
	}

	for(size_t i = 0 ; i < v_observations.size() ; i++)
	{
		glBegin(GL_LINES);
		for(size_t j = 0 ; j < v_observations[i].planes.size(); j++)
		{
			plane_t plane = v_observations[i].planes[j];
			plane_t plane_t = transformPlane(plane, v_observations[i].m_pose);

			float v1[3];
			float v2[3];
			v1[0] = plane_t.rho * plane_t.nx;
			v1[1] = plane_t.rho * plane_t.ny;
			v1[2] = plane_t.rho * plane_t.nz;

			v2[0] = (plane_t.rho + 1.0f) * plane_t.nx;
			v2[1] = (plane_t.rho + 1.0f) * plane_t.ny;
			v2[2] = (plane_t.rho + 1.0f) * plane_t.nz;

			glColor3f(fabs(plane_t.nx), fabs(plane_t.ny), fabs(plane_t.nz));
			glVertex3fv(v1);
			glVertex3fv(v2);
		}
		glEnd();
	}
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
	std::vector<pair_local_observations_t> v_pair_local_observations;

	for(size_t i = 0 ; i < v_observations.size() ; i++)
	{
		pair_local_observations_t pair;
		pair.m_pose = v_observations[i].m_pose;
		pair.om = v_observations[i].om;
		pair.fi = v_observations[i].fi;
		pair.ka = v_observations[i].ka;
		pair.tx = v_observations[i].tx;
		pair.ty = v_observations[i].ty;
		pair.tz = v_observations[i].tz;

		for(size_t j = 0 ; j < v_observations.size() ; j++)
		{
			if(i!=j)
			{


				for(size_t k = 0 ; k < v_observations[j].planes.size(); k++)
				{
					plane_t plane_ref = v_observations[j].planes[k];
					plane_t plane_ref_t = transformPlane(plane_ref, v_observations[j].m_pose);

					plane_t plane_to_register = v_observations[i].planes[k];
					pair.planes_reference.push_back(plane_ref_t);
					pair.planes_to_register.push_back(plane_to_register);
				}
			}
		}
		v_pair_local_observations.push_back(pair);
	}

	if(cudaWrapper.computeSingleIterationOfPlanarFeatureMatchingMultiplePoseCUDA(
			v_pair_local_observations
				))
	{
		for(size_t i = 0 ; i < v_pair_local_observations.size(); i++)
		{
			Eigen::Affine3f mR, mT;
			mR = Eigen::AngleAxisf(v_pair_local_observations[i].om, Eigen::Vector3f::UnitX())
						  * Eigen::AngleAxisf(v_pair_local_observations[i].fi, Eigen::Vector3f::UnitY())
						  * Eigen::AngleAxisf(v_pair_local_observations[i].ka, Eigen::Vector3f::UnitZ());

			mT = Eigen::Translation3f(v_pair_local_observations[i].tx, v_pair_local_observations[i].ty, v_pair_local_observations[i].tz);

			v_observations[i].m_pose = mT * mR;
			v_observations[i].om = v_pair_local_observations[i].om;
			v_observations[i].fi = v_pair_local_observations[i].fi;
			v_observations[i].ka = v_pair_local_observations[i].ka;
			v_observations[i].tx = v_pair_local_observations[i].tx;
			v_observations[i].ty = v_pair_local_observations[i].ty;
			v_observations[i].tz = v_pair_local_observations[i].tz;
		}
	}else
	{
		std::cout << "PROBLEM: cudaWrapper.computeSingleIterationOfPlanarFeatureMatchingMultipleCaseCUDA return false" << std::endl;
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


