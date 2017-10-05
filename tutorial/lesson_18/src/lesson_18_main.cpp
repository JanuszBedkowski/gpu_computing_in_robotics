#include <GL/freeglut.h>
#include "GPUMatching.h"
#include "BFROSTMatcher.h"

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -2.0;
float translate_x, translate_y = 0.0;

CBFROSTMatcher matcher;

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();

int main(int argc, char **argv)
{
	if (false == initGL(&argc, argv))
	{
		return -1;
	}

	if(argc != 3)
	{
		std::cout << "USAGE: " << argv[0] << " file1 file2" << std::endl;
		std::cout << "Default: " << argv[0] << " ../data/1.jpg ../data/2.jpg" << std::endl;
		matcher.loadPhotos("../data/1.jpg", "../data/2.jpg");
	}else
	{
		matcher.loadPhotos(argv[1], argv[2]);
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
    glutCreateWindow("Lesson 18 - BFROST: Binary Features from Robust Orientation Segment Tests");
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

    glClearColor(1.0, 1.0, 1.0, 1.0);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_2D);

    return true;
}

void display()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	matcher.paintGL();

	glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            glutDestroyWindow(glutGetWindow());
            return;
        case 'm' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

			matcher.computeKeypointsWithRotation();

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "matching() computation_time: " << computation_time << std::endl;
        	break;
        }
        case 'r':
        {
        	matcher.showROIForDescriptors = !matcher.showROIForDescriptors;
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
	std::cout << "press 'm': matching" << std::endl;
	std::cout << "press 'r': show ROI on|off" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}
