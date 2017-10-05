#include <GL/freeglut.h>
#include "cudaWrapper.h"

const unsigned int window_width  = 512;
const unsigned int window_height = 512;
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -2.0;
float translate_x, translate_y = 0.0;

CCudaWrapper cudaWrapper;

int sizePathMap = 256;
int robotX = 1;
int robotY = 1;
int goalX = 254;
int goalY = 245;
unsigned char *map = 0;
int xpath[10000];
int ypath[10000];
int resPath;
unsigned int texPath=0;

bool initGL(int *argc, char **argv);
void display();
void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void reshape(int width, int height);
void motion(int x, int y);
void printHelp();
void computePath();
void updatePathTexture();
void genCase(int _case);

int main(int argc, char **argv)
{
	if (false == initGL(&argc, argv))
	{
		return -1;
	}

	map = new unsigned char [sizePathMap*sizePathMap*3];
	for(int i = 0; i < sizePathMap; i++)
	{
		for(int j = 0; j < sizePathMap; j++)
		{
			map[3*(i+j*sizePathMap)] = 255;
			map[3*(i+j*sizePathMap)+1] = 255;
			map[3*(i+j*sizePathMap)+2] = 255;
		}
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
    glutCreateWindow("Lesson 17 - path planning (via diffusion process)");
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

    glPushAttrib(GL_ALL_ATTRIB_BITS);
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);

		glColor3f(1,1,1);

		glBindTexture(GL_TEXTURE_2D, texPath);
		glBegin(GL_QUADS);
		glNormal3f( 0.0f, 0.0f, 0.5f);
			 glNormal3f( 0.0f, 1.0f, 0.0f);
			 glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f, -1.0f, 0.0f);
			 glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f, 0.0f);
			 glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f, 0.0f);
			 glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f, -1.0f, 0.0f);
		glEnd();
	glPopAttrib();

	float x = float(((robotX *2) - sizePathMap)) / (float)sizePathMap;
	float y = (float)(  (sizePathMap- robotY - 1) *2 - sizePathMap) / (float)sizePathMap;

	glBegin(GL_LINES);
		glColor3f( 1.0f, 0.0f, 0.0f);

		glVertex3f(x,y,0.0f);
		glVertex3f(x,y,0.2f);
	glEnd();

	x = float(((goalX *2) - sizePathMap)) / (float)sizePathMap;
	y = (float)(  (sizePathMap - goalY - 1) *2 - sizePathMap) / (float)sizePathMap;

	glBegin(GL_LINES);
		glColor3f( 0.0f, 0.0f, 1.0f);

		glVertex3f(x,y,0.0f);
		glVertex3f(x,y,0.2f);
	glEnd();

	glutSwapBuffers();
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
        case (27) :
            glutDestroyWindow(glutGetWindow());
        	delete [] map;
            return;
        case 'p' :
        {
        	clock_t begin_time;
			double computation_time;
			begin_time = clock();

			computePath();

			computation_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
			std::cout << "computePath() computation_time: " << computation_time << std::endl;
        	break;
        }
        case '1':
        {
        	genCase(1);
        	break;
        }
        case '2':
		{
			genCase(2);
			break;
		}
		case '3':
		{
			genCase(3);
			break;
		}
		case '4':
		{
			genCase(4);
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
	std::cout << "press '1': case 1" << std::endl;
	std::cout << "press '2': case 2" << std::endl;
	std::cout << "press '3': case 3" << std::endl;
	std::cout << "press '4': case 4 (random obstacles)" << std::endl;
	std::cout << "press 'p': compute path" << std::endl;
	std::cout << "press 'Esc': EXIT" << std::endl;
}

void computePath()
{
	bool *map2D = new bool [sizePathMap*sizePathMap];

	for(int x=0 ; x < sizePathMap; x++)
	{
		for(int z=0 ; z < sizePathMap; z++)
		{
			if(map[3*(x+z*sizePathMap)] == 0) map2D[x+z*sizePathMap] = 1;
			if(map[3*(x+z*sizePathMap)] == 100) map2D[x+z*sizePathMap] = 1;
			if(map[3*(x+z*sizePathMap)] == 255) map2D[x+z*sizePathMap] = 0;
		}
	}

	char *map2DtoDraw = new char[sizePathMap*sizePathMap];
	for(int i = 0; i< sizePathMap*sizePathMap; i++)map2DtoDraw[i] = 255;

	int _PATH_MAX_LENGTH = 10000;

	resPath = cudaWrapper.computePath(map2D, sizePathMap, goalX, goalY, robotX, robotY, sizePathMap, map2DtoDraw, _PATH_MAX_LENGTH, xpath, ypath);

	//for(int i = 0 ; i< resPath; i++)
	//{
	//	printf("path node: %d x%d y%d\n", i, xpath[i] ,ypath[i]);
	//}
	printf("path length: %d\n", resPath);

	for(int x=0 ; x < sizePathMap; x++)
	{
		for(int z=0 ; z < sizePathMap; z++)
		{
			map[3*(x+z*sizePathMap)] = map2DtoDraw[x+z*sizePathMap];
			map[3*(x+z*sizePathMap)+1] = map2DtoDraw[x+z*sizePathMap];
			map[3*(x+z*sizePathMap)+2] = map2DtoDraw[x+z*sizePathMap];
		}
	}

	delete [] map2D;
	delete [] map2DtoDraw;
	updatePathTexture();
}

void updatePathTexture()
{
	if(texPath)
		glDeleteTextures(1, &texPath);
	glGenTextures(1,&texPath);
	glBindTexture(GL_TEXTURE_2D, texPath);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,sizePathMap,sizePathMap,0,GL_RGB,GL_UNSIGNED_BYTE,map);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,GL_REPEAT);
}

void genCase(int _case)
{
	//TRAVERSABLE
	for(int i = 0; i < sizePathMap; i++)
	{
		for(int j = 0; j < sizePathMap; j++)
		{
			map[3*(i+j*sizePathMap)] = 255;
			map[3*(i+j*sizePathMap)+1] = 255;
			map[3*(i+j*sizePathMap)+2] = 255;
		}
	}

	//NOTTRAVERSABLE
	switch(_case)
	{
		case 1:
		{
			robotX = 1;
			robotY = 1;
			goalX = 254;
			goalY = 245;

			for(int i = 50; i < 200; i++)
			{
				for(int j = 40; j < 200; j++)
				{
					map[3*(i+j*sizePathMap)] = 0;
					map[3*(i+j*sizePathMap)+1] = 0;
					map[3*(i+j*sizePathMap)+2] = 0;
				}
			}
			break;
		}
		case 2:
		{
			robotX = 1;
			robotY = 50;
			goalX = 254;
			goalY = 200;

			for(int i = 10; i < 20; i++)
			{
				for(int j = 40; j < 200; j++)
				{
					map[3*(i+j*sizePathMap)] = 0;
					map[3*(i+j*sizePathMap)+1] = 0;
					map[3*(i+j*sizePathMap)+2] = 0;
				}
			}

			for(int i = 180; i < 250; i++)
			{
				for(int j = 140; j < 200; j++)
				{
					map[3*(i+j*sizePathMap)] = 0;
					map[3*(i+j*sizePathMap)+1] = 0;
					map[3*(i+j*sizePathMap)+2] = 0;
				}
			}
			break;
		}
		case 3:
		{
			robotX = 1;
			robotY = 10;
			goalX = 254;
			goalY = 200;

			for(int i = 10; i < 200; i++)
			{
				for(int j = 40; j < 50; j++)
				{
					map[3*(i+j*sizePathMap)] = 0;
					map[3*(i+j*sizePathMap)+1] = 0;
					map[3*(i+j*sizePathMap)+2] = 0;
				}
			}

			for(int i = 180; i < 250; i++)
			{
				for(int j = 140; j < 200; j++)
				{
					map[3*(i+j*sizePathMap)] = 0;
					map[3*(i+j*sizePathMap)+1] = 0;
					map[3*(i+j*sizePathMap)+2] = 0;
				}
			}
			break;
		}
		case 4:
		{
			robotX = 1;
			robotY = rand()%254;
			goalX = 254;
			goalY = rand()%254;

			for(int k = 0 ; k < 10; k++)
			{
				int indx = rand()%190 + 10;
				int indy = rand()%190 + 10;

				for(int i = indx; i < indx+50; i++)
				{
					for(int j = indy; j < indy+50; j++)
					{
						map[3*(i+j*sizePathMap)] = 0;
						map[3*(i+j*sizePathMap)+1] = 0;
						map[3*(i+j*sizePathMap)+2] = 0;
					}
				}
			}
			break;
		}
	}
	updatePathTexture();
}


