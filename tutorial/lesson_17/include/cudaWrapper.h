#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_17.h"
#include <pcl/point_cloud.h>

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();

	void findMax(int tx, int bx, int &newIndexX, int &newIndexY, double *h_dyf, int sizeofmap1dim);
	int computePath(bool *map2D, int sizeofmap1dim, int xgoal, int ygoal, int robotXpos, int robotYpos,
		 int steps, char *_my_grid, int maxpathlength, int *path_x, int *path_y);
};

#endif
