#ifndef __CUDAWRAPPER__
#define __CUDAWRAPPER__

#include "lesson_13.h"

#define _11 0
#define _12 1
#define _13 2
#define _21 3
#define _22 4
#define _23 5
#define _31 6
#define _32 7
#define _33 8

class CCudaWrapper
{
public:
	CCudaWrapper();
	~CCudaWrapper();

	void warmUpGPU();
	int getNumberOfAvailableThreads();

	bool computeSingleIterationOfPlanarFeatureMatchingCUDA(std::vector<plane> planes1, std::vector<plane> planes2,
				double &om, double &fi, double &ka, double &tx, double &ty, double &tz);

	void fillA_PlanarFeatureMatching(double *A, double om, double fi, double ka, double tx, double ty, double tz,
			std::vector<plane> &planes1, std::vector<plane> &planes2);

	void fill_delta_PlanarFeatureMatching(double *delta, double om, double fi, double ka, double tx, double ty, double tz,
				std::vector<plane> &planes1, std::vector<plane> &planes2);

	void computeR(double om, double fi, double ka, double *R);
	double compute_a10(double *r, double x0, double y0, double z0);
	double compute_a20(double *r, double x0, double y0, double z0);
	double compute_a30(double *r, double x0, double y0, double z0);
	double compute_a11();
	double compute_a12(double m, double om, double fi, double ka, double x0, double y0, double z0);
	double compute_a13(double m, double *r, double x0, double y0);
	double compute_a21(double m, double *r, double x0, double y0, double z0);
	double compute_a22(double m, double om, double fi, double ka, double x0, double y0, double z0);
	double compute_a23(double m, double *r, double x0, double y0);
	double compute_a31(double m, double *r, double x0, double y0, double z0);
	double compute_a32(double m, double om, double fi, double ka, double x0, double y0, double z0);
	double compute_a33(double m, double *r, double x0, double y0);

	plane_t transformPlane(plane_t plane, Eigen::Affine3f m);

	bool computeSingleIterationOfPlanarFeatureMatchingMultiplePoseCUDA(std::vector<pair_local_observations_t> &v_pair_local_observations );

	int threads;
};



#endif
