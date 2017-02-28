#include "cudaWrapper.h"
#include <iostream>

#include "CCUDAAXBSolverWrapper.h"

CCudaWrapper::CCudaWrapper()
{

}

CCudaWrapper::~CCudaWrapper()
{

}

void CCudaWrapper::warmUpGPU()
{
	cudaError_t err = ::cudaSuccess;
	err = cudaSetDevice(0);
		if(err != ::cudaSuccess)return;

	err = cudaWarmUpGPU();
		if(err != ::cudaSuccess)return;

	this->threads = getNumberOfAvailableThreads();
}

int CCudaWrapper::getNumberOfAvailableThreads()
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);

	int threads = 0;
	if(prop.major == 2)
	{
		threads=prop.maxThreadsPerBlock/2;
	}else if(prop.major > 2)
	{
		threads=prop.maxThreadsPerBlock;
	}else
	{
		return 0;
	}

	return threads;
}

bool CCudaWrapper::computeSingleIterationOfPlanarFeatureMatchingCUDA(std::vector<plane> planes1, std::vector<plane> planes2,
		double &om, double &fi, double &ka, double &tx, double &ty, double &tz)
{
	int _size = planes1.size();

	double *A=(double*)malloc( _size* 6 * 4 * sizeof(double));
	double *delta = (double*)malloc( _size * 4 * sizeof(double));
	double *x = (double *)malloc (6 * sizeof(double));
	double *P=(double*)malloc(_size* 6 *sizeof(double));
	for(int i = 0 ; i < _size* 6; i++)P[i] = 1.0;

	fillA_PlanarFeatureMatching(A, om, fi, ka, tx, ty, tz,
				planes1, planes2);

	fill_delta_PlanarFeatureMatching(delta, om, fi, ka, tx, ty, tz,
			planes1, planes2);

	cudaError_t errCUDA = ::cudaSuccess;
	double *d_A = NULL;
	errCUDA  = cudaMalloc((void**)&d_A,  _size* 6 * 4 *sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}
	errCUDA = cudaMemcpy(d_A, A, _size* 6 * 4 *sizeof(double), cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){return false;}

	double *d_P = NULL;
	errCUDA  = cudaMalloc((void**)&d_P,  _size* 6 *sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}
	errCUDA = cudaMemcpy(d_P, P, _size* 6 *sizeof(double), cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){return false;}

	double *d_l=NULL;
	errCUDA  = cudaMalloc((void**)&d_l,  _size * 4 * sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}
	errCUDA = cudaMemcpy(d_l, delta, _size * 4 * sizeof(double), cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){return false;}

	CCUDA_AX_B_SolverWrapper * wr = new CCUDA_AX_B_SolverWrapper(false, 0);

	CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper_error errAXB =
					wr->Solve_ATPA_ATPl_x_data_on_GPU(this->threads, d_A, d_P, d_l, x, 6, _size * 4, CCUDA_AX_B_SolverWrapper::chol);
	if(errAXB!=CCUDA_AX_B_SolverWrapper::success)
	{
		std::cout << "problem with solving Ax=B" << std::endl;
		delete wr;
		return false;
	}

	om -= x[0];
	fi -= x[1];
	ka -= x[2];
	tx -= x[3];
	ty -= x[4];
	tz -= x[5];

	delete wr;

	errCUDA = cudaFree(d_A); d_A = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	errCUDA = cudaFree(d_P); d_P = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	errCUDA = cudaFree(d_l); d_l = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	free (A);
	free(delta);
	free (x);
	free (P);

	return true;
}

void CCudaWrapper::fill_delta_PlanarFeatureMatching(double *delta, double om, double fi, double ka, double tx, double ty, double tz,
			std::vector<plane> &planes1, std::vector<plane> &planes2)
{
	Eigen::Affine3f mR, mT;
	mR = Eigen::AngleAxisf(om, Eigen::Vector3f::UnitX())
				  * Eigen::AngleAxisf(fi, Eigen::Vector3f::UnitY())
				  * Eigen::AngleAxisf(ka, Eigen::Vector3f::UnitZ());
	mT = Eigen::Translation3f(tx, ty, tz);
	Eigen::Affine3f m = mT * mR;

	for(int i=0;i<planes1.size(); i++)
	{
		plane tplate = transformPlane(planes2[i], m);

		double delta1 = planes1[i].nx - tplate.nx;
		double delta2 = planes1[i].ny - tplate.ny;
		double delta3 = planes1[i].nz - tplate.nz;
		double delta4 = -(planes1[i].rho - tplate.rho);
		delta[4 * i + 0] = delta1;
		delta[4 * i + 1] = delta2;
		delta[4 * i + 2] = delta3;
		delta[4 * i + 3] = delta4;
	}


	/* approximated method
	double r[9];
	computeR(om, fi, ka, r);

	for(int i=0;i<planes1.size(); i++)
	{
		double nx1 = planes1[i].nx;
		double ny1 = planes1[i].ny;
		double nz1 = planes1[i].nz;
		double q1  = planes1[i].rho;

		double nx2 = planes2[i].nx;
		double ny2 = planes2[i].ny;
		double nz2 = planes2[i].nz;
		double q2  = planes2[i].rho;

		double delta1 = nx1 - (r[_11]*nx2 + r[_12]*ny2 + r[_13]*nz2);
		double delta2 = ny1 - (r[_21]*nx2 + r[_22]*ny2 + r[_23]*nz2);
		double delta3 = nz1 - (r[_31]*nx2 + r[_32]*ny2 + r[_33]*nz2);
		double delta4 = q2 - q1 + (r[_11]*nx2 + r[_12]*ny2 + r[_13]*nz2) * tx +
								  (r[_21]*nx2 + r[_22]*ny2 + r[_23]*nz2) * ty +
								  (r[_31]*nx2 + r[_32]*ny2 + r[_33]*nz2) * tz;
		delta[4 * i + 0] = delta1;
		delta[4 * i + 1] = delta2;
		delta[4 * i + 2] = delta3;
		delta[4 * i + 3] = delta4;
	}*/

}



void CCudaWrapper::fillA_PlanarFeatureMatching(double *A, double om, double fi, double ka, double tx, double ty, double tz,
			std::vector<plane> &planes1, std::vector<plane> &planes2)
{
	double r[9];
	computeR(om, fi, ka, r);


	for(int i=0;i<planes1.size(); i++)
	{
		double x0 = planes2[i].nx;
		double y0 = planes2[i].ny;
		double z0 = planes2[i].nz;

		double a11 = compute_a11();
		double a12 = compute_a12(1.0, om, fi, ka, x0, y0, z0);
		double a13 = compute_a13(1.0, r, x0, y0);
		double a21 = compute_a21(1.0, r, x0, y0, z0);
		double a22 = compute_a22(1.0, om, fi, ka, x0, y0, z0);
		double a23 = compute_a23(1.0, r, x0, y0);
		double a31 = compute_a31(1.0, r, x0, y0, z0);
		double a32 = compute_a32(1.0, om, fi, ka, x0, y0, z0);
		double a33 = compute_a33(1.0, r, x0, y0);

		double a41 = a11 * tx + a21 * ty + a31 * tz;
		double a42 = a12 * tx + a22 * ty + a32 * tz;
		double a43 = a13 * tx + a23 * ty + a33 * tz;

		double a14 = 0.0;
		double a15 = 0.0;
		double a16 = 0.0;

		double a24 = 0.0;
		double a25 = 0.0;
		double a26 = 0.0;

		double a34 = 0.0;
		double a35 = 0.0;
		double a36 = 0.0;

		double a44 = r[_11]*x0 + r[_12]*y0 + r[_13]*z0;
		double a45 = r[_21]*x0 + r[_22]*y0 + r[_23]*z0;
		double a46 = r[_31]*x0 + r[_32]*y0 + r[_33]*z0;

		A[i * 4 + 0 + 0 * planes1.size() * 4] = -a11;
		A[i * 4 + 1 + 0 * planes1.size() * 4] = -a21;
		A[i * 4 + 2 + 0 * planes1.size() * 4] = -a31;
		A[i * 4 + 3 + 0 * planes1.size() * 4] =  a41;

		A[i * 4 + 0 + 1 * planes1.size() * 4] = -a12;
		A[i * 4 + 1 + 1 * planes1.size() * 4] = -a22;
		A[i * 4 + 2 + 1 * planes1.size() * 4] = -a32;
		A[i * 4 + 3 + 1 * planes1.size() * 4] =  a42;

		A[i * 4 + 0 + 2 * planes1.size() * 4] = -a13;
		A[i * 4 + 1 + 2 * planes1.size() * 4] = -a23;
		A[i * 4 + 2 + 2 * planes1.size() * 4] = -a33;
		A[i * 4 + 3 + 2 * planes1.size() * 4] =  a43;

		A[i * 4 + 0 + 3 * planes1.size() * 4] = a14;
		A[i * 4 + 1 + 3 * planes1.size() * 4] = a24;
		A[i * 4 + 2 + 3 * planes1.size() * 4] = a34;
		A[i * 4 + 3 + 3 * planes1.size() * 4] = a44;

		A[i * 4 + 0 + 4 * planes1.size() * 4] = a15;
		A[i * 4 + 1 + 4 * planes1.size() * 4] = a25;
		A[i * 4 + 2 + 4 * planes1.size() * 4] = a35;
		A[i * 4 + 3 + 4 * planes1.size() * 4] = a45;

		A[i * 4 + 0 + 5 * planes1.size() * 4] = a16;
		A[i * 4 + 1 + 5 * planes1.size() * 4] = a26;
		A[i * 4 + 2 + 5 * planes1.size() * 4] = a36;
		A[i * 4 + 3 + 5 * planes1.size() * 4] = a46;
	}
}

void CCudaWrapper::computeR(double om, double fi, double ka, double *R)
{
	//R[11 12 13; 21 22 23; 31 32 33]
	//R[0  1  2 ; 3  4  5 ; 6  7  8]
	R[_11] = cos(fi) * cos(ka);
	R[_12] = -cos(fi) * sin(ka);
	R[_13] = sin(fi);

	R[_21] = cos(om)*sin(ka) + sin(om)*sin(fi)*cos(ka);
	R[_22] = cos(om) *cos(ka) - sin(om)*sin(fi)*sin(ka);
	R[_23] = -sin(om) * cos(fi);

	R[_31] = sin(om) * sin(ka) - cos(om)*sin(fi)*cos(ka);
	R[_32] = sin(om) * cos(ka) + cos(om)*sin(fi)*sin(ka);
	R[_33] = cos(om) * cos(fi);
}



double CCudaWrapper::compute_a10(double *r, double x0, double y0, double z0)
{
	return r[_11]*x0 + r[_12] * y0 + r[_13] * z0;
}

double CCudaWrapper::compute_a20(double *r, double x0, double y0, double z0)
{
	return r[_21]*x0 + r[_22] * y0 + r[_23] * z0;
}

double CCudaWrapper::compute_a30(double *r, double x0, double y0, double z0)
{
	return r[_31] * x0 + r[_32] * y0 + r[_33]*z0;
}

double CCudaWrapper::compute_a11()
{
	return 0.0;
}

double CCudaWrapper::compute_a12(double m, double om, double fi, double ka, double x0, double y0, double z0)
{
	return m*(-sin(fi)*cos(ka)*x0 + sin(fi)*sin(ka)*y0 + cos (fi) *z0);
}

double CCudaWrapper::compute_a13(double m, double *r, double x0, double y0)
{
	return m*(r[_12]*x0-r[_11]*y0);
}

double CCudaWrapper::compute_a21(double m, double *r, double x0, double y0, double z0)
{
	return m*(-r[_31]*x0-r[_32]*y0-r[_33]*z0);
}

double CCudaWrapper::compute_a22(double m, double om, double fi, double ka, double x0, double y0, double z0)
{
	return m*(sin(om)*cos(fi)*cos(ka)*x0 - sin(om)*cos(fi)*sin(ka)*y0+sin(om)*sin(fi)*z0);
}

double CCudaWrapper::compute_a23(double m, double *r, double x0, double y0)
{
	return m*(r[_22]*x0-r[_21]*y0);
}

double CCudaWrapper::compute_a31(double m, double *r, double x0, double y0, double z0)
{
	return m*(r[_21]*x0+r[_22]*y0 +r[_23]*z0);
}

double CCudaWrapper::compute_a32(double m, double om, double fi, double ka, double x0, double y0, double z0)
{
	return m * (-cos(om)*cos(fi)*cos(ka)*x0 + cos(om)*cos(fi)*sin(ka)*y0 - cos(om)*sin(fi)*z0);
}

double CCudaWrapper::compute_a33(double m, double *r, double x0, double y0)
{
	return m*(r[_32]*x0 - r[_31]*y0);
}

plane_t CCudaWrapper::transformPlane(plane_t plane, Eigen::Affine3f m)
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

bool CCudaWrapper::computeSingleIterationOfPlanarFeatureMatchingMultiplePoseCUDA(std::vector<pair_local_observations_t> &v_pair_local_observations )
{
	int _size_poses = v_pair_local_observations.size();
	int _size_observations = 0;

	for(size_t i = 0 ; i < v_pair_local_observations.size(); i++)
	{
		_size_observations += v_pair_local_observations[i].planes_reference.size();
	}

	double *A=(double*)malloc( _size_observations* 6 * 4 * _size_poses *sizeof(double));
	for(int i = 0 ; i < _size_observations* 6 * 4 * _size_poses; i++)A[i] = 0.0;

	double *delta = (double*)malloc( _size_observations * 4 * sizeof(double));
	for(int i = 0 ; i < _size_observations * 4; i++)delta[i] = 0.0;

	double *At=(double*)malloc( _size_observations* 6 * 4 * _size_poses * sizeof(double));
	for(int i = 0 ; i < _size_observations* 6 * 4 * _size_poses; i++)At[i] = 0.0;

	double *AtA = (double *)malloc (6*6*_size_poses*_size_poses *sizeof(double));
	for(int i = 0 ; i < 6*6*_size_poses; i++)AtA[i] = 0;

	double *Atl = (double *)malloc (6 * 1 * _size_poses * sizeof(double));
	for(int i = 0 ; i < 6 * 1 * _size_poses; i++)Atl[i] = 0.0;

	double *x = (double *)malloc (6 * _size_poses* sizeof(double));
	for(int i = 0 ; i < 6 * _size_poses; i++)x[i] = 0.0;

	int offset = 0;

	for(size_t k = 0 ; k < v_pair_local_observations.size(); k++)
	{
		double om = v_pair_local_observations[k].om;
		double fi = v_pair_local_observations[k].fi;
		double ka = v_pair_local_observations[k].ka;
		double tx = v_pair_local_observations[k].tx;
		double ty = v_pair_local_observations[k].ty;
		double tz = v_pair_local_observations[k].tz;

		if(k > 0)
		{
			offset += v_pair_local_observations[k-1].planes_reference.size();
		}

		double r[9];
		computeR(om, fi, ka, r);

		for(int i=0;i < v_pair_local_observations[k].planes_to_register.size(); i++)
		{
			double x0 = v_pair_local_observations[k].planes_to_register[i].nx;
			double y0 = v_pair_local_observations[k].planes_to_register[i].ny;
			double z0 = v_pair_local_observations[k].planes_to_register[i].nz;

			double a11 = compute_a11();
			double a12 = compute_a12(1.0, om, fi, ka, x0, y0, z0);
			double a13 = compute_a13(1.0, r, x0, y0);
			double a21 = compute_a21(1.0, r, x0, y0, z0);
			double a22 = compute_a22(1.0, om, fi, ka, x0, y0, z0);
			double a23 = compute_a23(1.0, r, x0, y0);
			double a31 = compute_a31(1.0, r, x0, y0, z0);
			double a32 = compute_a32(1.0, om, fi, ka, x0, y0, z0);
			double a33 = compute_a33(1.0, r, x0, y0);

			double a41 = a11 * tx + a21 * ty + a31 * tz;
			double a42 = a12 * tx + a22 * ty + a32 * tz;
			double a43 = a13 * tx + a23 * ty + a33 * tz;

			double a14 = 0.0;
			double a15 = 0.0;
			double a16 = 0.0;

			double a24 = 0.0;
			double a25 = 0.0;
			double a26 = 0.0;

			double a34 = 0.0;
			double a35 = 0.0;
			double a36 = 0.0;

			double a44 = r[_11]*x0 + r[_12]*y0 + r[_13]*z0;
			double a45 = r[_21]*x0 + r[_22]*y0 + r[_23]*z0;
			double a46 = r[_31]*x0 + r[_32]*y0 + r[_33]*z0;

			A[i * 4 + 0 + (0 + k * 6) * _size_observations * 4 + offset * 4] = -a11;
			A[i * 4 + 1 + (0 + k * 6) * _size_observations * 4 + offset * 4] = -a21;
			A[i * 4 + 2 + (0 + k * 6) * _size_observations * 4 + offset * 4] = -a31;
			A[i * 4 + 3 + (0 + k * 6) * _size_observations * 4 + offset * 4] =  a41;

			A[i * 4 + 0 + (1 + k * 6) * _size_observations * 4 + offset * 4] = -a12;
			A[i * 4 + 1 + (1 + k * 6) * _size_observations * 4 + offset * 4] = -a22;
			A[i * 4 + 2 + (1 + k * 6) * _size_observations * 4 + offset * 4] = -a32;
			A[i * 4 + 3 + (1 + k * 6) * _size_observations * 4 + offset * 4] =  a42;

			A[i * 4 + 0 + (2 + k * 6) * _size_observations * 4 + offset * 4] = -a13;
			A[i * 4 + 1 + (2 + k * 6) * _size_observations * 4 + offset * 4] = -a23;
			A[i * 4 + 2 + (2 + k * 6) * _size_observations * 4 + offset * 4] = -a33;
			A[i * 4 + 3 + (2 + k * 6) * _size_observations * 4 + offset * 4] =  a43;

			A[i * 4 + 0 + (3 + k * 6) * _size_observations * 4 + offset * 4] = a14;
			A[i * 4 + 1 + (3 + k * 6) * _size_observations * 4 + offset * 4] = a24;
			A[i * 4 + 2 + (3 + k * 6) * _size_observations * 4 + offset * 4] = a34;
			A[i * 4 + 3 + (3 + k * 6) * _size_observations * 4 + offset * 4] = a44;

			A[i * 4 + 0 + (4 + k * 6) * _size_observations * 4 + offset * 4] = a15;
			A[i * 4 + 1 + (4 + k * 6) * _size_observations * 4 + offset * 4] = a25;
			A[i * 4 + 2 + (4 + k * 6) * _size_observations * 4 + offset * 4] = a35;
			A[i * 4 + 3 + (4 + k * 6) * _size_observations * 4 + offset * 4] = a45;

			A[i * 4 + 0 + (5 + k * 6) * _size_observations * 4 + offset * 4] = a16;
			A[i * 4 + 1 + (5 + k * 6) * _size_observations * 4 + offset * 4] = a26;
			A[i * 4 + 2 + (5 + k * 6) * _size_observations * 4 + offset * 4] = a36;
			A[i * 4 + 3 + (5 + k * 6) * _size_observations * 4 + offset * 4] = a46;
		}
	}

	offset = 0;
	for(size_t k = 0 ; k < v_pair_local_observations.size(); k++)
	{
		if(k > 0)
		{
			offset += v_pair_local_observations[k-1].planes_reference.size();
		}

		double om = v_pair_local_observations[k].om;
		double fi = v_pair_local_observations[k].fi;
		double ka = v_pair_local_observations[k].ka;
		double tx = v_pair_local_observations[k].tx;
		double ty = v_pair_local_observations[k].ty;
		double tz = v_pair_local_observations[k].tz;

		Eigen::Affine3f mR, mT;
		mR = Eigen::AngleAxisf(om, Eigen::Vector3f::UnitX())
					  * Eigen::AngleAxisf(fi, Eigen::Vector3f::UnitY())
					  * Eigen::AngleAxisf(ka, Eigen::Vector3f::UnitZ());
		mT = Eigen::Translation3f(tx, ty, tz);
		Eigen::Affine3f m = mT * mR;

		for(int i=0;i < v_pair_local_observations[k].planes_to_register.size(); i++)
		{
			plane tplate = transformPlane(v_pair_local_observations[k].planes_to_register[i], m);
			double delta1 = v_pair_local_observations[k].planes_reference[i].nx - tplate.nx;
			double delta2 = v_pair_local_observations[k].planes_reference[i].ny - tplate.ny;
			double delta3 = v_pair_local_observations[k].planes_reference[i].nz - tplate.nz;
			double delta4 = -(v_pair_local_observations[k].planes_reference[i].rho - tplate.rho);

			delta[4 * i + 0 + offset *4] = delta1;
			delta[4 * i + 1 + offset *4] = delta2;
			delta[4 * i + 2 + offset *4] = delta3;
			delta[4 * i + 3 + offset *4] = delta4;
		}
	}

	cudaError_t errCUDA = ::cudaSuccess;
	double *d_A = NULL;
	errCUDA  = cudaMalloc((void**)&d_A,  _size_observations* 6 * 4 * _size_poses *sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}
	errCUDA = cudaMemcpy(d_A, A, _size_observations* 6 * 4 * _size_poses *sizeof(double), cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){return false;}

	double *P=(double*)malloc(_size_observations* 6 *sizeof(double));
	for(int i = 0 ; i < _size_observations* 6; i++)P[i] = 1.0;

	double *d_P = NULL;
	errCUDA  = cudaMalloc((void**)&d_P,  _size_observations* 6 *sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}
	errCUDA = cudaMemcpy(d_P, P, _size_observations* 6 *sizeof(double), cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){return false;}

	double *d_l=NULL;
	errCUDA  = cudaMalloc((void**)&d_l,  _size_observations * 4 * sizeof(double));
	if(errCUDA != ::cudaSuccess){return false;}
	errCUDA = cudaMemcpy(d_l, delta, _size_observations * 4 * sizeof(double), cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){return false;}

	CCUDA_AX_B_SolverWrapper * wr = new CCUDA_AX_B_SolverWrapper(false, 0);

	CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper_error errAXB =
					wr->Solve_ATPA_ATPl_x_data_on_GPU(this->threads, d_A, d_P, d_l, x, _size_poses * 6, _size_observations * 4, CCUDA_AX_B_SolverWrapper::chol);
	if(errAXB!=CCUDA_AX_B_SolverWrapper::success)
	{
		std::cout << "problem with solving Ax=B" << std::endl;
		delete wr;
		return false;
	}

	for(size_t i = 0 ; i < v_pair_local_observations.size(); i++)
	{
		v_pair_local_observations[i].om -= x[0 + i * 6];
		v_pair_local_observations[i].fi -= x[1 + i * 6];
		v_pair_local_observations[i].ka -= x[2 + i * 6];
		v_pair_local_observations[i].tx -= x[3 + i * 6];
		v_pair_local_observations[i].ty -= x[4 + i * 6];
		v_pair_local_observations[i].tz -= x[5 + i * 6];
	}

	delete wr;

	errCUDA = cudaFree(d_A); d_A = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	errCUDA = cudaFree(d_P); d_P = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	errCUDA = cudaFree(d_l); d_l = 0;
	if(errCUDA != ::cudaSuccess){return false;}

	free (A);
	free (AtA);
	free (At);
	free(delta);
	free (Atl);
	free (x);
	free (P);

return true;
}
