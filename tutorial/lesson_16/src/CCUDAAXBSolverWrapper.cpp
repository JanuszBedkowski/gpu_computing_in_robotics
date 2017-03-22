#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


#include <iostream>
#include <iomanip>

#include "CCUDAAXBSolverWrapper.h"
#include "helper_cuda.h"
#include "lesson_16.h"
#include "helper_cusolver.h"

#define TEST

CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper(bool _CCUDA_AX_B_SolverWrapperDEBUG, int cuda_device) {

	CCUDA_AX_B_SolverWrapperDEBUG = _CCUDA_AX_B_SolverWrapperDEBUG;

	handle = NULL;
	cublasHandle = NULL; // used in residual evaluation
	stream = NULL;

	cudaError_t errCUDA = ::cudaSuccess;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	errCUDA = cudaSetDevice(cuda_device);
	assert(::cudaSuccess == errCUDA);


	checkCudaErrors(cusolver_status = cusolverDnCreate(&handle));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cusolverStatus_t(cusolver_status, "cusolverDnCreate(&handle)");
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	checkCudaErrors(cublas_status = cublasCreate(&cublasHandle));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cublasStatus_t(cublas_status, "cublasCreate(&cublasHandle)");
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

	checkCudaErrors(errCUDA = cudaStreamCreate(&stream));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaStreamCreate(&stream)");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(cusolver_status = cusolverDnSetStream(handle, stream));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cusolverStatus_t(cusolver_status, "cusolverDnSetStream(handle, stream)");
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

	checkCudaErrors(cublas_status = cublasSetStream(cublasHandle, stream));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cublasStatus_t(cublas_status, "cublasSetStream(cublasHandle, stream)");
	assert(CUBLAS_STATUS_SUCCESS == cublas_status);

}

CCUDA_AX_B_SolverWrapper::~CCUDA_AX_B_SolverWrapper() {
	cudaError_t errCUDA = ::cudaSuccess;
	cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	if (handle)
	{
	  	checkCudaErrors(cusolver_status = cusolverDnDestroy(handle));
	    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cusolverStatus_t(cusolver_status, "checkCudaErrors(cusolverDnDestroy(handle))");
	   	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	}

	if (cublasHandle)
	{
	   	checkCudaErrors(cublas_status = cublasDestroy(cublasHandle));
	  	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cublasStatus_t(cublas_status, "cublasDestroy(cublasHandle)");
	    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
	}

	if (stream)
	{
	   	checkCudaErrors(errCUDA = cudaStreamDestroy(stream));
	   	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaStreamDestroy(stream)");
	   	assert(::cudaSuccess == errCUDA);
	}
}

double CCUDA_AX_B_SolverWrapper::Solve(double *a,double *b,double *x, int a_rows, int a_cols, int b_cols, char method)
{
		clock_t begin_time;
		double solve_time;

		begin_time = clock();

		    int rowsA = 0; // number of rows of A
		    int colsA = 0; // number of columns of A
		    int lda   = 0; // leading dimension in dense matrix

		    lda = a_rows;
		    rowsA = a_rows;
		    colsA = a_cols;

		    double *d_A = NULL; // a copy of h_A
		    double *d_x = NULL; // x = A \ b
		    double *d_b = NULL; // a copy of h_b

		    	// verify if A is symmetric or not.
		        if ( method == chol )
		        {
		            int issym = 1;
		            for(int j = 0 ; j < colsA ; j++)
		            {
		                for(int i = j ; i < rowsA ; i++)
		                {
		                    double Aij = a[i + j*lda];
		                    double Aji = a[j + i*lda];
		                    if ( Aij != Aji )
		                    {
		                        issym = 0;
		                        break;
		                    }
		                }
		            }
		            if (!issym)
		            {
		                printf("Error: A has no symmetric pattern, please use LU or QR \n");
		                exit(EXIT_FAILURE);
		            }
		        }

		    cudaError_t errCUDA = ::cudaSuccess;
		    checkCudaErrors(errCUDA = cudaMalloc((void **)&d_A, sizeof(double)*lda*colsA));
		    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_A");
		    assert(::cudaSuccess == errCUDA);

		    checkCudaErrors(errCUDA = cudaMalloc((void **)&d_x, sizeof(double)*colsA));
		    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_x");
		    assert(::cudaSuccess == errCUDA);

		    checkCudaErrors(errCUDA = cudaMalloc((void **)&d_b, sizeof(double)*rowsA));
		    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_b");
		    assert(::cudaSuccess == errCUDA);

		    checkCudaErrors(errCUDA = cudaMemcpy(d_A, a, sizeof(double)*lda*colsA, cudaMemcpyHostToDevice));
		    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(d_A, a)");
		    assert(::cudaSuccess == errCUDA);

		    checkCudaErrors(errCUDA = cudaMemcpy(d_b, b, sizeof(double)*rowsA, cudaMemcpyHostToDevice));
		    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(d_b, b)");
		   	assert(::cudaSuccess == errCUDA);

		    if ( method == chol)
		    {
		         linearSolverCHOL(handle, rowsA, d_A, lda, d_b, d_x);
		    }
		    else if ( method == lu )
		    {
		         linearSolverLU(handle, rowsA, d_A, lda, d_b, d_x);
		    }
		    else if ( method ==  qr)
		    {
		         linearSolverQR(handle, rowsA, d_A, lda, d_b, d_x);
		    }
		    else
		    {
		        fprintf(stderr, "Error: %d is unknown function\n", method);
		        exit(EXIT_FAILURE);
		    }

		    checkCudaErrors(errCUDA = cudaMemcpy(x, d_x, sizeof(double)*colsA, cudaMemcpyDeviceToHost));
		    if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(x, d_x)");
		    assert(::cudaSuccess == errCUDA);

		    if (d_A)
		    {
		    	checkCudaErrors(errCUDA = cudaFree(d_A));
		    	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_A)");
		    	assert(::cudaSuccess == errCUDA);
		    }

		    if (d_x)
		    {
		    	checkCudaErrors(errCUDA = cudaFree(d_x));
		    	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_x)");
		    	assert(::cudaSuccess == errCUDA);
		    }
		    if (d_b)
		    {
		    	checkCudaErrors(errCUDA = cudaFree(d_b));
		    	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_b)");
		    	assert(::cudaSuccess == errCUDA);
		    }

		solve_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
return solve_time;
}



double CCUDA_AX_B_SolverWrapper::Compute_AtP(int threads, double *A, double *P, double *AtP, int rows, int columns)
{
	clock_t begin_time;
	double solve_time;
	begin_time = clock();

	double *d_A = NULL;
	double *d_P = NULL;
	double *d_AtP = NULL;

	cudaError_t errCUDA = ::cudaSuccess;
	checkCudaErrors(errCUDA = cudaMalloc((void **)&d_A, sizeof(double)*rows*columns));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_A");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMalloc((void **)&d_P, sizeof(double)*columns));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_x");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMalloc((void **)&d_AtP, sizeof(double)*rows*columns));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_b");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMemcpy(d_A, A, sizeof(double)*rows*columns, cudaMemcpyHostToDevice));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(d_A, A)");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMemcpy(d_P, P, sizeof(double)*columns, cudaMemcpyHostToDevice));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(d_P, P)");
	assert(::cudaSuccess == errCUDA);

	errCUDA = cudaCompute_AtP(threads, d_A, d_P, d_AtP, rows, columns);

	checkCudaErrors(errCUDA = cudaMemcpy(AtP, d_AtP, sizeof(double)*rows*columns, cudaMemcpyDeviceToHost));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(AtP, d_AtP)");
	assert(::cudaSuccess == errCUDA);

	if (d_A)
	{
	  	checkCudaErrors(errCUDA = cudaFree(d_A));
	  	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_A)");
	   	assert(::cudaSuccess == errCUDA);
	}

	if (d_P)
	{
	   	checkCudaErrors(errCUDA = cudaFree(d_P));
	   	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_P)");
	   	assert(::cudaSuccess == errCUDA);
	}

	if (d_AtP)
	{
	  	checkCudaErrors(errCUDA = cudaFree(d_AtP));
	  	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_AtP)");
	  	assert(::cudaSuccess == errCUDA);
	}

	solve_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
	return solve_time;
}

double CCUDA_AX_B_SolverWrapper::Multiply(double *a, double *b,double *c, int a_rows, int a_cols, int b_cols)
{
	clock_t begin_time;
	double solve_time;
	begin_time = clock();

	double *d_a = NULL;
	double *d_b = NULL;
	double *d_c = NULL;

	cudaError_t errCUDA = ::cudaSuccess;
	checkCudaErrors(errCUDA = cudaMalloc((void **)&d_a, sizeof(double)*a_rows*a_cols));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_a");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMalloc((void **)&d_b, sizeof(double)*a_cols*b_cols));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_b");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMalloc((void **)&d_c, sizeof(double)*a_rows*b_cols));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMalloc((void **)&d_c");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMemcpy(d_a, a, sizeof(double)*a_rows*a_cols, cudaMemcpyHostToDevice));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(d_a, a)");
	assert(::cudaSuccess == errCUDA);

	checkCudaErrors(errCUDA = cudaMemcpy(d_b, b, sizeof(double)*a_cols*b_cols, cudaMemcpyHostToDevice));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(d_b, b)");
	assert(::cudaSuccess == errCUDA);

	multiplyCUBLAS( cublasHandle, d_a, d_b, d_c, a_rows, a_cols, b_cols);

	checkCudaErrors(errCUDA = cudaMemcpy(c, d_c, sizeof(double)*a_rows*b_cols, cudaMemcpyDeviceToHost));
	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaMemcpy(c, d_c)");
	assert(::cudaSuccess == errCUDA);

	if (d_a)
	{
	  	checkCudaErrors(errCUDA = cudaFree(d_a));
	  	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_a)");
	   	assert(::cudaSuccess == errCUDA);
	}

	if (d_b)
	{
	   	checkCudaErrors(errCUDA = cudaFree(d_b));
	   	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_b)");
	   	assert(::cudaSuccess == errCUDA);
	}

	if(d_c)
	{
	  	checkCudaErrors(errCUDA = cudaFree(d_c));
	  	if(this->CCUDA_AX_B_SolverWrapperDEBUG)cout_cudaError_t(errCUDA, "cudaFree(d_c)");
	  	assert(::cudaSuccess == errCUDA);
	}


	solve_time=(double)( clock () - begin_time ) /  CLOCKS_PER_SEC;
	return solve_time;
}


CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper_error CCUDA_AX_B_SolverWrapper::Solve_ATPA_ATPl_x(int threads, double *A, double *P, double *l, double *x,
		int rows, int columns, CCUDA_AX_B_SolverWrapper::Solver_Method solver_method)
{
	double *d_A = NULL;
	double *d_P = NULL;
	double *d_AtP = NULL;

	cudaError_t errCUDA = ::cudaSuccess;
	errCUDA = cudaMalloc((void **)&d_A, sizeof(double)*rows*columns);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaMemcpy(d_A, A, sizeof(double)*rows*columns, cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaMalloc((void **)&d_P, sizeof(double)*columns);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaMemcpy(d_P, P, sizeof(double)*columns, cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaMalloc((void **)&d_AtP, sizeof(double)*rows*columns);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaCompute_AtP(threads, d_A, d_P, d_AtP, rows, columns);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if (d_P)
	{
	   	errCUDA = cudaFree(d_P);
	   	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	double *d_AtPA = NULL;
	errCUDA = cudaMalloc((void **)&d_AtPA, sizeof(double)*rows*rows); //
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	multiplyCUBLAS( cublasHandle, d_AtP, d_A, d_AtPA, rows, columns, rows);
	errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if (d_A)
	{
	  	errCUDA = cudaFree(d_A);
	  	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	double *d_l = NULL;
	errCUDA = cudaMalloc((void **)&d_l, sizeof(double)*columns);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaMemcpy(d_l, l, sizeof(double)*columns, cudaMemcpyHostToDevice);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	double *d_AtPl = NULL;
	errCUDA = cudaMalloc((void **)&d_AtPl, sizeof(double)*rows);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	multiplyCUBLAS(cublasHandle, d_AtP, d_l, d_AtPl, rows, columns, 1);
	errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if (d_l)
	{
	  	errCUDA = cudaFree(d_l);
	  	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	if (d_AtP)
	{
	  	errCUDA = cudaFree(d_AtP);
	  	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	double *d_x = NULL;
	errCUDA = cudaMalloc((void **)&d_x, sizeof(double)*rows);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if ( solver_method == chol)
	{
		linearSolverCHOL(handle, rows, d_AtPA, rows, d_AtPl, d_x);
		errCUDA = cudaGetLastError();
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}
	else if ( solver_method == lu )
	{
		linearSolverLU(handle, rows, d_AtPA, rows, d_AtPl, d_x);
		errCUDA = cudaGetLastError();
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}
	else if ( solver_method ==  qr)
	{
	    linearSolverQR(handle, rows, d_AtPA, rows, d_AtPl, d_x);
	    errCUDA = cudaGetLastError();
	    if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}
	else
	{
		cudaDeviceReset();
		return fail_problem_with_CUDA_AX_B_Solver;

	}

	errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaMemcpy(x, d_x, sizeof(double)*rows, cudaMemcpyDeviceToHost);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if (d_AtPA)
	{
	  	errCUDA = cudaFree(d_AtPA);
	  	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	if (d_AtPl)
	{
	  	errCUDA = cudaFree(d_AtPl);
	  	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	if (d_x)
	{
	  	errCUDA = cudaFree(d_x);
	  	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}
	errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	return success;
}

CCUDA_AX_B_SolverWrapper::CCUDA_AX_B_SolverWrapper_error CCUDA_AX_B_SolverWrapper::Solve_ATPA_ATPl_x_data_on_GPU(int threads, double *d_A, double *d_P, double *d_l,
			double *x, int rows, int columns, Solver_Method solver_method)
{
	double *d_AtP = NULL;
	cudaError_t errCUDA = ::cudaSuccess;
	errCUDA = cudaMalloc((void **)&d_AtP, sizeof(double)*rows*columns);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaCompute_AtP(threads, d_A, d_P, d_AtP, rows, columns);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	double *d_AtPA = NULL;
	errCUDA = cudaMalloc((void **)&d_AtPA, sizeof(double)*rows*rows); //
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	multiplyCUBLAS( cublasHandle, d_AtP, d_A, d_AtPA, rows, columns, rows);
	errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	double *d_AtPl = NULL;
	errCUDA = cudaMalloc((void **)&d_AtPl, sizeof(double)*rows);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	multiplyCUBLAS(cublasHandle, d_AtP, d_l, d_AtPl, rows, columns, 1);
	errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if (d_AtP)
	{
		errCUDA = cudaFree(d_AtP);
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	double *d_x = NULL;
	errCUDA = cudaMalloc((void **)&d_x, sizeof(double)*rows);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if ( solver_method == chol)
	{
		linearSolverCHOL(handle, rows, d_AtPA, rows, d_AtPl, d_x);
		errCUDA = cudaGetLastError();
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}
	else if ( solver_method == lu )
	{
		linearSolverLU(handle, rows, d_AtPA, rows, d_AtPl, d_x);
		errCUDA = cudaGetLastError();
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}
	else if ( solver_method ==  qr)
	{
		linearSolverQR(handle, rows, d_AtPA, rows, d_AtPl, d_x);
		errCUDA = cudaGetLastError();
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}
	else
	{
		cudaDeviceReset();
		return fail_problem_with_CUDA_AX_B_Solver;
	}

	errCUDA = cudaGetLastError();
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	errCUDA = cudaMemcpy(x, d_x, sizeof(double)*rows, cudaMemcpyDeviceToHost);
	if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}

	if (d_AtPA)
	{
		errCUDA = cudaFree(d_AtPA);
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	if (d_AtPl)
	{
		errCUDA = cudaFree(d_AtPl);
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	if (d_x)
	{
		errCUDA = cudaFree(d_x);
		if(errCUDA != ::cudaSuccess){cudaDeviceReset(); return fail_problem_with_CUDA_AX_B_Solver;}
	}

	return success;
}

////////////////////////////////////////////////////////////////////////////////////
/*
 *  solve A*x = b by Cholesky factorization
 *
 */

int CCUDA_AX_B_SolverWrapper::linearSolverCHOL(
	    cusolverDnHandle_t handle,
	    int n,
	    const double *Acopy,
	    int lda,
	    const double *b,
	    double *x)
{
		int bufferSize = 0;
	    int *info = NULL;
	    double *buffer = NULL;
	    double *A = NULL;
	    int h_info = 0;
	    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	    checkCudaErrors(cusolverDnDpotrf_bufferSize(handle, uplo, n, (double*)Acopy, lda, &bufferSize));

	    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
	    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
	    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));


	    // prepare a copy of A because potrf will overwrite A with L
	    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
	    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

	    checkCudaErrors(cusolverDnDpotrf(handle, uplo, n, A, lda, buffer, bufferSize, info));

	    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

	    if ( 0 != h_info ){
	        fprintf(stderr, "Error: linearSolverCHOL failed\n");
	    }

	    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

	    checkCudaErrors(cusolverDnDpotrs(handle, uplo, n, 1, A, lda, x, n, info));

	    checkCudaErrors(cudaDeviceSynchronize());

	    if (info  ) { checkCudaErrors(cudaFree(info)); }
	    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
	    if (A     ) { checkCudaErrors(cudaFree(A)); }

	    return 0;
}

/*
 *  solve A*x = b by LU with partial pivoting
 *
 */
int CCUDA_AX_B_SolverWrapper::linearSolverLU(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    int *ipiv = NULL; // pivoting sequence
    int h_info = 0;

    checkCudaErrors(cusolverDnDgetrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc(&ipiv, sizeof(int)*n));

    // prepare a copy of A because getrf will overwrite A with L
    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    checkCudaErrors(cusolverDnDgetrf(handle, n, n, A, lda, buffer, ipiv, info));
    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: linearSolverLU failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, 1, A, lda, ipiv, x, n, info));
    checkCudaErrors(cudaDeviceSynchronize());

    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (ipiv  ) { checkCudaErrors(cudaFree(ipiv));}

    return 0;
}

/*
 *  solve A*x = b by QR
 *
 */
int CCUDA_AX_B_SolverWrapper::linearSolverQR(
    cusolverDnHandle_t handle,
    int n,
    const double *Acopy,
    int lda,
    const double *b,
    double *x)
{
    cublasHandle_t cublasHandle = NULL;
    int bufferSize = 0;
    int *info = NULL;
    double *buffer = NULL;
    double *A = NULL;
    double *tau = NULL;
    int h_info = 0;
    const double one = 1.0;
    checkCudaErrors(cublasCreate(&cublasHandle));

    checkCudaErrors(cusolverDnDgeqrf_bufferSize(handle, n, n, (double*)Acopy, lda, &bufferSize));

    checkCudaErrors(cudaMalloc(&info, sizeof(int)));
    checkCudaErrors(cudaMalloc(&buffer, sizeof(double)*bufferSize));
    checkCudaErrors(cudaMalloc(&A, sizeof(double)*lda*n));
    checkCudaErrors(cudaMalloc ((void**)&tau, sizeof(double)*n));

    checkCudaErrors(cudaMemcpy(A, Acopy, sizeof(double)*lda*n, cudaMemcpyDeviceToDevice));

    checkCudaErrors(cudaMemset(info, 0, sizeof(int)));

    checkCudaErrors(cusolverDnDgeqrf(handle, n, n, A, lda, tau, buffer, bufferSize, info));

    checkCudaErrors(cudaMemcpy(&h_info, info, sizeof(int), cudaMemcpyDeviceToHost));

    if ( 0 != h_info ){
        fprintf(stderr, "Error: linearSolverQR failed\n");
    }

    checkCudaErrors(cudaMemcpy(x, b, sizeof(double)*n, cudaMemcpyDeviceToDevice));

    // compute Q^T*b
    checkCudaErrors(cusolverDnDormqr(
        handle,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        n,
        1,
        n,
        A,
        lda,
        tau,
        x,
        n,
        buffer,
        bufferSize,
        info));

    // x = R \ Q^T*b
    checkCudaErrors(cublasDtrsm(
         cublasHandle,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N,
         CUBLAS_DIAG_NON_UNIT,
         n,
         1,
         &one,
         A,
         lda,
         x,
         n));
    checkCudaErrors(cudaDeviceSynchronize());

    if (cublasHandle) { checkCudaErrors(cublasDestroy(cublasHandle)); }
    if (info  ) { checkCudaErrors(cudaFree(info  )); }
    if (buffer) { checkCudaErrors(cudaFree(buffer)); }
    if (A     ) { checkCudaErrors(cudaFree(A)); }
    if (tau   ) { checkCudaErrors(cudaFree(tau)); }

    return 0;
}

int CCUDA_AX_B_SolverWrapper::multiplyCUBLAS( cublasHandle_t handle, const double *d_a, const double *d_b, double *d_c, int a_rows, int a_cols, int b_cols)
{
	const double alpha = 1.0;
	const double beta  = 0.0;

	checkCudaErrors(cublasDgemm(handle,
			CUBLAS_OP_N, CUBLAS_OP_N,
			a_rows, b_cols, a_cols,
			&alpha,
			d_a, a_rows,
			d_b, a_cols,
			&beta, d_c, a_rows));
	return 0;
}


void CCUDA_AX_B_SolverWrapper::cout_cudaError_t(cudaError_t err, string message)
{

	switch(err)
	{
		case ::cudaSuccess:
		{
			std::cout << message << " ::cudaSuccess"  << std::endl;
			break;
		}
		case ::cudaErrorMissingConfiguration:
		{
			std::cout<< message << " ::cudaErrorMissingConfiguration"  << std::endl;
			break;
		}
		case ::cudaErrorMemoryAllocation:
		{
			std::cout<< message << " ::cudaErrorMemoryAllocation"  << std::endl;
			break;
		}
		case ::cudaErrorInitializationError:
		{
			std::cout<< message <<" ::cudaErrorInitializationError"  << std::endl;
			break;
		}
		case ::cudaErrorLaunchFailure:
		{
			std::cout<< message<<" ::cudaErrorLaunchFailure"  << std::endl;
			break;
		}
		case ::cudaErrorLaunchTimeout:
		{
			std::cout<< message << "::cudaErrorLaunchTimeout"  << std::endl;
			break;
		}
		case ::cudaErrorLaunchOutOfResources:
		{
			std::cout<<message<< " ::cudaErrorLaunchOutOfResources"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidDeviceFunction:
		{
			std::cout<< message <<" ::cudaErrorInvalidDeviceFunction"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidConfiguration:
		{
			std::cout<< message << " ::cudaErrorInvalidConfiguration"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidDevice:
		{
			std::cout<< message << " ::cudaErrorInvalidDevice"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidValue:
		{
			std::cout<< message << " ::cudaErrorInvalidValue"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidPitchValue:
		{
			std::cout<< message << "::cudaErrorInvalidPitchValue"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidSymbol:
		{
			std::cout<< message << " ::cudaErrorInvalidSymbol"  << std::endl;
			break;
		}
		case ::cudaErrorUnmapBufferObjectFailed:
		{
			std::cout<< message << " ::cudaErrorUnmapBufferObjectFailed"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidHostPointer:
		{
			std::cout<< message << " ::cudaErrorInvalidHostPointer"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidDevicePointer:
		{
			std::cout<< message << " ::cudaErrorInvalidDevicePointer"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidTexture:
		{
			std::cout<< message << " ::cudaErrorInvalidTexture"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidTextureBinding:
		{
			std::cout<< message << " ::cudaErrorInvalidTextureBinding"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidChannelDescriptor:
		{
			std::cout<< message << " ::cudaErrorInvalidChannelDescriptor"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidMemcpyDirection:
		{
			std::cout<< message << " ::cudaErrorInvalidMemcpyDirection"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidFilterSetting:
		{
			std::cout<< message << " ::cudaErrorInvalidFilterSetting"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidNormSetting:
		{
			std::cout<< message << " ::cudaErrorInvalidNormSetting"  << std::endl;
			break;
		}
		case ::cudaErrorUnknown:
		{
			std::cout<< message << " ::cudaErrorUnknown"  << std::endl;
			break;
		}
		case ::cudaErrorInvalidResourceHandle:
		{
			std::cout<< message << " ::cudaErrorInvalidResourceHandle"  << std::endl;
			break;
		}
		case ::cudaErrorInsufficientDriver:
		{
			std::cout<< message << " ::cudaErrorInsufficientDriver"  << std::endl;
			break;
		}
		case ::cudaErrorSetOnActiveProcess:
		{
			std::cout<< message << " ::cudaErrorSetOnActiveProcess"  << std::endl;
			break;
		}
		case ::cudaErrorStartupFailure:
		{
			std::cout<< message << " ::cudaErrorStartupFailure"  << std::endl;
			break;
		}
		case ::cudaErrorIllegalAddress:
		{
			std::cout<< message << " ::cudaErrorIllegalAddress"  << std::endl;
			break;
		}
		default:
		{
			std::cout<< message << " error_code: "  << err << std::endl;
			break;
		}
	}
}

void CCUDA_AX_B_SolverWrapper::cout_cusolverStatus_t(cusolverStatus_t err, string message)
{
	switch(err)
	{
		case ::CUSOLVER_STATUS_SUCCESS:
		{
			std::cout << message << " ::CUSOLVER_STATUS_SUCCESS"  << std::endl;
			break;
		}
		case ::CUSOLVER_STATUS_NOT_INITIALIZED:
		{
			std::cout<< message << " ::CUSOLVER_STATUS_NOT_INITIALIZED"  << std::endl;
			break;
		}
		case ::CUSOLVER_STATUS_ALLOC_FAILED:
		{
			std::cout<< message << " ::CUSOLVER_STATUS_ALLOC_FAILED"  << std::endl;
			break;
		}
		case ::CUSOLVER_STATUS_ARCH_MISMATCH:
		{
			std::cout<< message << " ::CUSOLVER_STATUS_ARCH_MISMATCH"  << std::endl;
			break;
		}
		default:
		{
			std::cout<< message << " error_code: "  << err << std::endl;
			break;
		}
	}
}

void CCUDA_AX_B_SolverWrapper::cout_cublasStatus_t(cublasStatus_t err, string message)
{
		switch(err)
		{
			case ::CUBLAS_STATUS_SUCCESS:
			{
				std::cout << message << " ::CUBLAS_STATUS_SUCCESS"  << std::endl;
				break;
			}
			case ::CUBLAS_STATUS_NOT_INITIALIZED:
			{
				std::cout<< message << " ::CUBLAS_STATUS_NOT_INITIALIZED"  << std::endl;
				break;
			}
			case ::CUBLAS_STATUS_ALLOC_FAILED:
			{
				std::cout<< message << " ::CUBLAS_STATUS_ALLOC_FAILED"  << std::endl;
				break;
			}

			default:
			{
				std::cout<< message << " error_code: "  << err << std::endl;
				break;
			}
		}
}
