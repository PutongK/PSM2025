///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// HLBFGS                                                                    //
// http://www.loria.fr/~liuyang/software/HLBFGS/							 //
//                                                                           //
// HLBFGS is a hybrid L-BFGS optimization framework which unifies L-BFGS     //
// method, Preconditioned L-BFGS method and                                  //
// Preconditioned Conjugate Gradient method.                                 //
//                                                                           //
// Version 1.2                                                               //
// March 09, 2010                                                            //
//                                                                           //
// Copyright (C) 2009--2010                                                  //
// Yang Liu                                                                  //
//																			 //
// xueyuhanlang@gmail.com                                                    //
//                                                                           //
// HLBFGS is HLBFGS is freely available for non-commercial purposes.		 //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
#ifdef HLBFGS_EIGEN
#include <Eigen/Dense>
#else
#ifdef HLBFGS_OPENMP
#include <omp.h>
#endif
#endif

#include "HLBFGS_BLAS.h"
#include <cmath>

double HLBFGS_DDOT(const int n, const double *x, const double *y)
{
#ifdef HBLFGS_EIGEN
    Eigen::Map<const Eigen::VectorXd> xv(x, n);
    Eigen::Map<const Eigen::VectorXd> yv(y, n);
    const double result = xv.dot(yv);
#else
    double result = 0;
    int i = 0;
#ifdef HLBFGS_OPENMP
#pragma omp parallel for private(i) reduction(+:result)
#endif
	for (i = 0; i < n; i++)
	{
		result += x[i] * y[i];
	}
#endif
	return result;
}

void HLBFGS_DAXPY(const int n, const double alpha, const double *x, double *y)
{
#ifdef HBLFGS_EIGEN
    Eigen::Map<const Eigen::VectorXd> xv(x, n);
    Eigen::Map<Eigen::VectorXd> yv(y, n);
    yv += alpha*xv;
#else
	int i = 0;
#ifdef HLBFGS_OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < n; i++)
	{
		y[i] += alpha * x[i];
	}
#endif
}

double HLBFGS_DNRM2(const int n, const double *x)
{
#ifdef HBLFGS_EIGEN
    Eigen::Map<const Eigen::VectorXd> xv(x, n);
    const double result = xv.norm();
#else
	double result = 0;
	int i = 0;
#ifdef HLBFGS_OPENMP
#pragma omp parallel for private(i) reduction(+:result)
#endif
	for (i = 0; i < n; i++)
	{
		result += x[i] * x[i];
	}
	result = std::sqrt(result);
#endif
    return result;
}

void HLBFGS_DSCAL(const int n, const double a, double *x)
{
#ifdef HBLFGS_EIGEN
    Eigen::Map<Eigen::VectorXd> xv(x, n);
    xv *= a;
#else
	int i = 0;
#ifdef HLBFGS_OPENMP
#pragma omp parallel for private(i)
#endif
	for (i = 0; i < n; i++)
	{
		x[i] *= a;
	}
#endif
}

