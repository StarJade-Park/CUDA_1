#pragma once
#include "CUDA_1.cuh"
#include "helper_timer.h"


#define	ValM	1.0f
#define	ValN	0.01f

#define EPS		1.e-6

// Matrix Data Type
typedef float MATRIX_DT;

class MatrixMultiplication
{
public:
	MatrixMultiplication();
	~MatrixMultiplication();

	void	setTimer();
	double	getTimer();

	void InitMatrix(MATRIX_DT dataArr[], const int matrixSize, const MATRIX_DT value);

	bool CheckResult(const dim3& dimsM, const dim3& dimsP, const MATRIX_DT* f_P);

	bool MatrixMultiplyUsingCPU(const dim3& dimsM, const dim3& dimsN);
	bool MatrixMultiplyUsingCUDA(const dim3& dimsM, const dim3& dimsN);
	// 추가
	bool MatrixMultiplyUsingCUBLAS(const dim3& dimsM, const dim3& dimsN);


private:
	StopWatchWin mTimer;				// helper_timer.h

};
