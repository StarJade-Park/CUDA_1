#include "Matrix_Multiplication.h"
#include <iostream>

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "cublas_v2.h"

using std::cout;
using std::endl;
using std::cerr;

MatrixMultiplication::MatrixMultiplication()
{
	/* empty */
}

MatrixMultiplication::~MatrixMultiplication()
{
	/* empty */
}

void MatrixMultiplication::setTimer()
{
	mTimer.reset();
	mTimer.start();
}

double MatrixMultiplication::getTimer()
{
	mTimer.stop();
	return mTimer.getTime();
}

bool MatrixMultiplication::MatrixMultiplyUsingCPU(const dim3& dimsM, const dim3& dimsN){
	// M 
	unsigned int size_M = dimsM.x * dimsM.y;		// BLOCKSIZE, BLOCKSIZE
	unsigned int sizeOfMemory_M = size_M * sizeof(MATRIX_DT);
	MATRIX_DT* DT_M = new MATRIX_DT[sizeOfMemory_M];

	// N
	unsigned int size_N = dimsN.x * dimsN.y;
	unsigned int sizeOfMemory_N = size_N * sizeof(MATRIX_DT);
	MATRIX_DT* DT_N = new MATRIX_DT[sizeOfMemory_N];

	// P
	dim3 dimsP(dimsM.x, dimsN.y, 1);
	unsigned int sizeOfMemory_P = dimsP.x * dimsP.y * sizeof(MATRIX_DT);
	MATRIX_DT* DT_P = new MATRIX_DT[sizeOfMemory_P];

	if (DT_M == NULL || DT_N == NULL || DT_P == NULL) {
		cerr << "Failed to allocate matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// init host
	initMatrix(DT_M, size_M, ValA);
	initMatrix(DT_N, size_N, ValB);

	// multiply
	setTimer();
	for (unsigned int i = 0; i < dimsP.x; i++) {
		for (unsigned int j = 0; j < dimsP.y; j++) {
			MATRIX_DT Csub = 0;
			for (unsigned int k = j, l = i; k <= j + dimsM.x - 1; k++, l += dimsP.y) {
				// P = M * N
				Csub += DT_M[k] * DT_N[l];
			}
			DT_P[i * dimsP.x + j] = Csub;
		}
	}
	cout << "Matrix Multiply Using CPU : " << getTimer() << endl;

	// copy (device -> host)
	bool correct = CheckResult(dimsM, dimsP, DT_P);

	// free array
	delete[] DT_M;
	delete[] DT_N;
	delete[] DT_P;

	return correct;
}


// Allocate device memory for M, N and P
// copy M and N to allocated device memory location
// Kernel invocation code to let the device perform the actual multiplication
// Read P from the device
// Free device matrices	​

bool MatrixMultiplication::MatrixMultiplyUsingCUDA(const dim3& dimsM, const dim3& dimsN) {
	return true;
}

bool MatrixMultiplication::MatrixMultiplyUsingCUBLAS(const dim3& dimsM, const dim3& dimsN) {
	return true;
}

void MatrixMultiplication::initMatrix(MATRIX_DT dataArr[], const int matrixSize, const MATRIX_DT value) {
	for (int i = 0; i < matrixSize; i++) {
		dataArr[i] = value;
		//cout << dataArr[i] << endl;
	}
}

bool MatrixMultiplication::CheckResult(const dim3& dimsM, const dim3& dimsP, const MATRIX_DT* f_P) {

	// Check the result
	cout << "Checking computed result for correctness :" << endl;

	bool correct = true;

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps

	for (int i = 0; i < (int)(dimsP.x * dimsP.y); i++) {
		double abs_err = fabs(f_P[i] - (dimsM.x * ValB));
		double dot_length = dimsM.x;
		double abs_val = fabs(f_P[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > EPS) {
			cerr << "Error! Matrix[" << i << " ]=" << f_P[i]
				<< " , ref=" << dimsM.x * ValB
				<< " error term is > " << EPS << endl;
			correct = false;
		}
	}

	if (correct) {
		cout << "Result = PASS" << endl;
	}
	else {
		cerr << "Result = FAIL" << endl;
	}

	return correct;
}