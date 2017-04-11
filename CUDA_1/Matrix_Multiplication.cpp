﻿#include "Matrix_Multiplication.h"

// System includes
#include <iostream>
#include <cassert>

// CUDA runtime
#include "nvrtc_helper.h"
#include <cudaProfiler.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"
#include "helper_cuda_drvapi.h"
#include <cuda.h>
#include "cublas_v2.h"

// Machine zero
#define EPS		1.e-6

// Initial values of matrices A and B
#define	ValA	1.0f
#define	ValB	0.01f

// Single-precision scalar multiplier of CUBLAS matrix multiplication
#define	ALPHA	1.0f;	// A, B
#define	BETA	0.0f;	// C

// Repeat number of matrix multiplication
#define NLTER	10

// For to access "MatrixMul_kernel.cu"
#define CU_MatrixMul_Kernel			"CUDA_1.cu"
#define FN_matrixMulCUDA_block16	"matrixMulCUDA_block16"
#define FN_matrixMulCUDA_block32	"matrixMulCUDA_block32"

// Used names
using std::cout;
using std::cerr;
using std::endl;


int MatrixMultiplyUsingCPU(const dim3 &dimsA, const dim3 &dimsB) {

	// Allocate memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(MATRIX_DT);
	MATRIX_DT *h_A = new MATRIX_DT[memSizeA];

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(MATRIX_DT);
	MATRIX_DT *h_B = new MATRIX_DT[memSizeB];

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memSizeC = dimsC.x * dimsC.y * sizeof(MATRIX_DT);
	MATRIX_DT *h_C = new MATRIX_DT[memSizeC];

	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		cerr << "Failed to allocate matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);

	// Invoke the multiply function with timer
	cout << "Computing result using CPU..." << endl;

	double totalLeadTime = 0;
	StopWatchWin pfTimer;
	pfTimer.start();
	for (int i = 0; i < NLTER; i++) {
		StopWatchWin timer;
		timer.start();
		MatrixMultiplyCPU(h_C, h_A, h_B, dimsC.x, dimsC.y, dimsA.x);
		timer.stop();

		totalLeadTime += timer.getTime();
		cout << "Lead time (" << i << "): " << timer.getTime() << endl;
	}
	pfTimer.stop();
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;
	cout << "Performance time: " << pfTimer.getTime() << endl;

	// Check the result
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Clean up memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	if (correct) {
		return EXIT_SUCCESS;
	}
	else {
		return EXIT_FAILURE;
	}
}

int MatrixMultiplyUsingCUDA(const dim3 &dimsA, const dim3 &dimsB, const int blockSize) {

	// Allocate host memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(MATRIX_DT);
	MATRIX_DT *h_A = new MATRIX_DT[memSizeA];

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(MATRIX_DT);
	MATRIX_DT *h_B = new MATRIX_DT[memSizeB];

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memSizeC = dimsC.x * dimsC.y * sizeof(MATRIX_DT);
	MATRIX_DT *h_C = new MATRIX_DT[memSizeC];

	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		cerr << "Failed to allocate host matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize host memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);

	// Initialize device
	cudaFree(0);

	// Allocate device memory
	CUdeviceptr d_A, d_B, d_C;
	checkCudaErrors(cuMemAlloc(&d_A, memSizeA));
	checkCudaErrors(cuMemAlloc(&d_B, memSizeB));
	checkCudaErrors(cuMemAlloc(&d_C, memSizeC));

	// Copy host memory to device
	checkCudaErrors(cuMemcpyHtoD(d_A, h_A, memSizeA));
	checkCudaErrors(cuMemcpyHtoD(d_B, h_B, memSizeB));

	// Setup execution parameters
	dim3 threads(blockSize, blockSize);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	checkCudaErrors(cuCtxSynchronize());

	// Execute the kernel with timer
	double totalLeadTime = 0;
	StopWatchWin pfTimer;
	pfTimer.start();
	for (int i = 0; i < NLTER; i++) {
		StopWatchWin timer;
		timer.start();

		// matrixMulCUDA(h_C, h_A, h_B, dimsA.x, dimsB.x);

		timer.stop();
		checkCudaErrors(cuCtxSynchronize());

		totalLeadTime += timer.getTime();
		cout << "Lead time (" << i << "): " << timer.getTime() << endl;
	}
	pfTimer.stop();
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;
	cout << "Performance time: " << pfTimer.getTime() << endl;

	// Copy result from device to host
	checkCudaErrors(cuMemcpyDtoH(h_C, d_C, memSizeC));
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Clean up memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	checkCudaErrors(cuMemFree(d_A));
	checkCudaErrors(cuMemFree(d_B));
	checkCudaErrors(cuMemFree(d_C));

	cuProfilerStop();

	if (correct) {
		return EXIT_SUCCESS;
	}
	else {
		return EXIT_FAILURE;
	}
}

int MatrixMultiplyUsingCUBLAS(const dim3 &dimsA, const dim3 &dimsB, int blockSize) {

	// Allocate host memory for matrices A, B and C
	unsigned int sizeA = dimsA.x * dimsA.y;
	unsigned int memSizeA = sizeA * sizeof(MATRIX_DT);
	MATRIX_DT *h_A = new MATRIX_DT[memSizeA];

	unsigned int sizeB = dimsB.x * dimsB.y;
	unsigned int memSizeB = sizeB * sizeof(MATRIX_DT);
	MATRIX_DT *h_B = new MATRIX_DT[memSizeB];

	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int memSizeC = dimsC.x * dimsC.y * sizeof(MATRIX_DT);
	MATRIX_DT *h_C = new MATRIX_DT[memSizeC];

	if (h_A == NULL || h_B == NULL || h_C == NULL) {
		cerr << "Failed to allocate host matrix!" << endl;
		exit(EXIT_FAILURE);
	}

	// Initialize host memory
	constantInit(h_A, sizeA, ValA);
	constantInit(h_B, sizeB, ValB);

	// Allocate device memory
	MATRIX_DT *d_A, *d_B, *d_C;
	checkCudaErrors(cudaMalloc((void **)&d_A, memSizeA));
	checkCudaErrors(cudaMalloc((void **)&d_B, memSizeB));
	checkCudaErrors(cudaMalloc((void **)&d_C, memSizeC));

	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_A, h_A, memSizeA, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, memSizeB, cudaMemcpyHostToDevice));

	// Perform matrix mutiply function of CUBLAS with timer
	cout << "Computing result using CUBLAS..." << endl;

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));

	const float alpha = ALPHA;
	const float beta = BETA;

	// Perform warmup operation with CUBLAS
	checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
		dimsB.x, dimsA.y, dimsA.x,
		&alpha, d_B, dimsB.x, d_A, dimsA.x, &beta, d_C, dimsB.x));
	checkCudaErrors(cuCtxSynchronize());

	double totalLeadTime = 0;
	StopWatchWin pfTimer;
	pfTimer.start();
	for (int i = 0; i < NLTER; i++) {
		StopWatchWin timer;
		timer.start();

		checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
			dimsB.x, dimsA.y, dimsA.x,
			&alpha, d_B, dimsB.x, d_A, dimsA.x, &beta, d_C, dimsB.x));

		timer.stop();
		checkCudaErrors(cuCtxSynchronize());

		totalLeadTime += timer.getTime();
		cout << "Lead time (" << i << "): " << timer.getTime() << endl;
	}
	pfTimer.stop();
	cout << "Total lead time: " << totalLeadTime << endl;
	cout << "Average lead time: " << totalLeadTime / NLTER << endl;
	cout << "Performance time: " << pfTimer.getTime() << endl;

	// Copy result from device to host
	checkCudaErrors(cudaMemcpy(h_C, d_C, memSizeC, cudaMemcpyDeviceToHost));
	bool correct = CheckResult(dimsA, dimsC, h_C);

	// Destroy the handle
	checkCudaErrors(cublasDestroy(handle));

	// Clean up memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));

	cudaDeviceReset();

	if (correct) {
		return EXIT_SUCCESS;	// return value = 1
	}
	else {
		return EXIT_FAILURE;	// return value = 0
	}
}


void MatrixMultiplyCPU(MATRIX_DT* C, MATRIX_DT* A, MATRIX_DT* B, int wC, int hC, int wA) {

	for (int i = 0; i < hC; i++) {
		for (int j = 0; j < wC; j++) {
			//C[i * wC + j] = matrixSubMulCPU(i, j, A, B, wA, wC);
			// Csub is used to store the element
			MATRIX_DT Csub = 0;

			// Loop over all the sub-matrices of A and B
			for (int k = j, l = i; k <= j + wA - 1; k++, l += wC) {
				// Multiply the two matrices together
				Csub += A[k] * B[l];
			}
			C[i * wC + j] = Csub;
		}
	}
}

bool CheckResult(const dim3& dimsA, const dim3& dimsC, const MATRIX_DT* h_C) {

	// Check the result
	cout << "Checking computed result for correctness: ";

	bool correct = true;

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps

	for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++) {
		double abs_err = fabs(h_C[i] - (dimsA.x * ValB));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err / abs_val / dot_length;

		if (rel_err > EPS) {
			cerr << "Error! Matrix[" << i << " ]=" << h_C[i]
				<< " , ref=" << dimsA.x * ValB
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

void constantInit(MATRIX_DT dataArr[], const int arrSize, const MATRIX_DT value) {

	for (int i = 0; i < arrSize; i++) {
		dataArr[i] = value;
	}
}