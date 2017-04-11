#pragma once

// CUDA runtime
#include <cuda_runtime.h>
#include "CUDA_1.cuh"

// Matrix data type
typedef float MATRIX_DT;

// Prototypes

/**
* Run a matrix multiplication using CUDA
*/
int MatrixMultiplyUsingCUDA(const dim3 &dimsA, const dim3 &dimsB, const int blockSize = BLOCKSIZE);

/**
* Run a matrix multiplication using CUBLAS
*/
int MatrixMultiplyUsingCUBLAS(const dim3 &dimsA, const dim3 &dimsB, int blockSize = BLOCKSIZE);

/**
* Run a matrix multiplication using CPU
*/
int MatrixMultiplyUsingCPU(const dim3 &dimsA, const dim3 &dimsB);

/**
* Matrix multiplication on the CPU: C = A * B
* wC is C's width, hC is C's height and wA is A's width
*/
void MatrixMultiplyCPU(MATRIX_DT* C, MATRIX_DT* A, MATRIX_DT* B, int wC, int hC, int wA);

/**
* Check result of multiplication
*/
bool CheckResult(const dim3& dimsA, const dim3& dimsC, const MATRIX_DT* h_C);

/**
* Initialize DATA array
*/
void constantInit(MATRIX_DT dataArr[], const int arrSize, const MATRIX_DT value);