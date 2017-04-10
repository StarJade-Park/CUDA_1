/*
void MatrixMultiplication(float* M, float* N, float* P, int Width)‏
​{
​   int size = Width * Width * sizeof(float); 
​    float* Md, Nd, Pd;

​ // Allocate device memory for M, N and P

​ // and copy M and N to allocated device memory location
​
​ // Kernel invocation code to let the device perform the actual multiplication
​
​ // Read P from the device

​ // Free device matrices
*/

#pragma once
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cufft.h>
#include "cuda.h"
#include "device_launch_parameters.h"
#include "helper_functions.h"
#include "helper_timer.h"
#include "helper_cuda.h"
#include "cublas_v2.h"

#ifdef __cplusplus 
extern "C" {
#endif

#define BLOCKSIZE 32

	class MulCUDA
	{
	public:
		MulCUDA(void);
		virtual ~MulCUDA(void);

		char* cudaExample(char *str);
		//Error	illegal combination of memory qualifiers
		//__global__ void mulMatrixCUDA(float * P, float * M, float * N, int widthM, int widthN);
	};

#ifdef __cplusplus 
}
#endif