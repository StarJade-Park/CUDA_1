#include "CUDA_1.cuh"
#include "cuda.h"
#include <iostream>
#include <cufft.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include "device_launch_parameters.h"


GPUCUDA::GPUCUDA(void)
{

}

GPUCUDA::~GPUCUDA(void)
{
}

__global__ void helloWorld(char *str)
{
	int idx = __cudaGet_blockIdx().x * __cudaGet_blockDim().x + __cudaGet_threadIdx().x;
	str[idx] += idx;
}

__global__ void multiplicationKernel(float* lf, int Width)
{
	int idx = __cudaGet_blockIdx().x * __cudaGet_blockDim().x + __cudaGet_threadIdx().x;
	int idy = __cudaGet_blockIdx().y * __cudaGet_blockDim().y + __cudaGet_threadIdx().y;



}

char* GPUCUDA::cuda_example(char *str)
{
	// allocate memory on the device
	char *d_str;
	size_t size = sizeof(str);
	cudaMalloc((void**)&d_str, size);

	// copy the string to the device
	cudaMemcpy(d_str, str, size, cudaMemcpyHostToDevice);

	// set the grid and block sizes
	dim3 dimGrid(2);	// one block per word
	dim3 dimBlock(6);	// one thread per character

	// invoke the kernel
	helloWorld << < dimGrid, dimBlock >> > (d_str);

	// retrieve the results from the device
	cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);

	// free up the allocated memory on the device
	cudaFree(d_str);

	return str;
}

void GPUCUDA::MatrixMultiplication(float* M, float* N, float* P, int Width)‏
{
	int size = Width * Width * sizeof(float);
	float* Md, Nd, Pd;

	// Allocate device memory for M, N and P

	cudaArray* cuArray;

	// copy M and N to allocated device memory location

	// Kernel invocation code to let the device perform the actual multiplication

	// Read P from the device

	// Free device matrices	​

}