#include "CUDA_1.cuh"

using std::cout;
using std::endl;

CMatrixMultiply::CMatrixMultiply(void)
{

}

CMatrixMultiply::~CMatrixMultiply(void)
{

}

__global__ void helloWorld(char *str)
{
	int idx = __cudaGet_blockIdx().x * __cudaGet_blockDim().x + __cudaGet_threadIdx().x;
	str[idx] += idx;
}

char* CMatrixMultiply::cuda_example(char *str)
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
	helloWorld <<< dimGrid, dimBlock >>> (d_str);

	// retrieve the results from the device
	cudaMemcpy(str, d_str, size, cudaMemcpyDeviceToHost);

	// free up the allocated memory on the device
	delete[] d_str;
	cudaFree(d_str);

	return str;
}

// Allocate device memory for M, N and P
// copy M and N to allocated device memory location
// Kernel invocation code to let the device perform the actual multiplication
// Read P from the device
// Free device matrices	​

bool CMatrixMultiply::MatrixMultiplyUsingCPU(const dim3& dimsM, const dim3& dimsN)‏
{
	// Allocate device memory for M, N and P
	// M
	int size_M = dimsM.x * dimsM.y * dimsM.z;		// BLOCKSIZE, BLOCKSIZE, 1
	int sizeOfMemory_M = size_M * sizeof(float);
	float* f_M = new float[sizeOfMemory_M];
	// N
	int size_N = dimsN.x * dimsN.y * dimsN.z;		// BLOCKSIZE, BLOCKSIZE, 1
	int sizeOfMemory_N = size_N * sizeof(float);
	float* f_N = new float[sizeOfMemory_N];
	// P
	dim3 dimsP;
	int sizeOfMemory_P = dimsP.x * dimsP.y * dimsP.z * sizeof(float);
	float* f_P = new float[sizeOfMemory_P];


	// copy M and N to allocated device memory location

	// Kernel invocation code to let the device perform the actual multiplication

	// Read P from the device

	// Free device matrices

}

bool CMatrixMultiply::MatrixMultiplyUsingCUDA(const dim3& dimsM, const dim3& dimsN)‏
{

}

bool CMatrixMultiply::matrixMultiplyUsingCUBLAS(const dim3& dimsM, const dim3& dimsN)
{
	return false;
}