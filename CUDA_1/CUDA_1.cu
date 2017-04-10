#include "CUDA_1.cuh"

using std::cout;
using std::endl;

CUDAExampleClass::CUDAExampleClass(void)
{

}

CUDAExampleClass::~CUDAExampleClass(void)
{

}

__global__ void helloWorld(char *str)
{
	int idx = __cudaGet_blockIdx().x * __cudaGet_blockDim().x + __cudaGet_threadIdx().x;
	str[idx] += idx;
}

char* CUDAExampleClass::cudaExample(char *str)
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