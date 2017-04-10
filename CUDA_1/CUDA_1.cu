#include "CUDA_1.cuh"

using std::cout;
using std::endl;

MulCUDA::MulCUDA(void)
{

}

MulCUDA::~MulCUDA(void)
{

}

__global__ void helloWorld(char *str)
{
	int idx = __cudaGet_blockIdx().x * __cudaGet_blockDim().x + __cudaGet_threadIdx().x;
	str[idx] += idx;
}

char* MulCUDA::cudaExample(char *str)
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

__global__ void mulMatrixCUDA(float *P, float *M, float *N, int widthM, int widthN)
{
	int mBegin	= widthM * widthN * BLOCKSIZE;
	int mEnd	= mBegin + widthM - 1;
	int mStep	= BLOCKSIZE;
	
	int nBegin	= BLOCKSIZE * __cudaGet_blockIdx().x;
	int nStep	= BLOCKSIZE * widthN;

	float Csub	= 0;

	for (int mIndex = mBegin, nIndex = nBegin; mIndex <= mEnd; mIndex += mStep, nIndex += nStep)
	{
		__shared__ float Ms[BLOCKSIZE][BLOCKSIZE];
		__shared__ float Ns[BLOCKSIZE][BLOCKSIZE];

		Ms[__cudaGet_threadIdx().y][__cudaGet_threadIdx().x]
			= M[mBegin + widthM * __cudaGet_threadIdx().y + __cudaGet_threadIdx().x];

		Ns[__cudaGet_threadIdx().y][__cudaGet_threadIdx().x]
			= N[nBegin + widthN * __cudaGet_threadIdx().y + __cudaGet_threadIdx().x];


		for (int i = 0; i < BLOCKSIZE; ++i)
		{
			Csub += Ms[__cudaGet_threadIdx().y][i] * Ns[i][__cudaGet_threadIdx().x];
		}

	}


	int pIndex = widthM * BLOCKSIZE * __cudaGet_blockIdx().y + BLOCKSIZE * __cudaGet_blockIdx().x;
	P[pIndex + widthM * __cudaGet_threadIdx().y + __cudaGet_threadIdx().x];
}

