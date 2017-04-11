#pragma once
#include "device_launch_parameters.h"

// GPU block size
#define BLOCKSIZE	32
#define BLOCK_SIZE	BLOCKSIZE

__global__ void matrixMulCUDA(float *C, float *A, float *B, int wA, int wB);