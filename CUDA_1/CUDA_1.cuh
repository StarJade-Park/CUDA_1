#pragma once
#include "Header.h"

#define BLOCKSIZE 32

#ifdef __cplusplus 
extern "C" {
#endif

	class CMatrixMultiply
	{
	public:
		CMatrixMultiply(void);
		virtual ~CMatrixMultiply(void);

		char* cuda_example(char *str);

		bool MatrixMultiplyUsingCPU(const dim3& dimsM, const dim3& dimsN)‏;
		bool MatrixMultiplyUsingCUDA(const dim3& dimsM, const dim3& dimsN)‏;

		// 추가
		bool matrixMultiplyUsingCUBLAS(const dim3& dimsM, const dim3& dimsN);
	};

#ifdef __cplusplus 
}
#endif