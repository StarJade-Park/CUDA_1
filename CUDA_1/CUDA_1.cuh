#pragma once
#include "Header.h"

#ifdef __cplusplus 
extern "C" {//<-- extern 시작
#endif

	class CMatrixMultiply
	{
	public:
		CMatrixMultiply(void);
		virtual ~CMatrixMultiply(void);

		char* cuda_example(char *str);

		bool MatrixMultiplyUsingCPU()‏;
		bool MatrixMultiplyUsingCUDA()‏;
	};

#ifdef __cplusplus 
}
#endif