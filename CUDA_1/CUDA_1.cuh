#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus 
extern "C" {//<-- extern 시작
#endif

	class GPUCUDA
	{
	public:
		GPUCUDA(void);
		virtual ~GPUCUDA(void);
		char* cuda_example(char *str);
		void MatrixMultiplication(float* M, float* N, float* P, int Width)‏;
	};

#ifdef __cplusplus 
}
#endif