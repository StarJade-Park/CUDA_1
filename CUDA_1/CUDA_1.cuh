#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus 
extern "C" {//<-- extern ½ÃÀÛ
#endif

	class GPUCUDA
	{
	public:
		GPUCUDA(void);
		virtual ~GPUCUDA(void);
		char* cuda_example(char *str);
	};

#ifdef __cplusplus 
}
#endif