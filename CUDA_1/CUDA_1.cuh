#pragma once
#include <cuda_runtime.h>

#ifdef __cplusplus 
extern "C" {//<-- extern ����
#endif

	class CGPUACC
	{
	public:
		CGPUACC(void);
		virtual ~CGPUACC(void);
		char* cuda_example(char *str);
	};

#ifdef __cplusplus 
}
#endif