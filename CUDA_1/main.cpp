/*
	# Matrix multiplication
	- 정방형의 행렬 곱셈
	- P = M * N

	행렬의 크기를 변화시켜 처리 시간 측정 및 속도 향상 계산
	CPU VS with GPU?

	## 제출물
	CUDA 프로젝트, Screentshot(Result), discussion(report - .doc)
	file format - .zip

	## 추가
	메모리 활용 또는 CUBLAS를 사용하여 속도 향상 여부 확인
*/

#include <iostream>
#include "CUDA_1.cuh"

void example_cuda();

int main(int argc, char **argv)
{
	example_cuda();

	printf("\nEnd of program, press any key. ");
	getchar();
	return 0;
}

void example_cuda()
{
	// desired output
	char str[] = "Hello World!";

	// mangle contents of output
	// the null character is left intact for simplicity
	for (int i = 0; i < sizeof(str) - 1; i++)
		str[i] -= i;
	printf("%s\n", str);

	// cuda part
	GPUCUDA cuda;
	char* cuda_str = cuda.cuda_example(str);

	printf("%s\n", cuda_str);
}