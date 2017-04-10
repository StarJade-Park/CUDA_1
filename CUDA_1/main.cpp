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
// #pragma warning(disable: 4996)
#include <iostream>
#include "Matrix_Multiplication.h"

using namespace std;

void exampleCuda();
int MatrixMultiplicationAll(const dim3& dimsA, const dim3& dimsB);

int main(int argc, char **argv)
{
	//example_cuda();

	dim3 dimSmall(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 dimBig(BLOCKSIZE * 50, BLOCKSIZE * 50, 1);

	cout << "32 x 32 matrix" << endl;
	int MMResult = MatrixMultiplicationAll(dimSmall, dimSmall);
	if (MMResult != 0) {
		exit(1);
	}

	cout << endl << "==================================" << endl;
	cout << "1600 x 1600 matrix" << endl;
	MMResult = MatrixMultiplicationAll(dimBig, dimBig);

	// for debug
	printf("\nEnd of program, press any key. ");
	getchar();

	exit(MMResult);
}

void exampleCuda()
{
	// desired output
	char str[] = "Hello World!";

	// mangle contents of output
	// the null character is left intact for simplicity
	for (int i = 0; i < sizeof(str) - 1; i++)
		str[i] -= i;
	printf("%s\n", str);

	// cuda part
	MulCUDA cuda;
	char* cuda_str = cuda.cudaExample(str);

	printf("%s\n", cuda_str);
}

int MatrixMultiplicationAll(const dim3& dimsM, const dim3& dimsN)
{
	MatrixMultiplication MM_parameter;	// CUDA_1.h
	bool MatrixMultiplyResult;

	// init
	MatrixMultiplyResult = true;

	// CPU part
	cout << "CPU matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MM_parameter.MatrixMultiplyUsingCPU(dimsM, dimsN);
	if (!MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	cout << endl;

	// GPU part
	cout << "GPU matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MM_parameter.MatrixMultiplyUsingCUDA(dimsM, dimsN);

	if (!MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	cout << endl;

	// TODO : 추가 CUBLAS
	cout << "CUBLAS matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MM_parameter.MatrixMultiplyUsingCUBLAS(dimsM, dimsN);
	if (!MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	cout << endl;

	cout << "Matrix multiplication function end." << endl;

	return 0;
}