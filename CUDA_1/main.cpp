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

using namespace std;

#define BLOCKSIZE 32

void example_cuda();
int MatrixMultiplication(const dim3& dimsA, const dim3& dimsB);

int main(int argc, char **argv)
{
	//example_cuda();

	dim3 dimSmall(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 dimBig(BLOCKSIZE * 50, BLOCKSIZE * 50, 1);

	int MMResult = MatrixMultiplication(dimSmall, dimBig);

	if (MMResult != 0) {
		exit(1);
	}

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
	CMatrixMultiply cuda;
	char* cuda_str = cuda.cuda_example(str);

	printf("%s\n", cuda_str);
}

int MatrixMultiplication(const dim3& dimsA, const dim3& dimsB)
{

	CMatrixMultiply MM_parameter;
	bool MatrixMultiplyResult;

	cout << "GPU matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MM_parameter.MatrixMultiplyUsingCUDA();

	if (!MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	cout << endl;

	cout << "CPU matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MM_parameter.MatrixMultiplyUsingCPU();
	if (!MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}

	cout << "All(CPU, GPU) matrix multiplication is complete!" << endl;

	return 0;
}