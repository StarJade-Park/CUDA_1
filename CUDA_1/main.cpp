/*
	# Matrix multiplication
	- �������� ��� ����
	- P = M * N

	����� ũ�⸦ ��ȭ���� ó�� �ð� ���� �� �ӵ� ��� ���
	CPU VS with GPU?

	## ���⹰
	CUDA ������Ʈ, Screentshot(Result), discussion(report - .doc)
	file format - .zip

	## �߰�
	�޸� Ȱ�� �Ǵ� CUBLAS�� ����Ͽ� �ӵ� ��� ���� Ȯ��
*/
// #pragma warning(disable: 4996)
#include <iostream>
#include "CUDA_1.cuh"

using namespace std;

void example_cuda();
int MatrixMultiplication(const dim3& dimsA, const dim3& dimsB);

int main(int argc, char **argv)
{
	//example_cuda();

	dim3 dimSmall(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 dimBig(BLOCKSIZE * 50, BLOCKSIZE * 50, 1);

	int MMResult = MatrixMultiplication(dimSmall, dimSmall);
	if (MMResult != 0) {
		exit(1);
	}

	cout << endl << "==================" << endl;

	int MMResult = MatrixMultiplication(dimBig, dimBig);

	// for debug
	printf("\nEnd of program, press any key. ");
	getchar();

	exit(MMResult);
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

int MatrixMultiplication(const dim3& dimsM, const dim3& dimsN)
{

	CMatrixMultiply MM_parameter;	// CUDA_1.h
	StopWatchWin watch;				// helper_timer.h
	bool MatrixMultiplyResult;

	// init
	MatrixMultiplyResult = false;
	watch.reset();

	// GPU part
	cout << "GPU matrix multiplication is start!" << endl;
	watch.start();
	MatrixMultiplyResult = MM_parameter.MatrixMultiplyUsingCUDA(dimsM, dimsN);

	if (!MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	watch.stop();

	cout << "GPU matrix multiplication complete time : " << watch.getTime() << endl;
	cout << endl;

	// CPU part
	cout << "CPU matrix multiplication is start!" << endl;
	watch.reset();
	watch.start();
	MatrixMultiplyResult = MM_parameter.MatrixMultiplyUsingCPU(dimsM, dimsN);
	if (!MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	watch.stop();
	cout << "CPU matrix multiplication complete time : " << watch.getTime() << endl;
	cout << endl;

	cout << "All(CPU, GPU) matrix multiplication is complete!" << endl;

	// TODO : �߰� CUBLAS
	return 0;
}