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
#include "Matrix_Multiplication.h"

using namespace std;

int MatrixMultiplicationAll(const dim3& dimsA, const dim3& dimsB);

int main(int argc, char **argv)
{
	//example_cuda();

	dim3 dimSmall(BLOCKSIZE, BLOCKSIZE, 1);
	dim3 dimBig(BLOCKSIZE * 50, BLOCKSIZE * 50, 1);

	cout << "32 x 32 matrix multiplication" << endl;
	int MMResult = MatrixMultiplicationAll(dimSmall, dimSmall);
	if (MMResult) {
		exit(MMResult);
	}

	cout << endl << "==================================" << endl;
	cout << "1600 x 1600 matrix multiplication" << endl;
	MMResult = MatrixMultiplicationAll(dimBig, dimBig);

	// for debug
	printf("\nEnd of program, press any key. ");
	getchar();

	exit(MMResult);
}

int MatrixMultiplicationAll(const dim3& dimsM, const dim3& dimsN)
{
	int MatrixMultiplyResult;

	// init
	MatrixMultiplyResult = true;

	// CPU part
	cout << "CPU matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MatrixMultiplyUsingCPU(dimsM, dimsN);
	if (MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	cout << endl;

	// GPU part
	cout << "GPU matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MatrixMultiplyUsingCUDA(dimsM, dimsN);

	if (MatrixMultiplyResult) {
		return MatrixMultiplyResult;
	}
	cout << endl;

	// TODO : �߰� CUBLAS
	cout << "CUBLAS matrix multiplication is start!" << endl;
	MatrixMultiplyResult = MatrixMultiplyUsingCUBLAS(dimsM, dimsN);
	cout << "Matrix multiplication function end." << endl;

	return MatrixMultiplyResult;
}