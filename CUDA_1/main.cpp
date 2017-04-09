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