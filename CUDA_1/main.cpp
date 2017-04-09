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