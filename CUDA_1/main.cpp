#include <iostream>
#include "CUDA_1.cuh"

int sum_int(int a, int b);
int sum_int(int a, int b)
{
	int c;
	c = a + b;
	return c;
}

int main(int argc, char **argv)
{
	int i;

	// desired output
	char str[] = "Hello World!";

	// mangle contents of output
	// the null character is left intact for simplicity
	for (i = 0; i < 12; i++)
		str[i] -= i;

	printf("%s\n", str);

	// cuda part
	CGPUACC gpuacc;
	printf("%d\n", gpuacc.cuda_example(str));

	getchar();
	return 0;
}