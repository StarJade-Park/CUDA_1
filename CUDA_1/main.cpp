#include <iostream>
#include "CUDA_1.cuh"

int main(int argc, char **argv)
{
	int i;

	// desired output
	char str[] = "Hello World!";

	// mangle contents of output
	// the null character is left intact for simplicity
	for (i = 0; i < sizeof(str)-1; i++)
		str[i] -= i;
	printf("%s\n", str);

	// cuda part
	GPUCUDA cuda;
	char* cuda_str = cuda.cuda_example(str);
	printf("%s\n", cuda_str);

	printf("\nEnd of program, press any key. ");
	getchar();
	return 0;
}