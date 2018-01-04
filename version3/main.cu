#include "CUDASupport.h"

#include "radixsort.cuh"

#include <algorithm>
#include <cstring> //memset
#include <time.h>

void printArray(unsigned int* const h_input, const size_t numElems)
 {
	for (unsigned int i = 0; i < numElems; ++i)
		printf("%d  ", h_input[i]);
	printf("\n");
 }

__device__
 void dev_printArray(unsigned int* const d_input, const size_t numElems)
 {
	for (unsigned int i = 0; i < numElems; ++i)
		printf("%d \t", d_input[i]);
	printf("\n");
 }

int main()
{
	// srand(time(NULL));
	const dim3 blockSize(64);
	const size_t numElems = 10000000;
	const unsigned int numBits = 3;

	unsigned int* h_inputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	for (int i = 0; i < numElems; ++i)
	{
		h_inputVals[i] = rand() % 1000000000 + 1;
	}
	
	// printArray(h_inputVals, numElems);

	unsigned int* d_inputVals;
	checkCudaErrors(cudaMalloc(&d_inputVals, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMemcpy(d_inputVals, h_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));

	unsigned int* d_outputVals;
	checkCudaErrors(cudaMalloc(&d_outputVals, sizeof(unsigned int) * numElems));

	clock_t begin = clock();
	radix_sort(h_inputVals, d_inputVals, d_outputVals, numElems, numBits, blockSize);
	clock_t end = clock();
	double duration = end - begin;
	printf("Time: %lf\n", duration);

	unsigned int* h_outputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	checkCudaErrors(cudaMemcpy(h_outputVals, d_outputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));

	// printArray(h_outputVals, numElems);
	bool check = true;
	for (int i = 1; i < numElems; ++i)
	{
		if (h_outputVals[i] < h_outputVals[i - 1])
		{
			printf("\nfalse at index : %d\n", i);
			check = false;
			break;
		}
	}

	if (check)
		printf("\nTRUE\n");
	else
		printf("\nFALSE\n");


	
	free(h_inputVals);
	free(h_outputVals);
	checkCudaErrors(cudaFree(d_inputVals));
	checkCudaErrors(cudaFree(d_outputVals));
	return 0;
}
