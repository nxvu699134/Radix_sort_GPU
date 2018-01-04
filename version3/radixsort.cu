#include "radixsort.cuh"
#include "histogram.cuh"
#include "exscan.cuh"
#include "scatter.cuh"

#include <algorithm>

#define MAX_INT_BITS	32

unsigned int getMaxNumOfBits(unsigned int* h_input, const size_t numElems)
{
	unsigned int maxElem = *std::max_element(h_input, h_input + numElems);
	unsigned int mask = 1 << 31;
	unsigned int count = 0;
	while (! (mask & maxElem))
	{
		++count;
		mask >>= 1;
	}
	return MAX_INT_BITS - count;
}

void radix_sort(unsigned int* h_inputVals,
				unsigned int* d_inputVals, 
				unsigned int* d_outputVals,
				const size_t numElems,
				const size_t numBits,
				const dim3 blockSize)
{
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);
	size_t numBins = 1 << numBits;
	
	unsigned int* pInVals = d_inputVals;
	unsigned int* pOutVals = d_outputVals;

	unsigned int maxBits = getMaxNumOfBits(h_inputVals, numElems);

	if (maxBits % numBits)
		maxBits += numBits;

	// loop through digits
	for (unsigned int i = 0; i <= maxBits; i += numBits)
	{
		unsigned int mask = (numBins - 1) << i;

		// printf("mask: %d\n", mask);
		//histogram 
		unsigned int* d_hist = host_histogram(pInVals, numElems, numBins, mask, i, blockSize);

		// unsigned int* h_hist = (unsigned int*) malloc (sizeof(unsigned int) * numBins * gridSize.x);
		// checkCudaErrors(cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * numBins * gridSize.x, cudaMemcpyDeviceToHost));
		// printArray(h_hist, numBins * gridSize.x);
		// free(h_hist);

		// exclusive scan hist
		unsigned int* d_histScan = host_exclusive_scan(d_hist, numBins * gridSize.x, blockSize);

		// unsigned int* h_histScan = (unsigned int*) malloc (sizeof(unsigned int) * numBins * gridSize.x);
		// checkCudaErrors(cudaMemcpy(h_histScan, d_histScan, sizeof(unsigned int) * numBins * gridSize.x, cudaMemcpyDeviceToHost));
		// printArray(h_histScan, numBins * gridSize.x);
		// free(h_histScan);

		//scatter
		host_scatter(pInVals, pOutVals, numElems, numBins, d_histScan, mask, i, blockSize);

		std::swap(pInVals, pOutVals);
		// unsigned int* h_result = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
		// checkCudaErrors(cudaMemcpy(h_result, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
		// printArray(h_result, numElems);
		// free(h_result);
		// printf("\n\n\n");

		checkCudaErrors(cudaFree(d_hist));
		checkCudaErrors(cudaFree(d_histScan));
		d_hist = NULL;
		d_histScan = NULL;
	}
	if (pInVals == d_outputVals)
	{
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	}
}


