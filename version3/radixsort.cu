#include "radixsort.cuh"
#include "histogram.cuh"
#include "exscan.cuh"
#include "scatter.cuh"

#include <algorithm>
#include <time.h>

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
	double duration[3] = { 0, 0, 0 };
	for (unsigned int i = 0; i <= maxBits; i += numBits)
	{
		unsigned int mask = (numBins - 1) << i;

		//histogram 
		unsigned int *d_index;
		checkCudaErrors(cudaMalloc(&d_index, sizeof(unsigned int) * numElems));
		clock_t begin = clock();
		unsigned int* d_hist = calc_histogram(pInVals, d_index, numElems, numBins, mask, i, blockSize);
		clock_t end = clock();
		duration[0] += end - begin;

		// exclusive scan hist
		begin = clock();
		unsigned int* d_histScan = calc_exclusive_scan(d_hist, numBins * gridSize.x, blockSize);
		end = clock();
		duration[1] += end - begin;

		//scatter
		begin = clock();
		host_scatter(pInVals, pOutVals, d_index, numElems, numBins, d_histScan, mask, i, blockSize);
		end = clock();
		duration[2] += end - begin;

		std::swap(pInVals, pOutVals);

		checkCudaErrors(cudaFree(d_hist));
		checkCudaErrors(cudaFree(d_histScan));
		checkCudaErrors(cudaFree(d_index));
		d_hist = NULL;
		d_histScan = NULL;
	}
	if (pInVals == d_outputVals)
	{
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	}

	printf("Time histogram: %lf\n", duration[0]);
	printf("Time exscan: %lf\n", duration[1]);
	printf("Time scatter: %lf\n", duration[2]);
}


