#include "CUDASupport.h"
#include "histogram.cuh"

#include <iostream>
#include <cstring>

using namespace std;

__global__
void histogram(const unsigned int* const d_inputVals,
				unsigned int* const d_index,
				const size_t numElems,
				const size_t numBins,
				const int numWorks,
				const int numHistCols,
				unsigned int* const d_hist,
				const unsigned int mask,
				const unsigned int digitOrder)
{
	unsigned int bin;
	int idxHist, currentIdx, currentCol;
	currentCol = threadIdx.x + blockDim.x * blockIdx.x;
	if (currentCol >= numHistCols) return;

	int beginIdx = (blockIdx.x * blockDim.x + threadIdx.x) * numWorks;
	for (int idx = 0; idx < numWorks; ++idx) {
		currentIdx = beginIdx + idx;
		if (currentIdx >= numElems) break;

		bin = (d_inputVals[currentIdx] & mask) >> digitOrder;
		idxHist = bin * numHistCols + currentCol;
		d_index[currentIdx] = d_hist[idxHist]++;
	}
}

unsigned int* calc_histogram(const unsigned int* const d_inputVals,
							unsigned int* const d_index,
							const size_t numElems,
							const size_t numBins,
							const unsigned int mask,
							const unsigned int digitOrder,
							const dim3 blockSize)
{
	// Cho mỗi thread tương ứng với 1 block (1 cột) trong d_hist
	const int numWorks = blockSize.x;
	const int numHistCols = (numElems - 1) / blockSize.x + 1;
	const dim3 gridSize(numHistCols);
	const int numThreads = numHistCols;
	const dim3 newBlockSize = dim3(std::min(numThreads, 32));
	const dim3 newGridSize((numThreads - 1) / newBlockSize.x + 1);

	unsigned int *d_hist;
	int histSize = sizeof(unsigned int) * numBins * gridSize.x;
	checkCudaErrors(cudaMalloc(&d_hist, histSize));
	checkCudaErrors(cudaMemset(d_hist, 0, histSize));

	histogram<<<newGridSize, newBlockSize>>>(d_inputVals, d_index, numElems, numBins, numWorks,
											numHistCols, d_hist, mask, digitOrder);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	return d_hist;
}
