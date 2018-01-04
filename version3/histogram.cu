#include "CUDASupport.h"
#include "histogram.cuh"


__global__
void histogram(const unsigned int* const d_inputVals,
				const size_t numElems,
				const size_t numBins,
				unsigned int* const d_hist,
				const unsigned int mask,
				const unsigned int digitOrder)
{
	extern __shared__ unsigned int s_hist[]; // numBins

	for (unsigned int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
	{
		s_hist[bin] = 0;
	}
	__syncthreads();

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElems)
	{
		unsigned int bin = (d_inputVals[idx] & mask) >> digitOrder;
		atomicAdd(&s_hist[bin], 1);
	}
	__syncthreads();
	
	//update d_hist by s_hist
	for (unsigned int bin = threadIdx.x; bin < numBins; bin += blockDim.x)
	{
		int idxHist = bin * gridDim.x + blockIdx.x;
		d_hist[idxHist] = s_hist[bin];
		// printf("bin - indexHist - val : %d - %d - %d - %d - %d\n", blockIdx.x, bin, idxHist, d_hist[idxHist]);
	}
}

unsigned int* host_histogram(const unsigned int* const d_inputVals, 
							const size_t numElems,
							const size_t numBins,
							const unsigned int mask,
							const unsigned int digitOrder,
							const dim3 blockSize)
{	
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);

	unsigned int *d_hist;
	checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numBins * gridSize.x));
	checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * numBins * gridSize.x));

	int sharedSize = numBins * sizeof(unsigned int);

	histogram<<<gridSize, blockSize, sharedSize>>>(d_inputVals, 
													numElems, 
													numBins, 
													d_hist, 
													mask, 
													digitOrder);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	return d_hist;
}
