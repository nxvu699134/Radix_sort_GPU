#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring> //memset

#define MAX_INT_BITS	32

#define checkCudaErrors(call)                                                  \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}




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

__device__ unsigned int d_count = 0;
__device__ unsigned int d_curBlock = 0;

__global__
void set_init_scan()
{
	d_count = 0;
	d_curBlock = 0;
}

__global__
void exclusive_scan(const unsigned int* const d_in, 
					unsigned int* const d_out, 
					unsigned int* const d_prevSums,
					const int n)
{
	extern __shared__ unsigned int s_blkIn[]; //blockSize

	__shared__ unsigned int s_prevSum;
	__shared__ unsigned int s_bid;
	__shared__ unsigned int s_endBlock;
	
	
	// get block idx
	if (0 == threadIdx.x)
	{
		s_bid = atomicAdd(&d_count, 1);
		s_endBlock = blockDim.x - 1;
	}
	__syncthreads();
	
	int i = s_bid * blockDim.x + threadIdx.x;
	if (i == n - 1)
		s_endBlock = threadIdx.x;

	if (i >= n)
		return;
	
	// load data
	if (0 == i)
		s_blkIn[0] = 0;
	else
		s_blkIn[threadIdx.x] = d_in[i - 1];
    __syncthreads();
        
    /// reduction
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
		int localThreadIdx = 2 * (threadIdx.x + 1) * stride - 1;
		if (localThreadIdx < blockDim.x)
		{	
			s_blkIn[localThreadIdx] += s_blkIn[localThreadIdx - stride];
		}
		__syncthreads();
	}

	//post reduction
	for (int stride = blockDim.x / 4; stride > 0; stride /= 2)
	{
		int localThreadIdx = 2 * (threadIdx.x + 1) * stride - 1;
		if (localThreadIdx + stride < blockDim.x)
		{	
			s_blkIn[localThreadIdx + stride] += s_blkIn[localThreadIdx];
		}
		__syncthreads();
	}

	if (0 == threadIdx.x)
	{
		while(atomicAdd(&d_curBlock, 0) < s_bid);
		s_prevSum = d_prevSums[s_bid];
		d_prevSums[s_bid + 1] = s_prevSum + s_blkIn[s_endBlock];
		__threadfence();
		atomicAdd(&d_curBlock, 1);
	}
	__syncthreads();

	 // update output
	d_out[i] = s_blkIn[threadIdx.x] + s_prevSum;
}

unsigned int* host_exclusive_scan(const unsigned int* const d_in, const size_t numElems, const dim3 blockSize)
{
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);
	set_init_scan<<<1, 1>>>();

	unsigned int* d_out;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * numElems));

	unsigned int* d_prevSums;
	checkCudaErrors(cudaMalloc(&d_prevSums, sizeof(unsigned int) * gridSize.x));
	checkCudaErrors(cudaMemset(d_prevSums, 0, sizeof(int) * gridSize.x));

	int sharedSize = blockSize.x * sizeof(unsigned int) + 2 * sizeof (unsigned int) + sizeof(int);

	exclusive_scan<<<gridSize, blockSize, sharedSize>>>(d_in, d_out, d_prevSums, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_prevSums));

	return d_out;
}

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

__global__ 
void scatter(unsigned int* const d_inputVals,
			unsigned int* const d_outputVals,
			const unsigned int* const d_histScan,
			const size_t numElems,
			const size_t numBins,
			const unsigned int mask,
			const unsigned int digitOrder)
{
	extern __shared__ unsigned int s_blkIn[]; // blockSize

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numElems)
		return;

	//load data
	s_blkIn[threadIdx.x] = d_inputVals[idx];
	__syncthreads();

	//get number of elements ( < this element and = this element but locate on prev block 
	unsigned int bin = (s_blkIn[threadIdx.x] & mask) >> digitOrder;
	unsigned int numPrevElems = d_histScan[bin * gridDim.x + blockIdx.x];

	//calc number of element = this element on this block
	unsigned int count = 0;
	for (int i = 0; i < threadIdx.x; ++i)
	{
		unsigned int prevBin = (s_blkIn[i] & mask) >> digitOrder;
		if (bin == prevBin)
			++count;
	}
	
	// scatter to result array
	unsigned int rank = numPrevElems + count;
	d_outputVals[rank] = s_blkIn[threadIdx.x];
	// printf("block - idx - rank - val: %d - %d - %d - %d\n", blockIdx.x, idx, rank, s_blkIn[threadIdx.x]);

}

void host_scatter(unsigned int* const d_inputVals,
				unsigned int* const d_outputVals,
				const size_t numElems,
				const size_t numBins,
				const unsigned int* const d_histScan,
				const unsigned int mask,
				const unsigned int digitOrder,
				const dim3 blockSize)
{
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);

	unsigned int sharedSize = blockSize.x * sizeof(unsigned int);
	scatter<<<gridSize, blockSize, sharedSize>>> (d_inputVals, d_outputVals, d_histScan, numElems,numBins, mask, digitOrder);
}

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

int main()
{
	// srand(time(NULL));
	const dim3 blockSize(64);
	const size_t numElems = 10000000;
	const unsigned int numBits = 2;

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

	radix_sort(h_inputVals, d_inputVals, d_outputVals, numElems, numBits, blockSize);

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
