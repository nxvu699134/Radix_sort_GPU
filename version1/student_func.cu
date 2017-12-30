//Udacity HW 4
//Radix Sorting
#include <algorithm>
#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include "timer.h"

#define MAX_INT_BITS	32

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

 
GpuTimer timer;

__global__ 
void max(const unsigned int* const d_inputVals, unsigned int* d_out, const size_t numElems)
{
	int startIdx = (blockIdx.x * blockDim.x) << 1;
	extern __shared__ unsigned int s_blkData[];
	if (startIdx + threadIdx.x < numElems)
		s_blkData[threadIdx.x] = d_inputVals[startIdx + threadIdx.x];
	if (startIdx + threadIdx.x + blockDim.x < numElems)
		s_blkData[threadIdx.x + blockDim.x] = d_inputVals[startIdx + threadIdx.x + blockDim.x];
	__syncthreads();
	
	for (int stride = blockDim.x; stride > 0; stride >>= 1)
	{
		if (threadIdx.x < stride)
		{
			int i = startIdx + threadIdx.x;
			if (i + stride < numElems)
				s_blkData[threadIdx.x] = umax(s_blkData[threadIdx.x], s_blkData[threadIdx.x + stride]);
		}
		__syncthreads();
	}

	if (0 == threadIdx.x)
	{
		d_out[blockIdx.x] = s_blkData[0];
	}
}	

unsigned int host_max(const unsigned int * const d_inputVals, 
						const size_t numElems, 
						const dim3 blockSize)
{
	const dim3 gridSize((numElems - 1) / (blockSize.x * 2) + 1);
	
	unsigned int* d_out;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * gridSize.x));
	unsigned int* h_out = (unsigned int*) malloc(gridSize.x * sizeof(unsigned int));

	
	max<<<gridSize, blockSize, (blockSize.x * 2 * sizeof(unsigned int))>>>(d_inputVals, d_out, numElems);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpy(h_out, d_out, gridSize.x * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	unsigned int uMax = h_out[0];
	for (int i = 1; i < gridSize.x; ++i)
	{
		if (h_out[i] > uMax)
			uMax = h_out[i];
	}

	checkCudaErrors(cudaFree(d_out));
	free(h_out);
	return uMax;
}

unsigned int getMaxNumOfBits_parallel(unsigned int* d_inputVals, const size_t numElems)
{
	const dim3 blockSize(128);
	unsigned int maxElem = host_max(d_inputVals, numElems, blockSize);
	unsigned int mask = 1 << 31;
	unsigned int count = 0;
	while (! (mask & maxElem)) 
	{
		++count;
		mask >>= 1;
	}
	return MAX_INT_BITS - count;
}

unsigned int getMaxNumOfBits(unsigned int* d_inputVals, const size_t numElems)
{
	unsigned int* h_inputVals = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	
	unsigned int maxElem = *std::max_element(h_inputVals, h_inputVals + numElems);
	unsigned int mask = 1 << 31;
	unsigned int count = 0;
	while (! (mask & maxElem)) 
	{
		++count;
		mask >>= 1;
	}
	free(h_inputVals);
	return MAX_INT_BITS - count;
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
	checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * numElems));

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
			unsigned int* const d_inputPos,
			unsigned int* const d_outputVals,
			unsigned int* const d_outputPos,
			const unsigned int* const d_histScan,
			const size_t numElems,
			const size_t numBins,
			const unsigned int mask,
			const unsigned int digitOrder)
{
	extern __shared__ unsigned int s_blkInVals[]; // blockSize
	

	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numElems)
		return;

	//load data
	s_blkInVals[threadIdx.x] = d_inputVals[idx];
	__syncthreads();

	//get number of elements ( < this element and = this element but locate on prev block 
	unsigned int bin = (s_blkInVals[threadIdx.x] & mask) >> digitOrder;
	unsigned int numPrevElems = d_histScan[bin * gridDim.x + blockIdx.x];

	//calc number of element = this element on this block
	unsigned int count = 0;
	for (int i = 0; i < threadIdx.x; ++i)
	{
		unsigned int prevBin = (s_blkInVals[i] & mask) >> digitOrder;
		if (bin == prevBin)
			++count;
	}
	
	// scatter to result array
	unsigned int rank = numPrevElems + count;
	d_outputVals[rank] = s_blkInVals[threadIdx.x];
	d_outputPos[rank] = d_inputPos[idx];
	// printf("block - idx - rank - val: %d - %d - %d - %d\n", blockIdx.x, idx, rank, s_blkIn[threadIdx.x]);

}

void host_scatter(unsigned int* const d_inputVals,
				unsigned int* const d_inputPos,
				unsigned int* const d_outputVals,
				unsigned int* const d_outputPos,
				const size_t numElems,
				const size_t numBins,
				const unsigned int* const d_histScan,
				const unsigned int mask,
				const unsigned int digitOrder,
				const dim3 blockSize)
{
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);

	unsigned int sharedSize = blockSize.x * sizeof(unsigned int);
	scatter<<<gridSize, blockSize, sharedSize>>> (d_inputVals, 
													d_inputPos, 
													d_outputVals, 
													d_outputPos, 
													d_histScan, 
													numElems,
													numBins, 
													mask, 
													digitOrder);
}

unsigned int* scan_serial(unsigned int* const d_hist, const size_t numElems)
{
	unsigned int* h_hist = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	checkCudaErrors(cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToHost));
	unsigned int* h_histScan = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	h_histScan[0] = 0;
	for (unsigned int j = 1; j < numElems; ++j)
	{
		h_histScan[j] += h_histScan[j - 1] + h_hist[j - 1];
	}
	unsigned int* d_histScan;
	checkCudaErrors(cudaMalloc(&d_histScan, sizeof(unsigned int) * numElems));
	checkCudaErrors(cudaMemcpy(d_histScan, h_histScan, sizeof(unsigned int) * numElems, cudaMemcpyHostToDevice));
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	free(h_hist);
	free(h_histScan);
	return d_histScan;
}
void radix_sort(unsigned int* d_inputVals,
				unsigned int* d_inputPos, 
				unsigned int* d_outputVals,
				unsigned int* d_outputPos,
				const size_t numElems,
				const size_t numBits,
				const dim3 blockSize)
{
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);
	size_t numBins = 1 << numBits;
	
	unsigned int* pInVals = d_inputVals;
	unsigned int* pOutVals = d_outputVals;
	unsigned int* pInPos = d_inputPos;
	unsigned int* pOutPos = d_outputPos;

	float histKernelTime = 0;
	float scanKernelTime = 0;
	float scatterKernelTime = 0;

	timer.Start();
	unsigned int maxBits = getMaxNumOfBits_parallel(d_inputVals, numElems);
	timer.Stop();
	float getMaxTime = timer.Elapsed();

	if (maxBits % numBits)
		maxBits += numBits;
	
	// loop through digits
	for (unsigned int i = 0; i <= maxBits; i += numBits)
	{
		unsigned int mask = (numBins - 1) << i;

		//histogram 
		timer.Start();
		unsigned int* d_hist = host_histogram(pInVals, numElems, numBins, mask, i, blockSize);
		timer.Stop();
		histKernelTime += timer.Elapsed();

		// exclusive scan hist
		timer.Start();
		unsigned int* d_histScan = host_exclusive_scan(d_hist, numBins * gridSize.x, blockSize);
		// unsigned int* d_histScan = scan_serial(d_hist, numBins * gridSize.x);
		timer.Stop();
		scanKernelTime += timer.Elapsed();

		//scatter
		timer.Start();
		host_scatter(pInVals, pInPos, pOutVals, pOutPos, numElems, numBins, d_histScan, mask, i, blockSize);
		timer.Stop();
		scatterKernelTime += timer.Elapsed();

		std::swap(pInVals, pOutVals);
		std::swap(pInPos, pOutPos);

		checkCudaErrors(cudaFree(d_hist));
		checkCudaErrors(cudaFree(d_histScan));
		d_hist = NULL;
		d_histScan = NULL;
	}
	if (pInVals == d_outputVals)
	{
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
		checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals, sizeof(unsigned int) * numElems, cudaMemcpyDeviceToDevice));
	}

	printf("%15s%12d%12d%16.3f\n", "Max", 0, 0, getMaxTime);
	printf("%15s%12d%12d%16.3f\n", "Histogram", 0, 0, histKernelTime);
	printf("%15s%12d%12d%16.3f\n", "Scan", 0, 0, scanKernelTime);
	printf("%15s%12d%12d%16.3f\n", "Scatter", 0, 0, scatterKernelTime);
 }

void get_devices_info()
{
	int numdevs;
    cudaGetDeviceCount(&numdevs);
    printf("\n\n\nNum devices = %d\n", numdevs);
    
    for (int i = 0; i < numdevs; ++i)
    {
        printf("Device %d\n", i);
        cudaDeviceProp devprop;
        cudaGetDeviceProperties(&devprop, i);
        printf("CC %d.%d\n", devprop.major, devprop.minor);
        printf("Max num threads/block: %d\n", devprop.maxThreadsPerBlock);
        printf("Maximum block dimensions: %d x %d x %d\n", devprop.maxThreadsDim[0], devprop.maxThreadsDim[1], devprop.maxThreadsDim[2]);
        printf("Maximum grid dimensions: %d x %d x %d\n\n", devprop.maxGridSize[0], devprop.maxGridSize[1], devprop.maxGridSize[2]);
        printf("-----------------------------------------------------------------------------\n\n");
    }
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{ 
	const dim3 blockSize(128);
	unsigned int numBits = 4;
	
	printf("\nSize: %d\n", numElems);

	// unsigned int* h_inputVals = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	// checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// unsigned int* h_inputPos = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	// checkCudaErrors(cudaMemcpy(h_inputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	printf("\n%15s%12s%12s%16s\n", 
	"Function", "BlockSize", "GridSize", "TotalTime(ms)");

	// unsigned int* h_outputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	// unsigned int* h_outputPos = (unsigned int*) malloc(sizeof(unsigned int) * numElems);

	radix_sort(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems, numBits, blockSize);
	
	// checkCudaErrors(cudaMemcpy(d_outputVals, h_outputVals, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(d_outputPos, h_outputPos, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));

	// free(h_inputVals);
	// free(h_inputPos);
	// free(h_outputVals);
	// free(h_outputPos);
}
