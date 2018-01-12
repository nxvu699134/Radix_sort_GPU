#include <algorithm>
#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
#include "timer.h"

#define MAX_INT_BITS	32

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
	for (unsigned int i = 1; i < gridSize.x; ++i)
	{
		if (h_out[i] > uMax)
			uMax = h_out[i];
	}

	checkCudaErrors(cudaFree(d_out));
	free(h_out);
	return uMax;
}

unsigned int getMaxNumOfBits_parallel(unsigned int* d_inputVals, const size_t numElems, const dim3 blockSize)
{
	unsigned int maxElem = host_max(d_inputVals, numElems, blockSize);
	unsigned int mask = (unsigned int)0x00000001 << 31;
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
	unsigned int mask = (unsigned int)0x00000001 << 31;
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
// __device__ unsigned int d_curBlock = 0;
__device__ volatile unsigned int d_curBlock = 0;

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
	
	// get block idx
	if (0 == threadIdx.x)
		s_bid = atomicAdd(&d_count, 1);
	__syncthreads();
	
	int i = s_bid * blockDim.x + threadIdx.x;
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
		// while(atomicAdd(&d_curBlock, 0) < s_bid);
		while(d_curBlock < s_bid);
		s_prevSum = d_prevSums[s_bid];
		d_prevSums[s_bid + 1] = s_prevSum + s_blkIn[blockDim.x - 1];
		__threadfence();
		// atomicAdd(&d_curBlock, 1);
		++d_curBlock;
	}
	__syncthreads();

	 // update output
	d_out[i] = s_blkIn[threadIdx.x] + s_prevSum;
}

unsigned int* host_exclusive_scan(const unsigned int* const d_in, const size_t numElemsScan, const dim3 blockSize)
{
	const dim3 gridSize((numElemsScan - 1) / blockSize.x + 1);
	set_init_scan<<<1, 1>>>();

	unsigned int* d_out;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * numElemsScan));

	unsigned int* d_prevSums;
	checkCudaErrors(cudaMalloc(&d_prevSums, sizeof(unsigned int) * gridSize.x));
	checkCudaErrors(cudaMemset(d_prevSums, 0, sizeof(int) * gridSize.x));

	int sharedSize = blockSize.x * sizeof(unsigned int);

	exclusive_scan<<<gridSize, blockSize, sharedSize>>>(d_in, d_out, d_prevSums, numElemsScan);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaFree(d_prevSums));

	return d_out;
}

 __global__
void histogram(const unsigned int* const d_inputVals,
				const size_t numElems,
				const size_t numBins,
				const size_t numElemsHist,
				unsigned int* const d_hist,
				const unsigned int mask,
				const unsigned int digitOrder)
{
	extern __shared__ unsigned int s_blkData[]; // 2 * numBins
	unsigned int* s_hist0 = (unsigned int*)s_blkData;
	unsigned int* s_hist1 = (unsigned int*)&s_hist0[numBins];

	if (threadIdx.x < numBins)
	{
		s_hist0[threadIdx.x] = 0;
		s_hist1[threadIdx.x] = 0;
	}
	__syncthreads();

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numElems)
	{
		unsigned int bin = (d_inputVals[idx] & mask) >> digitOrder;
		if (threadIdx.x < blockDim.x / 2)
			atomicAdd(&s_hist0[bin], 1);
		else
			atomicAdd(&s_hist1[bin], 1);
	}
	__syncthreads();
	
	//update d_hist by s_hist
	if (threadIdx.x < numBins)
	{
		int idxHist0 = threadIdx.x * (numElemsHist / numBins) + 2 * blockIdx.x;
		d_hist[idxHist0] = s_hist0[threadIdx.x];
		if (2 * blockIdx.x + 1 < numElemsHist / numBins)
			d_hist[idxHist0 + 1] = s_hist1[threadIdx.x];
		// printf("bin - indexHist - val : %d - %d - %d - %d - %d\n", blockIdx.x, bin, idxHist, d_hist[idxHist]);
	}
}

unsigned int* host_histogram(const unsigned int* const d_inputVals, 
							const size_t numElems,
							const size_t numBins,
							const size_t numElemsHist,
							const unsigned int mask,
							const unsigned int digitOrder,
							const dim3 blockSize)
{	
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);

	unsigned int *d_hist;
	checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * numElemsHist));
	checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * numElemsHist));

	int sharedSize = 2 * numBins * sizeof(unsigned int);

	histogram<<<gridSize, blockSize, sharedSize>>>(d_inputVals, 
													numElems, 
													numBins,
													numElemsHist,
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


void radix_sort(unsigned int* d_inputVals,
				unsigned int* d_inputPos, 
				unsigned int* d_outputVals,
				unsigned int* d_outputPos,
				const size_t numElems,
				const size_t numBits,
				const dim3 blockSizeMax,
				const dim3 blockSizeHist,
				const dim3 blockSizeScan,
				const dim3 blockSizeScatter)
{
	size_t numBins = 1 << numBits;

	const dim3 gridSizeScatter((numElems - 1) / blockSizeScatter.x + 1);
	size_t numElemsHist = numBins * gridSizeScatter.x;
	const dim3 gridSizeHist((numElemsHist - 1) / blockSizeHist.x + 1);
	const dim3 gridSizeScan((numElemsHist - 1) / blockSizeScan.x + 1);
	const dim3 gridSizeMax((numElems - 1) / blockSizeMax.x + 1);
	
	unsigned int* pInVals = d_inputVals;
	unsigned int* pOutVals = d_outputVals;
	unsigned int* pInPos = d_inputPos;
	unsigned int* pOutPos = d_outputPos;

	float histKernelTime = 0;
	float scanKernelTime = 0;
	float scatterKernelTime = 0;

	timer.Start();
	unsigned int maxBits = getMaxNumOfBits_parallel(d_inputVals, numElems, blockSizeMax);
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
		unsigned int* d_hist = host_histogram(pInVals, numElems, numBins, numElemsHist, mask, i, blockSizeHist);
		timer.Stop();
		histKernelTime += timer.Elapsed();


		// exclusive scan hist
		timer.Start();
		unsigned int* d_histScan = host_exclusive_scan(d_hist, numElemsHist, blockSizeScan);
		timer.Stop();
		scanKernelTime += timer.Elapsed();

		//scatter
		timer.Start();
		host_scatter(pInVals, pInPos, pOutVals, pOutPos, numElems, numBins, d_histScan, mask, i, blockSizeScatter);
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

	printf("%15s%12d%12d%12zu%16.3f\n", "Max", blockSizeMax.x, gridSizeMax.x, numBits, getMaxTime);
	printf("%15s%12d%12d%12zu%16.3f\n", "Histogram", blockSizeHist.x, gridSizeHist.x, numBits, histKernelTime);
	printf("%15s%12d%12d%12zu%16.3f\n", "Scan", blockSizeScan.x, gridSizeScan.x, numBits, scanKernelTime);
	printf("%15s%12d%12d%12zu%16.3f\n", "Scatter", blockSizeScatter.x, gridSizeScatter.x, numBits, scatterKernelTime);
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
	const dim3 blockSizeMax(256);
	const dim3 blockSizeHist(256); // larger to better, blockSizeHist must equal 2 * blockSizeScatter
	const dim3 blockSizeScan(256); //larger to better
	const dim3 blockSizeScatter(128); //smaller to better
	unsigned int numBits = 5;
	
	printf("\nSize: %zu\n", numElems);

	// unsigned int* h_inputVals = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	// checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	// unsigned int* h_inputPos = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	// checkCudaErrors(cudaMemcpy(h_inputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	printf("\n%15s%12s%12s%12s%16s\n", 
	"Function", "BlockSize", "GridSize", "NumBits", "TotalTime(ms)");

	// unsigned int* h_outputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	// unsigned int* h_outputPos = (unsigned int*) malloc(sizeof(unsigned int) * numElems);

	radix_sort(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems, numBits, blockSizeMax, blockSizeHist, blockSizeScan, blockSizeScatter);
	
	// checkCudaErrors(cudaMemcpy(d_outputVals, h_outputVals, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
	// checkCudaErrors(cudaMemcpy(d_outputPos, h_outputPos, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));

	// free(h_inputVals);
	// free(h_inputPos);
	// free(h_outputVals);
	// free(h_outputPos);
}
