#include "scatter.cuh"

__global__
void scatter1(unsigned int* const d_inputVals,
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

__global__
void scatter(const unsigned int* const d_inputVals,
			unsigned int* const d_outputVals,
			const unsigned int* const d_index,
			const unsigned int* const d_histScan,
			const size_t numElems,
			const size_t numBins,
			const unsigned int mask,
			const unsigned int digitOrder)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numElems) return;

	unsigned int value = d_inputVals[idx];
	unsigned int bin = (value & mask) >> digitOrder;
	unsigned int rank = d_histScan[bin * gridDim.x + blockIdx.x] + d_index[idx];
	d_outputVals[rank] = value;

}

void host_scatter(const unsigned int* const d_inputVals,
				unsigned int* const d_outputVals,
				const unsigned int* const d_index,
				const size_t numElems,
				const size_t numBins,
				const unsigned int* const d_histScan,
				const unsigned int mask,
				const unsigned int digitOrder,
				const dim3 blockSize)
{
	const dim3 gridSize((numElems - 1) / blockSize.x + 1);
	scatter<<<gridSize, blockSize>>> (d_inputVals, d_outputVals, d_index, d_histScan, numElems,numBins, mask, digitOrder);
}