#include "exscan.cuh"

__device__ int d_cnt = 0;
__global__
void reset_scan()
{
	d_cnt = 0;
}
__global__ void exclusive_scan(const unsigned int* const idata, unsigned int* const odata, int N,
					 volatile unsigned int *prevSums, volatile bool *flags)
{
	__shared__ int bid;
	if (threadIdx.x == 0) {
		bid = atomicAdd(&d_cnt, 1);
		if (bid == 0) {
			flags[0] = true;
		}
	}
	__syncthreads();

	// 1. Each block loads data from GMEM to SMEM
	//    (each thread will load 2 elements)
	extern __shared__ unsigned int s_data[]; // It's size will be 2*blockDim.x (elements)
	int dataIdx1 = bid * (2 * blockDim.x) + threadIdx.x;
	int dataIdx2 = bid * (2 * blockDim.x) + threadIdx.x + blockDim.x;
	if (dataIdx1 < N)
		s_data[threadIdx.x + 1] = idata[dataIdx1];
	if (dataIdx2 < N)
		s_data[threadIdx.x + blockDim.x + 1] = idata[dataIdx2];
	if (threadIdx.x == 0)
		s_data[0] = 0;
	__syncthreads();
	
	// 2. Each block does scan with data on SMEM
	// 2.1. Reduction phase
	for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // So active threads will be consecutive
		if (s_dataIdx < 2 * blockDim.x)
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
		__syncthreads();
	}
	// 2.2. Post-reduction phase
	for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
	{
		int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride; // Wow
		if (s_dataIdx < 2 * blockDim.x)
			s_data[s_dataIdx] += s_data[s_dataIdx - stride];
		__syncthreads();
	}

	__shared__ unsigned int prevSum;
	if (threadIdx.x == 0) {
		while (flags[bid] == false);
		prevSum = prevSums[bid];
		if ((bid + 1) < N) {
			unsigned int idx = (bid * (2 * blockDim.x)) + 2*blockDim.x - 1;
			if (idx > ((unsigned int)N - 1)) idx = ((unsigned int)N - 1);
			prevSums[bid + 1] = prevSum + s_data[2*blockDim.x - 1] + idata[idx];
			flags[bid + 1] = true;
		}
	}
	__syncthreads();
	
	// 3. Each block writes result from SMEM to GMEM
	//    (each thread will write 2 elements)
	if (dataIdx1 < N)
		odata[dataIdx1] = s_data[threadIdx.x] + prevSum;
	if (dataIdx2 < N)
		odata[dataIdx2] = s_data[threadIdx.x + blockDim.x] + prevSum;
}

unsigned int* host_exclusive_scan(const unsigned int* const d_in, const size_t numElems, const dim3 blockSize)
{
	reset_scan<<<1, 1>>>();

	int sharedSize = 2 * blockSize.x * sizeof(unsigned int) + 1;
	dim3 gridSize((numElems - 1) / (2*blockSize.x) + 1);
	//cout << "Size scan: " << gridSize.x << endl;

	unsigned int* d_out;
	checkCudaErrors(cudaMalloc(&d_out, sizeof(unsigned int) * numElems));

	bool *d_flags;
	unsigned int *d_prevSums;
	checkCudaErrors(cudaMalloc(&d_prevSums, gridSize.x * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_prevSums, 0, gridSize.x * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_flags, gridSize.x * sizeof(unsigned int)));
	checkCudaErrors(cudaMemset(d_flags, false, gridSize.x * sizeof(unsigned int)));
	exclusive_scan<<<gridSize, blockSize, sharedSize>>>(d_in, d_out, numElems, d_prevSums, d_flags);
	cudaDeviceSynchronize();

	checkCudaErrors(cudaFree(d_flags));
	checkCudaErrors(cudaFree(d_prevSums));

	return d_out;
}
