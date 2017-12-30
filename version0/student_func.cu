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
				unsigned int* h_inputPos,
				unsigned int* h_outputVals,
				unsigned int* h_outputPos,
				const size_t numElems,
				const unsigned int numBits)
 {
	unsigned int numBins = 1 << numBits;
   	unsigned int* h_hist = (unsigned int*) malloc (sizeof(unsigned int) * numBins);
	unsigned int* h_histScan = (unsigned int*) malloc (sizeof(unsigned int) * numBins);
	
	unsigned int* pInVals = h_inputVals;
	unsigned int* pInPos = h_inputPos;
	unsigned int* pOutVals = h_outputVals;
	unsigned int* pOutPos = h_outputPos;
	
	float histKernelTime = 0;
	float scanKernelTime = 0;
	float scatterKernelTime = 0;

	unsigned int maxBits = getMaxNumOfBits(h_inputVals, numElems);
	
	if (maxBits % numBits)
		maxBits += numBits;
 
   // loop through digits
   	for (unsigned int i = 0; i <= maxBits; i += numBits)
   	{
    	unsigned int mask = (numBins - 1) << i;
 
     	//init bin histogram
     	memset(h_hist, 0, sizeof(unsigned int) * numBins);
    	memset(h_histScan, 0, sizeof(unsigned int) * numBins);
		
		//histogram 
		timer.Start();
		for (unsigned int j = 0; j < numElems; ++j)
		{
			unsigned int bin = (pInVals[j] & mask) >> i;
			++h_hist[bin];
		}
		timer.Stop();
		histKernelTime += timer.Elapsed();

		// exclusive scan hist
		timer.Start();
		for (unsigned int j = 1; j < numBins; ++j)
		{
			h_histScan[j] += h_histScan[j - 1] + h_hist[j - 1];
		}
		timer.Stop();
		scanKernelTime += timer.Elapsed();

		timer.Start();
		for (int j = 0; j < numElems; ++j)
		{
			unsigned int bin = (pInVals[j] & mask) >> i;
			pOutVals[h_histScan[bin]] = pInVals[j];
			pOutPos[h_histScan[bin]] = pInPos[j];
			++h_histScan[bin];
		}
		timer.Stop();
		scatterKernelTime += timer.Elapsed();

		std::swap(pInVals, pOutVals);
		std::swap(pInPos, pOutPos);
	}
	
	if (pInVals == h_outputVals)
	{
		std::copy(h_inputVals, h_inputVals + numElems, h_outputVals);
		std::copy(h_inputPos, h_inputPos + numElems, h_outputPos);
	}

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
	unsigned int numBits = 2;
	
	printf("\nSize: %d\n", numElems);

	unsigned int* h_inputVals = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	unsigned int* h_inputPos = (unsigned int*) malloc (sizeof(unsigned int) * numElems);
	checkCudaErrors(cudaMemcpy(h_inputPos, d_inputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	printf("\n%15s%12s%12s%16s\n", 
	"Function", "BlockSize", "GridSize", "TotalTime(ms)");

	unsigned int* h_outputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	unsigned int* h_outputPos = (unsigned int*) malloc(sizeof(unsigned int) * numElems);

	radix_sort(h_inputVals, h_inputPos, h_outputVals, h_outputPos, numElems, numBits);
	
	checkCudaErrors(cudaMemcpy(d_outputVals, h_outputVals, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_outputPos, h_outputPos, numElems * sizeof(unsigned int), cudaMemcpyHostToDevice));

	free(h_inputVals);
	free(h_inputPos);
	free(h_outputVals);
	free(h_outputPos);
}
