#include <stdio.h>
#include <algorithm>
#include <cstring> //memset
#include <ctime>
#include <time.h>
#define MAX_INT_BITS	32

void printArray(unsigned int* const h_input, const size_t numElems)
 {
	for (unsigned int i = 0; i < numElems; ++i)
		printf("%u  ", h_input[i]);
	printf("\n");
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
				unsigned int* h_outputVals,
				const size_t numElems,
				const unsigned int numBits)
{
	unsigned int blockSize = 64;
	unsigned gridSize = (numElems - 1) / blockSize + 1;

	unsigned int numBins = 1 << numBits;
	unsigned int* h_hist = (unsigned int*) malloc (sizeof(unsigned int) * numBins * gridSize);
	unsigned int* h_histScan = (unsigned int*) malloc (sizeof(unsigned int) * numBins * gridSize);
	
	unsigned int* pInVals = h_inputVals;
	unsigned int* pOutVals = h_outputVals;


	unsigned int maxBits = getMaxNumOfBits(h_inputVals, numElems);

	if (maxBits % numBits)
		maxBits += numBits;

	double histTime = 0;
	// loop through digits
	for (unsigned int i = 0; i <= maxBits; i += numBits)
	{
		unsigned int mask = (numBins - 1) << i;

		//init bin histogram
		memset(h_hist, 0, sizeof(unsigned int) * numBins * gridSize);
    	memset(h_histScan, 0, sizeof(unsigned int) * numBins * gridSize);

		//histogram 
		clock_t start = clock();
		for (unsigned int j = 0; j < numElems; ++j)
		{
			unsigned int blockIdx = j / blockSize;
			unsigned int threadIdx = j % blockSize;
			unsigned int bin = (pInVals[j] & mask) >> i;
			int idxHist = bin * gridSize + blockIdx;
			h_hist[idxHist]++;
		}
		clock_t stop = clock();
		histTime += (stop - start);
		// printArray(h_hist, numBins * gridSize);
		// exclusive scan hist
		for (unsigned int j = 1; j < numBins * gridSize; ++j)
		{
			h_histScan[j] += h_histScan[j - 1] + h_hist[j - 1];
		}
		// printArray(h_histScan, numBins * gridSize);

		for (int j = 0; j < numElems; ++j)
		{
			unsigned int blockIdx = j / blockSize;
			unsigned int threadIdx = j % blockSize;

			unsigned int bin = (pInVals[j] & mask) >> i;
			unsigned int numPrevElems = h_histScan[bin * gridSize + blockIdx];
			unsigned int count = 0;
			for (int k = blockIdx * blockSize; k < blockIdx * blockSize + threadIdx; ++k)
			{
				unsigned int prevBin = (pInVals[k] & mask) >> i;
				if (bin == prevBin)
					++count;
			}
			unsigned int rank = numPrevElems + count;
			pOutVals[rank] = pInVals[blockIdx * blockSize + threadIdx];
		}
		// printArray(pOutVals, numElems);
		std::swap(pInVals, pOutVals);
		// printf("\n");
	}

	if (pInVals == h_outputVals)
	{
		std::copy(h_inputVals, h_inputVals + numElems, h_outputVals);
	}
	printf("hist time = %lf", histTime);
}

int main()
{
	srand(time(NULL));
	const size_t numElems = 250000;
	const unsigned int numBits = 2;

	unsigned int* h_inputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	for (int i = 0; i < numElems; ++i)
	{
		h_inputVals[i] = rand() % 2000000000 + 1;
	}
	// unsigned int h_inputVals[] = {84,87,78,16,94,36,87,93,50,22,63,28,91,60,64,27,41,27
	// 	,  73,37,12,69,68,30,83,31,63,24,68,36,30,3, 23,59,70
	// 	,  68,94,57,12,43,30,74,22,20,85,38,99,25,16,71,14,27
	// 	,  92,81,57,74,63,71,97,82,6, 26,85,28,37,6, 47,30,14
	// 	,  58,25,96,83,46,15,68,35,65,44,51,88,9, 77,79,89,85
	// 	,  4, 52,55,100 ,33,61,77,69,40,13,27};
	unsigned int* h_outputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);

	// printArray(h_inputVals, numElems);
	radix_sort(h_inputVals, h_outputVals, numElems, numBits);
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
	return 0;
}
