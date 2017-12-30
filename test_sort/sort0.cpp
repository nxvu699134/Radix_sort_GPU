#include <stdio.h>
#include <algorithm>
#include <cstring> //memset
#include <ctime>

#define MAX_INT_BITS	32

void printArray(unsigned int* const h_input, const size_t numElems)
 {
	for (unsigned int i = 0; i < numElems; ++i)
		printf("%u ", h_input[i]);
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
		for (unsigned int j = 0; j < numElems; ++j)
		{
			unsigned int bin = (pInVals[j] & mask) >> i;
			++h_hist[bin];
		}

		printArray(h_hist, numBins);
		// exclusive scan hist
		for (unsigned int j = 1; j < numBins; ++j)
		{
			h_histScan[j] += h_histScan[j - 1] + h_hist[j - 1];
		}
		printArray(h_histScan, numBins);

		for (int j = 0; j < numElems; ++j)
		{
			unsigned int bin = (pInVals[j] & mask) >> i;
			pOutVals[h_histScan[bin]] = pInVals[j];
			pOutPos[h_histScan[bin]] = pInPos[j];
			++h_histScan[bin];
		}
		
		std::swap(pInVals, pOutVals);
		std::swap(pInPos, pOutPos);
	}

	if (pInVals == h_outputVals)
	{
		std::copy(h_inputVals, h_inputVals + numElems, h_outputVals);
		std::copy(h_inputPos, h_inputPos + numElems, h_outputPos);
	}
}

int main()
{
	srand(time(NULL));
	const size_t numElems = 97;
	const unsigned int numBits = 2;

	unsigned int* h_inputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	unsigned int* h_inputPos = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	for (int i = 0; i < numElems; ++i)
	{
		h_inputVals[i] = rand() % 1000000000 + 1;
		h_inputPos[i] = i;
	}

	unsigned int* h_outputVals = (unsigned int*) malloc(sizeof(unsigned int) * numElems);
	unsigned int* h_outputPos = (unsigned int*) malloc(sizeof(unsigned int) * numElems);

	// printArray(h_inputVals, numElems);
	radix_sort(h_inputVals, h_inputPos, h_outputVals, h_outputPos, numElems, numBits);
	//printArray(h_outputVals, numElems);
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
	

	
	// free(h_inputVals);
	free(h_inputPos);
	free(h_outputVals);
	free(h_outputPos);
	return 0;
}
