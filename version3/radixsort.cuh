#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include "CUDASupport.h"

void radix_sort(unsigned int* h_inputVals,
				unsigned int* d_inputVals, 
				unsigned int* d_outputVals,
				const size_t numElems,
				const size_t numBits,
				const dim3 blockSize);
#endif
