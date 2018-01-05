#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "CUDASupport.h"

unsigned int* calc_histogram(const unsigned int* const d_inputVals,
							unsigned int* const d_index,
							const size_t numElems,
							const size_t numBins,
							const unsigned int mask,
							const unsigned int digitOrder,
							const dim3 blockSize);

#endif
