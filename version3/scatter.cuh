#ifndef SCATTER_H
#define SCATTER_H

#include "CUDASupport.h"

void host_scatter(unsigned int* const d_inputVals,
				unsigned int* const d_outputVals,
				const size_t numElems,
				const size_t numBins,
				const unsigned int* const d_histScan,
				const unsigned int mask,
				const unsigned int digitOrder,
				const dim3 blockSize);

#endif
