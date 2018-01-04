#ifndef EXSCAN_H
#define EXSCAN_H

#include "CUDASupport.h"

unsigned int* host_exclusive_scan(const unsigned int* const d_in, const size_t numElems, const dim3 blockSize);

#endif
