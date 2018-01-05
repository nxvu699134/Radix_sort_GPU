#ifndef EXSCAN_H
#define EXSCAN_H

#include "CUDASupport.h"

unsigned int* calc_exclusive_scan(const unsigned int* const d_in, const size_t numElems, const dim3 blockSize);

#endif
