/* ------------
 * This code is provided solely for the personal and private use of 
 * students taking the CSC367H5 course at the University of Toronto.
 * Copying for purposes other than this use is expressly prohibited. 
 * All forms of distribution of this code, whether as given or with 
 * any changes, are expressly prohibited. 
 * 
 * Authors: Bogdan Simion, Felipe de Azevedo Piovezan
 * 
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2019 Bogdan Simion
 * -------------
*/

#include "kernels.h"

__global__ void reduce4(int *g_idata, int *g_odata) {

	extern __shared__ int sdata[];
	
	unsigned int tid = threadIdx.x;
	
	// Global thread id
	unsigned int i = blockIdx.x * (blockDim.x*2) + threadIdx.x;

	sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
	__syncthreads();

    // do reduction in shared memory
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) { 
        if (tid < s) {  
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
	// write result for this block back to global memory
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}
