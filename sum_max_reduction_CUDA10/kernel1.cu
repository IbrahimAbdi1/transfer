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

// Attempt #1: interleaved addressing + divergent branch
__global__ void reduce1(int *g_idata, int *g_odata) {

	extern __shared__ int sdata[];
	
	/* This is not a global tid, it's the tid within the block.
	Since the compiler creates a copy of sdata per block, then 
	to index into the sdata array we only need a thread's id within 
	its block. */
	unsigned int tid = threadIdx.x;
	
	// Global thread id
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2*s) == 0)  { // In a warp, only thread ids divisible by the step participate
			sdata[tid] += sdata[tid + s];
		} 
		__syncthreads();
	}

	// write result for this block back to global memory
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}
