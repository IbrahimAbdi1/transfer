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

__global__ void reduce2(int *g_idata, int *g_odata) {

	extern __shared__ int sdata[];
	
	unsigned int tid = threadIdx.x;
	
	// Global thread id
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	sdata[tid] = g_idata[i];
	__syncthreads();

	// do reduction in shared memory
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int idx = 2 * s * tid; // change this to reduce divergence
		if (idx < blockDim.x) { // In a warp, all threads participate (or don't)
			sdata[idx] += sdata[idx + s];
		} 
		__syncthreads();
	}

	// write result for this block back to global memory
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}


