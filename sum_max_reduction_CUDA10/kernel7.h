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

template <unsigned int blockSize>
__global__ void reduce7(int *g_idata, int *g_odata, unsigned int n) {

	extern __shared__ int sdata[];
	
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockSize*2) + threadIdx.x;
	unsigned int gridSize = (blockSize * 2 * gridDim.x);

	sdata[tid] = 0;

	while(i < n) {
		sdata[tid] += g_idata[i] + g_idata[i+blockSize];
		i += gridSize;
	}
	__syncthreads();

	// do reduction in shared memory
	if (blockSize >= 512) { if (tid < 256){ sdata[tid] += sdata[tid + 256];} __syncthreads();}
	if (blockSize >= 256) { if (tid < 128){ sdata[tid] += sdata[tid + 128];} __syncthreads();}
	if (blockSize >= 128) {	if (tid <  64){ sdata[tid] += sdata[tid +  64];} __syncthreads();}

	if (tid < 32) {
		volatile int* smem = sdata;
		if (blockSize >= 64) smem[tid] += smem[tid + 32];
		if (blockSize >= 32) smem[tid] += smem[tid + 16];
		if (blockSize >= 16) smem[tid] += smem[tid + 8];
		if (blockSize >=  8) smem[tid] += smem[tid + 4];
		if (blockSize >=  4) smem[tid] += smem[tid + 2];
		if (blockSize >=  2) smem[tid] += smem[tid + 1];
	} 

	// write result for this block back to global memory
	if (tid == 0) { g_odata[blockIdx.x] = sdata[0]; }
}
