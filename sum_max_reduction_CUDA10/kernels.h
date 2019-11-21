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

#ifndef __KERNELS__H
#define __KERNELS__H

#include "kernel_max.h"

#define NUM_KERNELS 10

constexpr int maxThreads = 512; // threads per block
constexpr int maxBlocks = 64;
//------------------------------------------------------------
//---- Kernels for reduction of SUM - step-by-step attempts ---
//-----------------------------------------------------------

// Attempt #1: interleaved addressing + divergent branch
__global__ void reduce1(int *g_idata, int *g_odata);

// Attempt #2: interleaved addressing + bank conflicts
__global__ void reduce2(int *g_idata, int *g_odata); 

// Attempt #3: sequential addressing
__global__ void reduce3(int *g_idata, int *g_odata);

// Attempt #4: first add during global load
__global__ void reduce4(int *g_idata, int *g_odata);

// Attempt #5: unroll the last warp
__global__ void reduce5(int *g_idata, int *g_odata);

// Attempt #6: completely unrolled
template <unsigned int blockSize>
__global__ void reduce6(int *g_idata, int *g_odata);
#include "kernel6.h"

// Attempt #7: multiple adds per thread
template <unsigned int blockSize>
__global__ void reduce7(int *g_idata, int *g_odata, unsigned int n);
#include "kernel7.h"

// Attempt #8: shfl instructions
__global__ void reduce8(int *g_idata, int *g_odata, unsigned int n);
// Attempt #9: shfl instructions + warp atomic
__global__ void reduce9(int *g_idata, int *g_odata, unsigned int n);
// Attempt #10: shfl instructions + block atomic
__global__ void reduce10(int *g_idata, int *g_odata, unsigned int n);
#endif
