#include "kernels.h"
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <math.h>


void gpu_min_max_switch_threads(int pixelCount, int numThreads, int numBlocks, int32_t *indata, int32_t *max, int32_t *min, int first)
{
  dim3 dimBlock(numThreads,1,1);
  dim3 dimGrid (numBlocks, 1,1);
  //int shMemSize = 2 * ((numThreads <= 32) ? 2 * numThreads * sizeof(int32_t) : numThreads * sizeof(int32_t));
  int shMemSize = 2 * 1024 * sizeof(int32_t);

  switch (numThreads)
  {
    case 1024:
      if (first == 1) {find_min_max_f<1024><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<1024><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case 512:
      if (first) {find_min_max_f<512><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<512><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case 256:
      if (first) {find_min_max_f<256><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<256><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case 128:
      if (first) {find_min_max_f<128><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<128><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case 64:
      if (first) {find_min_max_f<64><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<64><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case 32:
      if (first) {find_min_max_f<32><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<32><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case 16:
      if (first) {find_min_max_f<16><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<16><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case  8:
      if (first) {find_min_max_f<8><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<8><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case  4:
      if (first) {find_min_max_f<4><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<4><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case  2:
      if (first) {find_min_max_f<2><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<2><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    case  1:
      if (first) {find_min_max_f<1><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
      else {find_min_max<1><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
      break;
    default:
      printf("invalid number of threads, exiting...\n");
      exit(1);
  }
}



bool calculate_blocks_and_threads(int n, int &blocks, int &threads)
{
  threads = 1024;
  blocks = n/1024 +1;
  if (n <= 512) threads = 512;
  if (n <= 256) threads = 256;
  if (n <= 128) threads = 128;
  if (n <= 64) threads = 64;
  if (n <= 32) threads = 32;
  if (n <= 16) threads = 16;
  if (n <= 8) threads = 8;
  if (n <= 4) threads = 4;
  if (n <= 2) threads = 2;
  
  return blocks != 1;
}

template <unsigned int blockSize>
__global__ void find_min_max_f(int32_t *indata,int32_t *max, int32_t *min,int pixelCount)
{

extern __shared__ double max_min_data[2][1024];

unsigned int threadID = threadIdx.x;
unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

if(tid < pixelCount){
  int32_t g_pixel = indata[tid];
  max_min_data[0][threadID] = (double)g_pixel;
  max_min_data[1][threadID] = (double)g_pixel;
}

else{
  max_min_data[0][threadID] = -INFINITY;
  max_min_data[1][threadID] = INFINITY;
}
__syncthreads();

// do reduction in shared memory
if (blockSize >= 1024) { if(threadID < 512){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+512]){
      max_min_data[0][threadID] = max_min_data[0][threadID+512];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+512]){
      max_min_data[1][threadID] = max_min_data[1][threadID+512];
  }

}
__syncthreads();}
if (blockSize >= 512) { if(threadID < 256){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+256]){
      max_min_data[0][threadID] = max_min_data[0][threadID+256];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+256]){
      max_min_data[1][threadID] = max_min_data[1][threadID+256];
  }

}
__syncthreads();}
if (blockSize >= 256) { if(threadID < 128){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+128]){
      max_min_data[0][threadID] = max_min_data[0][threadID+128];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+128]){
      max_min_data[1][threadID] = max_min_data[1][threadID+128];
  }

}
__syncthreads();}
if (blockSize >= 128) {	if(threadID < 64){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+64]){
      max_min_data[0][threadID] = max_min_data[0][threadID+64];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+64]){
      max_min_data[1][threadID] = max_min_data[1][threadID+64];
  }

}__syncthreads();}

if (blockSize >= 64) {	if(threadID < 32){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+32]){max_min_data[0][threadID] = max_min_data[0][threadID+32];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+32]){max_min_data[1][threadID] = max_min_data[1][threadID+32];}

}}

if (blockSize >= 32) {	if(threadID < 16){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+16]){max_min_data[0][threadID] = max_min_data[0][threadID+16];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+16]){max_min_data[1][threadID] = max_min_data[1][threadID+16];}

}}

if (blockSize >= 16) {	if(threadID < 8){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+8]){max_min_data[0][threadID] = max_min_data[0][threadID+8];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+8]){max_min_data[1][threadID] = max_min_data[1][threadID+8];}

}}

if (blockSize >= 8) {	if(threadID < 4){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+4]){max_min_data[0][threadID] = max_min_data[0][threadID+4];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+4]){max_min_data[1][threadID] = max_min_data[1][threadID+4];}

}}

if (blockSize >= 4) {	if(threadID < 2){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+2]){max_min_data[0][threadID] = max_min_data[0][threadID+2];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+2]){max_min_data[1][threadID] = max_min_data[1][threadID+2];}

}}

if (blockSize >= 2) {	if(threadID < 1){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+1]){max_min_data[0][threadID] = max_min_data[0][threadID+1];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+1]){max_min_data[1][threadID] = max_min_data[1][threadID+1];}

}}

// write result for this block back to global memory
  if (threadID == 0) {
    printf("vlock %d max %d min %d\n", blockIdx.x,(int)max_min_data[0][0],(int)max_min_data[1][0]);
    max[blockIdx.x] = (int32_t) max_min_data[0][0];
    min[blockIdx.x] = (int32_t) max_min_data[1][0];
  }
}

 // tid switch to threadId
 template <unsigned int blockSize>
 __global__ void find_min_max(int32_t *max, int32_t *min,int pixelCount)
{

extern __shared__ double max_min_data[2][1024];

unsigned int threadID = threadIdx.x;
unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

if(tid < pixelCount){
  max_min_data[0][threadID] = (double)max[tid];
  max_min_data[1][threadID] = (double)min[tid];
}
else{
  max_min_data[0][threadID] = -INFINITY;
  max_min_data[1][threadID] = INFINITY;
}
__syncthreads();

// do reduction in shared memory
if (blockSize >= 1024) { if(threadID < 512){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+512]){
      max_min_data[0][threadID] = max_min_data[0][threadID+512];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+512]){
      max_min_data[1][threadID] = max_min_data[1][threadID+512];
  }

}
__syncthreads();}
if (blockSize >= 512) { if(threadID < 256){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+256]){
      max_min_data[0][threadID] = max_min_data[0][threadID+256];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+256]){
      max_min_data[1][threadID] = max_min_data[1][threadID+256];
  }

}
__syncthreads();}
if (blockSize >= 256) { if(threadID < 128){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+128]){
      max_min_data[0][threadID] = max_min_data[0][threadID+128];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+128]){
      max_min_data[1][threadID] = max_min_data[1][threadID+128];
  }

}
__syncthreads();}
if (blockSize >= 128) {	if(threadID < 64){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+64]){
      max_min_data[0][threadID] = max_min_data[0][threadID+64];
  }
  if(max_min_data[1][threadID] > max_min_data[1][threadID+64]){
      max_min_data[1][threadID] = max_min_data[1][threadID+64];
  }

}__syncthreads();}

if (blockSize >= 64) {	if(threadID < 32){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+32]){max_min_data[0][threadID] = max_min_data[0][threadID+32];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+32]){max_min_data[1][threadID] = max_min_data[1][threadID+32];}

}}

if (blockSize >= 32) {	if(threadID < 16){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+16]){max_min_data[0][threadID] = max_min_data[0][threadID+16];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+16]){max_min_data[1][threadID] = max_min_data[1][threadID+16];}

}}

if (blockSize >= 16) {	if(threadID < 8){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+8]){max_min_data[0][threadID] = max_min_data[0][threadID+8];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+8]){max_min_data[1][threadID] = max_min_data[1][threadID+8];}

}}

if (blockSize >= 8) {	if(threadID < 4){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+4]){max_min_data[0][threadID] = max_min_data[0][threadID+4];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+4]){max_min_data[1][threadID] = max_min_data[1][threadID+4];}

}}

if (blockSize >= 4) {	if(threadID < 2){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+2]){max_min_data[0][threadID] = max_min_data[0][threadID+2];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+2]){max_min_data[1][threadID] = max_min_data[1][threadID+2];}

}}

if (blockSize >= 2) {	if(threadID < 1){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+1]){max_min_data[0][threadID] = max_min_data[0][threadID+1];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+1]){max_min_data[1][threadID] = max_min_data[1][threadID+1];}

}}

// write result for this block back to global memory
  if (threadID == 0) {
    printf("vlock %d max %d min %d\n", blockIdx.x,(int)max_min_data[0][0],(int)max_min_data[1][0]);
    max[blockIdx.x] = max_min_data[0][0];
    min[blockIdx.x] = max_min_data[1][0];
  }
}