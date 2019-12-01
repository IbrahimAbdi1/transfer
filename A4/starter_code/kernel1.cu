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
 #include <stdio.h>
 #include <string>
 #include <unistd.h>
 #include <math.h>


 #define MY_MIN(x,y) ((x < y) ? x : y)

// void gpu_min_max_switch_threads(int pixelCount, int numThreads, int numBlocks, int32_t *indata, int32_t *max, int32_t *min, int first)
// {
//   dim3 dimBlock(numThreads,1,1);
//   dim3 dimGrid (numBlocks, 1,1);
//   //int shMemSize = 2 * ((numThreads <= 32) ? 2 * numThreads * sizeof(int32_t) : numThreads * sizeof(int32_t));
//   int shMemSize = 2 * 1024 * sizeof(int32_t);

//   switch (numThreads)
//   {
//     case 1024:
//       if (first == 1) {
//           find_min_max_f<1024><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<1024><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case 512:
//       if (first) {find_min_max_f<512><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<512><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case 256:
//       if (first) {find_min_max_f<256><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<256><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case 128:
//       if (first) {find_min_max_f<128><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<128><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case 64:
//       if (first) {find_min_max_f<64><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<64><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case 32:
//       if (first) {find_min_max_f<32><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<32><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case 16:
//       if (first) {find_min_max_f<16><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<16><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case  8:
//       if (first) {find_min_max_f<8><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<8><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case  4:
//       if (first) {find_min_max_f<4><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<4><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case  2:
//       if (first) {find_min_max_f<2><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<2><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     case  1:
//       if (first) {find_min_max_f<1><<<numBlocks,numThreads,shMemSize>>>(indata, max, min,pixelCount);}
//       else {find_min_max<1><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);}
//       break;
//     default:
//       printf("invalid number of threads, exiting...\n");
//       exit(1);
//   }
// }

 bool calculate_blocks_and_threads(int n, int &blocks, int &threads)
{
  threads = 1024; // (n < 1024*2) ? (n/2) : 
  if (n < 2) threads = 1;
  if (n < 4) threads = 2;
  if (n < 8) threads = 4;
  if (n < 16) threads = 8;
  if (n < 32) threads = 16;
  if (n < 64) threads = 32;
  if (n < 128) threads = 64;
  if (n < 256) threads = 128;
  if (n < 512) threads = 256;
  if (n < 1024) threads = 512;
  blocks = (n + (threads * 2 - 1)) / (threads * 2);
  return blocks != 1;
}
 
 void run_kernel1(const int8_t *filter, int32_t dimension, const int32_t *input,
                  int32_t *output, int32_t width, int32_t height) {
   // Figure out how to split the work into threads and call the kernel below.
   int pixelCount = width*height;
   int32_t *g_min_max;
   int32_t *deviceMatrix_IN,*deviceMatrix_OUT;
   int8_t *deviceFilter;
   int size = height*width*sizeof(int32_t);
   int numBlocks = pixelCount / 1024;
   
   int numThreads, nblocks;
   int iteration_n = pixelCount;
   printf("pixelCount %d numBlocks %d\n",pixelCount,numBlocks);
 
   cudaMalloc((void**)&deviceMatrix_IN,size);
   cudaMalloc((void**)&deviceMatrix_OUT,size);
   cudaMalloc((void**)&deviceFilter,dimension*dimension*sizeof(int8_t));
   cudaMalloc((void**)&g_min_max,2*(numBlocks+1)*sizeof(int32_t));
   
   cudaMemcpy(deviceMatrix_IN,input,size, cudaMemcpyHostToDevice);
   cudaMemcpy(deviceMatrix_OUT,output,size, cudaMemcpyHostToDevice);
   cudaMemcpy(deviceFilter,filter,dimension*dimension*sizeof(int8_t),cudaMemcpyHostToDevice);
 
   kernel1<<<numBlocks+1,1024>>>(deviceFilter,dimension,deviceMatrix_IN,deviceMatrix_OUT,width,height); 

   int32_t *max = g_min_max;
   int32_t *min = g_min_max + (numBlocks +1);
    bool should_repeat = calculate_blocks_and_threads(iteration_n, nblocks, numThreads);
    find_min_max_f<numThreads><<<numBlocks,numThreads,shMemSize>>>(deviceMatrix_OUT, max, min,pixelCount);
    //gpu_min_max_switch_threads(iteration_n, numThreads, nblocks, deviceMatrix_OUT, max, min, first);

    
 
     while(should_repeat)
     {
       iteration_n = nblocks;
       printf("HERE: %d blocks \n", nblocks);
       should_repeat = calculate_blocks_and_threads(iteration_n, nblocks, numThreads);
       find_min_max<numThreads><<<numBlocks,numThreads,shMemSize>>>(max, min,pixelCount);
       //gpu_min_max_switch_threads(iteration_n, numThreads, nblocks, g_min_max, max, min, first);
     }
   
   normalize1<<<numBlocks + 1,1024>>>(deviceMatrix_OUT,width,height,g_min_max);
 
    cudaMemcpy(output,deviceMatrix_OUT,size, cudaMemcpyDeviceToHost);
    
    cudaFree(deviceMatrix_IN);
    cudaFree(deviceMatrix_OUT);
    cudaFree(deviceFilter);
    cudaFree(g_min_max);
 }
 
 
 
 __global__ void kernel1(const int8_t *filter, int32_t dimension, const int32_t *input, 
 int32_t *output, int32_t width,int32_t height) {
 
   // shared memory   
  
   // get index given tid
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   // call apply2d on input @ index and store it  on output @ index
   if(idx < height*width){
     int row = idx/width;
     int column = idx%width; // expensive 
    
     // apply2d function (really bad need fix)
     int32_t sum = 0;
     int filter_centre = dimension/2;
     
     int s_row = row - filter_centre;
     int s_column = column - filter_centre;
     for(int r = 0;r<dimension;r++){
         int n_row = s_row + r;
         for(int c = 0;c<dimension;c++){
             int n_column = s_column + c;
             if((n_row >= 0) && (n_column >= 0) && (n_column < width) && (n_row < height)){
                 sum += (filter[r*dimension + c]) * (input[n_row*width + n_column]);
                 
             }
         }
     }
     
     output[idx] = sum;
     
   }
 
   
                           
 }
 
 __global__ void normalize1(int32_t *image, int32_t width, int32_t height, int32_t *smallest_biggest) {
     
    
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if(smallest_biggest[0] != smallest_biggest[1] && idx < width * height){
     image[idx] = ((image[idx] - smallest_biggest[1]) * 255) / (smallest_biggest[0] - smallest_biggest[1]);
   }
 }
 
 
 // tid switch to threadId
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

}}

if(threadID < 32){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+32]){max_min_data[0][threadID] = max_min_data[0][threadID+32];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+32]){max_min_data[1][threadID] = max_min_data[1][threadID+32];}

}

if(threadID < 16){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+16]){max_min_data[0][threadID] = max_min_data[0][threadID+16];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+16]){max_min_data[1][threadID] = max_min_data[1][threadID+16];}

}

if(threadID < 8){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+8]){max_min_data[0][threadID] = max_min_data[0][threadID+8];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+8]){max_min_data[1][threadID] = max_min_data[1][threadID+8];}

}

if(threadID < 4){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+4]){max_min_data[0][threadID] = max_min_data[0][threadID+4];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+4]){max_min_data[1][threadID] = max_min_data[1][threadID+4];}

}

if(threadID < 2){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+2]){max_min_data[0][threadID] = max_min_data[0][threadID+2];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+2]){max_min_data[1][threadID] = max_min_data[1][threadID+2];}

}

if(threadID < 1){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+1]){max_min_data[0][threadID] = max_min_data[0][threadID+1];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+1]){max_min_data[1][threadID] = max_min_data[1][threadID+1];}

}

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

}}

if(threadID < 32){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+32]){max_min_data[0][threadID] = max_min_data[0][threadID+32];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+32]){max_min_data[1][threadID] = max_min_data[1][threadID+32];}

}

if(threadID < 16){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+16]){max_min_data[0][threadID] = max_min_data[0][threadID+16];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+16]){max_min_data[1][threadID] = max_min_data[1][threadID+16];}

}

if(threadID < 8){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+8]){max_min_data[0][threadID] = max_min_data[0][threadID+8];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+8]){max_min_data[1][threadID] = max_min_data[1][threadID+8];}

}

if(threadID < 4){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+4]){max_min_data[0][threadID] = max_min_data[0][threadID+4];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+4]){max_min_data[1][threadID] = max_min_data[1][threadID+4];}

}

if(threadID < 2){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+2]){max_min_data[0][threadID] = max_min_data[0][threadID+2];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+2]){max_min_data[1][threadID] = max_min_data[1][threadID+2];}

}

if(threadID < 1){
  if(max_min_data[0][threadID] < max_min_data[0][threadID+1]){max_min_data[0][threadID] = max_min_data[0][threadID+1];}
  if(max_min_data[1][threadID] > max_min_data[1][threadID+1]){max_min_data[1][threadID] = max_min_data[1][threadID+1];}

}

// write result for this block back to global memory
  if (threadID == 0) {
    printf("vlock %d max %d min %d\n", blockIdx.x,(int)max_min_data[0][0],(int)max_min_data[1][0]);
    max[blockIdx.x] = max_min_data[0][0];
    min[blockIdx.x] = max_min_data[1][0];
  }
}